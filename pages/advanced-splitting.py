import streamlit as st
import pandas as pd
from prophet import Prophet
import holidays
import io
import numpy as np
import os
import plotly.express as px

st.set_page_config(page_title="Holiday-Aware Proration", layout="wide")

# --- Date Format Options ---
DATE_FORMATS = {
    "D-Mon-YY (e.g., 2-Jan-25)": "%-d-%b-%y",
    "DD-Mon-YY (e.g., 02-Jan-25)": "%d-%b-%y",
    "D-Mon-YYYY (e.g., 2-Jan-2025)": "%-d-%b-%Y",
    "DD-Mon-YYYY (e.g., 02-Jan-2025)": "%d-%b-%Y",
    "MM/DD/YY (e.g., 01/31/25)": "%m/%d/%y",
    "MM/DD/YYYY (e.g., 01/31/2025)": "%m/%d/%Y",
    "DD/MM/YY (e.g., 31/01/25)": "%d/%m/%y",
    "DD/MM/YYYY (e.g., 31/01/2025)": "%d/%m/%Y",
    "YYYY-MM-DD (e.g., 2025-01-31)": "%Y-%m-%d",
}

# --- Helper Functions & Processing Logic ---
@st.cache_data
def get_country_code_map():
    """Generates and caches a dictionary mapping full country names to their 2-letter codes."""
    country_map = {}
    holiday_country_codes = holidays.list_supported_countries()
    for code in sorted(holiday_country_codes):
        try:
            country = pycountry.countries.get(alpha_2=code)
            if country:
                country_map[country.name] = code
        except Exception:
            continue
    return country_map

try:
    import pycountry
    COUNTRY_CODES = get_country_code_map()
except ImportError:
    st.error("The 'pycountry' library is not installed. Please install it by running 'pip install pycountry'.")
    COUNTRY_CODES = {"United States": "US"}

def train_holiday_model(df_train, country_code, date_col, value_col):
    """Trains a Prophet model to learn the multiplicative effects of holidays and weekends."""
    df_prophet = df_train[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    model = Prophet(seasonality_mode='multiplicative', holidays_prior_scale=10.0, weekly_seasonality=True)
    model.add_country_holidays(country_name=country_code)
    model.fit(df_prophet)
    effects = {}
    if hasattr(model, 'train_holiday_names') and 'beta' in model.params:
        holiday_params = model.params['beta'][0]
        holiday_names = list(model.train_holiday_names)
        for i, name in enumerate(holiday_names):
            effects[name] = 1 + holiday_params[i]
        last_date = df_prophet['ds'].max()
        days_to_saturday = (5 - last_date.weekday() + 7) % 7
        future_saturday = last_date + pd.Timedelta(days=days_to_saturday)
        future_sunday = future_saturday + pd.Timedelta(days=1)
        weekend_df = pd.DataFrame({'ds': [future_saturday, future_sunday]})
        forecast = model.predict(weekend_df)
        if 'weekly' in forecast.columns:
            effects['saturday'] = 1 + forecast.loc[0, 'weekly']
            effects['sunday'] = 1 + forecast.loc[1, 'weekly']
    return model, effects

def perform_proration(df_process, date_col, value_col, date_format, model=None, holiday_effects=None):
    """Performs proration. If a model is provided, it uses weighted proration."""
    df = df_process.copy()
    df[date_col] = df[date_col].astype(str)
    df['date_obj'] = pd.to_datetime(df[date_col], errors='coerce')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    holiday_dates = model.holidays[['ds', 'holiday']].set_index('ds')['holiday'].to_dict() if model and model.holidays is not None else {}
    new_rows = []
    for index, row in df.iterrows():
        start_date = row['date_obj']
        if pd.isna(start_date):
            new_rows.append(row.to_dict())
            continue
        end_of_week = start_date + pd.Timedelta(days=6)
        if start_date.month == end_of_week.month:
            row['weight'] = 7.0
            new_rows.append(row.to_dict())
            continue
        
        daily_weights = []
        week_dates = [start_date + pd.Timedelta(days=i) for i in range(7)]
        if model and holiday_effects:
            for day in week_dates:
                weight = 1.0 
                holiday_name = holiday_dates.get(pd.Timestamp(day))
                if holiday_name and holiday_name in holiday_effects: weight = holiday_effects[holiday_name]
                elif day.weekday() == 5: weight = holiday_effects.get('saturday', 1.0)
                elif day.weekday() == 6: weight = holiday_effects.get('sunday', 1.0)
                daily_weights.append(weight)
        else:
            daily_weights = [1.0] * 7
            
        total_week_weight = sum(daily_weights) if sum(daily_weights) > 0 else 7
        last_day_of_month = start_date + pd.offsets.MonthEnd(0)
        days_in_first_part = (last_day_of_month - start_date).days + 1
        weight_first_part = sum(daily_weights[:days_in_first_part])
        
        row1 = row.to_dict()
        row1[value_col] = round((row[value_col] / total_week_weight) * weight_first_part, 2) if total_week_weight != 0 else 0
        row1['weight'] = round(weight_first_part, 2)
        new_rows.append(row1)

        row2 = row.to_dict()
        row2['date_obj'] = last_day_of_month + pd.Timedelta(days=1)
        row2[value_col] = row[value_col] - row1[value_col]
        row2['weight'] = round(total_week_weight - weight_first_part, 2)
        new_rows.append(row2)

    processed_df = pd.DataFrame(new_rows)
    processed_df = processed_df.convert_dtypes()
    processed_df['date'] = pd.to_datetime(processed_df['date_obj'])
    processed_df[date_col] = processed_df['date'].dt.strftime(date_format)
    return processed_df.drop(columns=['date_obj'])

# --- Streamlit UI ---
st.title("Holiday-Aware Proration")

if 'trained_models' not in st.session_state: st.session_state.trained_models = {}
if 'result_df' not in st.session_state: st.session_state.result_df = None
if 'simple_result_df' not in st.session_state: st.session_state.simple_result_df = None

with st.container(border=True):
    st.header("Step 1: Train a Holiday Impact Model")
    st.caption("Upload a historical dataset (ideally 2+ years) to teach the app how holidays affect your data.")
    training_file = st.file_uploader("Upload your historical training data", type=["csv", "xlsx"], key="trainer")
    if training_file:
        df_train_full = pd.read_csv(training_file, dtype=str) if training_file.name.endswith('.csv') else pd.read_excel(training_file, dtype=str)
        col1, col2 = st.columns(2)
        with col1: train_date_col = st.selectbox("Select Training Date Column", df_train_full.columns.tolist())
        with col2: train_value_col = st.selectbox("Select Training Value Column", df_train_full.columns.tolist())
        try:
            all_training_dates = pd.to_datetime(df_train_full[train_date_col], errors='coerce')
            available_training_years = sorted(all_training_dates.dt.year.dropna().unique().astype(int))
            selected_training_years = st.multiselect("Select year(s) to train on:", options=available_training_years, default=available_training_years)
        except Exception:
            selected_training_years = []
        
        col3, col4 = st.columns(2)
        with col3:
            country_options = list(COUNTRY_CODES.keys())
            default_index = country_options.index("United States") if "United States" in country_options else 0
            selected_country_name = st.selectbox("Select Country", options=country_options, index=default_index)
            country_code = COUNTRY_CODES[selected_country_name]
        with col4: model_name = st.text_input("Name this Model", f"{selected_country_name} Model")
        
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training '{model_name}'..."):
                try:
                    df_to_train = df_train_full[all_training_dates.dt.year.isin(selected_training_years)].copy()
                    model_object, holiday_effects = train_holiday_model(df_to_train, country_code, train_date_col, train_value_col)
                    st.session_state.trained_models[model_name] = {"model": model_object, "effects": holiday_effects}
                    st.success(f"Model '{model_name}' trained and ready to use!")
                except Exception as e: st.error(f"Model training failed: {e}")

with st.container(border=True):
    st.header("Step 2: Process a New File")
    if not st.session_state.trained_models:
        st.info("No trained models available. Please train a model in Step 1 first.", icon="ℹ️")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_model_name = st.selectbox("Choose a trained model to apply:", list(st.session_state.trained_models.keys()))
        with col2:
            processing_file = st.file_uploader("Upload the file you want to process", type=["csv", "xlsx"], key="processor")

        if processing_file and selected_model_name:
            df_process_full = pd.read_csv(io.BytesIO(processing_file.getvalue()), dtype=str) if processing_file.name.endswith('.csv') else pd.read_excel(io.BytesIO(processing_file.getvalue()), dtype=str)
            process_cols = df_process_full.columns.tolist()
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1: process_date_col = st.selectbox("Select Date Column:", process_cols, key="process_date")
            with p_col2: process_value_col = st.selectbox("Select Value Column:", process_cols, key="process_value")
            with p_col3:
                selected_format_name = st.selectbox("Select Date Format:", list(DATE_FORMATS.keys()), key="process_format")
                process_date_format = DATE_FORMATS[selected_format_name]
            
            if st.button("Process with Holiday Model", use_container_width=True, type="primary"):
                with st.spinner("Applying holiday model and splitting weeks..."):
                    st.session_state.simple_result_df = perform_proration(df_process_full.copy(), process_date_col, process_value_col, process_date_format)
                    trained_model_data = st.session_state.trained_models[selected_model_name]
                    st.session_state.result_df = perform_proration(df_process_full.copy(), process_date_col, process_value_col, process_date_format, trained_model_data['model'], trained_model_data['effects'])
                    st.session_state.comparison_possible = True
                    st.success("File processed successfully!")

if st.session_state.result_df is not None:
    st.divider()
    st.header("Results")
    
    all_result_dates = pd.to_datetime(st.session_state.result_df['date'], errors='coerce')
    available_result_years = sorted(all_result_dates.dt.year.dropna().unique().astype(int))
    selected_result_years = st.multiselect("Filter results by year:", options=available_result_years, default=available_result_years)
    
    display_df = st.session_state.result_df[all_result_dates.dt.year.isin(selected_result_years)]
    
    df_for_display = display_df.drop(columns=['weight', 'date'], errors='ignore')
    st.dataframe(df_for_display, use_container_width=True, hide_index=True)
    
    output_buffer = io.BytesIO()
    base_name, extension = os.path.splitext(processing_file.name)
    output_filename = f"{base_name}_holiday_aware_output{extension}"
    
    if extension == '.csv':
        df_for_display.to_csv(output_buffer, index=False)
        mime = "text/csv"
    else:
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            df_for_display.to_excel(writer, index=False, sheet_name='Processed_Data')
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
    st.download_button(
        label="⬇️ Download Processed File",
        data=output_buffer.getvalue(),
        file_name=output_filename,
        mime=mime,
        use_container_width=True
    )
    st.divider()
    
    if st.session_state.get('comparison_possible', False):
        st.subheader("Proration Comparison Graph")
        granularity = st.radio("Select Graph Granularity:", ("Monthly", "Weekly"), horizontal=True)
        try:
            simple_dates = pd.to_datetime(st.session_state.simple_result_df['date'], errors='coerce')
            df_simple_filtered = st.session_state.simple_result_df[simple_dates.dt.year.isin(selected_result_years)]
            df_holiday_filtered = display_df
            
            freq = 'M' if granularity == "Monthly" else 'W-MON'
            
            holiday_agg = df_holiday_filtered.groupby(pd.Grouper(key='date', freq=freq))[process_value_col].sum().rename("Holiday-Aware Proration")
            simple_agg = df_simple_filtered.groupby(pd.Grouper(key='date', freq=freq))[process_value_col].sum().rename("Simple Proration")
            chart_df = pd.concat([simple_agg, holiday_agg], axis=1).fillna(0)
            
            xaxis_title = "Month" if granularity == "Monthly" else "Week"
            
            xaxis_title = "Month" if granularity == "Monthly" else "Week"
            
            fig = px.line(chart_df, x=chart_df.index, y=chart_df.columns, title="Proration Method Comparison",
                          labels={"value": "Total Value", "variable": "Method"}, 
                          color_discrete_map={"Simple Proration": "blue", "Holiday-Aware Proration": "red"})
            
            fig.update_layout(xaxis_title=xaxis_title)
            
            # Vertical lines for month or week boundaries
            if granularity == "Monthly":
                boundary_dates = chart_df.index.to_period('M').unique().to_timestamp()
            else: # Weekly
                boundary_dates = chart_df.index
                
            for date in boundary_dates:
                fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="grey")

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate comparison graph: {e}")
