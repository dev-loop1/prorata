import streamlit as st
import pandas as pd
import polars as pl
from datetime import timedelta
import io
import numpy as np
import os
import platform

# --- Page Configuration ---
st.set_page_config(
    page_title="Fiscal Week Converter",
    page_icon="🗓️",
    layout="wide"
)

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

# --- Data Processing ---
def process_data_with_polars(source_file, date_col, value_col, date_format):
    lf = pl.scan_csv(source_file, infer_schema_length=0).with_row_index(name="_row_index")
    original_cols = lf.collect_schema().names()
    lf = lf.with_columns(
        pl.col(date_col).str.to_date(format=date_format, strict=True).alias("date_obj"),
        pl.col(value_col).cast(pl.Float64, strict=False)
    )
    split_mask = ((pl.col("date_obj").is_not_null()) & (pl.col("date_obj").dt.day() >= 23) & (pl.col("date_obj").dt.month() != (pl.col("date_obj") + pl.duration(days=6)).dt.month()))
    lf_to_split = lf.filter(split_mask)
    lf_no_split = lf.filter(~split_mask)
    lf_split_calculated = lf_to_split.with_columns(pl.col("date_obj").dt.month_end().alias("last_day_of_month"),).with_columns(((pl.col("last_day_of_month") - pl.col("date_obj")).dt.total_days().cast(pl.Int8) + 1).alias("days_in_first_week"),).with_columns(first_week_value=(pl.col(value_col) / 7 * pl.col("days_in_first_week")).round(2),).with_columns(second_week_value=(pl.col(value_col) - pl.col("first_week_value")))
    lf_split1 = lf_split_calculated.with_columns(pl.col("first_week_value").alias(value_col))
    lf_split2 = lf_split_calculated.with_columns((pl.col("last_day_of_month") + pl.duration(days=1)).alias("date_obj"),pl.col("second_week_value").alias(value_col),)
    cols_to_keep_for_concat = lf.collect_schema().names()
    final_lf = pl.concat([lf_no_split, lf_split1.select(cols_to_keep_for_concat), lf_split2.select(cols_to_keep_for_concat)]).sort("_row_index")
    final_lf = final_lf.with_columns(pl.col("date_obj").dt.strftime(date_format).alias("Time.[Partial Week]")).drop("date_obj", "_row_index")
    final_col_order = ["Time.[Partial Week]" if col == date_col else col for col in original_cols if col != "_row_index"]
    return final_lf.select(final_col_order).collect()

def process_data_with_pandas(df, date_col, value_col, date_format):
    safe_date_format = date_format
    if platform.system() == "Windows" and '%-d' in date_format:
        safe_date_format = date_format.replace('%-d', '%#d')
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    start_dates = pd.to_datetime(df[date_col], errors='coerce')
    actual_split_mask = (start_dates.notna()) & (start_dates.dt.day >= 23) & (start_dates.dt.month != (start_dates + pd.Timedelta(days=6)).dt.month)
    df_no_split = df[~actual_split_mask].copy()
    df_to_split = df[actual_split_mask].copy()
    if df_to_split.empty:
        if not (platform.system() == "Windows" and '%-d' in date_format):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime(date_format)
        return df.rename(columns={date_col: "Time.[Partial Week]"})
    split_dates = pd.to_datetime(df_to_split[date_col], errors='coerce')
    last_day_of_month = split_dates + pd.offsets.MonthEnd(0)
    days_in_first_week = (last_day_of_month - split_dates).dt.days + 1
    first_week_value = round((df_to_split[value_col] / 7) * days_in_first_week, 2)
    second_week_value = df_to_split[value_col] - first_week_value
    df_split1 = df_to_split.copy()
    df_split1[value_col] = first_week_value
    df_split1['_split_order'] = 1
    df_split2 = df_to_split.copy()
    df_split2[date_col] = (last_day_of_month + pd.Timedelta(days=1))
    df_split2[value_col] = second_week_value
    df_split2['_split_order'] = 2
    df_no_split['_split_order'] = 0
    result_df = pd.concat([df_no_split, df_split1, df_split2]).sort_index(kind='mergesort').drop(columns=['_split_order'])
    temp_dates = pd.to_datetime(result_df[date_col], errors='coerce')
    if platform.system() == "Windows" and '%-d' in date_format:
        day = temp_dates.dt.day.astype(str)
        month = temp_dates.dt.strftime('%b')
        year = temp_dates.dt.strftime('%y') if '%y' in date_format and '%Y' not in date_format else temp_dates.dt.strftime('%Y')
        delimiter = '-' if '-' in date_format else '/'
        result_df[date_col] = day + delimiter + month + delimiter + year
    else:
        result_df[date_col] = temp_dates.dt.strftime(safe_date_format)
    return result_df.rename(columns={date_col: "Time.[Partial Week]"})

# --- Streamlit UI ---

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None

# --- Hero Section ---
st.markdown("<h1 style='text-align: center;'>Fiscal Week Converter</h1>", unsafe_allow_html=True)

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your CSV or Excel file to begin",
    type=["csv", "xlsx"],
    key="file_uploader",
    label_visibility="collapsed"
)

if uploaded_file:
    # --- Configuration Section ---
    st.divider()
    
    st.subheader("Data Preview")
    file_extension = uploaded_file.name.split('.')[-1].lower()
    df_preview = pd.read_csv(uploaded_file, nrows=5, dtype=str) if file_extension == 'csv' else pd.read_excel(uploaded_file, nrows=5, dtype=str)
    st.dataframe(df_preview, use_container_width=True, hide_index=True)
    uploaded_file.seek(0)
    
    column_names = df_preview.columns.tolist()
    col1, col2, col3 = st.columns(3)
    with col1:
        date_column = st.selectbox("Select the fiscal week starting date column:", column_names)
    with col2:
        value_column = st.selectbox("Select the value column to split:", column_names)
    with col3:
        selected_format_name = st.selectbox("Select the date format that matches your file:", list(DATE_FORMATS.keys()))
    
    st.write("")
    if st.button("Process File", type="primary", use_container_width=True):
        date_format_code = DATE_FORMATS[selected_format_name]
        with st.spinner("Processing file... Please wait."):
            try:
                if file_extension == 'csv':
                    result_df = process_data_with_polars(uploaded_file, date_column, value_column, date_format_code).to_pandas()
                    uploaded_file.seek(0)
                    original_sum = pl.read_csv(uploaded_file, ignore_errors=True)[value_column].sum()
                else: # Excel
                    df = pd.read_excel(uploaded_file, dtype=str)
                    result_df = process_data_with_pandas(df, date_column, value_column, date_format_code)
                    original_sum = pd.to_numeric(df[value_column], errors='coerce').sum()
                
                processed_sum = result_df[value_column].sum()
                st.session_state.result = result_df
                st.session_state.sums = (original_sum, processed_sum)

            except Exception as e:
                st.error(f"An error occurred. Please check your column and format selections. Details: {e}", icon="🚨")
                # Clear previous results on error
                st.session_state.result = None


# --- Results Section ---
if st.session_state.result is not None:
    st.divider()
    st.subheader("Results")
    
    tab1, tab2 = st.tabs(["**Processed Data Preview**", "**Verification**"])

    with tab1:
        st.dataframe(st.session_state.result.head(), use_container_width=True, hide_index=True)
        
        # --- Spinner for download button preparation ---
        with st.spinner("Getting file ready for download..."):
            output_buffer = io.BytesIO()
            base_name, extension = os.path.splitext(uploaded_file.name)
            output_filename = f"{base_name}_partial_week_output{extension}"

            if uploaded_file.name.endswith('.csv'):
                 st.session_state.result.to_csv(output_buffer, index=False)
                 mime = "text/csv"
            else:
                 with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                     st.session_state.result.to_excel(writer, index=False, sheet_name='Processed_Data')
                 mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
            st.download_button(
                label="⬇️ Download Processed File",
                data=output_buffer.getvalue(),
                file_name=output_filename,
                mime=mime,
                use_container_width=True
            )

    with tab2:
        st.caption("This is checked as a cautionary step to avoid any mismatch of the values.")
        original_sum, processed_sum = st.session_state.sums
        
        col1, col2 = st.columns(2)
        col1.metric("Sum of Original Values", f"{original_sum:,.2f}")
        col2.metric("Sum of Processed Values", f"{processed_sum:,.2f}")

        if np.isclose(original_sum, processed_sum):
            st.success("Totals match!")
        else:
            st.warning("Totals do not match.")
