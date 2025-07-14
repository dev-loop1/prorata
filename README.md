# Prorata

A powerful, dual-page **Streamlit** web app that enables both **calendar-based** and **holiday-aware** proration of weekly data—ensuring accurate and context-sensitive reporting for any dataset.

* **For lighter use, the app is hosted on Streamlit Server.** Access Link: 

---

##  Core Features

### 🔹 Simple Proration Page (Home)

* **Smart Week Splitting**  
  Automatically separates any week that spans across months into two rows.  
  *Example: A week starting Jan 29th is split into 3 days for January and 4 days for February.*

* **Proportional Value Distribution**  
  Distributes numeric values across split weeks based on calendar day count.

* **Dual-Engine Performance**  
  - **Polars** for ultra-fast, parallel `.csv` processing  
  - **Pandas** for robust `.xlsx` (Excel) file compatibility

* **Built-in Verification**  
  Ensures the total sum of the selected value column remains unchanged after processing.

---

### 🔹 Holiday-Aware Proration Page

* **Intelligent Holiday Impact Modeling**  
  Uses **Prophet** to learn the multiplicative effect of weekends and public holidays from historical data.

* **Weighted Proration**  
  Splits weekly values using a "business effort" score per day—adjusting for holiday and weekend impact.

* **Dynamic Model Management**  
  Train and save multiple models (e.g., per country or business unit) and choose the appropriate one when processing new files.

* **Advanced Visual Analytics**  
  Generates comparison charts to visualize the difference between simple and holiday-aware proration, with toggle options for **monthly** and **weekly** views.

---

##  Technology Stack

* **Framework:** Streamlit  
* **Data Processing:** Pandas, Polars  
* **Forecasting & Modeling:** Prophet (by Meta)  
* **Country & Holiday Data:** PyCountry, Holidays  
* **Dependencies:** `streamlit`, `pandas`, `polars`, `numpy`, `openpyxl`, `prophet`, `pycountry`, `holidays`

---

## Getting Started (Run Locally)

### Prerequisites

* **Python 3.9+** – [Download here](https://www.python.org/downloads/)
* Alternatively, Python can be downloaded from the Microsoft Store [Download here](https://apps.microsoft.com/detail/9PNRBTZXMB4Z?hl=en-us&gl=US&ocid=pdpshare)
* `pip` (comes with Python)

---

### 1. Clone the Repository

```bash
git clone https://github.com/dev-loop1/prorata.git
cd prorata
```

---

### 2. Create a Virtual Environment (Recommended but not Mandatory)

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

### 4. (Optional) Increase File Upload Limit

If you're working with large files (>10 GB), adjust Streamlit’s upload size limit:

1. Go to the folder named `.streamlit` in the project root.
2. Inside it, there is a file called `config.toml` with the following:

```toml
# .streamlit/config.toml
[server]
maxUploadSize = 100000  # Size in MB (You can adjust the size here)
```

---
### 5. Run the Application

```bash
streamlit run app.py
```

This will open the app in your default browser. **Use the left sidebar to switch between Simple Proration and Holiday-Aware Proration pages.**

---

## How to Use the App

### 🔹 Simple Proration (Home Page)

Use this for quick, calendar-day-based splitting:

1. **Upload Your File**
   Supports `.csv` and `.xlsx`.

2. **Configure Columns**
   Select your **start date** column and the **value column** to be prorated.

3. **Choose Date Format**
   Match the exact format used in your file (e.g., `YYYY-MM-DD`, `MM/DD/YYYY`).

4. **Process & Download**
   Click **"Process File"** to split the data and download the results.

---

### 🔹 Holiday-Aware Proration Page

A two-step process for context-rich proration:

#### Step 1: Train a Model

1. **Upload Historical Data**
   Minimum 2 years of weekly data is recommended.

2. **Configure Model Training**

   * Select **date** and **value** columns
   * Choose **training years** from your data
   * Select the **country** to include public holidays
   * Provide a **unique model name**

3. **Train Model**
   Click **"Train Model"**. You may train and manage multiple models.

#### Step 2: Process a New File

1. **Choose a Model**
   Select a trained model from the dropdown.

2. **Upload Your File**
   Upload the weekly dataset to process.

3. **Configure & Process**
   Select the appropriate columns and date format, then click **"Process with Holiday Model"**.

4. **Review & Download**

   * View processed data
   * Analyze the **comparison graph** (Simple vs. Holiday-Aware)
   * Download the finalized result
