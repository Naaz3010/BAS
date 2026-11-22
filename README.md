# 📊 AI Business Analytics Studio

### AI-Powered Business Analytics, Forecasting & Automated Reporting Platform

**AI Business Analytics Studio** is an end-to-end Python and Streamlit platform that transforms raw business datasets into **actionable insights, interactive analytics, forecasts, anomaly detection, customer intelligence, product analytics, AI-generated summaries, and executive-ready reports.**

The project combines **Data Analytics + Machine Learning + Generative AI + Automation** into a single workflow instead of treating them as separate notebooks or dashboards.

Live APP : https://ai-business-analytics-studiogit-9kns6gx8xm9grnhnpodnkf.streamlit.app/

---

## 🚀 Project Overview

Traditional dashboards usually require users to manually explore charts and identify what matters.

This project goes further:

```text
Raw Business Data
        ↓
Data Profiling
        ↓
Cleaning & Transformation
        ↓
Business Analytics
        ↓
AI Insights
        ↓
Forecasting
        ↓
Anomaly Detection
        ↓
Customer & Product Intelligence
        ↓
Executive Report / Alert
```

The platform is designed to answer not only:

> **"What happened?"**

but also:

> **"Why does it matter, what might happen next, and what should we investigate?"**

---

## ✨ Key Features

### 📤 Data Ingestion

Supports business datasets through:

* CSV
* Excel
* JSON
* SQLite
* SQL/database-oriented workflows

### 🔍 Automated Data Profiling

Provides:

* Data quality assessment
* Missing-value analysis
* Duplicate detection
* Descriptive statistics
* Correlation analysis
* Data-type inspection

### 🧹 Data Cleaning & Transformation

Includes:

* Missing-value handling
* Duplicate removal
* Outlier handling
* Data-type conversion
* Transformations
* Filtering

Cleaning operations are designed to be reversible within the application workflow.

### 📊 Business Analytics

Generates business KPIs and interactive visualizations including:

* Line charts
* Bar charts
* Grouped bar charts
* Pie charts
* Scatter plots
* Histograms
* Box plots
* Heatmaps
* Area charts
* Treemaps
* Funnel charts
* Sunburst charts
* Violin plots
* Gauge visualizations

### 🤖 AI-Powered Insights

Uses the **Google Gemini API** to generate:

* Executive summaries
* Key trends
* Business opportunities
* Potential risks
* Recommendations

The AI layer works on top of the analytical results rather than replacing the underlying calculations.

### 💬 Chat With Data

Users can ask natural-language questions about their dataset.

Example questions:

```text
Which products generated the highest revenue?

Which month had the strongest sales?

What are the major trends in the dataset?

Are there unusual changes in sales?

Which customer segments appear most valuable?
```

### 🔮 Forecasting

Supports:

* Linear Trend
* Moving Average
* Holt-Winters

Forecast performance can be evaluated using:

* MAE
* MAPE
* RMSE
* Historical backtesting

### 🚨 Anomaly Detection

Multiple approaches are available:

* Z-score
* IQR
* Time-series spike/drop detection
* Isolation Forest

This allows unusual business behavior to be identified automatically.

### 👥 Customer Analytics

Includes:

* RFM segmentation
* Customer Lifetime Value estimation
* Churn-risk scoring
* Cohort retention analysis

### 📦 Product Analytics

Provides:

* Best-selling products
* Slow-moving products
* Revenue analysis
* Margin analysis
* ABC/Pareto classification

### 🗄️ SQL Workspace

Allows loaded datasets to be queried using a SQLite-based SQL workflow.

### 📄 Automated Reporting

Generates reports in:

* PDF
* Excel
* CSV
* PowerPoint

---

# 🧠 Key Technical Feature: Semantic Column Mapping

One of the main design decisions in this project is that the application **does not depend on one fixed dataset schema**.

The system uses semantic column mapping to identify business roles such as:

```text
Revenue
Date
Customer ID
Product
Quantity
Cost
Profit
```

For example, different datasets may use:

```text
revenue
sales
total_sales
net_revenue
amount
```

Instead of requiring every dataset to use the same column names, the system uses a combination of:

* Keyword matching
* Fuzzy matching
* Data types
* Numeric validation
* Business-role heuristics

The resulting role map is then consumed by the analytics modules.

This makes the analytics layer more reusable across different business datasets.

---

# 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │      User / Data     │
                    │      Upload / SQL    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit App      │
                    │       app.py         │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Shared Analytics Layer    │
              │                                 │
              │ Data Utilities                  │
              │ Cleaning                        │
              │ Analytics                       │
              │ Forecasting                     │
              │ Anomaly Detection               │
              │ Customer Analytics              │
              │ Product Analytics               │
              │ AI Insights                     │
              │ Reporting                       │
              └───────────────┬─────────────────┘
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
        Analytics         ML Models        Gemini AI
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Reports / Insights   │
                    │ Forecasts / Alerts   │
                    └──────────────────────┘
```

---

# ⚙️ Automation Pipeline

The project also includes a standalone automation engine:

```text
pipeline.py
```

This separates the interactive application from unattended analytics execution.

### Automated workflow

```text
Data Source
    ↓
Load Data
    ↓
Clean & Transform
    ↓
Anomaly Detection
    ↓
Forecasting
    ↓
AI Executive Summary
    ↓
PDF Report
    ↓
Email / Alert
    ↓
Execution Log
```

The pipeline is designed to:

1. Load configured data
2. Clean and preprocess it
3. Detect anomalies
4. Generate forecasts
5. Evaluate forecasting performance
6. Generate an AI executive summary
7. Generate a PDF report
8. Send reports/alerts through email
9. Maintain execution logging

The same `modules/` logic is shared between `app.py` and `pipeline.py`, reducing duplicated analytics logic.

---

# 🔗 Automation Integrations

The automation pipeline can be used with:

### Windows Task Scheduler

The repository includes:

```text
setup_scheduled_task.bat
```

for scheduled execution.

### n8n

Because `pipeline.py` is a command-line Python script, it can also be integrated into an n8n workflow.

Example:

```text
Cron / Webhook
      ↓
Execute Command
      ↓
pipeline.py
      ↓
Analytics
      ↓
Report / Alert
```

This allows the analytics engine to become part of a larger automated workflow.

---

# 📁 Project Structure

```text
ai-business-analytics-studio/
│
├── app.py
├── pipeline.py
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
│
├── pipeline_config.example.json
├── setup_scheduled_task.bat
│
├── sample_data/
│   └── sample_sales.csv
│
└── modules/
    ├── __init__.py
    ├── data_utils.py
    ├── cleaning.py
    ├── analytics.py
    ├── forecasting.py
    ├── anomaly.py
    ├── customer.py
    ├── product.py
    ├── ai_insights.py
    └── reports.py
```

---

# 🛠️ Tech Stack

| Category             | Technologies                     |
| -------------------- | -------------------------------- |
| **Language**         | Python                           |
| **Application**      | Streamlit                        |
| **Data Processing**  | Pandas, NumPy                    |
| **Visualization**    | Plotly                           |
| **Machine Learning** | Scikit-learn                     |
| **Forecasting**      | Statsmodels                      |
| **Generative AI**    | Google Gemini API                |
| **Database / SQL**   | SQLite, SQLAlchemy               |
| **Reporting**        | ReportLab, OpenPyXL, python-pptx |
| **Deployment**       | Docker                           |
| **Automation**       | Windows Task Scheduler, n8n      |

---

# 💻 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/naaz-719/ai-business-analytics-studio.git
cd ai-business-analytics-studio
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure AI Features

Create a local `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
# optional - model names are retired over time, so this is configurable
GEMINI_MODEL=gemini-2.5-flash
```

A ready-made template is in `.env.example`.
**Never commit `.env` or API credentials to GitHub.**

> **Chat with Data** works by having the model write one read-only SQL query,
> running it on your *full* dataset, and explaining the real result (the SQL is
> shown for verification). **AI Insights** are built from aggregates, not just
> the first rows. Column names, aggregates and a few sample rows are sent to the
> Gemini API, so avoid sensitive data unless that is permitted.

## 5. Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# 🧪 Run the Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests cover role detection on text-heavy data, time-series anomaly detection,
forecasting at all frequencies, CLV/RFM logic, the SQL safety guard, PDF export,
and an end-to-end pipeline run. They run automatically on every push via GitHub Actions.

---

# 🐳 Run With Docker

Build the Docker image:

```bash
docker build -t ai-business-analytics-studio .
```

Run it:

```bash
docker run -p 8501:8501 ai-business-analytics-studio
```

---

# 🧪 Included Sample Dataset

The repository includes a synthetic sales dataset:

```text
sample_data/sample_sales.csv
```

This allows the application to be tested without requiring an external dataset.

---

# 📸 Screenshots

> Add screenshots of the application here. Recommended screenshots:
>
> 1. Main dashboard
> 2. Data profiling / cleaning
> 3. Analytics dashboard
> 4. AI Insights / Chat With Data
> 5. Forecasting / anomaly detection
> 6. Generated report

Example:

```markdown
![Dashboard](screenshots/dashboard.png)

![AI Insights](screenshots/ai-insights.png)

![Forecasting](screenshots/forecasting.png)
```

---

# 📈 Example Business Questions

The platform is designed around practical business questions such as:

**Sales**

* Which products generate the most revenue?
* Which periods have the strongest sales?
* How is revenue changing over time?

**Customers**

* Which customers are most valuable?
* Which customer segments show higher churn risk?
* How does retention change across cohorts?

**Products**

* Which products are best sellers?
* Which products are slow-moving?
* Which products contribute most to overall margin?

**Operations**

* Are there unusual sales spikes or drops?
* What does the next period's demand look like?
* Which areas require further investigation?

---

# ⚠️ Known Limitations

### Semantic Mapping

Column mapping is heuristic and may require manual correction for unusual datasets.

### Data Requirements

Certain analytics require specific fields.

For example:

```text
Customer Analytics
→ customer_id + date + revenue

Forecasting
→ sufficient historical observations

Isolation Forest
→ sufficient complete multivariate data
```

### Database Drivers

External databases may require additional drivers depending on the database system being connected
(e.g. `psycopg2-binary` for PostgreSQL). The **SQL Connection** upload option is hidden by default
because it is unsafe on a public deployment; set `ENABLE_SQL_CONNECTION=1` to enable it locally,
and use a read-only database user.

### Datasets Without Dates or Revenue

Forecasting, time-series anomalies and customer analytics need a date column (and revenue/customer
columns). On datasets without them (e.g. HR or survey data) those tabs explain what is missing
instead of failing; profiling, cleaning, charts, SQL, row-level anomaly detection and reports still work.

### CLV and Churn Risk

CLV is a simple historical-rate projection (average order value × purchases per year × assumed
lifespan), and churn risk is a relative 0–100 ranking within the dataset, not a probability.

### AI Features

AI functionality requires a valid Gemini API key.

The core data analysis functionality does not depend entirely on the AI layer.

---

# 🔐 Security

Sensitive credentials are intentionally excluded from the repository.

Do **not** commit:

```text
.env
API keys
Passwords
Gmail App Passwords
Database credentials
```

Use environment variables (`.env`) for sensitive configuration. `pipeline_config.json`,
`pipeline_outputs/` and `pipeline_log.csv` are git-ignored because they can contain credentials or business data.

The repository's `.gitignore` excludes local environment files and development artifacts.

---

# 🎯 What I Learned / Demonstrated

This project demonstrates practical experience across multiple areas of modern data and AI development:

* End-to-end data analytics application development
* Data cleaning and preprocessing
* Exploratory and business analytics
* Semantic schema mapping
* Statistical analysis
* Machine learning
* Time-series forecasting
* Anomaly detection
* Customer segmentation
* Product analytics
* SQL-based analysis
* Generative AI integration
* Natural-language data interaction
* Automated reporting
* Workflow automation
* Docker-based deployment
* Modular Python architecture

The key engineering focus was to build a **reusable analytics system rather than a single-dataset notebook**.

---

# 🔮 Future Improvements

Potential improvements include:

* Cloud deployment
* Authentication and user management
* Persistent project storage
* Additional database connectors
* Advanced forecasting models
* Automated model selection
* More LLM providers
* RAG-based business knowledge integration
* Real-time data ingestion
* Data-quality monitoring
* Pipeline observability
* API access for external applications

---

# 👩‍💻 Author

## Naaz Mulla

**Computer Science & Engineering — AI & ML**
**Honors in Data Science**

Interested in:

**Data Analytics • Data Science • Artificial Intelligence • Machine Learning • Generative AI • Data Engineering • AI Automation**

### Connect

* GitHub: https://github.com/naaz-719
* LinkedIn: Add your LinkedIn profile URL here
* Portfolio: Add your portfolio URL here

---

# ⭐ Project

If you find the project useful, consider giving the repository a ⭐.

**Repository:**
https://github.com/naaz-719/ai-business-analytics-studio
