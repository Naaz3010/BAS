"""
AI Business Analytics Studio
A Python/Streamlit platform that turns raw business data into actionable
insights: Upload -> Profile -> Clean -> Analyze -> AI Insights -> Chat ->
Forecast -> Anomaly Detection -> Customer Analytics -> Product Analytics ->
SQL Workspace -> Report Generator.

Run with:  streamlit run app.py
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from modules import data_utils, cleaning, analytics, forecasting, anomaly, customer, product, ai_insights, reports

load_dotenv()  # loads GEMINI_API_KEY (and others) from a local .env file, if present

st.set_page_config(page_title="AI Business Analytics Studio", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "history": [],          # undo stack of dataframes
        "df": None,
        "roles": {},
        "gemini_model": None,
        "chat_history": [],
        "sql_conn": None,
        "loaded_tables": {},    # name -> df, for SQL workspace joins
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def _merge_roles(old_roles: dict, df: pd.DataFrame) -> dict:
    """Re-infer roles but keep any the user already set, as long as the column still exists."""
    inferred = data_utils.infer_semantic_roles(df)
    return {r: (old_roles.get(r) if old_roles.get(r) in df.columns else inferred[r]) for r in inferred}


def set_df(new_df: pd.DataFrame, push_history: bool = True):
    """push_history=True -> a cleaning step (keeps history + manual role mapping).
    push_history=False -> a fresh dataset load (resets history and derived results)."""
    if push_history and st.session_state.df is not None:
        st.session_state.history.append(st.session_state.df.copy())
        st.session_state.history = st.session_state.history[-20:]  # cap history
        roles = _merge_roles(st.session_state.roles, new_df)
    else:
        roles = data_utils.infer_semantic_roles(new_df)
        st.session_state.history = []
        for k in ("rfm", "product_perf", "abc", "last_summary", "forecast_df", "sql_result", "report_files"):
            st.session_state.pop(k, None)
        st.session_state.chat_history = []
    # Parse the date column once so KPIs, charts and forecasts see real datetimes.
    new_df = data_utils.try_parse_dates(new_df, roles.get("date"))
    st.session_state.df = new_df
    st.session_state.roles = roles


def load_once(kind: str, file, loader):
    """Streamlit re-runs the script on every click while an uploaded file stays in the
    uploader. Without this guard the file was re-loaded on each rerun, silently
    discarding cleaning steps. Load only when the file actually changes."""
    sig = (kind, getattr(file, "name", ""), getattr(file, "size", 0))
    if st.session_state.get("loaded_sig") == sig:
        return False
    try:
        set_df(loader(file), push_history=False)
    except Exception as e:
        st.error(f"Could not read this file: {e}")
        return False
    st.session_state.loaded_sig = sig
    return True


def undo():
    if st.session_state.history:
        prev = st.session_state.history.pop()
        st.session_state.roles = _merge_roles(st.session_state.roles, prev)
        st.session_state.df = prev
        st.success("Reverted to previous step.")
    else:
        st.warning("Nothing to undo.")


@st.cache_data(show_spinner=False)
def cached_profile(df: pd.DataFrame) -> dict:
    return data_utils.profile_dataframe(df)


def date_columns(df: pd.DataFrame) -> list:
    """Columns that are datetimes, or text columns that parse as dates (>=90% of a sample)."""
    import warnings
    cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            cols.append(c)
        elif not pd.api.types.is_numeric_dtype(df[c]):
            sample = df[c].dropna().head(50)
            if len(sample):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ok = pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.9
                if ok:
                    cols.append(c)
    return cols


def numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=np.number).columns.tolist()


ALLOW_SQL_CONNECTION = os.environ.get("ENABLE_SQL_CONNECTION", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

st.sidebar.title("📊 AI Business Analytics Studio")
st.sidebar.caption("Raw Data → Understand → Clean → Analyze → Ask AI → Predict → Detect → Report")

with st.sidebar.expander("🔑 AI Settings (Gemini API)", expanded=False):
    env_key_present = bool(os.environ.get("GEMINI_API_KEY"))
    if env_key_present and st.session_state.gemini_model is None:
        st.session_state.gemini_model = ai_insights.configure_gemini(os.environ.get("GEMINI_API_KEY"))
        if st.session_state.gemini_model:
            st.success("Loaded GEMINI_API_KEY from .env file.")
    api_key = st.text_input("Gemini API Key (overrides .env)", type="password",
                             help="Used for AI Insights & Chat with Data", key="sidebar_gemini_key")
    st.caption("Privacy: AI features send column names, aggregates and a few sample rows to Google's Gemini API. "
               "Don't use sensitive data unless that is permitted.")
    if api_key:
        st.session_state.gemini_model = ai_insights.configure_gemini(api_key)
        if st.session_state.gemini_model:
            st.success("Gemini connected.")

if st.session_state.df is not None:
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Rows:** {st.session_state.df.shape[0]:,}")
    st.sidebar.write(f"**Columns:** {st.session_state.df.shape[1]}")
    if st.sidebar.button("↩️ Undo last change"):
        undo()

TABS = [
    "📤 Upload", "🔍 Profile", "🧹 Clean & Transform", "📊 Analyze",
    "🤖 AI Insights", "💬 Chat with Data", "🔮 Forecast", "🚨 Anomaly Detection",
    "👥 Customer Analytics", "📦 Product Analytics", "🗄️ SQL Workspace", "📄 Report Generator",
]
tabs = st.tabs(TABS)

# ---------------------------------------------------------------------------
# 1. UPLOAD
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("📤 Upload Data")
    sources = ["CSV", "Excel", "JSON", "SQLite"] + (["SQL Connection"] if ALLOW_SQL_CONNECTION else [])
    source = st.radio("Data source", sources, horizontal=True, key="upload_source")

    if source == "CSV":
        file = st.file_uploader("Upload CSV", type=["csv"], key="upload_csv_file")
        if file:
            load_once("csv", file, data_utils.load_csv)

    elif source == "Excel":
        file = st.file_uploader("Upload Excel", type=["xlsx", "xls"], key="upload_excel_file")
        if file:
            try:
                sheets = data_utils.get_excel_sheets(file)
            except Exception as e:
                sheets = []
                st.error(f"Could not read this Excel file: {e}")
            if sheets:
                sheet = st.selectbox("Sheet", sheets, key="upload_excel_sheet")
                # A single-sheet workbook loads automatically; otherwise use the button.
                if len(sheets) == 1:
                    load_once("excel:" + sheet, file, lambda f: data_utils.load_excel(f, sheet))
                elif st.button("Load sheet", key="upload_excel_load_btn"):
                    try:
                        set_df(data_utils.load_excel(file, sheet), push_history=False)
                        st.success(f"Loaded sheet '{sheet}' with {st.session_state.df.shape[0]} rows.")
                    except Exception as e:
                        st.error(f"Could not read sheet '{sheet}': {e}")

    elif source == "JSON":
        file = st.file_uploader("Upload JSON", type=["json"], key="upload_json_file")
        if file:
            load_once("json", file, data_utils.load_json)

    elif source == "SQLite":
        file = st.file_uploader("Upload SQLite .db file", type=["db", "sqlite", "sqlite3"], key="upload_sqlite_file")
        if file:
            try:
                _, tables = data_utils.load_sqlite(file)
            except Exception as e:
                tables = []
                st.error(f"Could not read this database: {e}")
            table = st.selectbox("Table", tables, key="upload_sqlite_table") if tables else None
            if table and st.button("Load table", key="upload_sqlite_load_btn"):
                try:
                    df_loaded, _ = data_utils.load_sqlite(file, table_name=table)
                    set_df(df_loaded, push_history=False)
                    st.success(f"Loaded table '{table}' with {st.session_state.df.shape[0]} rows.")
                except Exception as e:
                    st.error(f"Could not load table '{table}': {e}")

    elif source == "SQL Connection":
        st.warning("Runs the query with the credentials you provide. Use a READ-ONLY database user. "
                   "Needs the matching driver installed (e.g. psycopg2-binary, pymysql).")
        conn_str = st.text_input("SQLAlchemy connection string", type="password",
                                  placeholder="postgresql://user:pass@host:5432/dbname", key="upload_sql_connstr")
        query = st.text_area("SQL query", value="SELECT * FROM your_table LIMIT 1000;", key="upload_sql_query")
        if st.button("Run query", key="upload_sql_run_btn"):
            try:
                df_loaded = data_utils.load_from_sql_connection(conn_str, query)
                set_df(df_loaded, push_history=False)
                st.success(f"Loaded {st.session_state.df.shape[0]} rows.")
            except Exception as e:
                st.error(f"Connection failed: {e}")

    if st.session_state.df is not None:
        st.success(f"Dataset loaded: {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} columns.")
        st.subheader("Preview")
        st.dataframe(st.session_state.df.head(20), width="stretch")

        st.subheader("Semantic Column Mapping (auto-detected)")
        st.caption("Analytics modules use these to work on any dataset, regardless of column names.")
        roles_df = pd.DataFrame(
            [(k, v) for k, v in st.session_state.roles.items()], columns=["Business Role", "Mapped Column"]
        )
        edited_roles = st.data_editor(roles_df, width="stretch", key="roles_editor")
        if st.button("Save role mapping", key="upload_save_roles_btn"):
            cleaned = {}
            for role, col in zip(edited_roles["Business Role"], edited_roles["Mapped Column"]):
                col = None if (col is None or (isinstance(col, float) and np.isnan(col)) or str(col).strip() == "") else str(col).strip()
                if col is not None and col not in st.session_state.df.columns:
                    st.warning(f"'{col}' is not a column in the dataset — ignored for role '{role}'.")
                    col = None
                cleaned[role] = col
            st.session_state.roles = cleaned
            st.session_state.df = data_utils.try_parse_dates(st.session_state.df, cleaned.get("date"))
            st.success("Role mapping updated.")

# ---------------------------------------------------------------------------
# 2. PROFILE
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("🔍 Data Profile")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        profile = cached_profile(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{profile['n_rows']:,}")
        c2.metric("Columns", profile["n_cols"])
        c3.metric("Duplicate Rows", profile["duplicates"])
        c4.metric("Data Quality Score", f"{profile['quality_score']}/100")

        st.subheader("Column Summary")
        st.dataframe(profile["col_summary"], width="stretch")

        if not profile["numeric_stats"].empty:
            st.subheader("Numeric Statistics")
            st.dataframe(profile["numeric_stats"], width="stretch")

        if not profile["correlation"].empty:
            st.subheader("Correlation Matrix")
            fig = analytics.chart_heatmap(profile["correlation"])
            st.plotly_chart(fig, width="stretch")

        st.subheader("Missing Values by Column")
        missing = profile["col_summary"]["missing_%"]
        missing = missing[missing > 0].sort_values(ascending=False)
        if not missing.empty:
            fig = px.bar(missing, title="Missing % by Column")
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("No missing values detected. 🎉")

# ---------------------------------------------------------------------------
# 3. CLEAN & TRANSFORM
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("🧹 Clean & Transform")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        st.caption("Every operation is undoable — use the sidebar 'Undo last change' button.")

        def _apply(fn):
            try:
                set_df(fn())
                st.success("Applied.")
            except Exception as e:
                st.error(f"Could not apply this step: {e}")

        op = st.selectbox("Operation", [
            "Fill missing values", "Drop duplicates", "Handle outliers",
            "Convert data type", "Rename column", "Drop columns",
            "Apply transformation", "Filter rows",
        ], key="clean_op")

        if op == "Fill missing values":
            col = st.selectbox("Column", df.columns, key="clean_fill_col")
            method = st.selectbox("Method", ["mean", "median", "mode", "custom", "ffill", "bfill", "drop_rows"], key="clean_fill_method")
            custom_val = None
            if method == "custom":
                custom_val = st.text_input("Custom fill value", key="clean_fill_custom_val")
            if st.button("Apply", key="clean_fill_apply"):
                _apply(lambda: cleaning.fill_missing(df, col, method, custom_val))

        elif op == "Drop duplicates":
            subset = st.multiselect("Subset columns (optional)", df.columns, key="clean_dedup_subset")
            if st.button("Apply", key="clean_dedup_apply"):
                _apply(lambda: cleaning.drop_duplicates(df, subset or None))

        elif op == "Handle outliers":
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            col = st.selectbox("Numeric column", numeric_cols, key="clean_outlier_col")
            method = st.selectbox("Detection method", ["iqr", "zscore"], key="clean_outlier_method")
            action = st.selectbox("Action", ["cap", "remove", "flag"], key="clean_outlier_action")
            if st.button("Apply", key="clean_outlier_apply"):
                _apply(lambda: cleaning.handle_outliers(df, col, method, action))

        elif op == "Convert data type":
            col = st.selectbox("Column", df.columns, key="clean_dtype_col")
            new_type = st.selectbox("New type", ["datetime", "numeric", "string", "category"], key="clean_dtype_new")
            if st.button("Apply", key="clean_dtype_apply"):
                _apply(lambda: cleaning.convert_dtype(df, col, new_type))

        elif op == "Rename column":
            col = st.selectbox("Column", df.columns, key="clean_rename_col")
            new_name = st.text_input("New name", key="clean_rename_new")
            if st.button("Apply", key="clean_rename_apply") and new_name:
                _apply(lambda: cleaning.rename_column(df, col, new_name))

        elif op == "Drop columns":
            cols = st.multiselect("Columns to drop", df.columns, key="clean_drop_cols")
            if st.button("Apply", key="clean_drop_apply") and cols:
                _apply(lambda: cleaning.drop_columns(df, cols))

        elif op == "Apply transformation":
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            col = st.selectbox("Numeric column", numeric_cols, key="clean_transform_col")
            transform = st.selectbox("Transform", ["log", "sqrt", "zscore", "minmax", "abs"], key="clean_transform_type")
            new_name = st.text_input("New column name (optional)", key="clean_transform_new")
            if st.button("Apply", key="clean_transform_apply"):
                _apply(lambda: cleaning.apply_transform(df, col, transform, new_name or None))

        elif op == "Filter rows":
            col = st.selectbox("Column", df.columns, key="clean_filter_col")
            operator = st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "contains"], key="clean_filter_op")
            value = st.text_input("Value", key="clean_filter_val")
            if st.button("Apply", key="clean_filter_apply") and value != "":
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value
                _apply(lambda: cleaning.filter_rows(df, col, operator, parsed_value))

        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.df.head(20), width="stretch")

# ---------------------------------------------------------------------------
# 4. ANALYZE
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("📊 Analyze")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        roles = st.session_state.roles

        st.subheader("KPI Dashboard")
        kpis = analytics.compute_kpis(df, roles)
        cols = st.columns(min(4, max(1, len(kpis))))
        for i, (k, v) in enumerate(kpis.items()):
            display_val = f"{v:,.2f}" if isinstance(v, float) else v
            cols[i % len(cols)].metric(k.replace("_", " ").title(), display_val)

        st.markdown("---")
        st.subheader("Visualization Builder")
        chart_type = st.selectbox("Chart type", list(analytics.CHART_REGISTRY.keys()), key="analyze_chart_type")
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        try:
            if chart_type in ["Line Chart", "Area Chart"]:
                x = st.selectbox("X axis", all_cols, key="x1")
                y = st.selectbox("Y axis", numeric_cols, key="y1")
                fig = analytics.CHART_REGISTRY[chart_type](df, x, y)
            elif chart_type == "Bar Chart":
                x = st.selectbox("X axis", all_cols, key="x2")
                y = st.selectbox("Y axis", numeric_cols, key="y2")
                fig = analytics.CHART_REGISTRY[chart_type](df, x, y)
            elif chart_type == "Grouped Bar Chart":
                x = st.selectbox("X axis", all_cols, key="x3")
                y = st.selectbox("Y axis", numeric_cols, key="y3")
                color = st.selectbox("Group by", all_cols, key="c3")
                fig = analytics.chart_grouped_bar(df, x, y, color)
            elif chart_type == "Pie Chart":
                names = st.selectbox("Category", all_cols, key="n4")
                values = st.selectbox("Values", numeric_cols, key="v4")
                fig = analytics.chart_pie(df, names, values)
            elif chart_type == "Scatter Plot":
                x = st.selectbox("X axis", numeric_cols, key="x5")
                y = st.selectbox("Y axis", numeric_cols, key="y5")
                color = st.selectbox("Color by (optional)", [None] + all_cols, key="c5")
                fig = analytics.chart_scatter(df, x, y, color)
            elif chart_type == "Histogram":
                x = st.selectbox("Column", numeric_cols, key="x6")
                fig = analytics.chart_histogram(df, x)
            elif chart_type in ["Box Plot", "Violin Plot"]:
                x = st.selectbox("Category (X)", all_cols, key="x7")
                y = st.selectbox("Value (Y)", numeric_cols, key="y7")
                fig = analytics.CHART_REGISTRY[chart_type](df, x, y)
            elif chart_type == "Correlation Heatmap":
                corr = df[numeric_cols].corr()
                fig = analytics.chart_heatmap(corr)
            elif chart_type in ["Treemap", "Sunburst Chart"]:
                path = st.multiselect("Hierarchy path", all_cols, key="p8")
                values = st.selectbox("Values", numeric_cols, key="v8")
                fig = analytics.CHART_REGISTRY[chart_type](df, path, values) if path else None
            elif chart_type == "Funnel Chart":
                x = st.selectbox("Stage values", numeric_cols, key="x9")
                y = st.selectbox("Stage labels", all_cols, key="y9")
                fig = analytics.chart_funnel(df, x, y)
            elif chart_type == "KPI Gauge":
                metric_name = st.selectbox("Metric", list(kpis.keys()), key="gauge_metric") if kpis else None
                val = kpis.get(metric_name, 0) if metric_name else 0
                max_val = st.number_input("Gauge max", value=float(val) * 2 if val else 100.0, key="gauge_max")
                fig = analytics.chart_gauge(val, title=metric_name or "KPI", max_value=max_val)
            else:
                fig = None

            if fig is not None:
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Could not render chart with selected columns: {e}")

# ---------------------------------------------------------------------------
# 5. AI INSIGHTS
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("🤖 AI Insights")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        roles = st.session_state.roles
        kpis = analytics.compute_kpis(df, roles)

        if st.button("Generate Executive Summary", key="ai_insights_generate_btn"):
            with st.spinner("Analyzing dataset..."):
                summary = ai_insights.generate_executive_summary(df, kpis, roles, st.session_state.gemini_model)
                st.session_state.last_summary = summary

        if "last_summary" in st.session_state:
            st.markdown(st.session_state.last_summary)

        if not st.session_state.gemini_model:
            st.info("Add a Gemini API key in the sidebar for full AI-generated narrative insights.")

# ---------------------------------------------------------------------------
# 6. CHAT WITH DATA
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("💬 Chat with Data")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        st.caption("Questions are answered by running a read-only SQL query on your full dataset "
                   "(not just a sample). The query is shown so you can verify it.")
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["q"])
            with st.chat_message("assistant"):
                st.write(turn["a"])
                if turn.get("sql"):
                    with st.expander("SQL used"):
                        st.code(turn["sql"], language="sql")
                if turn.get("table") is not None:
                    st.dataframe(turn["table"], width="stretch")

        question = st.chat_input("Ask a question about your dataset...")
        if question:
            with st.spinner("Thinking..."):
                result = ai_insights.chat_with_data(
                    question, st.session_state.df, st.session_state.roles,
                    st.session_state.gemini_model, st.session_state.chat_history
                )
            st.session_state.chat_history.append({"q": question, "a": result["answer"],
                                                  "sql": result["sql"], "table": result["table"]})
            st.rerun()

# ---------------------------------------------------------------------------
# 7. FORECAST
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("🔮 Forecast")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    elif not date_columns(st.session_state.df) or not numeric_columns(st.session_state.df):
        st.info("Forecasting needs a date column and at least one numeric column. "
                "None was found in this dataset (e.g. HR/survey data has no time dimension).")
    else:
        df = st.session_state.df
        roles = st.session_state.roles
        date_default = roles.get("date")
        rev_default = roles.get("revenue")

        c1, c2, c3 = st.columns(3)
        dcols = date_columns(df)
        date_col = c1.selectbox("Date column", dcols, index=dcols.index(date_default) if date_default in dcols else 0, key="forecast_date_col")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        value_col = c2.selectbox("Value to forecast", numeric_cols, index=numeric_cols.index(rev_default) if rev_default in numeric_cols else 0, key="forecast_value_col")
        model_name = c3.selectbox("Model", list(forecasting.MODEL_REGISTRY.keys()), key="forecast_model")

        c4, c5 = st.columns(2)
        periods = c4.number_input("Periods to forecast", min_value=1, max_value=365, value=30, key="forecast_periods")
        freq = c5.selectbox("Frequency", ["D", "W", "M"], format_func=lambda f: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[f], key="forecast_freq")

        if st.button("Run Forecast", key="forecast_run_btn"):
            try:
                ts, forecast_df, backtest = forecasting.run_forecast(df, date_col, value_col, model_name, int(periods), freq)

                hist_df = ts.reset_index()
                hist_df.columns = ["date", "value"]

                fig = px.line(hist_df, x="date", y="value", title=f"{value_col} — Historical + Forecast ({model_name})")
                fig.add_scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines", name="Forecast")
                fig.add_scatter(x=forecast_df["date"], y=forecast_df["upper_ci"], mode="lines", name="Upper CI", line=dict(dash="dot"))
                fig.add_scatter(x=forecast_df["date"], y=forecast_df["lower_ci"], mode="lines", name="Lower CI", line=dict(dash="dot"))
                st.plotly_chart(fig, width="stretch")

                st.subheader("Forecast Table")
                st.dataframe(forecast_df, width="stretch")

                if backtest:
                    st.subheader("Backtest Accuracy")
                    b1, b2, b3 = st.columns(3)
                    b1.metric("MAE", backtest["mae"])
                    b2.metric("MAPE (%)", backtest["mape"])
                    b3.metric("RMSE", backtest["rmse"])
                else:
                    st.info("Not enough historical data points for a reliable backtest.")

                st.session_state.forecast_df = forecast_df
            except Exception as e:
                st.error(f"Forecast failed: {e}")

# ---------------------------------------------------------------------------
# 8. ANOMALY DETECTION
# ---------------------------------------------------------------------------
with tabs[7]:
    st.header("🚨 Anomaly Detection")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        roles = st.session_state.roles
        method = st.radio("Method", ["Time-series (spikes/drops)", "Z-score (row-level)", "IQR (row-level)", "Isolation Forest (multivariate)"], horizontal=False, key="anomaly_method")

        dcols = date_columns(df)
        if method == "Time-series (spikes/drops)" and (not dcols or not numeric_columns(df)):
            st.info("Time-series detection needs a date column and a numeric column. "
                    "Use the row-level or Isolation Forest methods for this dataset.")
        elif method == "Time-series (spikes/drops)":
            date_col = st.selectbox("Date column", dcols, index=dcols.index(roles.get("date")) if roles.get("date") in dcols else 0, key="anomaly_ts_date_col")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            value_col = st.selectbox("Value column", numeric_cols, key="anomaly_ts_value_col")
            freq = st.selectbox("Aggregation frequency", ["D", "W", "M"], format_func=lambda f: {"D": "Daily", "W": "Weekly", "M": "Monthly"}[f], key="anomaly_ts_freq")
            threshold = st.slider("Sensitivity (z-score threshold)", 1.0, 5.0, 3.0, 0.1, key="anomaly_ts_threshold")
            if st.button("Detect", key="anomaly_ts_detect_btn"):
                agg = anomaly.time_series_anomalies(df, date_col, value_col, freq, threshold)
                fig = px.line(agg, x=date_col, y=value_col, title="Time Series with Anomalies")
                anomalies = agg[agg["is_anomaly"]]
                fig.add_scatter(x=anomalies[date_col], y=anomalies[value_col], mode="markers",
                                 marker=dict(color="red", size=10), name="Anomaly")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(anomalies, width="stretch")

        elif method in ["Z-score (row-level)", "IQR (row-level)"]:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            col = st.selectbox("Column", numeric_cols, key="anomaly_rowlevel_col")
            if method == "Z-score (row-level)":
                threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1, key="anomaly_zscore_threshold")
                if st.button("Detect", key="anomaly_zscore_detect_btn"):
                    result = anomaly.detect_zscore_anomalies(df, col, threshold)
                    st.dataframe(result[result["is_anomaly"]], width="stretch")
                    fig = px.scatter(result, x=result.index, y=col, color="is_anomaly", title="Z-score Anomalies")
                    st.plotly_chart(fig, width="stretch")
            else:
                multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1, key="anomaly_iqr_multiplier")
                if st.button("Detect", key="anomaly_iqr_detect_btn"):
                    result = anomaly.detect_iqr_anomalies(df, col, multiplier)
                    st.dataframe(result[result["is_anomaly"]], width="stretch")
                    fig = px.scatter(result, x=result.index, y=col, color="is_anomaly", title="IQR Anomalies")
                    st.plotly_chart(fig, width="stretch")

        else:  # Isolation Forest
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            cols = st.multiselect("Columns to include", numeric_cols, default=numeric_cols[:3], key="anomaly_iforest_cols")
            contamination = st.slider("Expected anomaly proportion", 0.01, 0.2, 0.03, 0.01, key="anomaly_iforest_contamination")
            if st.button("Detect", key="anomaly_iforest_detect_btn") and cols:
                result = anomaly.detect_isolation_forest(df, cols, contamination)
                st.dataframe(result[result["is_anomaly"]], width="stretch")
                if len(cols) >= 2:
                    fig = px.scatter(result, x=cols[0], y=cols[1], color="is_anomaly", title="Isolation Forest Anomalies")
                    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------------
# 9. CUSTOMER ANALYTICS
# ---------------------------------------------------------------------------
with tabs[8]:
    st.header("👥 Customer Analytics")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    elif not date_columns(st.session_state.df) or not numeric_columns(st.session_state.df):
        st.info("Customer analytics needs a customer ID, a date column and a numeric revenue column. "
                "No date column was found in this dataset.")
    else:
        df = st.session_state.df
        roles = st.session_state.roles

        c1, c2, c3, c4 = st.columns(4)
        cust_col = c1.selectbox("Customer ID column", df.columns, index=list(df.columns).index(roles.get("customer_id")) if roles.get("customer_id") in df.columns else 0, key="cust_customer_col")
        dcols = date_columns(df)
        date_col = c2.selectbox("Date column", dcols, index=dcols.index(roles.get("date")) if roles.get("date") in dcols else 0, key="cust_date_col")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        rev_col = c3.selectbox("Revenue column", numeric_cols, index=numeric_cols.index(roles.get("revenue")) if roles.get("revenue") in numeric_cols else 0, key="cust_revenue_col")
        order_col = c4.selectbox("Order ID column (optional)", [None] + list(df.columns), key="cust_order_col")

        if st.button("Run Customer Analytics", key="cust_run_btn"):
            try:
                rfm = customer.compute_rfm(df, cust_col, date_col, rev_col, order_col)
                rfm = customer.estimate_clv(rfm)
                rfm = customer.churn_risk_score(rfm)
                st.session_state.rfm = rfm

                st.subheader("RFM Segmentation")
                seg_counts = rfm["segment"].value_counts().reset_index()
                seg_counts.columns = ["segment", "count"]
                fig = px.bar(seg_counts, x="segment", y="count", title="Customer Segments")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(rfm, width="stretch")

                st.subheader("Churn Risk Distribution")
                risk_counts = rfm["churn_risk_level"].value_counts().reset_index()
                risk_counts.columns = ["risk_level", "count"]
                fig2 = px.pie(risk_counts, names="risk_level", values="count", title="Churn Risk Levels")
                st.plotly_chart(fig2, width="stretch")

                st.subheader("CLV Distribution")
                fig3 = px.histogram(rfm, x="clv_estimate", nbins=30, title="Estimated Customer Lifetime Value")
                st.plotly_chart(fig3, width="stretch")

                st.subheader("Cohort Retention (%)")
                cohort = customer.cohort_retention(df, cust_col, date_col)
                st.dataframe(cohort.style.background_gradient(cmap="Blues", axis=None), width="stretch")
            except Exception as e:
                st.error(f"Customer analytics failed: {e}")

# ---------------------------------------------------------------------------
# 10. PRODUCT ANALYTICS
# ---------------------------------------------------------------------------
with tabs[9]:
    st.header("📦 Product Analytics")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    elif not numeric_columns(st.session_state.df):
        st.info("Product analytics needs a numeric revenue column.")
    else:
        df = st.session_state.df
        roles = st.session_state.roles

        c1, c2, c3, c4 = st.columns(4)
        prod_col = c1.selectbox("Product column", df.columns, index=list(df.columns).index(roles.get("product_name") or roles.get("product_id")) if (roles.get("product_name") or roles.get("product_id")) in df.columns else 0, key="prod_product_col")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        rev_col = c2.selectbox("Revenue column", numeric_cols, index=numeric_cols.index(roles.get("revenue")) if roles.get("revenue") in numeric_cols else 0, key="prod_revenue_col")
        qty_col = c3.selectbox("Quantity column (optional)", [None] + numeric_cols, key="prod_qty_col")
        cost_col = c4.selectbox("Cost column (optional)", [None] + numeric_cols, key="prod_cost_col")

        if st.button("Run Product Analytics", key="prod_run_btn"):
            try:
                perf = product.product_performance(df, prod_col, rev_col, qty_col, cost_col)
                st.session_state.product_perf = perf

                best, slow = product.best_and_slow_movers(perf, top_n=10)

                st.subheader("Best Sellers")
                fig = px.bar(best, x=prod_col, y="revenue", title="Top 10 Products by Revenue")
                st.plotly_chart(fig, width="stretch")

                st.subheader("Slow Movers")
                st.dataframe(slow, width="stretch")

                if "margin" in perf.columns:
                    st.subheader("Margin Analysis")
                    fig2 = px.scatter(perf, x="revenue", y="margin_pct", size="revenue",
                                       hover_name=prod_col, title="Revenue vs Margin %")
                    st.plotly_chart(fig2, width="stretch")

                st.subheader("ABC Classification")
                abc = product.abc_classification(perf)
                st.session_state.abc = abc
                fig3 = px.bar(abc, x=prod_col, y="revenue", color="abc_class", title="ABC Classification")
                st.plotly_chart(fig3, width="stretch")
                st.dataframe(abc, width="stretch")
            except Exception as e:
                st.error(f"Product analytics failed: {e}")

# ---------------------------------------------------------------------------
# 11. SQL WORKSPACE
# ---------------------------------------------------------------------------
with tabs[10]:
    st.header("🗄️ SQL Workspace")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        st.caption("Your loaded dataset is available as table `data`. Query it with standard SQL (SQLite dialect).")
        query = st.text_area("SQL Query", value="SELECT * FROM data LIMIT 20;", height=120, key="sql_query")
        if st.button("Run Query", key="sql_run_btn"):
            try:
                conn = sqlite3.connect(":memory:")
                st.session_state.df.to_sql("data", conn, index=False, if_exists="replace")
                result = pd.read_sql_query(query, conn)
                st.dataframe(result, width="stretch")
                st.session_state.sql_result = result
                conn.close()
            except Exception as e:
                st.error(f"Query failed: {e}")

# ---------------------------------------------------------------------------
# 12. REPORT GENERATOR
# ---------------------------------------------------------------------------
with tabs[11]:
    st.header("📄 Report Generator")
    if st.session_state.df is None:
        st.info("Upload a dataset first.")
    else:
        df = st.session_state.df
        roles = st.session_state.roles
        kpis = analytics.compute_kpis(df, roles)

        report_title = st.text_input("Report title", value="Business Analytics Report", key="report_title")
        summary_text = st.session_state.get("last_summary") or ai_insights._fallback_summary(df, kpis)
        st.text_area("Summary to include (editable)", value=summary_text, height=150, key="report_summary")

        formats = st.multiselect("Export formats", ["CSV", "Excel", "PDF", "PowerPoint"], default=["PDF"], key="report_formats")

        if st.button("Generate Report", key="report_generate_btn"):
            summary_final = st.session_state.report_summary
            files = {}
            try:
                if "CSV" in formats:
                    files["CSV"] = (reports.export_csv(df), "data_export.csv", "text/csv")

                if "Excel" in formats:
                    sheets = {"Data": df}
                    if "rfm" in st.session_state:
                        sheets["Customer_RFM"] = st.session_state.rfm
                    if "product_perf" in st.session_state:
                        sheets["Product_Performance"] = st.session_state.product_perf
                    files["Excel"] = (reports.export_excel(sheets), "business_report.xlsx",
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                if "PDF" in formats:
                    tables = {}
                    if "product_perf" in st.session_state:
                        tables["Top Products"] = st.session_state.product_perf.head(10)
                    if "rfm" in st.session_state:
                        tables["Customer Segments (sample)"] = st.session_state.rfm.head(10)
                    files["PDF"] = (reports.export_pdf(report_title, kpis, summary_final, tables or None),
                                    "business_report.pdf", "application/pdf")

                if "PowerPoint" in formats:
                    top_products = st.session_state.get("product_perf")
                    files["PowerPoint"] = (
                        reports.export_pptx(report_title, kpis, summary_final,
                                            top_products.head(10) if top_products is not None else None),
                        "business_report.pptx",
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation")
                st.session_state.report_files = files
            except Exception as e:
                st.error(f"Report generation failed: {e}")

        # Rendered OUTSIDE the button block: a download click triggers a rerun, and
        # buttons nested inside `if st.button(...)` would vanish on that rerun.
        for label, (data, fname, mime) in st.session_state.get("report_files", {}).items():
            st.download_button(f"⬇️ Download {label}", data, file_name=fname, mime=mime, key=f"dl_{label}")

st.sidebar.markdown("---")
st.sidebar.caption("AI Business Analytics Studio · Built with Python, Streamlit, Pandas, Plotly & scikit-learn")
