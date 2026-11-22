"""
pipeline.py
============
Autonomous BI & Alert Pipeline — runs the core analytics logic from
`modules/` with ZERO manual interaction, on a schedule or trigger.

This is what turns AI Business Analytics Studio from a click-through
dashboard into an automation system: point it at a data source, and it
will clean the data, detect anomalies, forecast revenue, generate an
AI executive summary, build a PDF report, and email it out — completely
unattended.

USAGE
-----
Manual test run:
    python pipeline.py --source sample_data/sample_sales.csv

Scheduled run (Windows Task Scheduler / cron / n8n Execute Command node):
    python pipeline.py --config pipeline_config.json

WHAT IT DOES (in order)
------------------------
1. Load data (CSV or Excel file)
2. Auto-detect business roles (revenue, date, customer_id, etc.)
3. Clean: fill missing values, drop duplicates (outliers are KEPT so anomalies stay visible)
4. Detect anomalies (spikes/drops in revenue over time)
5. Forecast next N days of revenue with backtested accuracy
6. Generate an AI executive summary (Gemini, if key is configured)
7. Build a PDF report
8. Email the report — and if anomalies were found, send a separate
   urgent alert email with the anomaly details inline
9. Log every run to pipeline_log.csv so you have an audit trail

This script has NO Streamlit dependency — it can be triggered by
Windows Task Scheduler, a cron job, an n8n "Execute Command" node,
a FastAPI endpoint, or anything else that can call `python pipeline.py`.
"""

import argparse
import copy
import json
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from modules import data_utils, cleaning, analytics, forecasting, anomaly, ai_insights, reports

load_dotenv()  # loads GEMINI_API_KEY, EMAIL_SENDER, EMAIL_APP_PASSWORD from a local .env file

LOG_PATH = Path(__file__).parent / "pipeline_log.csv"
OUTPUT_DIR = Path(__file__).parent / "pipeline_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# CONFIG LOADING
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "source_path": "sample_data/sample_sales.csv",
    "source_type": "csv",              # csv | excel
    "forecast_periods": 14,
    "forecast_freq": "D",
    "anomaly_zscore_threshold": 3.0,
    "email": {
        "enabled": False,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "",
        "sender_app_password": "",     # use an app password, never your real password
        "recipients": [],
        "alert_recipients": [],        # who gets the urgent anomaly alert (can differ from report list)
    },
    "gemini_api_key": "",
}


BASE_DIR = Path(__file__).parent.resolve()


def _resolve(path_str: str) -> str:
    """Resolve relative paths against the project folder, not the current working
    directory (Task Scheduler / cron start in System32 or /)."""
    p = Path(path_str)
    return str(p if p.is_absolute() else BASE_DIR / p)


def load_config(config_path: str = None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)   # deep copy: don't mutate the shared defaults
    if config_path and not Path(config_path).is_absolute() and not Path(config_path).exists():
        config_path = str(BASE_DIR / config_path)
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = json.load(f)
        config.update(user_config)
        if "email" in user_config:
            config["email"] = {**copy.deepcopy(DEFAULT_CONFIG["email"]), **user_config["email"]}

    # .env values fill in anything not explicitly set in the JSON config,
    # so credentials never need to be committed to pipeline_config.json
    if not config["email"].get("sender_email") and os.environ.get("EMAIL_SENDER"):
        config["email"]["sender_email"] = os.environ["EMAIL_SENDER"]
    if not config["email"].get("sender_app_password") and os.environ.get("EMAIL_APP_PASSWORD"):
        config["email"]["sender_app_password"] = os.environ["EMAIL_APP_PASSWORD"]
    if not config.get("gemini_api_key") and os.environ.get("GEMINI_API_KEY"):
        config["gemini_api_key"] = os.environ["GEMINI_API_KEY"]

    return config


# ---------------------------------------------------------------------------
# EMAIL
# ---------------------------------------------------------------------------

def send_email(config: dict, subject: str, body: str, recipients: list, attachment_path: str = None):
    email_cfg = config["email"]
    if not email_cfg.get("enabled"):
        print(f"[email disabled] Would have sent: '{subject}' to {recipients}")
        return

    if not recipients:
        print(f"[email skipped] No recipients configured for: '{subject}'")
        return

    msg = MIMEMultipart()
    msg["From"] = email_cfg["sender_email"]
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    if attachment_path and Path(attachment_path).exists():
        with open(attachment_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=Path(attachment_path).name)
        part["Content-Disposition"] = f'attachment; filename="{Path(attachment_path).name}"'
        msg.attach(part)

    with smtplib.SMTP(email_cfg["smtp_host"], email_cfg["smtp_port"], timeout=30) as server:
        server.starttls()
        server.login(email_cfg["sender_email"], email_cfg["sender_app_password"])
        server.sendmail(email_cfg["sender_email"], recipients, msg.as_string())

    print(f"Email sent: '{subject}' -> {recipients}")


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

def log_run(status: str, rows: int, anomalies_found: int, notes: str = ""):
    entry = pd.DataFrame([{
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "rows_processed": rows,
        "anomalies_found": anomalies_found,
        "notes": notes,
    }])
    if LOG_PATH.exists():
        entry.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        entry.to_csv(LOG_PATH, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, source_path: str = None):
    source_path = _resolve(source_path or config["source_path"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}\nPipeline run started: {timestamp}\nSource: {source_path}\n{'='*60}")

    # 1. LOAD -----------------------------------------------------------------
    if config["source_type"] == "csv":
        df = pd.read_csv(source_path)
    elif config["source_type"] == "excel":
        df = pd.read_excel(source_path)
    else:
        raise ValueError(f"Unsupported source_type: {config['source_type']}")
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. SEMANTIC ROLE DETECTION -----------------------------------------------
    roles = data_utils.infer_semantic_roles(df)
    date_col, revenue_col = roles.get("date"), roles.get("revenue")
    print(f"Detected roles -> date: {date_col}, revenue: {revenue_col}")

    if date_col:
        df = data_utils.try_parse_dates(df, date_col)

    # 3. CLEAN ------------------------------------------------------------------
    # NOTE: outliers are deliberately NOT capped here. Capping revenue before anomaly
    # detection removes the very spikes we want to alert on, and understates KPIs.
    if revenue_col and df[revenue_col].isna().any():
        df = cleaning.fill_missing(df, revenue_col, "median")
    df = cleaning.drop_duplicates(df)
    print(f"After cleaning: {df.shape[0]} rows")

    # 4. KPIs ---------------------------------------------------------------
    kpis = analytics.compute_kpis(df, roles)

    # 5. ANOMALY DETECTION --------------------------------------------------
    anomalies_found = 0
    anomaly_summary = "No anomalies detected."
    if date_col and revenue_col:
        agg = anomaly.time_series_anomalies(df, date_col, revenue_col, config["forecast_freq"],
                                             config["anomaly_zscore_threshold"])
        flagged = agg[agg["is_anomaly"]]
        anomalies_found = len(flagged)
        if anomalies_found:
            lines = [f"  - {row[date_col].date()}: {row['anomaly_type']} "
                     f"({row[revenue_col]:,.2f}, z={row['z_score']:.2f})" for _, row in flagged.iterrows()]
            anomaly_summary = f"{anomalies_found} anomalies detected:\n" + "\n".join(lines)
    print(anomaly_summary)

    # 6. FORECAST -------------------------------------------------------------
    forecast_df, backtest = None, None
    if date_col and revenue_col:
        try:
            _, forecast_df, backtest = forecasting.run_forecast(
                df, date_col, revenue_col, "Holt-Winters (Exponential Smoothing)",
                config["forecast_periods"], config["forecast_freq"]
            )
            print(f"Forecast generated for next {config['forecast_periods']} periods "
                  f"(backtest MAE: {backtest['mae'] if backtest else 'n/a'})")
        except Exception as e:
            print(f"Forecast skipped: {e}")

    # 7. AI EXECUTIVE SUMMARY ---------------------------------------------------
    gemini_model = ai_insights.configure_gemini(config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY"))
    summary = ai_insights.generate_executive_summary(df, kpis, roles, gemini_model)

    # 8. BUILD PDF REPORT ---------------------------------------------------
    report_title = f"Automated Business Report — {timestamp}"
    extra_tables = {}
    if forecast_df is not None:
        extra_tables["Forecast (next periods)"] = forecast_df.head(10)
    pdf_bytes = reports.export_pdf(report_title, kpis, summary + "\n\n" + anomaly_summary, extra_tables or None)

    pdf_path = OUTPUT_DIR / f"report_{timestamp}.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Report saved: {pdf_path}")

    # 9. EMAIL ----------------------------------------------------------------
    body = (
        f"Automated Business Analytics Report — {timestamp}\n\n"
        f"Rows processed: {df.shape[0]}\n"
        f"Key metrics:\n" + "\n".join(f"  - {k}: {v}" for k, v in kpis.items()) +
        f"\n\n{anomaly_summary}\n\nFull report attached."
    )
    send_email(config, report_title, body, config["email"]["recipients"], str(pdf_path))

    if anomalies_found:
        send_email(
            config,
            f"🚨 URGENT: {anomalies_found} anomalies detected in latest data",
            f"Anomaly alert from automated pipeline run {timestamp}:\n\n{anomaly_summary}",
            config["email"]["alert_recipients"] or config["email"]["recipients"],
        )

    log_run("success", df.shape[0], anomalies_found)
    print(f"\nPipeline run complete: {timestamp}\n{'='*60}\n")
    return {"kpis": kpis, "anomalies_found": anomalies_found, "report_path": str(pdf_path)}


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autonomous BI & Alert Pipeline")
    parser.add_argument("--source", type=str, help="Path to data file (overrides config)")
    parser.add_argument("--config", type=str, default="pipeline_config.json", help="Path to JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)

    try:
        run_pipeline(config, source_path=args.source)
    except Exception as e:
        print(f"PIPELINE FAILED: {e}")
        traceback.print_exc()
        log_run("failed", 0, 0, notes=str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
