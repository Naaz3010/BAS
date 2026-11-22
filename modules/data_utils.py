"""
data_utils.py
Handles data loading (CSV, Excel, JSON, SQLite, SQL), data profiling,
and semantic column mapping so analytics modules work on any dataset.
"""

import io
import os
import re
import sqlite3
import tempfile
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_csv(file) -> pd.DataFrame:
    if hasattr(file, "seek"):
        file.seek(0)
    return pd.read_csv(file)


def load_excel(file, sheet_name=0) -> pd.DataFrame:
    if hasattr(file, "seek"):
        file.seek(0)
    return pd.read_excel(file, sheet_name=sheet_name)


def load_json(file) -> pd.DataFrame:
    if hasattr(file, "seek"):
        file.seek(0)
    return pd.read_json(file)


def load_sqlite(file, table_name=None):
    """Load a table from an uploaded SQLite file. Returns (df, list_of_tables).

    Uses a private temp file (safe for concurrent users and Windows) and a
    quoted table identifier."""
    data = file.getbuffer() if hasattr(file, "getbuffer") else file.read()
    fd, tmp_path = tempfile.mkstemp(suffix=".db")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        conn = sqlite3.connect(tmp_path)
        try:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';", conn)["name"].tolist()
            if table_name is None and tables:
                table_name = tables[0]
            if table_name and table_name not in tables:
                raise ValueError(f"Table '{table_name}' not found in database.")
            safe_name = '"' + str(table_name).replace('"', '""') + '"' if table_name else None
            df = pd.read_sql(f"SELECT * FROM {safe_name}", conn) if safe_name else pd.DataFrame()
        finally:
            conn.close()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return df, tables


def load_from_sql_connection(conn_string: str, query: str) -> pd.DataFrame:
    engine = create_engine(conn_string)
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def get_excel_sheets(file) -> list:
    if hasattr(file, "seek"):
        file.seek(0)
    xls = pd.ExcelFile(file)
    names = xls.sheet_names
    if hasattr(file, "seek"):
        file.seek(0)
    return names


# ---------------------------------------------------------------------------
# SEMANTIC COLUMN MAPPING
# ---------------------------------------------------------------------------

SEMANTIC_KEYWORDS = {
    "date": ["date", "order_date", "transaction_date", "created_at", "timestamp", "invoice_date", "time"],
    "revenue": ["revenue", "sales", "amount", "total", "price", "net_sales", "gross_sales", "total_amount"],
    "quantity": ["qty", "quantity", "units", "units_sold", "count"],
    "customer_id": ["customer_id", "cust_id", "client_id", "customer", "user_id", "buyer_id"],
    "product_id": ["product_id", "sku", "item_id", "product", "item"],
    "product_name": ["product_name", "item_name", "product_title", "description"],
    "category": ["category", "product_category", "segment", "department"],
    "cost": ["cost", "cogs", "unit_cost", "purchase_price"],
    "profit": ["profit", "margin", "net_profit", "gross_profit"],
    "region": ["region", "state", "country", "city", "location"],
    "order_id": ["order_id", "invoice_id", "transaction_id", "order_no"],
}


# Roles that must resolve to a NUMERIC column. Used to prevent text columns
# (e.g. "country") from being fuzzy-matched into a role like "quantity" via a
# stray substring (e.g. "count" inside "country").
NUMERIC_ROLES = {"revenue", "quantity", "cost", "profit"}


def infer_semantic_roles(df: pd.DataFrame) -> dict:
    """
    Map dataframe columns to business roles (revenue, date, customer_id, etc.)
    using fuzzy keyword matching + dtype checks. Returns {role: column_name or None}.
    """
    roles = {role: None for role in SEMANTIC_KEYWORDS}
    lower_cols = {c: c.lower().strip().replace(" ", "_") for c in df.columns}
    numeric_cols = set(df.select_dtypes(include=np.number).columns)

    for role, keywords in SEMANTIC_KEYWORDS.items():
        candidates = numeric_cols if role in NUMERIC_ROLES else set(df.columns)
        best_match = None

        # Pass 1: exact match against the normalized column name
        for col, lc in lower_cols.items():
            if col not in candidates:
                continue
            if any(kw == lc for kw in keywords):
                best_match = col
                break

        # Pass 2: whole-word match only (word boundaries), never a raw substring —
        # this is what previously let "count" match inside "country".
        if best_match is None:
            for col, lc in lower_cols.items():
                if col not in candidates:
                    continue
                if any(re.search(rf"(^|_){re.escape(kw)}($|_)", lc) for kw in keywords):
                    best_match = col
                    break

        roles[role] = best_match

    # dtype-based fallback for date
    if roles["date"] is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                roles["date"] = col
                break

    # NOTE: no 'largest numeric column' fallback for revenue. Guessing produced
    # misleading KPIs on non-sales data (HR, surveys...). Unmapped roles stay None
    # and the user can map them manually in the Upload tab.

    return roles


def try_parse_dates(df: pd.DataFrame, date_col: str, min_success: float = 0.8) -> pd.DataFrame:
    """Convert date_col to datetime only if at least `min_success` of its
    non-null values parse. Leaves numeric columns (e.g. 'Age') untouched."""
    if not (date_col and date_col in df.columns):
        return df
    col = df[date_col]
    if pd.api.types.is_datetime64_any_dtype(col) or pd.api.types.is_numeric_dtype(col):
        return df
    parsed = pd.to_datetime(col, errors="coerce")
    non_null = col.notna().sum()
    if non_null and parsed.notna().sum() / non_null >= min_success:
        df = df.copy()
        df[date_col] = parsed
    return df


# ---------------------------------------------------------------------------
# PROFILING
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    missing = df.isna().sum()
    missing_pct = (missing / max(n_rows, 1) * 100).round(2)
    dup_count = df.duplicated().sum()

    numeric_df = df.select_dtypes(include=np.number)
    stats = numeric_df.describe().T if not numeric_df.empty else pd.DataFrame()

    corr = numeric_df.corr(numeric_only=True) if numeric_df.shape[1] > 1 else pd.DataFrame()

    dtypes = df.dtypes.astype(str)

    col_summary = pd.DataFrame({
        "dtype": dtypes,
        "missing": missing,
        "missing_%": missing_pct,
        "unique": df.nunique(),
    })

    quality_score = 100.0
    quality_score -= min(40, missing_pct.mean())
    quality_score -= min(20, (dup_count / max(n_rows, 1)) * 100)
    quality_score = max(0, round(quality_score, 1))

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicates": int(dup_count),
        "col_summary": col_summary,
        "numeric_stats": stats,
        "correlation": corr,
        "quality_score": quality_score,
    }
