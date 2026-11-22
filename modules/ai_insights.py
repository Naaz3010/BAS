"""
ai_insights.py
AI-powered executive summaries and "Chat with Data" using the Gemini API
(google-genai SDK).

Design notes
------------
* The model never sees the whole dataset. Summaries are built from *aggregates*
  (KPIs, numeric statistics, top categories, monthly trend) plus a tiny sample.
* "Chat with Data" does NOT answer from a sample. The model writes one
  read-only SQL SELECT, we run it against the FULL dataframe in an in-memory
  SQLite database, and the model then explains the real result. The SQL is
  shown to the user so the answer is auditable.
* Falls back to a rule-based summary if no key is configured, so the app stays
  usable offline.

Privacy: prompts include column names, aggregates and a few sample rows. Do not
use real customer/employee data with a third-party API unless that is allowed.

Set GEMINI_MODEL to change the model (default below). Model names are retired
over time, so keep this configurable.
"""

import os
import re
import sqlite3

import numpy as np
import pandas as pd

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:  # SDK not installed
    GEMINI_AVAILABLE = False

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SAMPLE_ROWS = 5


class GeminiModel:
    """Tiny wrapper so the rest of the app depends on one method: generate()."""

    def __init__(self, client, model_name: str):
        self._client = client
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(model=self.model_name, contents=prompt)
        return (response.text or "").strip()


def configure_gemini(api_key: str = None):
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key or not GEMINI_AVAILABLE:
        return None
    try:
        return GeminiModel(genai.Client(api_key=key), DEFAULT_MODEL)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Context building (aggregates, not raw rows)
# ---------------------------------------------------------------------------

def _dataset_context(df: pd.DataFrame, kpis: dict, roles: dict, max_rows: int = SAMPLE_ROWS) -> str:
    parts = [f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns"]

    role_str = "\n".join(f"- {k}: {v}" for k, v in roles.items() if v)
    if role_str:
        parts.append(f"Detected business roles:\n{role_str}")
    if kpis:
        parts.append("Key metrics:\n" + "\n".join(f"- {k}: {v}" for k, v in kpis.items()))

    parts.append("Column dtypes:\n" + df.dtypes.astype(str).to_string())

    num = df.select_dtypes(include=np.number)
    if not num.empty:
        desc = num.describe().T[["mean", "std", "min", "50%", "max"]].round(2)
        parts.append("Numeric column statistics:\n" + desc.head(25).to_string())

    cat_lines = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        n_unique = df[col].nunique()
        if 1 < n_unique <= 30:
            top = df[col].value_counts().head(5)
            cat_lines.append(f"- {col} ({n_unique} values): " + ", ".join(f"{k} ({v})" for k, v in top.items()))
    if cat_lines:
        parts.append("Top categories:\n" + "\n".join(cat_lines[:10]))

    date_col, rev_col = roles.get("date"), roles.get("revenue")
    if (date_col in df.columns and rev_col in df.columns
            and pd.api.types.is_datetime64_any_dtype(df[date_col])
            and pd.api.types.is_numeric_dtype(df[rev_col])):
        monthly = df.set_index(date_col)[rev_col].resample("ME").sum().round(2)
        if len(monthly) > 1:
            parts.append("Monthly totals of " + rev_col + ":\n" + monthly.tail(18).to_string())

    if max_rows:
        parts.append(f"Sample rows (first {max_rows}):\n{df.head(max_rows).to_string()}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------

def generate_executive_summary(df: pd.DataFrame, kpis: dict, roles: dict, model=None) -> str:
    if model is None:
        return _fallback_summary(df, kpis)

    prompt = f"""You are a senior business analyst. Based on the dataset context below, write a
concise executive summary (5-8 bullet points) covering:
1. Overall performance summary
2. Key trends observed
3. Notable opportunities
4. Potential risks
5. 2-3 concrete, actionable recommendations

Context (aggregates computed from the full dataset):
{_dataset_context(df, kpis, roles)}

Write in clear business language suitable for a leadership audience. Only state facts
supported by the context; do not invent numbers. Avoid restating raw numbers without
interpretation."""

    try:
        return model.generate(prompt)
    except Exception as e:
        return f"(AI generation failed: {e})\n\n" + _fallback_summary(df, kpis)


def _fallback_summary(df: pd.DataFrame, kpis: dict) -> str:
    lines = ["**Executive Summary (rule-based, no AI key configured)**", ""]
    lines.append(f"- Dataset contains {kpis.get('total_records', len(df)):,} records and {df.shape[1]} columns.")
    if "total_revenue" in kpis:
        lines.append(f"- Total revenue: {kpis['total_revenue']:,.2f}")
    if "avg_transaction_value" in kpis:
        lines.append(f"- Average transaction value: {kpis['avg_transaction_value']:,.2f}")
    if "unique_customers" in kpis:
        lines.append(f"- Unique customers: {kpis['unique_customers']}")
    if "date_range" in kpis:
        lines.append(f"- Data spans: {kpis['date_range']}")
    missing_pct = df.isna().mean().mean() * 100
    lines.append(f"- Overall missing values: {missing_pct:.1f}%")
    lines.append("- Add a GEMINI_API_KEY in the sidebar to unlock AI-generated narrative insights,")
    lines.append("  trend detection, opportunities, risks, and tailored recommendations.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat with data (text-to-SQL, executed on the full dataframe)
# ---------------------------------------------------------------------------

_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|trigger|load_extension)\b",
    re.IGNORECASE,
)


def validate_sql(sql: str) -> str:
    """Accept exactly one read-only SELECT/WITH statement; raise ValueError otherwise."""
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip().rstrip(";").strip()
    if not sql:
        raise ValueError("empty query")
    if ";" in sql:
        raise ValueError("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise ValueError("only SELECT queries are allowed")
    if _FORBIDDEN_SQL.search(sql):
        raise ValueError("query contains a forbidden keyword")
    return sql


def run_readonly_sql(df: pd.DataFrame, sql: str, max_rows: int = 200) -> pd.DataFrame:
    """Run a validated query against the full dataframe as table `data` (in-memory, query_only)."""
    sql = validate_sql(sql)
    safe = df.copy()
    for col in safe.columns:
        if isinstance(safe[col].dtype, pd.CategoricalDtype) or str(safe[col].dtype).startswith("period"):
            safe[col] = safe[col].astype(str)
    conn = sqlite3.connect(":memory:")
    try:
        safe.to_sql("data", conn, index=False, if_exists="replace")
        conn.execute("PRAGMA query_only = ON")
        return pd.read_sql_query(sql, conn).head(max_rows)
    finally:
        conn.close()


def chat_with_data(question: str, df: pd.DataFrame, roles: dict, model=None, history=None) -> dict:
    """Returns {"answer": str, "sql": str | None, "table": DataFrame | None}."""
    if model is None:
        return {"answer": ("AI chat requires a Gemini API key. Add one in the sidebar to ask "
                           "natural-language questions about your data. In the meantime, use the "
                           "SQL Workspace or Analyze tab."), "sql": None, "table": None}

    schema = "\n".join(f'- "{c}" ({t})' for c, t in df.dtypes.astype(str).items())
    history_str = ""
    if history:
        history_str = "\n".join(f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:])

    sql_prompt = f"""You translate business questions into ONE read-only SQLite query.

The data is in a table named `data` with these columns:
{schema}

Sample rows:
{df.head(SAMPLE_ROWS).to_string()}

Rules:
- Output ONLY the SQL, no explanation, no markdown fences.
- A single SELECT statement. No INSERT/UPDATE/DELETE/DDL.
- Wrap column names containing spaces or capitals in double quotes.
- Use aggregation (SUM, AVG, COUNT, GROUP BY) rather than returning raw rows when possible.
- Add LIMIT 50 unless the question needs an aggregate.
- If the question cannot be answered with this table, output exactly: CANNOT_ANSWER

Previous conversation:
{history_str}

Question: {question}"""

    try:
        raw_sql = model.generate(sql_prompt)
    except Exception as e:
        return {"answer": f"AI request failed: {e}", "sql": None, "table": None}

    if raw_sql.strip().upper().startswith("CANNOT_ANSWER"):
        return {"answer": "I can't answer that from the loaded table. Try rephrasing the question "
                          "or check that the relevant columns exist.", "sql": None, "table": None}

    try:
        sql = validate_sql(raw_sql)
        table = run_readonly_sql(df, sql)
    except Exception as e:
        return {"answer": f"I couldn't run a safe query for that question ({e}). "
                          "Try rephrasing, or use the SQL Workspace tab.",
                "sql": raw_sql, "table": None}

    answer_prompt = f"""A user asked a question about their business data. A query was run on the
FULL dataset and returned the result below. Explain the answer clearly and concisely in plain
language, using only numbers that appear in the result. If the result is empty, say so.

Question: {question}

SQL used:
{sql}

Result ({len(table)} rows shown):
{table.head(50).to_string(index=False)}"""

    try:
        answer = model.generate(answer_prompt)
    except Exception as e:
        answer = f"(Could not generate an explanation: {e}) The raw result is shown below."
    return {"answer": answer, "sql": sql, "table": table}
