"""
analytics.py
KPI computation and the 14 visualization builders used in the Analyze tab.
Each chart function takes a dataframe + column names and returns a Plotly figure.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

def _numeric_or_none(df: pd.DataFrame, col: str):
    """Return the column only if it exists and is actually numeric; else None.
    Prevents crashes when a role gets mapped to a text column (manually or
    via imperfect auto-detection)."""
    if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        return col
    return None


def compute_kpis(df: pd.DataFrame, roles: dict) -> dict:
    kpis = {}
    rev_col = _numeric_or_none(df, roles.get("revenue"))
    qty_col = _numeric_or_none(df, roles.get("quantity"))
    cust_col = roles.get("customer_id")
    order_col = roles.get("order_id")
    date_col = roles.get("date")

    kpis["total_records"] = len(df)

    if rev_col and rev_col in df.columns:
        kpis["total_revenue"] = float(df[rev_col].sum())
        kpis["avg_transaction_value"] = float(df[rev_col].mean())
    if qty_col and qty_col in df.columns:
        kpis["total_units"] = float(df[qty_col].sum())
    if cust_col and cust_col in df.columns:
        kpis["unique_customers"] = int(df[cust_col].nunique())
        if rev_col:
            kpis["revenue_per_customer"] = kpis.get("total_revenue", 0) / max(kpis["unique_customers"], 1)
    if order_col and order_col in df.columns:
        kpis["unique_orders"] = int(df[order_col].nunique())
    if date_col and date_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        kpis["date_range"] = f"{df[date_col].min().date()} to {df[date_col].max().date()}"

    return kpis


# ---------------------------------------------------------------------------
# 14 VISUALIZATION TYPES
# ---------------------------------------------------------------------------

def chart_line(df, x, y, title="Trend Over Time"):
    return px.line(df, x=x, y=y, title=title, markers=True)

def chart_bar(df, x, y, title="Bar Comparison"):
    return px.bar(df, x=x, y=y, title=title)

def chart_grouped_bar(df, x, y, color, title="Grouped Bar"):
    return px.bar(df, x=x, y=y, color=color, barmode="group", title=title)

def chart_pie(df, names, values, title="Distribution"):
    return px.pie(df, names=names, values=values, title=title)

def chart_scatter(df, x, y, color=None, title="Scatter Plot"):
    return px.scatter(df, x=x, y=y, color=color, title=title, trendline=None)

def chart_histogram(df, x, title="Distribution Histogram", nbins=30):
    return px.histogram(df, x=x, nbins=nbins, title=title)

def chart_box(df, x, y, title="Box Plot"):
    return px.box(df, x=x, y=y, title=title)

def chart_heatmap(corr_df, title="Correlation Heatmap"):
    return px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", title=title)

def chart_area(df, x, y, title="Area Chart"):
    return px.area(df, x=x, y=y, title=title)

def chart_treemap(df, path, values, title="Treemap"):
    return px.treemap(df, path=path, values=values, title=title)

def chart_funnel(df, x, y, title="Funnel"):
    return px.funnel(df, x=x, y=y, title=title)

def chart_sunburst(df, path, values, title="Sunburst"):
    return px.sunburst(df, path=path, values=values, title=title)

def chart_violin(df, x, y, title="Violin Plot"):
    return px.violin(df, x=x, y=y, box=True, title=title)

def chart_gauge(value, title="KPI Gauge", max_value=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={"axis": {"range": [0, max_value]}}
    ))
    return fig


CHART_REGISTRY = {
    "Line Chart": chart_line,
    "Bar Chart": chart_bar,
    "Grouped Bar Chart": chart_grouped_bar,
    "Pie Chart": chart_pie,
    "Scatter Plot": chart_scatter,
    "Histogram": chart_histogram,
    "Box Plot": chart_box,
    "Correlation Heatmap": chart_heatmap,
    "Area Chart": chart_area,
    "Treemap": chart_treemap,
    "Funnel Chart": chart_funnel,
    "Sunburst Chart": chart_sunburst,
    "Violin Plot": chart_violin,
    "KPI Gauge": chart_gauge,
}
