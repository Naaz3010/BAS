"""
customer.py
Customer Analytics: RFM segmentation, CLV estimation, churn-risk scoring,
and cohort retention analysis.
"""

import numpy as np
import pandas as pd


def compute_rfm(df: pd.DataFrame, customer_col: str, date_col: str, revenue_col: str,
                 order_col: str = None, snapshot_date=None) -> pd.DataFrame:
    use_order_col = bool(order_col) and order_col in df.columns and order_col not in (customer_col, date_col, revenue_col)
    cols = [customer_col, date_col, revenue_col] + ([order_col] if use_order_col else [])
    data = df[cols].dropna(subset=[customer_col, date_col, revenue_col]).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    if snapshot_date is None:
        snapshot_date = data[date_col].max() + pd.Timedelta(days=1)

    freq_col = order_col if use_order_col else date_col
    freq_agg = "nunique" if use_order_col else "count"   # distinct orders, not line items
    rfm = data.groupby(customer_col).agg(
        recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        tenure_days=(date_col, lambda x: (snapshot_date - x.min()).days),
        frequency=(freq_col, freq_agg),
        monetary=(revenue_col, "sum"),
    ).reset_index()

    n_bins = min(5, rfm[customer_col].nunique())
    n_bins = max(n_bins, 2)

    def safe_qcut(series, labels_desc=False):
        labels = list(range(n_bins, 0, -1)) if labels_desc else list(range(1, n_bins + 1))
        try:
            return pd.qcut(series.rank(method="first"), n_bins, labels=labels).astype(int)
        except ValueError:
            return pd.Series([labels[len(labels) // 2]] * len(series), index=series.index)

    rfm["R_score"] = safe_qcut(rfm["recency"], labels_desc=True)
    rfm["F_score"] = safe_qcut(rfm["frequency"], labels_desc=False)
    rfm["M_score"] = safe_qcut(rfm["monetary"], labels_desc=False)
    rfm["RFM_score"] = rfm["R_score"].astype(str) + rfm["F_score"].astype(str) + rfm["M_score"].astype(str)
    rfm["RFM_total"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    def segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New Customers"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Lost"
        else:
            return "Needs Attention"

    rfm["segment"] = rfm.apply(segment, axis=1)
    return rfm


def estimate_clv(rfm_df: pd.DataFrame, avg_lifespan_years: float = 3.0) -> pd.DataFrame:
    """Simple historical-rate CLV projection:

        CLV = avg order value x purchases per year x assumed lifespan (years)

    Purchases per year = frequency / observation window, where the window is the
    span of the whole dataset (same for every customer). The previous version
    divided by *recency*, which inflated CLV for anyone who bought recently.
    This is a heuristic projection, not a probabilistic model (e.g. BG/NBD).
    """
    df = rfm_df.copy()
    avg_order_value = df["monetary"] / df["frequency"].replace(0, 1)
    window_days = df["tenure_days"].max() if "tenure_days" in df.columns else 365
    window_years = max(float(window_days), 30.0) / 365.0
    purchases_per_year = df["frequency"] / window_years
    df["clv_estimate"] = (avg_order_value * purchases_per_year * avg_lifespan_years).round(2)
    return df


def churn_risk_score(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Relative churn-risk score (0-100): higher recency + lower frequency => higher.
    Min-max scaled within this dataset, so it ranks customers; it is not a probability."""
    df = rfm_df.copy()
    recency_norm = (df["recency"] - df["recency"].min()) / (df["recency"].max() - df["recency"].min() + 1e-9)
    freq_norm = 1 - (df["frequency"] - df["frequency"].min()) / (df["frequency"].max() - df["frequency"].min() + 1e-9)
    df["churn_risk_score"] = ((0.6 * recency_norm + 0.4 * freq_norm) * 100).round(1)
    df["churn_risk_level"] = pd.cut(
        df["churn_risk_score"], bins=[-1, 33, 66, 101], labels=["Low", "Medium", "High"]
    )
    return df


def cohort_retention(df: pd.DataFrame, customer_col: str, date_col: str) -> pd.DataFrame:
    data = df[[customer_col, date_col]].dropna().copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    data["order_period"] = data[date_col].dt.to_period("M")
    data["cohort"] = data.groupby(customer_col)[date_col].transform("min").dt.to_period("M")

    cohort_data = data.groupby(["cohort", "order_period"])[customer_col].nunique().reset_index()
    cohort_data["period_number"] = (cohort_data["order_period"] - cohort_data["cohort"]).apply(lambda x: x.n)

    cohort_pivot = cohort_data.pivot_table(index="cohort", columns="period_number", values=customer_col)
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0).round(3) * 100

    retention.index = retention.index.astype(str)
    return retention
