"""
product.py
Product Analytics: best sellers, slow movers, margin analysis, ABC classification.
"""

import numpy as np
import pandas as pd


def product_performance(df: pd.DataFrame, product_col: str, revenue_col: str,
                         quantity_col: str = None, cost_col: str = None) -> pd.DataFrame:
    agg = {"revenue": (revenue_col, "sum")}
    if quantity_col and quantity_col in df.columns:
        agg["units_sold"] = (quantity_col, "sum")
    if cost_col and cost_col in df.columns:
        agg["total_cost"] = (cost_col, "sum")

    perf = df.groupby(product_col).agg(**agg).reset_index()

    if "total_cost" in perf.columns:
        perf["margin"] = perf["revenue"] - perf["total_cost"]
        perf["margin_pct"] = (perf["margin"] / perf["revenue"].replace(0, np.nan) * 100).round(2)

    perf = perf.sort_values("revenue", ascending=False).reset_index(drop=True)
    return perf


def best_and_slow_movers(perf_df: pd.DataFrame, top_n: int = 10):
    best = perf_df.sort_values("revenue", ascending=False).head(top_n)
    slow = perf_df.sort_values("revenue", ascending=True).head(top_n)
    return best, slow


def abc_classification(perf_df: pd.DataFrame, value_col: str = "revenue") -> pd.DataFrame:
    """Classic ABC / Pareto classification: A = top 80% cumulative value, B = next 15%, C = last 5%."""
    df = perf_df.sort_values(value_col, ascending=False).reset_index(drop=True).copy()
    total = df[value_col].sum()
    df["cum_value"] = df[value_col].cumsum()
    df["cum_pct"] = df["cum_value"] / total * 100

    def classify(pct):
        if pct <= 80:
            return "A"
        elif pct <= 95:
            return "B"
        else:
            return "C"

    df["abc_class"] = df["cum_pct"].apply(classify)
    return df
