"""
anomaly.py
Anomaly detection for spikes, drops, and unusual transactions.
Uses Z-score, IQR, and Isolation Forest methods.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_zscore_anomalies(df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    mean, std = df[column].mean(), df[column].std()
    if std == 0 or np.isnan(std):
        df["z_score"] = 0.0
        df["is_anomaly"] = False
        df["anomaly_type"] = "Normal"
        return df
    df["z_score"] = (df[column] - mean) / std
    df["is_anomaly"] = df["z_score"].abs() > threshold
    df["anomaly_type"] = np.where(df["z_score"] > threshold, "Spike",
                           np.where(df["z_score"] < -threshold, "Drop", "Normal"))
    return df


def detect_iqr_anomalies(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    df["is_anomaly"] = (df[column] < lower) | (df[column] > upper)
    df["anomaly_type"] = np.where(df[column] > upper, "Spike",
                           np.where(df[column] < lower, "Drop", "Normal"))
    return df


def detect_isolation_forest(df: pd.DataFrame, columns: list, contamination: float = 0.03) -> pd.DataFrame:
    df = df.copy()
    data = df[columns].dropna()
    df["is_anomaly"] = False
    if data.empty or len(data) < 10:
        return df
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(data)
    df.loc[data.index, "is_anomaly"] = preds == -1
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    return df


def normalize_freq(freq: str) -> str:
    """pandas >= 2.2 deprecated 'M'/'Y'/'Q' (removed in 3.0) in favour of 'ME'/'YE'/'QE'."""
    return {"M": "ME", "Y": "YE", "A": "YE", "Q": "QE"}.get(freq, freq)


def time_series_anomalies(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "D", threshold: float = 3.0):
    """Aggregate by time period, then flag spikes/drops against a *trailing* baseline.

    Each point is compared with the mean/std of the PREVIOUS `window` periods
    (the point itself is excluded via shift(1)). Including the point in its own
    baseline caps the achievable z-score at (n-1)/sqrt(n) -- about 2.27 for a
    7-point window -- so a threshold of 3 could never fire.
    """
    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col)
    agg = ts.set_index(date_col)[value_col].resample(normalize_freq(freq)).sum().reset_index()

    window = min(14, max(4, len(agg) // 4))
    base = agg[value_col].shift(1)
    agg["rolling_mean"] = base.rolling(window, min_periods=3).mean()
    agg["rolling_std"] = base.rolling(window, min_periods=3).std()
    agg["z_score"] = (agg[value_col] - agg["rolling_mean"]) / agg["rolling_std"].replace(0, np.nan)
    agg["z_score"] = agg["z_score"].fillna(0.0)
    agg["is_anomaly"] = agg["z_score"].abs() > threshold
    agg["anomaly_type"] = np.where(agg["z_score"] > threshold, "Spike",
                             np.where(agg["z_score"] < -threshold, "Drop", "Normal"))
    return agg
