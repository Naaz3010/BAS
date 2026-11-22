"""
forecasting.py
Revenue forecasting using multiple models with backtesting and confidence intervals.
Models: Linear Trend, Moving Average, Exponential Smoothing (Holt-Winters).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from modules.anomaly import normalize_freq

SEASONAL_PERIODS = {"D": 7, "W": 52, "ME": 12}


def prepare_series(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "D") -> pd.Series:
    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col]).sort_values(date_col)
    ts = ts.set_index(date_col)[value_col].resample(normalize_freq(freq)).sum()
    return ts


def forecast_linear(ts: pd.Series, periods: int):
    y = ts.values
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    preds = model.predict(future_X)

    residuals = y - model.predict(X)
    std = residuals.std()
    lower = preds - 1.96 * std
    upper = preds + 1.96 * std
    return preds, lower, upper


def forecast_moving_average(ts: pd.Series, periods: int, window: int = 7):
    y = ts.values
    window = min(window, len(y))
    avg = y[-window:].mean()
    std = y[-window:].std()
    preds = np.full(periods, avg)
    lower = preds - 1.96 * std
    upper = preds + 1.96 * std
    return preds, lower, upper


def forecast_holt_winters(ts: pd.Series, periods: int, seasonal_periods: int = None):
    if seasonal_periods is None:
        key = normalize_freq(ts.index.freqstr) if getattr(ts.index, "freqstr", None) else "D"
        key = "ME" if key.startswith("M") else ("W" if key.startswith("W") else "D")
        seasonal_periods = SEASONAL_PERIODS.get(key, 7)
    try:
        model = ExponentialSmoothing(
            ts, trend="add",
            seasonal="add" if len(ts) >= 2 * seasonal_periods else None,
            seasonal_periods=seasonal_periods if len(ts) >= 2 * seasonal_periods else None,
        ).fit()
        preds = model.forecast(periods)
        resid_std = model.resid.std()
        lower = preds - 1.96 * resid_std
        upper = preds + 1.96 * resid_std
        return preds.values, lower.values, upper.values
    except Exception:
        return forecast_linear(ts, periods)


MODEL_REGISTRY = {
    "Linear Trend": forecast_linear,
    "Moving Average": forecast_moving_average,
    "Holt-Winters (Exponential Smoothing)": forecast_holt_winters,
}


def backtest_model(ts: pd.Series, model_fn, test_size: int = 7):
    """Hold out the last `test_size` points, forecast them, and compute error metrics."""
    if len(ts) <= test_size + 3:
        return None
    train, test = ts[:-test_size], ts[-test_size:]
    preds, lower, upper = model_fn(train, test_size)
    actual = test.values
    mae = np.mean(np.abs(actual - preds))
    nonzero = actual != 0
    mape = (np.mean(np.abs((actual[nonzero] - preds[nonzero]) / actual[nonzero])) * 100
            if nonzero.any() else float("nan"))
    rmse = np.sqrt(np.mean((actual - preds) ** 2))
    return {
        "mae": round(mae, 2),
        "mape": round(mape, 2),
        "rmse": round(rmse, 2),
        "test_dates": test.index,
        "actual": actual,
        "predicted": preds,
    }


def run_forecast(df, date_col, value_col, model_name, periods, freq="D"):
    ts = prepare_series(df, date_col, value_col, freq)
    model_fn = MODEL_REGISTRY[model_name]
    preds, lower, upper = model_fn(ts, periods)

    freq = normalize_freq(freq)
    future_index = pd.date_range(ts.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    forecast_df = pd.DataFrame({
        "date": future_index,
        "forecast": preds,
        "lower_ci": lower,
        "upper_ci": upper,
    })

    backtest = backtest_model(ts, model_fn, test_size=min(7, max(2, len(ts) // 5)))

    return ts, forecast_df, backtest
