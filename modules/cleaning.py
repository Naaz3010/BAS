"""
cleaning.py
Data cleaning & transformation operations with an undoable history stack.
Each operation returns a new dataframe; the caller (app.py) manages the
history list in st.session_state so any step can be undone.
"""

import pandas as pd
import numpy as np


def fill_missing(df: pd.DataFrame, column: str, method: str, custom_value=None) -> pd.DataFrame:
    df = df.copy()
    if method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif method == "median":
        df[column] = df[column].fillna(df[column].median())
    elif method == "mode":
        mode_val = df[column].mode()
        if not mode_val.empty:
            df[column] = df[column].fillna(mode_val[0])
    elif method == "custom":
        if pd.api.types.is_numeric_dtype(df[column]):
            custom_value = float(custom_value)   # raises a clear error if not a number
        df[column] = df[column].fillna(custom_value)
    elif method == "ffill":
        df[column] = df[column].ffill()
    elif method == "bfill":
        df[column] = df[column].bfill()
    elif method == "drop_rows":
        df = df.dropna(subset=[column])
    return df


def drop_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def handle_outliers(df: pd.DataFrame, column: str, method: str = "iqr", action: str = "cap") -> pd.DataFrame:
    df = df.copy()
    series = df[column]

    if method == "iqr":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    elif method == "zscore":
        mean, std = series.mean(), series.std()
        lower, upper = mean - 3 * std, mean + 3 * std
    else:
        return df

    if action == "cap":
        df[column] = series.clip(lower, upper)
    elif action == "remove":
        df = df[(series >= lower) & (series <= upper)]
    elif action == "flag":
        df[f"{column}_is_outlier"] = (series < lower) | (series > upper)

    return df.reset_index(drop=True)


def convert_dtype(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    df = df.copy()
    try:
        if new_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif new_type == "numeric":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif new_type == "string":
            df[column] = df[column].astype(str)
        elif new_type == "category":
            df[column] = df[column].astype("category")
    except Exception:
        pass
    return df


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    return df.rename(columns={old_name: new_name})


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=[c for c in columns if c in df.columns])


def apply_transform(df: pd.DataFrame, column: str, transform: str, new_col_name: str = None) -> pd.DataFrame:
    df = df.copy()
    target = new_col_name or f"{column}_{transform}"
    if transform == "log":
        df[target] = np.log1p(df[column].clip(lower=0))
    elif transform == "sqrt":
        df[target] = np.sqrt(df[column].clip(lower=0))
    elif transform == "zscore":
        df[target] = (df[column] - df[column].mean()) / df[column].std()
    elif transform == "minmax":
        df[target] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    elif transform == "abs":
        df[target] = df[column].abs()
    return df


def filter_rows(df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
    if operator == "==":
        return df[df[column] == value]
    elif operator == "!=":
        return df[df[column] != value]
    elif operator == ">":
        return df[df[column] > value]
    elif operator == "<":
        return df[df[column] < value]
    elif operator == ">=":
        return df[df[column] >= value]
    elif operator == "<=":
        return df[df[column] <= value]
    elif operator == "contains":
        return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
    return df
