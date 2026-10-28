from datetime import timedelta
import numpy as np
import pandas as pd

FEATURES = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28", "weekday", "month"]


def one_row(history: pd.Series, target_date: pd.Timestamp) -> dict:
    if len(history) < 28 or history.index[-1] != target_date - timedelta(days=1):
        raise ValueError("Need at least 28 consecutive historical days ending before target")
    values = history.to_numpy(dtype=float)
    return {"lag_1": values[-1], "lag_7": values[-7], "lag_14": values[-14], "lag_28": values[-28],
            "rolling_mean_7": float(np.mean(values[-7:])), "rolling_mean_28": float(np.mean(values[-28:])),
            "weekday": target_date.dayofweek, "month": target_date.month}


def training_matrix(history: pd.Series):
    rows = [one_row(history.iloc[:i], history.index[i]) for i in range(28, len(history))]
    return pd.DataFrame(rows, columns=FEATURES), history.iloc[28:].to_numpy(dtype=float)
