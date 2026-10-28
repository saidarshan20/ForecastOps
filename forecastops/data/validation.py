import numpy as np
import pandas as pd

START = pd.Timestamp("2011-01-01")
END = pd.Timestamp("2012-12-31")


def validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"dteday", "cnt"}.issubset(frame.columns):
        raise ValueError("Dataset must contain dteday and cnt")
    dates = pd.to_datetime(frame["dteday"], errors="raise", format="%Y-%m-%d")
    values = pd.to_numeric(frame["cnt"], errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be valid and unique")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Targets must be finite and nonmissing")
    if (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Targets must be nonnegative integers")
    clean = pd.DataFrame({"dteday": dates, "cnt": values.astype(int)}).sort_values("dteday").reset_index(drop=True)
    if len(clean) != 731 or not clean["dteday"].equals(pd.Series(pd.date_range(START, END), name="dteday")):
        raise ValueError("Expected exactly 731 consecutive dates from 2011-01-01 to 2012-12-31")
    return clean
