from pathlib import Path
import pandas as pd
from .validation import validate_daily
from forecastops.config import ROOT

RAW = ROOT / "datasets/bike+sharing+dataset/day.csv"
PROCESSED = ROOT / "data/processed/daily_rentals.csv"


def load_raw(path: Path = RAW) -> pd.DataFrame:
    return validate_daily(pd.read_csv(path, usecols=["dteday", "cnt"]))


def load_processed(path: Path = PROCESSED) -> pd.Series:
    frame = validate_daily(pd.read_csv(path))
    return pd.Series(frame["cnt"].to_numpy(dtype=float), index=pd.DatetimeIndex(frame["dteday"]), name="cnt")
