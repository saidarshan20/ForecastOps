import pandas as pd
import pytest
from forecastops.data.loader import load_raw
from forecastops.data.validation import validate_daily


def test_original_contract():
    frame = load_raw()
    assert len(frame) == 731
    assert str(frame.dteday.iloc[0].date()) == "2011-01-01"
    assert str(frame.dteday.iloc[-1].date()) == "2012-12-31"


@pytest.mark.parametrize("change", ["duplicate", "gap", "negative", "fraction", "missing", "nonfinite", "short"])
def test_reject_bad_data(change):
    frame = load_raw()
    frame["cnt"] = frame["cnt"].astype(float)
    if change == "duplicate": frame.loc[1, "dteday"] = frame.loc[0, "dteday"]
    if change == "gap": frame.loc[1, "dteday"] = pd.Timestamp("2013-01-01")
    if change == "negative": frame.loc[1, "cnt"] = -1
    if change == "fraction": frame.loc[1, "cnt"] = 1.5
    if change == "missing": frame.loc[1, "cnt"] = None
    if change == "nonfinite": frame.loc[1, "cnt"] = float("inf")
    if change == "short": frame = frame.iloc[:-1]
    with pytest.raises(ValueError): validate_daily(frame)
