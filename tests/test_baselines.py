import pandas as pd
from forecastops.models.baselines import SeasonalNaive, LastValueNaive


def test_weekly_recursion():
    h = pd.Series([1,2,3,4,5,6,7], index=pd.date_range("2012-01-01", periods=7))
    assert list(SeasonalNaive().fit(h).predict(14)) == [1,2,3,4,5,6,7]*2
    assert list(LastValueNaive().fit(h).predict(14)) == [7]*14
