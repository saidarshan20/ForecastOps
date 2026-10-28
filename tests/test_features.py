import pandas as pd
from forecastops.features.build import one_row, training_matrix
from forecastops.models.registry import NAMES, make_model


def history(length=80):
    return pd.Series(range(length), index=pd.date_range("2012-01-01", periods=length), dtype=float)


def test_shifted_lags():
    h = history()
    row = one_row(h, h.index[-1] + pd.Timedelta(days=1))
    assert row["lag_1"] == 79 and row["lag_7"] == 73
    assert row["rolling_mean_7"] == sum(range(73, 80)) / 7
    assert row["rolling_mean_28"] == sum(range(52, 80)) / 28
    x, y = training_matrix(h)
    assert len(x) == len(y) == 52


def test_future_actual_mutation_every_model():
    h = history(100)
    complete = pd.concat([h, pd.Series([12345.0] * 14, index=pd.date_range(h.index[-1] + pd.Timedelta(days=1), periods=14))])
    for name in NAMES:
        first = make_model(name).fit(h).predict(14, h.index[-1])
        second = make_model(name).fit(complete.loc[:h.index[-1]]).predict(14, h.index[-1])
        assert list(first) == list(second)
