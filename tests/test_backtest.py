import pandas as pd
from forecastops.data.loader import load_processed
from forecastops.evaluation.backtest import backtest
from forecastops.evaluation.metrics import score


def test_boundaries_and_folds():
    s = load_processed()
    assert len(s.iloc[:591]) == 591 and len(s.iloc[591:661]) == len(s.iloc[661:]) == 70
    assert s.index[590] < s.index[591] < s.index[660] < s.index[661]
    starts = pd.date_range("2012-08-14", periods=5, freq="14D")
    p, f, failed = backtest("seasonal_naive", s.loc[:"2012-10-22"], starts)
    assert failed.empty and len(p) == 70 and len(f) == 5
    assert p.target_date.nunique() == 70 and p.groupby("fold").horizon_step.max().eq(14).all()
    assert (pd.to_datetime(p.issue_date) < pd.to_datetime(p.target_date)).all()


def test_metrics_zero_denominator():
    assert score([0,0], [1,2])["wape"] is None
    assert score([1,3], [2,1])["mae"] == 1.5


def test_future_mutation_does_not_change_issued_forecast():
    series = load_processed()
    start = pd.Timestamp("2012-08-14")
    for name in ["seasonal_naive", "last_value_naive", "holt_winters", "ridge", "hist_gradient_boosting"]:
        before, _, _ = backtest(name, series.loc[:"2012-08-27"], [start])
        mutated = series.copy()
        mutated.loc[start:] = 1_000_000
        after, _, _ = backtest(name, mutated.loc[:"2012-08-27"], [start])
        assert before.prediction.tolist() == after.prediction.tolist()
