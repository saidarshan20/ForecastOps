import sqlite3
from pathlib import Path
from forecastops.monitoring.replay import run


def test_replay_idempotent(tmp_path):
    path = tmp_path / "monitor.sqlite"
    first = run(path)
    second = run(path)
    assert first["paired_days"] == second["paired_days"] == 70
    assert first["selected_day_ahead_mae"] == second["selected_day_ahead_mae"]
    assert first["records"][26]["selected_rolling_28_mae"] is None
    assert first["records"][27]["selected_rolling_28_mae"] is not None
    with sqlite3.connect(path) as db:
        assert db.execute("select count(*) from forecast_issues").fetchone()[0] == 140
        assert db.execute("select count(*) from actuals").fetchone()[0] == 70
