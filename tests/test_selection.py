import pandas as pd
from forecastops.evaluation.selection import choose
from forecastops.models.registry import NAMES


def frame(maes):
    return pd.DataFrame({"model": NAMES, "status": ["success"]*5, "mae": maes})


def test_gate_and_tie():
    assert choose(frame([100,99,110,120,130]))["selected_model"] == "last_value_naive"
    assert choose(frame([100,99.01,110,120,130]))["selected_model"] == "seasonal_naive"
    assert choose(frame([100,90,90,120,130]))["selected_model"] == "last_value_naive"
    assert choose(frame([0,0,1,2,3]))["selected_model"] == "seasonal_naive"


def test_failed_candidate_ineligible():
    data = frame([100,1,110,120,130]); data.loc[1,"status"]="failed"
    assert choose(data)["selected_model"] == "seasonal_naive"


def test_selection_ignores_test_columns():
    data = frame([100,90,110,120,130])
    data["test_mae"] = [1,1000,1,1,1]
    assert choose(data)["selected_model"] == "last_value_naive"
