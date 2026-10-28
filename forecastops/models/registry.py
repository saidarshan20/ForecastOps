from .baselines import SeasonalNaive, LastValueNaive
from .statistical import HoltWinters
from .regression import RecursiveRegressor

NAMES = ["seasonal_naive", "last_value_naive", "holt_winters", "ridge", "hist_gradient_boosting"]
PARAMETERS = {
    "seasonal_naive": {"lag_days": 7}, "last_value_naive": {},
    "holt_winters": {"trend": "add", "damped_trend": True, "seasonal": "add", "seasonal_periods": 7, "initialization_method": "estimated"},
    "ridge": {"alpha": 10.0, "scaler": "StandardScaler"},
    "hist_gradient_boosting": {"max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 15, "random_state": 42},
}


def make_model(name):
    if name == "seasonal_naive": return SeasonalNaive()
    if name == "last_value_naive": return LastValueNaive()
    if name == "holt_winters": return HoltWinters()
    if name == "ridge": return RecursiveRegressor("ridge")
    if name == "hist_gradient_boosting": return RecursiveRegressor("hist_gradient_boosting")
    raise ValueError(f"Unknown model: {name}")
