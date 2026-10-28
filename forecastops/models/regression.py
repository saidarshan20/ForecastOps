from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from forecastops.features.build import one_row, training_matrix, FEATURES


class RecursiveRegressor:
    def __init__(self, kind):
        self.kind = kind

    def fit(self, history):
        x, y = training_matrix(history)
        self.model = (make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if self.kind == "ridge" else
                      HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=42))
        self.model.fit(x, y)
        self.history = history.copy()
        return self

    def predict(self, horizon, origin_date=None):
        past = self.history.copy()
        if origin_date is not None and pd.Timestamp(origin_date) != past.index[-1]:
            raise ValueError("Origin does not match fitted history")
        out = []
        for step in range(1, horizon + 1):
            date = past.index[-1] + timedelta(days=1)
            row = pd.DataFrame([one_row(past, date)], columns=FEATURES)
            value = max(0.0, float(self.model.predict(row)[0]))
            if not np.isfinite(value):
                raise ValueError("Regressor returned nonfinite forecast")
            out.append(value)
            past.loc[date] = value
        return np.array(out)
