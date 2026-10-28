import numpy as np


class SeasonalNaive:
    def fit(self, history):
        if len(history) < 7:
            raise ValueError("Seasonal naive needs seven days")
        self.history = history.copy()
        return self

    def predict(self, horizon, origin_date=None):
        values = list(self.history.to_numpy(dtype=float))
        out = []
        for _ in range(horizon):
            value = values[-7]
            out.append(value)
            values.append(value)
        return np.maximum(0, np.array(out, dtype=float))


class LastValueNaive:
    def fit(self, history):
        self.last = float(history.iloc[-1])
        return self

    def predict(self, horizon, origin_date=None):
        return np.repeat(max(0, self.last), horizon).astype(float)
