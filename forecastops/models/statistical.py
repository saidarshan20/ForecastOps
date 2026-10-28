import numpy as np


class HoltWinters:
    def fit(self, history):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        self.fitted = ExponentialSmoothing(history.to_numpy(dtype=float), trend="add", damped_trend=True,
                                           seasonal="add", seasonal_periods=7,
                                           initialization_method="estimated").fit(optimized=True)
        return self

    def predict(self, horizon, origin_date=None):
        result = np.maximum(0, np.asarray(self.fitted.forecast(horizon), dtype=float))
        if not np.isfinite(result).all():
            raise ValueError("Holt-Winters returned nonfinite forecast")
        return result
