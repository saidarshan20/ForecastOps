from time import perf_counter
import numpy as np
import pandas as pd
from .metrics import score
from forecastops.models.registry import make_model


def backtest(name, series, starts, horizon=14):
    predictions, folds, failures = [], [], []
    for fold, start in enumerate(starts, 1):
        start = pd.Timestamp(start)
        history = series.loc[series.index < start]
        actual = series.loc[(series.index >= start) & (series.index < start + pd.Timedelta(days=horizon))]
        issue_date = start - pd.Timedelta(days=1)
        try:
            if len(actual) != horizon or history.index[-1] != issue_date:
                raise ValueError("Incomplete or nonconsecutive fold")
            t = perf_counter()
            model = make_model(name).fit(history)
            fit_seconds = perf_counter() - t
            t = perf_counter()
            forecast = np.asarray(model.predict(horizon, issue_date), dtype=float)
            predict_seconds = perf_counter() - t
            if len(forecast) != horizon or not np.isfinite(forecast).all() or (forecast < 0).any():
                raise ValueError("Forecast must contain finite nonnegative values for every step")
            issued = [{"model": name, "fold": fold, "issue_date": issue_date.date().isoformat(),
                       "target_date": date.date().isoformat(), "horizon_step": step,
                       "prediction": float(value)} for step, (date, value) in enumerate(zip(actual.index, forecast), 1)]
            # Only after the complete issued vector exists do target values enter scoring rows.
            for row, truth in zip(issued, actual.to_numpy(dtype=float)):
                row["actual"] = float(truth)
                row["absolute_error"] = abs(row["prediction"] - row["actual"])
            predictions.extend(issued)
            folds.append({"model": name, "fold": fold, "issue_date": issue_date.date().isoformat(),
                          **score(actual, forecast), "fit_seconds": fit_seconds, "predict_seconds": predict_seconds})
        except Exception as exc:
            failures.append({"model": name, "fold": fold, "issue_date": issue_date.date().isoformat(),
                             "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(predictions), pd.DataFrame(folds), pd.DataFrame(failures)


def summarize(predictions, folds, names):
    records = []
    for name in names:
        rows = predictions.loc[predictions.model == name] if not predictions.empty else pd.DataFrame()
        timing = folds.loc[folds.model == name] if not folds.empty else pd.DataFrame()
        valid = len(rows) == 70 and len(timing) == 5
        record = {"model": name, "status": "success" if valid else "failed", "forecast_count": len(rows)}
        if valid:
            record.update(score(rows.actual, rows.prediction))
            record.update({"fit_seconds": float(timing.fit_seconds.sum()), "predict_seconds": float(timing.predict_seconds.sum())})
        else:
            record.update({"mae": None, "rmse": None, "wape": None, "fit_seconds": None, "predict_seconds": None})
        records.append(record)
    return pd.DataFrame(records)


def horizon_summary(predictions, split):
    rows = []
    if predictions.empty:
        return pd.DataFrame(columns=["split", "model", "period", "mae", "count"])
    for name, group in predictions.groupby("model"):
        for period, subset in [(str(i), group[group.horizon_step == i]) for i in range(1, 15)] + [
            ("days_1_7", group[group.horizon_step <= 7]), ("days_8_14", group[group.horizon_step > 7])]:
            rows.append({"split": split, "model": name, "period": period,
                         "mae": float(subset.absolute_error.mean()) if len(subset) else None, "count": len(subset)})
    return pd.DataFrame(rows)
