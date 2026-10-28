import numpy as np


def score(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if len(a) != len(p) or not len(a) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction vector")
    errors = a - p
    denominator = np.abs(a).sum()
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "wape": float(np.abs(errors).sum() / denominator * 100) if denominator > 0 else None}
