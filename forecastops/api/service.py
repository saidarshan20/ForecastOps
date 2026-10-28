import json
import shutil
from datetime import timedelta
import joblib
import numpy as np
from forecastops.config import ARTIFACT_DIR, DB_PATH, SEED_DB, ROOT


def load_release():
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    payload = joblib.load(ARTIFACT_DIR / "model.joblib")
    history = payload["history"]
    if str(history.index[-1].date()) != manifest["trained_through"]:
        raise ValueError("Artifact history and manifest cutoff disagree")
    if not SEED_DB.exists() or not (ROOT / "reports/monitoring_summary.json").exists():
        raise FileNotFoundError("Monitoring snapshot or summary missing")
    return manifest, payload


def predict(manifest, payload, horizon):
    history = payload["history"]
    values = np.asarray(payload["model"].predict(horizon, history.index[-1]), dtype=float)
    if len(values) != horizon or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Persisted model returned invalid forecast")
    dates = [(history.index[-1] + timedelta(days=i)).date() for i in range(1, horizon + 1)]
    return {"model_name": manifest["model_id"], "model_version": manifest["version"], "mode": "historical_demo",
            "trained_through": history.index[-1].date(), "horizon": horizon,
            "forecast_start": dates[0], "forecast_end": dates[-1],
            "predictions": [{"date": date, "value": float(value)} for date, value in zip(dates, values)],
            "disclaimer": "Historical demonstration; not a forecast for the current date."}


def ensure_runtime_db():
    if not DB_PATH.exists():
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SEED_DB, DB_PATH)
