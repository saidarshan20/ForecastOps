import json
import hashlib
import shutil
import sqlite3
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
    selection_path = ROOT / "reports/selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (hashlib.sha256(selection_path.read_bytes()).hexdigest() != manifest["selection_sha256"]
            or selection["selected_model"] != manifest["model_id"]
            or selection["parameters"] != manifest["parameters"]
            or manifest["mode"] != "historical_demo"
            or manifest["trained_through"] != "2012-10-22"
            or len(history) != manifest["training_rows"]):
        raise ValueError("Release provenance or historical contract mismatch")
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


def read_monitoring(selected):
    import pandas as pd
    from forecastops.monitoring.metrics import summarize_replay
    ensure_runtime_db()
    with sqlite3.connect(f"{DB_PATH.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Corrupt runtime database")
        saved = json.loads((ROOT / "reports/monitoring_summary.json").read_text(encoding="utf-8"))
        runs = db.execute("SELECT source_checksum FROM monitoring_runs WHERE status='complete'").fetchall()
        if not runs or any(row[0] != saved["source_checksum"] for row in runs):
            raise ValueError("Replay provenance mismatch")
        if db.execute("SELECT COUNT(*) FROM forecast_issues").fetchone()[0] != 140 or db.execute("SELECT COUNT(*) FROM actuals").fetchone()[0] != 70:
            raise ValueError("Incomplete replay counts")
        frame = pd.read_sql_query(
            "SELECT f.target_date, f.model_version, ABS(f.prediction-a.actual) AS absolute_error "
            "FROM forecast_issues f JOIN actuals a USING(target_date) WHERE f.horizon=1", db)
    result = summarize_replay(frame, selected)
    if result["start"] != "2012-10-23" or result["end"] != "2012-12-31":
        raise ValueError("Unexpected replay dates")
    for field in ["selected_day_ahead_mae", "baseline_day_ahead_mae", "latest_selected_rolling_28_mae", "latest_baseline_rolling_28_mae"]:
        if not np.isfinite(result[field]) or not np.isclose(result[field], saved[field], rtol=1e-10, atol=1e-7):
            raise ValueError("Replay metrics disagree with snapshot")
    return {**result, "generated_at": saved["generated_at"], "source_checksum": saved["source_checksum"]}


def read_report(path, required_columns, expected_rows):
    import pandas as pd
    frame = pd.read_csv(path)
    if not set(required_columns) <= set(frame.columns) or len(frame) != expected_rows:
        raise ValueError("Incomplete dashboard report")
    for column in required_columns:
        if frame[column].isna().any():
            raise ValueError("Missing report values")
        if column in ["mae", "mae_validation", "mae_test", "rmse_test", "wape_test", "count"]:
            numeric = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(numeric).all() or (numeric < 0).any():
                raise ValueError("Invalid report values")
    return frame.replace({float("nan"): None}).to_dict(orient="records")
