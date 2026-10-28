import json
import logging
import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from forecastops.config import ROOT
from .schemas import PredictRequest, PredictResponse
from .service import load_release, predict, ensure_runtime_db
from forecastops.config import DB_PATH
from forecastops.monitoring.metrics import summarize_replay

app = FastAPI(title="ForecastOps", description="Historical 2011–2012 bike rental demand forecasting demo")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
log = logging.getLogger(__name__)


@app.get("/health")
def health():
    try:
        load_release()
        ensure_runtime_db()
        return {"status": "ready", "mode": "historical_demo"}
    except Exception as exc:
        log.exception("Readiness failed")
        raise HTTPException(status_code=503, detail="Required release artifact unavailable") from exc


@app.get("/model")
def model_info():
    try:
        manifest, _ = load_release()
        selection = json.loads((ROOT / "reports/selection.json").read_text(encoding="utf-8"))
        return {**manifest, "selection_rule": "validation MAE with 1% gate over seasonal naive",
                "selection_reason": selection["reason"], "validation_improvement_pct": selection["relative_improvement"] * 100,
                "disclaimer": "Historical demonstration; not a forecast for the current date."}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Required release artifact unavailable") from exc


@app.post("/predict", response_model=PredictResponse)
def make_prediction(request: PredictRequest):
    try:
        manifest, payload = load_release()
        return predict(manifest, payload, request.horizon)
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=503, detail="Prediction artifact unavailable") from exc


@app.get("/monitoring")
def monitoring():
    try:
        manifest, _ = load_release()
        ensure_runtime_db()
        import pandas as pd
        with sqlite3.connect(DB_PATH) as db:
            frame = pd.read_sql_query("SELECT f.target_date, f.model_version, ABS(f.prediction-a.actual) AS absolute_error "
                                      "FROM forecast_issues f JOIN actuals a USING(target_date) WHERE f.horizon=1", db)
        result = summarize_replay(frame, manifest["model_id"])
        saved = json.loads((ROOT / "reports/monitoring_summary.json").read_text(encoding="utf-8"))
        result["generated_at"] = saved["generated_at"]
        result["source_checksum"] = saved["source_checksum"]
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Monitoring snapshot unavailable") from exc


@app.get("/comparison")
def comparison():
    import pandas as pd
    try:
        return pd.read_csv(ROOT / "reports/model_comparison.csv").replace({float("nan"): None}).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Comparison report unavailable") from exc


@app.get("/horizons")
def horizons():
    import pandas as pd
    try:
        return pd.read_csv(ROOT / "reports/horizon_errors.csv").replace({float("nan"): None}).to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Horizon report unavailable") from exc


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "web/index.html")
