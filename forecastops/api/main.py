import json
import logging
import math
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from forecastops.config import ROOT
from .schemas import PredictRequest, PredictResponse
from .service import load_release, predict, read_monitoring, read_report

app = FastAPI(title="ForecastOps", description="Historical 2011–2012 bike rental demand forecasting demo")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
log = logging.getLogger(__name__)


@app.exception_handler(RequestValidationError)
async def invalid_request(request, exc):
    # Python's JSON decoder accepts NaN/Infinity, but JSON error responses cannot.
    # Preserve validation details while rendering only nonfinite inputs as strings.
    detail = jsonable_encoder(exc.errors(), custom_encoder={float: lambda value: value if math.isfinite(value) else str(value)})
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health")
def health():
    try:
        manifest, payload = load_release()
        predict(manifest, payload, 14)
        model_info()
        read_monitoring(manifest["model_id"])
        comparison()
        horizons()
        for name in ["index.html", "app.js", "styles.css"]:
            if not (ROOT / "web" / name).is_file():
                raise FileNotFoundError(name)
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


@app.post("/predict", response_model=PredictResponse, responses={200: {"description": "Artifact-generated forecasts beginning 2012-10-23. The documented example illustrates horizon=1; the default request returns 14 points through 2012-11-05."}})
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
        return read_monitoring(manifest["model_id"])
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Monitoring snapshot unavailable") from exc


@app.get("/comparison")
def comparison():
    try:
        return read_report(ROOT / "reports/model_comparison.csv",
                           ["model", "mae_validation", "mae_test", "rmse_test", "wape_test", "selected"], 5)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Comparison report unavailable") from exc


@app.get("/horizons")
def horizons():
    try:
        return read_report(ROOT / "reports/horizon_errors.csv", ["split", "model", "period", "mae", "count"], 160)
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Horizon report unavailable") from exc


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(ROOT / "web/index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(ROOT / "web/favicon.svg", media_type="image/svg+xml")
