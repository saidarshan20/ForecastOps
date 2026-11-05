from datetime import date
from pydantic import BaseModel, ConfigDict, Field, StrictInt


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: StrictInt = Field(default=14, ge=1, le=14, examples=[14], description="Consecutive historical days after 2012-10-22; integer 1–14")


class ForecastPoint(BaseModel):
    date: date
    value: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "model_name": "holt_winters", "model_version": "v1-20121022", "mode": "historical_demo",
        "trained_through": "2012-10-22", "horizon": 1, "forecast_start": "2012-10-23", "forecast_end": "2012-10-23",
        "predictions": [{"date": "2012-10-23", "value": 6998.057787694276}],
        "disclaimer": "Historical demonstration; not a forecast for the current date."
    }]})
    model_name: str
    model_version: str
    mode: str
    trained_through: date
    horizon: int
    forecast_start: date
    forecast_end: date
    predictions: list[ForecastPoint]
    disclaimer: str
