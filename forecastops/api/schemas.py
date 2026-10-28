from datetime import date
from pydantic import BaseModel, ConfigDict, Field, StrictInt


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon: StrictInt = Field(default=14, ge=1, le=14)


class ForecastPoint(BaseModel):
    date: date
    value: float


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    mode: str
    trained_through: date
    horizon: int
    forecast_start: date
    forecast_end: date
    predictions: list[ForecastPoint]
    disclaimer: str
