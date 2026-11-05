from fastapi.testclient import TestClient
from forecastops.api.main import app
from forecastops.api import service
import math
import shutil
import sqlite3
import pytest
from forecastops.api import main

client = TestClient(app)


def test_endpoints():
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/").status_code == 200
    icon = client.get("/favicon.ico")
    assert icon.status_code == 200
    assert icon.headers["content-type"] == "image/svg+xml"
    assert len(client.get("/comparison").json()) == 5
    assert len(client.get("/horizons").json()) == 160
    assert client.get("/monitoring").json()["paired_days"] == 70
    assert client.get("/model").json()["trained_through"] == "2012-10-22"
    for horizon in [1,14]:
        response = client.post("/predict", json={"horizon":horizon})
        assert response.status_code == 200
        body = response.json()
        assert body["horizon"] == len(body["predictions"]) == horizon
        assert body["forecast_start"] == "2012-10-23"
        assert body["trained_through"] == "2012-10-22"
        assert body["forecast_end"] == ("2012-10-23" if horizon == 1 else "2012-11-05")
        assert all(isinstance(p["value"],float) and math.isfinite(p["value"]) and p["value"]>=0 for p in body["predictions"])
        assert client.post("/predict", json={"horizon": horizon}).json() == body
    assert client.post("/predict", json={}).json()["horizon"] == 14
    for invalid in [0,15,1.5,"7",True,None,[],{}]:
        assert client.post("/predict",json={"horizon":invalid}).status_code == 422
    assert client.post("/predict",json={"horizon":7,"cutoff":"2012-12-31"}).status_code == 422


def test_missing_release_fails_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ARTIFACT_DIR", tmp_path)
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("target", ["manifest.json", "model.joblib", "seed", "runtime", "selection.json", "monitoring_summary.json", "model_comparison.csv", "horizon_errors.csv"])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_required_artifacts_fail_readiness(monkeypatch, tmp_path, target, damage):
    root = tmp_path / "fixture"
    shutil.copytree(service.ROOT / "reports", root / "reports")
    shutil.copytree(service.ARTIFACT_DIR, root / "release")
    shutil.copytree(service.ROOT / "web", root / "web")
    shutil.copyfile(service.SEED_DB, root / "seed.sqlite")
    monkeypatch.setattr(service, "ROOT", root)
    monkeypatch.setattr(main, "ROOT", root)
    monkeypatch.setattr(service, "ARTIFACT_DIR", root / "release")
    monkeypatch.setattr(service, "SEED_DB", root / "seed.sqlite")
    if target == "runtime":
        service.ensure_runtime_db()
        path = service.DB_PATH
    elif target == "seed":
        path = service.SEED_DB
    elif target in ["manifest.json", "model.joblib"]:
        path = root / "release" / target
    else:
        path = root / "reports" / target
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    # Missing runtime is recoverable; missing/corrupt sources or existing corrupt DB are not.
    expected = 200 if target == "runtime" and damage == "missing" else 503
    assert client.get("/health").status_code == expected


def test_runtime_restart_and_schema_damage():
    assert client.get("/health").status_code == 200
    before = service.DB_PATH.read_bytes()
    assert client.get("/health").status_code == 200
    assert service.DB_PATH.read_bytes() == before
    with sqlite3.connect(service.DB_PATH) as db:
        db.execute("DROP TABLE actuals")
    assert client.get("/health").status_code == 503


def test_swagger_historical_example():
    schema = client.get("/openapi.json").json()
    example = schema["components"]["schemas"]["PredictResponse"]["examples"][0]
    assert client.post("/predict", json={"horizon": 1}).json() == example


def test_serving_never_fits(monkeypatch):
    from forecastops.models.statistical import HoltWinters
    def forbidden(*args, **kwargs):
        raise AssertionError("Serving must use the prebuilt artifact")
    monkeypatch.setattr(HoltWinters, "fit", forbidden)
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json={"horizon": 14}).status_code == 200


def test_non_numeric_dashboard_metrics_fail_readiness(monkeypatch, tmp_path):
    import pandas as pd
    root = tmp_path / "reports-fixture"
    shutil.copytree(main.ROOT / "reports", root / "reports")
    monkeypatch.setattr(main, "ROOT", root)
    path = root / "reports/model_comparison.csv"
    frame = pd.read_csv(path)
    frame['mae_test'] = 'broken'
    frame.to_csv(path, index=False)
    assert client.get("/comparison").status_code == 503
    assert client.get("/health").status_code == 503


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_request_rejected(token):
    response = client.post('/predict', content='{"horizon":' + token + '}', headers={'content-type': 'application/json'})
    assert response.status_code == 422
