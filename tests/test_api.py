from fastapi.testclient import TestClient
from forecastops.api.main import app
from forecastops.api import service

client = TestClient(app)


def test_endpoints():
    assert client.get("/health").status_code == 200
    assert client.get("/docs").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/monitoring").json()["paired_days"] == 70
    assert client.get("/model").json()["trained_through"] == "2012-10-22"
    for horizon in [1,14]:
        response = client.post("/predict", json={"horizon":horizon})
        assert response.status_code == 200
        body = response.json()
        assert body["horizon"] == len(body["predictions"]) == horizon
        assert body["forecast_start"] == "2012-10-23"
        assert body["trained_through"] == "2012-10-22"
        assert all(isinstance(p["value"],float) and p["value"]>=0 for p in body["predictions"])
    for invalid in [0,15,1.5,"7",True]:
        assert client.post("/predict",json={"horizon":invalid}).status_code == 422
    assert client.post("/predict",json={"horizon":7,"cutoff":"2012-12-31"}).status_code == 422


def test_missing_release_fails_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "ARTIFACT_DIR", tmp_path)
    assert client.get("/health").status_code == 503
