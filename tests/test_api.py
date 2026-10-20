from fastapi.testclient import TestClient

from src.serving.app import app

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence():
    with TestClient(app) as client:
        r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_accepts_missing_optional_fields():
    payload = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_invalid_category():
    with TestClient(app) as client:
        assert client.post("/predict", json={**SAMPLE, "cp": 9}).status_code == 422


def test_predict_rejects_missing_required_field():
    payload = {k: v for k, v in SAMPLE.items() if k != "age"}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 422