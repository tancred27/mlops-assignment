from fastapi.testclient import TestClient

from src.serving.app import app

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_metrics_expose_request_and_prediction_counters():
    with TestClient(app) as client:
        client.post("/predict", json=SAMPLE)
        text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"}' in text
    assert "predictions_total" in text
    assert "http_request_duration_seconds_bucket" in text


def test_invalid_request_is_counted_as_422():
    with TestClient(app) as client:
        client.post("/predict", json={**SAMPLE, "cp": 9})
        text = client.get("/metrics").text
    assert 'status="422"' in text