"""FastAPI service that serves the heart-disease model, with request logging and metrics."""
from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

from src.serving.predictor import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency in seconds", ["path"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["prediction"])
QUIET_PATHS = {"/health"}  # probes hit these constantly, so skip them in the request log


class PatientFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6,
    }})

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float | None = None    # may be missing, imputed by the pipeline
    thal: float | None = None  # may be missing, imputed by the pipeline


class Prediction(BaseModel):
    prediction: int
    confidence: float
    probability_disease: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()  # fail fast at startup if the model file is missing
    log.info("Model loaded")
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Log every request and record Prometheus metrics (the scrape endpoint itself is excluded)."""
    if request.url.path == "/metrics":
        return await call_next(request)

    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = route.path if route else "unmatched"  # route template keeps label cardinality low
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        if request.url.path not in QUIET_PATHS:
            log.info("method=%s path=%s status=%d duration_ms=%.1f",
                     request.method, request.url.path, status, elapsed * 1000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=Prediction)
def predict_endpoint(features: PatientFeatures) -> dict:
    try:
        result = predict(features.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    PREDICTIONS.labels(prediction=str(result["prediction"])).inc()
    # Log only the outcome, never the patient's health values
    log.info("prediction=%d confidence=%.3f", result["prediction"], result["confidence"])
    return result