"""FastAPI service that serves the heart-disease model."""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.serving.predictor import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")


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


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(features: PatientFeatures) -> dict:
    try:
        return predict(features.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))