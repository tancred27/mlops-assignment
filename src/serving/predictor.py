"""Load the saved pipeline and make validated predictions."""
from pathlib import Path
from typing import Any
import os

import joblib
import pandas as pd

from src.features.preprocessing import FEATURES

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))

# Allowed values for categorical inputs (None = missing, imputed by the pipeline)
VALID_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}
# Only ca and thal may be missing, matching the training data
OPTIONAL = {"ca", "thal"}

_model = None


def load_model(path: Path = MODEL_PATH):
    """Load the pipeline once and cache it."""
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned record or raise ValueError with a clear message."""
    missing = [f for f in FEATURES if f not in record and f not in OPTIONAL]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    clean = {}
    for f in FEATURES:
        value = record.get(f)
        if value is None:
            if f not in OPTIONAL:
                raise ValueError(f"Feature '{f}' cannot be null")
            clean[f] = float("nan")
            continue
        try:
            clean[f] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Feature '{f}' must be numeric, got {value!r}")
        if f in VALID_VALUES and clean[f] not in VALID_VALUES[f]:
            raise ValueError(
                f"Feature '{f}' must be one of {sorted(VALID_VALUES[f])}, got {value!r}"
            )
    return clean


def predict(record: dict[str, Any]) -> dict[str, Any]:
    """Predict heart disease risk for one patient record."""
    clean = validate_record(record)
    X = pd.DataFrame([clean], columns=FEATURES)
    proba = float(load_model().predict_proba(X)[0, 1])
    return {
        "prediction": int(proba >= 0.5),
        "confidence": round(max(proba, 1 - proba), 4),
        "probability_disease": round(proba, 4),
    }