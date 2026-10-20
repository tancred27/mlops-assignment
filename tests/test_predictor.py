import pytest

from src.serving.predictor import predict, validate_record

SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_validate_accepts_sample():
    assert validate_record(SAMPLE)["age"] == 63.0


def test_missing_required_feature_raises():
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    with pytest.raises(ValueError, match="Missing required"):
        validate_record(bad)


def test_invalid_category_raises():
    with pytest.raises(ValueError, match="must be one of"):
        validate_record({**SAMPLE, "cp": 9})


def test_non_numeric_raises():
    with pytest.raises(ValueError, match="must be numeric"):
        validate_record({**SAMPLE, "chol": "abc"})


def test_optional_fields_can_be_missing():
    rec = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert validate_record(rec)["ca"] != validate_record(rec)["ca"]  # NaN


def test_predict_output_shape():
    out = predict(SAMPLE)
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert 0.0 <= out["probability_disease"] <= 1.0