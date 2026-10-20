"""Clean the raw Heart Disease data: validate, binarize the target, save."""
from pathlib import Path
import logging

import pandas as pd

RAW_PATH = Path("data/raw/heart.csv")
CLEAN_PATH = Path("data/processed/heart_clean.csv")

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

# Allowed values per categorical column (NaN is allowed for ca and thal)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if a categorical column has unexpected values."""
    for col, allowed in VALID_VALUES.items():
        bad = set(df[col].dropna().unique()) - allowed
        if bad:
            raise ValueError(f"Column '{col}' has unexpected values: {sorted(bad)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, binarize the target (num > 0 -> 1) and validate values.

    Missing values are intentionally left in place: they are imputed inside the
    modelling pipeline so that no information leaks from the test folds.
    """
    df = df.copy().drop_duplicates()
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    validate(df)
    return df.reset_index(drop=True)


def main() -> None:
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    cleaned = clean(raw)
    cleaned.to_csv(CLEAN_PATH, index=False)
    log.info("Saved %d rows x %d columns to %s", *cleaned.shape, CLEAN_PATH)
    log.info("Target balance:\n%s", cleaned[TARGET].value_counts().to_string())
    log.info("Missing values:\n%s", cleaned.isna().sum()[cleaned.isna().sum() > 0].to_string())


if __name__ == "__main__":
    main()