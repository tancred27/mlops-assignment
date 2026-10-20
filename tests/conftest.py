import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """40 synthetic rows in the same layout as data/raw/heart.csv."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "num": rng.integers(0, 5, n),
    })
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    return df