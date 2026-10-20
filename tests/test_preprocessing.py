import numpy as np

from src.data.clean_data import NUMERIC_FEATURES, clean
from src.features.preprocessing import FEATURES, build_preprocessor


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def test_preprocessor_imputes_scales_and_encodes(raw_df):
    df = clean(raw_df)
    X = _dense(build_preprocessor().fit_transform(df[FEATURES]))
    assert X.shape[0] == len(df)
    assert X.shape[1] > len(FEATURES)  # one-hot encoding adds columns
    assert not np.isnan(X).any()  # missing ca/thal were imputed
    # numeric columns come first and are standardised
    assert np.allclose(X[:, : len(NUMERIC_FEATURES)].mean(axis=0), 0, atol=1e-8)


def test_preprocessor_ignores_unseen_categories(raw_df):
    df = clean(raw_df)
    pre = build_preprocessor().fit(df[FEATURES])
    new = df[FEATURES].head(2).copy()
    new["cp"] = 99
    assert _dense(pre.transform(new)).shape[0] == 2