import pandas as pd
import pytest

from src.data.clean_data import TARGET, clean


def test_clean_binarizes_target_and_drops_num(raw_df):
    out = clean(raw_df)
    assert "num" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}
    assert (out[TARGET] == (raw_df["num"] > 0).astype(int)).all()


def test_clean_keeps_missing_values_for_the_pipeline(raw_df):
    out = clean(raw_df)
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1


def test_clean_drops_duplicate_rows(raw_df):
    duplicated = pd.concat([raw_df, raw_df.iloc[[2]]])
    assert len(clean(duplicated)) == len(raw_df)


def test_clean_rejects_unexpected_category(raw_df):
    bad = raw_df.copy()
    bad.loc[3, "cp"] = 9
    with pytest.raises(ValueError, match="cp"):
        clean(bad)


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)