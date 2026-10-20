"""Download the Heart Disease UCI dataset (id=45) and save it as a single CSV."""
from pathlib import Path
import logging

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATASET_ID = 45
RAW_PATH = Path("data/raw/heart.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def download_data(out_path: Path = RAW_PATH, verbose: bool = False) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Fetching UCI dataset id=%d", DATASET_ID)
    ds = fetch_ucirepo(id=DATASET_ID)

    if verbose:
        print(ds.metadata)
        print(ds.variables)

    # Column names come from the dataset itself, nothing hardcoded
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.to_csv(out_path, index=False)
    log.info("Saved %d rows x %d columns to %s", *df.shape, out_path)
    return df


if __name__ == "__main__":
    download_data(verbose=True)