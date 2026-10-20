"""Build a summary table of all MLflow runs (CSV + PNG) for the report."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

from src.models.train_mlflow import EXPERIMENT, TRACKING_URI

OUT_DIR = Path("reports")
METRICS = ["cv_accuracy", "cv_precision", "cv_recall", "cv_roc_auc"]


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])

    table = runs[["tags.model"]].rename(columns={"tags.model": "model"})
    for m in METRICS:
        mean, std = runs[f"metrics.{m}_mean"], runs[f"metrics.{m}_std"]
        table[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(mean, std)]
    for m in ["accuracy", "precision", "recall", "roc_auc"]:
        table[f"test_{m}"] = runs[f"metrics.test_{m}"].round(3)

    OUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUT_DIR / "run_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(16, 1.2 + 0.5 * len(table)))
    ax.axis("off")
    tbl = ax.table(cellText=table.values, colLabels=table.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    fig.savefig(OUT_DIR / "figures" / "06_run_summary.png", dpi=150, bbox_inches="tight")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()