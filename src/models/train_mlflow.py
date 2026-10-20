"""Train Logistic Regression and Random Forest with MLflow experiment tracking."""
from pathlib import Path
import json
import logging
import subprocess

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.data.clean_data import CLEAN_PATH
from src.models.train import (
    CV_FOLDS, METRICS_PATH, MODEL_DIR, RANDOM_STATE, SCORING, TEST_SIZE,
    evaluate, get_candidates, load_split, tune,
)

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "heart-disease-classification"

log = logging.getLogger(__name__)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def log_plots(model, X_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("ROC curve (test set)")
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title("Confusion matrix (test set)")
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)


def train_one(name, pipeline, grid, data) -> dict:
    X_train, X_test, y_train, y_test = data
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model": name, "git_sha": git_sha(), "dataset": str(CLEAN_PATH)})
        mlflow.log_params({
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "refit_metric": "roc_auc",
        })

        search = tune(name, pipeline, grid, X_train, y_train)
        best = search.best_estimator_
        i = search.best_index_

        mlflow.log_params({k.replace("clf__", "best_"): v for k, v in search.best_params_.items()})

        cv_metrics = {}
        for m in SCORING:
            mean = float(search.cv_results_[f"mean_test_{m}"][i])
            std = float(search.cv_results_[f"std_test_{m}"][i])
            cv_metrics[m] = {"mean": mean, "std": std}
            mlflow.log_metric(f"cv_{m}_mean", mean)
            mlflow.log_metric(f"cv_{m}_std", std)

        test_metrics = evaluate(best, X_test, y_test)
        for m, v in test_metrics.items():
            mlflow.log_metric(f"test_{m}", v)

        # Full tuning table (every hyperparameter combination) and plots
        cv_table = pd.DataFrame(search.cv_results_).drop(columns=["params"])
        mlflow.log_text(cv_table.to_csv(index=False), "tuning/cv_results.csv")
        log_plots(best, X_test, y_test)

        # Log the full pipeline (preprocessing + model) with an input/output signature
        X_example = X_train.astype("float64")
        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            signature=infer_signature(X_example, best.predict_proba(X_example)),
            input_example=X_example.head(3),
        )

        joblib.dump(best, MODEL_DIR / f"{name}.joblib")
        log.info("%s | CV AUC %.3f | TEST AUC %.3f | run %s",
                 name, cv_metrics["roc_auc"]["mean"], test_metrics["roc_auc"], run.info.run_id)

        return {
            "run_id": run.info.run_id,
            "best_params": search.best_params_,
            "cv": cv_metrics,
            "test": test_metrics,
        }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    data = load_split()
    results = {name: train_one(name, pipe, grid, data)
               for name, (pipe, grid) in get_candidates().items()}

    # Select by cross-validated ROC-AUC (never by test score)
    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc"]["mean"])
    joblib.dump(joblib.load(MODEL_DIR / f"{best_name}.joblib"), MODEL_DIR / "model.joblib")
    mlflow.MlflowClient().set_tag(results[best_name]["run_id"], "selected", "true")

    results["selected_model"] = best_name
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    log.info("Selected model: %s (saved to models/model.joblib)", best_name)


if __name__ == "__main__":
    main()