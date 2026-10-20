"""Train and tune Logistic Regression and Random Forest, evaluate, save models."""
import math
from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data.clean_data import CLEAN_PATH, TARGET
from src.features.preprocessing import FEATURES, build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = ["accuracy", "precision", "recall", "roc_auc"]
MODEL_DIR = Path("models")
METRICS_PATH = Path("reports/metrics.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_split():
    """Load the cleaned data and return a stratified train/test split."""
    df = pd.read_csv(CLEAN_PATH)
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def get_candidates() -> dict:
    """Model name -> (pipeline, hyperparameter grid)."""
    lr = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    rf = Pipeline([
        ("preprocess", build_preprocessor()),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    return {
        "logistic_regression": (lr, {"clf__C": [0.01, 0.1, 1, 10]}),
        "random_forest": (rf, {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        }),
    }


def evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }



def tune(name, pipeline, grid, X_train, y_train) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, grid, cv=cv, scoring=SCORING, refit="roc_auc", n_jobs=-1
    )
    n_candidates = math.prod(len(v) for v in grid.values())
    log.info("Tuning %s (%d candidates x %d folds)", name, n_candidates, CV_FOLDS)
    search.fit(X_train, y_train)
    return search


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_split()
    log.info("Train: %d rows, Test: %d rows", len(X_train), len(X_test))

    results = {}
    for name, (pipeline, grid) in get_candidates().items():
        search = tune(name, pipeline, grid, X_train, y_train)
        i = search.best_index_
        cv_metrics = {
            m: {"mean": search.cv_results_[f"mean_test_{m}"][i],
                "std": search.cv_results_[f"std_test_{m}"][i]}
            for m in SCORING
        }
        test_metrics = evaluate(search.best_estimator_, X_test, y_test)
        results[name] = {
            "best_params": search.best_params_,
            "cv": cv_metrics,
            "test": test_metrics,
        }
        joblib.dump(search.best_estimator_, MODEL_DIR / f"{name}.joblib")

        log.info("%s best params: %s", name, search.best_params_)
        log.info("%s CV   : %s", name,
                 {m: f"{v['mean']:.3f}+/-{v['std']:.3f}" for m, v in cv_metrics.items()})
        log.info("%s TEST : %s", name, {m: round(v, 3) for m, v in test_metrics.items()})

    # Select the final model by cross-validated ROC-AUC (never by test score)
    best_name = max(results, key=lambda n: results[n]["cv"]["roc_auc"]["mean"])
    joblib.dump(joblib.load(MODEL_DIR / f"{best_name}.joblib"), MODEL_DIR / "model.joblib")
    results["selected_model"] = best_name
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    log.info("Selected model: %s (saved to models/model.joblib)", best_name)


if __name__ == "__main__":
    main()