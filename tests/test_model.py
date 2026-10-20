from src.models.train import evaluate, get_candidates, load_split


def test_split_is_stratified_and_disjoint():
    X_train, X_test, y_train, y_test = load_split()
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_both_candidate_models_are_defined():
    assert set(get_candidates()) == {"logistic_regression", "random_forest"}


def test_logistic_regression_meets_quality_gate():
    X_train, X_test, y_train, y_test = load_split()
    pipeline, _ = get_candidates()["logistic_regression"]
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "roc_auc"}
    assert metrics["roc_auc"] > 0.85  # fails the pipeline if model quality regresses