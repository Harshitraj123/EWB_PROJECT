import joblib
import pytest

from src.config import RAW_DATA_PATH
from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.preprocessing import (
    build_pipeline,
    prepare_features_and_target,
    split_data,
)
from src.train import get_candidate_models, train_models


@pytest.fixture(scope="module")
def trained_pipelines():
    df = load_data(RAW_DATA_PATH)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipelines = train_models(X_train, y_train)

    return pipelines, X_test, y_test


def test_all_three_required_models_are_trained(trained_pipelines):
    pipelines, _, _ = trained_pipelines

    assert set(pipelines.keys()) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
    }


def test_evaluate_model_returns_expected_metrics(trained_pipelines):
    pipelines, X_test, y_test = trained_pipelines

    metrics = evaluate_model(
        pipelines["logistic_regression"],
        X_test,
        y_test,
    )

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

    assert set(metrics.keys()) == expected_keys

    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_model_reproducible_with_fixed_seed():
    """The same random seed should produce identical predictions."""
    df = load_data(RAW_DATA_PATH)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, _ = split_data(X, y)

    model_a = get_candidate_models()["random_forest"]
    model_b = get_candidate_models()["random_forest"]

    pipeline_a = build_pipeline(model_a)
    pipeline_b = build_pipeline(model_b)

    pipeline_a.fit(X_train, y_train)
    pipeline_b.fit(X_train, y_train)

    preds_a = pipeline_a.predict(X_test)
    preds_b = pipeline_b.predict(X_test)

    assert (preds_a == preds_b).all()


def test_model_save_and_load_roundtrip(tmp_path, trained_pipelines):
    pipelines, X_test, _ = trained_pipelines

    model_path = tmp_path / "test_model.joblib"

    joblib.dump(
        pipelines["logistic_regression"],
        model_path,
    )

    assert model_path.exists()

    loaded_pipeline = joblib.load(model_path)

    original_preds = pipelines["logistic_regression"].predict(X_test)
    loaded_preds = loaded_pipeline.predict(X_test)

    assert (original_preds == loaded_preds).all()