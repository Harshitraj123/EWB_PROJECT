import joblib
import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data_loader import load_data
from src.predict import ChurnPredictor
from src.preprocessing import (
    build_pipeline,
    prepare_features_and_target,
    split_data,
)
from src.train import get_candidate_models


@pytest.fixture(scope="module")
def saved_model_path(tmp_path_factory):
    """Train and save a real pipeline for prediction tests."""
    df = load_data(RAW_DATA_PATH)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, _ = split_data(X, y)

    model = get_candidate_models()["logistic_regression"]
    pipeline = build_pipeline(model)
    pipeline.fit(X_train, y_train)

    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "test_model.joblib"

    joblib.dump(pipeline, model_path)

    return model_path, X_test


def test_predictor_raises_when_model_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChurnPredictor(
            model_path=tmp_path / "missing.joblib"
        )


def test_predict_single_customer_returns_expected_shape(saved_model_path):
    model_path, X_test = saved_model_path

    predictor = ChurnPredictor(
        model_path=model_path
    )

    sample = X_test.iloc[0].to_dict()
    result = predictor.predict(sample)

    assert result["churn_prediction"] in {"Yes", "No"}
    assert 0.0 <= result["churn_probability"] <= 1.0


def test_predict_rejects_incomplete_input(saved_model_path):
    model_path, _ = saved_model_path

    predictor = ChurnPredictor(
        model_path=model_path
    )

    incomplete_input = {
        "tenure": 12,
    }

    with pytest.raises(ValueError):
        predictor.predict(incomplete_input)


def test_predict_rejects_unexpected_input_column(saved_model_path):
    model_path, X_test = saved_model_path

    predictor = ChurnPredictor(
        model_path=model_path
    )

    sample = X_test.iloc[0].to_dict()
    sample["unexpected_column"] = "invalid"

    with pytest.raises(
        ValueError,
        match="unexpected columns",
    ):
        predictor.predict(sample)


def test_predict_batch_matches_row_count(saved_model_path):
    model_path, X_test = saved_model_path

    predictor = ChurnPredictor(
        model_path=model_path
    )

    result = predictor.predict_batch(
        X_test.head(10)
    )

    assert len(result) == 10
    assert "churn_prediction" in result.columns
    assert "churn_probability" in result.columns