import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH
from src.data_loader import load_data
from src.preprocessing import (
    build_pipeline,
    prepare_features_and_target,
    split_data,
)
from src.train import get_candidate_models
from src.validation import clean_total_charges


def test_clean_total_charges_fills_blanks():
    df = pd.DataFrame(
        {
            "TotalCharges": [" ", "100.5", " "],
            "tenure": [0, 5, 0],
        }
    )

    cleaned = clean_total_charges(df)

    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 100.5
    assert cleaned["TotalCharges"].iloc[2] == 0.0


def test_prepare_features_and_target_encodes_binary_target():
    df = load_data(RAW_DATA_PATH)

    X, y = prepare_features_and_target(df)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


def test_split_data_is_stratified_and_disjoint():
    df = load_data(RAW_DATA_PATH)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    assert len(X_train) + len(X_test) == len(X)
    assert set(X_train.index).isdisjoint(set(X_test.index))

    # Stratification should keep the churn ratio similar in both splits.
    assert abs(y_train.mean() - y_test.mean()) < 0.03


def test_pipeline_fits_and_predicts():
    df = load_data(RAW_DATA_PATH)
    X, y = prepare_features_and_target(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = get_candidate_models()["logistic_regression"]
    pipeline = build_pipeline(model)

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert set(np.unique(predictions)) <= {0, 1}