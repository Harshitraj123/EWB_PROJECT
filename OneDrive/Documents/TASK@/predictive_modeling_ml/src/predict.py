"""
Prediction interface used by the Streamlit app and tests.

The saved pipeline already includes preprocessing, so callers can pass
raw feature values in the same format as the training data, excluding
customerID and Churn.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_PATH
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class ChurnPredictor:
    """Load the saved pipeline and provide single or batch predictions."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {self.model_path}. "
                "Run training first with 'python main.py'."
            )

        self.pipeline = joblib.load(self.model_path)
        logger.info("Loaded model from %s", self.model_path)

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate the columns and shape expected by the saved model."""
        if df.empty:
            raise ValueError("Prediction input is empty.")

        expected = set(EXPECTED_INPUT_COLUMNS)
        actual = set(df.columns)

        missing = expected - actual
        unexpected = actual - expected

        if missing:
            raise ValueError(
                f"Input is missing required columns: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Input contains unexpected columns: {sorted(unexpected)}"
            )

    def predict(self, features: dict) -> dict:
        """Predict churn for a single customer."""
        df = pd.DataFrame([features])
        self._validate_input(df)

        prediction = self.pipeline.predict(df)[0]
        probability = self.pipeline.predict_proba(df)[0, 1]

        return {
            "churn_prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": float(probability),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict churn for multiple customers."""
        self._validate_input(df)

        predictions = self.pipeline.predict(df)
        probabilities = self.pipeline.predict_proba(df)[:, 1]

        result = df.copy()
        result["churn_prediction"] = [
            "Yes" if prediction == 1 else "No"
            for prediction in predictions
        ]
        result["churn_probability"] = probabilities

        return result