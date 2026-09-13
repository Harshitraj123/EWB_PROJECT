"""
Preprocessing: turning the raw churn dataframe into model-ready features
without leaking information from the test set into training.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUMNS_TO_DROP, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from src.utils import get_logger
from src.validation import clean_total_charges

logger = get_logger(__name__)

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# SeniorCitizen is a binary indicator, so treating it as categorical
# keeps it consistent with the other yes/no-style features.
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split the raw dataframe into features and a binary target."""
    df = clean_total_charges(df)
    df = df.drop(columns=COLUMNS_TO_DROP)

    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split data into train=%s and test=%s "
        "(churn rate train=%.3f, test=%.3f)",
        X_train.shape,
        X_test.shape,
        y_train.mean(),
        y_test.mean(),
    )

    return X_train, X_test, y_train, y_test


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing steps used inside each model pipeline.

    This function only defines the transformations. Fitting happens later
    on the training data through the sklearn Pipeline.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(model) -> Pipeline:
    """Combine preprocessing and a model into a single pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )