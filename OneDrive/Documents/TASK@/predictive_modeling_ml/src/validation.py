"""
Data validation.

The Telco churn dataset has a known quirk: TotalCharges is stored as a
string and some new customers with 0 tenure have a blank value instead
of "0.0". This is handled explicitly before the value is converted to
numeric form.
"""

import pandas as pd

from src.config import TARGET_COLUMN
from src.utils import get_logger

logger = get_logger(__name__)

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def validate_raw_dataset(df: pd.DataFrame) -> None:
    """Validate the raw dataframe before preprocessing."""
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing expected columns: {sorted(missing_columns)}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    unexpected_labels = set(df[TARGET_COLUMN].dropna().unique()) - {"Yes", "No"}
    if unexpected_labels:
        raise ValueError(
            f"Target column has unexpected labels: {sorted(unexpected_labels)}"
        )

    duplicate_ids = df["customerID"].duplicated().sum()
    if duplicate_ids > 0:
        raise ValueError(
            f"Found {duplicate_ids} duplicate customerID rows."
        )

    logger.info("Raw dataset passed validation. Shape: %s", df.shape)


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Handle known blank TotalCharges values and convert to numeric."""
    df = df.copy()

    total_charges = df["TotalCharges"].astype("string").str.strip()
    blank_mask = total_charges.isna() | total_charges.eq("")

    if blank_mask.any():
        invalid_blank_rows = df.loc[blank_mask, "tenure"] != 0

        if invalid_blank_rows.any():
            raise ValueError(
                "Found blank TotalCharges values for customers with "
                "tenure greater than 0."
            )

        blank_count = int(blank_mask.sum())
        logger.info(
            "Filling %d blank TotalCharges values with 0.0 "
            "for 0-tenure customers.",
            blank_count,
        )

        total_charges = total_charges.mask(blank_mask, "0.0")

    numeric_total_charges = pd.to_numeric(
        total_charges,
        errors="coerce",
    )

    invalid_values = numeric_total_charges.isna()

    if invalid_values.any():
        bad_values = total_charges[invalid_values].unique().tolist()
        raise ValueError(
            f"Found invalid TotalCharges values: {bad_values}"
        )

    df["TotalCharges"] = numeric_total_charges

    return df