import pandas as pd
import pytest

from src.config import RAW_DATA_PATH
from src.data_loader import load_data
from src.validation import (
    EXPECTED_COLUMNS,
    clean_total_charges,
    validate_raw_dataset,
)


def _minimal_valid_df() -> pd.DataFrame:
    """Create a small dataframe with the expected dataset schema."""
    row = {column: "No" for column in EXPECTED_COLUMNS}

    row.update(
        {
            "customerID": "1234-ABCD",
            "tenure": 5,
            "MonthlyCharges": 50.0,
            "TotalCharges": "250.0",
        }
    )

    return pd.DataFrame([row])


def test_load_data_reads_real_dataset():
    df = load_data(RAW_DATA_PATH)

    assert not df.empty
    assert "Churn" in df.columns


def test_load_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_path)


def test_load_data_empty_file_raises(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("col1,col2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_data(empty_csv)


def test_validate_raw_dataset_accepts_valid_data():
    df = _minimal_valid_df()

    validate_raw_dataset(df)


def test_validate_raw_dataset_rejects_missing_columns():
    df = _minimal_valid_df().drop(columns=["Contract"])

    with pytest.raises(ValueError):
        validate_raw_dataset(df)


def test_validate_raw_dataset_rejects_bad_target_labels():
    df = _minimal_valid_df()
    df["Churn"] = "Maybe"

    with pytest.raises(ValueError):
        validate_raw_dataset(df)


def test_validate_raw_dataset_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        validate_raw_dataset(pd.DataFrame())


def test_validate_raw_dataset_rejects_duplicate_customer_ids():
    df = pd.concat(
        [_minimal_valid_df(), _minimal_valid_df()],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate customerID"):
        validate_raw_dataset(df)


def test_clean_total_charges_fills_blank_for_zero_tenure():
    df = _minimal_valid_df()
    df["tenure"] = 0
    df["TotalCharges"] = " "

    cleaned = clean_total_charges(df)

    assert cleaned.loc[0, "TotalCharges"] == 0.0


def test_clean_total_charges_rejects_blank_for_nonzero_tenure():
    df = _minimal_valid_df()
    df["TotalCharges"] = " "

    with pytest.raises(
        ValueError,
        match="tenure greater than 0",
    ):
        clean_total_charges(df)


def test_clean_total_charges_rejects_invalid_numeric_values():
    df = _minimal_valid_df()
    df["TotalCharges"] = "not-a-number"

    with pytest.raises(
        ValueError,
        match="invalid TotalCharges values",
    ):
        clean_total_charges(df)