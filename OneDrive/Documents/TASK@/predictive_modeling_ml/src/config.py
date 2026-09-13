"""
Project-wide configuration.

Keeps paths and constants in one place so nothing in the pipeline
hardcodes a personal directory or magic number.
"""

from pathlib import Path

# Project root: src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ID columns are not used as model features.
COLUMNS_TO_DROP = [ID_COLUMN]

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"