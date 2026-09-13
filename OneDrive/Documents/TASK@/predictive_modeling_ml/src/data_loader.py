"""Loading the raw dataset from disk."""

from pathlib import Path

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Read the churn CSV into a DataFrame.

    Raises FileNotFoundError or ValueError when the dataset cannot be
    loaded correctly so failures are caught early in the pipeline.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Check that the dataset is present at the expected path."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")

    logger.info("Loaded dataset with shape %s from %s", df.shape, path)
    return df