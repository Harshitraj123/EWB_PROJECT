"""Training the three candidate models."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_pipeline
from src.utils import get_logger

logger = get_logger(__name__)


def get_candidate_models() -> dict:
    """Return the baseline classification models used in the project.

    Class weights are balanced to account for the churn class being
    smaller than the non-churn class.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    """Fit each candidate model and return the fitted pipelines."""
    fitted_pipelines = {}

    for name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline
        logger.info("Trained %s", name)

    return fitted_pipelines


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Calculate cross-validated ROC-AUC for each candidate model."""
    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = {}

    for name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring="roc_auc",
        )

        scores[name] = float(cv_scores.mean())

        logger.info(
            "%s: CV ROC-AUC = %.4f (+/- %.4f)",
            name,
            cv_scores.mean(),
            cv_scores.std(),
        )

    return scores