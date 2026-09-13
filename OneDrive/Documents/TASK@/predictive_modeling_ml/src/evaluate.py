"""Evaluate trained models and generate comparison visualizations."""

import matplotlib

# Use a non-GUI backend because this project saves plots to files
# instead of opening interactive plot windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import VISUALIZATIONS_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics on held-out test data."""
    y_pred = pipeline.predict(X_test)

    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "The provided pipeline does not support probability predictions."
        )

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_proba,
        ),
    }


def evaluate_all_models(
    pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all trained models and return a comparison table."""
    rows = []

    for name, pipeline in pipelines.items():
        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
        )
        metrics["model"] = name
        rows.append(metrics)

    comparison = pd.DataFrame(rows).set_index("model")

    return comparison.sort_values(
        "roc_auc",
        ascending=False,
    )


def plot_confusion_matrix(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> None:
    """Save a confusion matrix for a single model."""
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    fig, ax = plt.subplots(figsize=(5, 4))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"],
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    ax.set_title(
        f"Confusion Matrix - {model_name}"
    )

    fig.tight_layout()

    output_path = (
        VISUALIZATIONS_DIR
        / f"confusion_matrix_{model_name}.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)

    logger.info(
        "Saved confusion matrix for %s to %s",
        model_name,
        output_path,
    )


def plot_roc_curves(
    pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Save one ROC curve comparing all candidate models."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, pipeline in pipelines.items():
        RocCurveDisplay.from_estimator(
            pipeline,
            X_test,
            y_test,
            name=name,
            ax=ax,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random guess",
    )

    ax.set_title(
        "ROC Curves - Model Comparison"
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    output_path = (
        VISUALIZATIONS_DIR
        / "roc_curves_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)

    logger.info(
        "Saved ROC curve comparison to %s",
        output_path,
    )


def plot_model_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Save a bar chart comparing model evaluation metrics."""
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    comparison[metrics].plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Model Comparison"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    output_path = (
        VISUALIZATIONS_DIR
        / "model_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)

    logger.info(
        "Saved model comparison chart to %s",
        output_path,
    )