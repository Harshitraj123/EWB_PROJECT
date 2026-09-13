"""
End-to-end pipeline:
load -> validate -> EDA -> preprocess -> train -> evaluate ->
select best model -> save.

Run with:
    python main.py
"""

import json

import matplotlib

# Use a non-GUI backend because this script only saves plots to files.
# This avoids Tkinter/main-thread errors on Windows.
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    VISUALIZATIONS_DIR,
)
from src.data_loader import load_data
from src.evaluate import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
)
from src.preprocessing import prepare_features_and_target, split_data
from src.train import cross_validate_models, train_models
from src.utils import get_logger
from src.validation import validate_raw_dataset

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


def run_eda(df) -> None:
    """Generate a small set of useful exploratory visualizations."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Target distribution helps show the class imbalance.
    fig, ax = plt.subplots(figsize=(5, 4))

    df["Churn"].value_counts().plot(
        kind="bar",
        ax=ax,
        color=["#4C72B0", "#DD8452"],
    )

    ax.set_title("Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Customer Count")

    fig.tight_layout()
    fig.savefig(
        VISUALIZATIONS_DIR / "churn_distribution.png",
        dpi=150,
    )
    plt.close(fig)

    # Compare tenure distributions between churn classes.
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(
        data=df,
        x="tenure",
        hue="Churn",
        multiple="stack",
        bins=30,
        ax=ax,
    )

    ax.set_title("Tenure Distribution by Churn")

    fig.tight_layout()
    fig.savefig(
        VISUALIZATIONS_DIR / "tenure_by_churn.png",
        dpi=150,
    )
    plt.close(fig)

    # Compare monthly charges between churn classes.
    fig, ax = plt.subplots(figsize=(6, 4))

    sns.boxplot(
        data=df,
        x="Churn",
        y="MonthlyCharges",
        ax=ax,
    )

    ax.set_title("Monthly Charges by Churn")

    fig.tight_layout()
    fig.savefig(
        VISUALIZATIONS_DIR / "monthly_charges_by_churn.png",
        dpi=150,
    )
    plt.close(fig)

    # Compare churn rates across contract types.
    churn_by_contract = (
        df.groupby("Contract")["Churn"]
        .apply(lambda values: (values == "Yes").mean())
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    churn_by_contract.plot(
        kind="bar",
        ax=ax,
        color="#C44E52",
    )

    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate")

    fig.tight_layout()
    fig.savefig(
        VISUALIZATIONS_DIR / "churn_rate_by_contract.png",
        dpi=150,
    )
    plt.close(fig)

    logger.info(
        "EDA plots saved to %s",
        VISUALIZATIONS_DIR,
    )


def select_best_model(cv_scores: dict[str, float]) -> str:
    """Select the model with the highest cross-validated ROC-AUC."""
    best_model_name = max(
        cv_scores,
        key=cv_scores.get,
    )

    logger.info(
        "Selected '%s' based on cross-validated ROC-AUC.",
        best_model_name,
    )

    return best_model_name


def main() -> None:
    """Run the complete machine learning pipeline."""
    logger.info("Starting pipeline run.")

    # ---------------------------------------------------------
    # 1. Load and validate the raw dataset
    # ---------------------------------------------------------
    df = load_data(RAW_DATA_PATH)
    validate_raw_dataset(df)

    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis
    # ---------------------------------------------------------
    run_eda(df)

    # ---------------------------------------------------------
    # 3. Prepare features and target
    # ---------------------------------------------------------
    X, y = prepare_features_and_target(df)

    # ---------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------------------------------------------------
    # 5. Cross-validation on training data
    #
    # The test set is kept untouched during model selection.
    # ---------------------------------------------------------
    cv_scores = cross_validate_models(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 6. Train all candidate models
    # ---------------------------------------------------------
    pipelines = train_models(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # 7. Select the best model using CV results only
    # ---------------------------------------------------------
    best_model_name = select_best_model(cv_scores)

    best_pipeline = pipelines[best_model_name]

    # ---------------------------------------------------------
    # 8. Evaluate all models on the untouched test set
    # ---------------------------------------------------------
    comparison = evaluate_all_models(
        pipelines,
        X_test,
        y_test,
    )

    logger.info(
        "Model comparison on held-out test set:\n%s",
        comparison.round(4).to_string(),
    )

    # ---------------------------------------------------------
    # 9. Generate evaluation visualizations
    # ---------------------------------------------------------
    plot_model_comparison(comparison)

    plot_roc_curves(
        pipelines,
        X_test,
        y_test,
    )

    for name, pipeline in pipelines.items():
        plot_confusion_matrix(
            pipeline,
            X_test,
            y_test,
            name,
        )

    # ---------------------------------------------------------
    # 10. Save the selected model
    # ---------------------------------------------------------
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_pipeline,
        MODEL_PATH,
    )

    logger.info(
        "Saved selected model ('%s') to %s",
        best_model_name,
        MODEL_PATH,
    )

    # ---------------------------------------------------------
    # 11. Save model metadata
    # ---------------------------------------------------------
    metadata = {
        "best_model": best_model_name,
        "test_metrics": comparison.loc[best_model_name].to_dict(),
        "cv_roc_auc": cv_scores,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "random_state": RANDOM_STATE,
    }

    with open(
        MODEL_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # 12. Save all model metrics
    # ---------------------------------------------------------
    comparison.round(4).to_json(
        METRICS_PATH,
        orient="index",
        indent=2,
    )

    logger.info("Pipeline run complete.")


if __name__ == "__main__":
    main()