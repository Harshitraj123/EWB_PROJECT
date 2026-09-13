# Telecom Customer Churn Prediction

An end-to-end supervised machine learning project that predicts whether a telecom customer is likely to churn.

The project compares **Logistic Regression, Decision Tree, and Random Forest** models and evaluates them using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.

It also includes data validation, leakage-safe preprocessing, model persistence, a reusable prediction interface, a Streamlit application, and automated tests with Pytest.

---

## 📌 Highlights

- Customer churn prediction using supervised machine learning
- Data validation and quality checks
- Exploratory Data Analysis (EDA)
- Leakage-safe preprocessing with `Pipeline` and `ColumnTransformer`
- Logistic Regression, Decision Tree, and Random Forest
- Model comparison using multiple evaluation metrics
- Confusion matrices and ROC curves
- Business-oriented model selection
- Model persistence using Joblib
- Reusable single and batch prediction interface
- Streamlit prediction application
- Automated testing with Pytest
- Reproducible training pipeline

---

## 📖 Problem Statement

Telecom companies lose recurring revenue when customers cancel their services.

Identifying customers who are likely to churn can help businesses prioritize retention efforts such as customer outreach, discounts, plan changes, and support interventions.

This project frames the problem as a **binary classification task**:

> Given a customer's demographic, account, and subscribed-service information, predict whether the customer will churn.

---

## 🎯 Objective

The objective of this project is to:

1. Explore and understand customer churn data.
2. Clean and validate the dataset.
3. Build a leakage-safe preprocessing pipeline.
4. Train and compare three classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
5. Evaluate the models using appropriate classification metrics.
6. Select a final model based on performance and business considerations.
7. Save the trained model for reuse.
8. Provide a reusable prediction interface.
9. Demonstrate the model through a Streamlit application.

---

## 📊 Dataset

### Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains telecom customer information and a binary churn target.

### Source

A publicly available copy of the IBM Telco Customer Churn dataset is used:

[Telco Customer Churn Dataset](https://github.com/treselle-systems/customer_churn_analysis)

The dataset is included in:

```text
data/raw/telco_customer_churn.csv
```

### Dataset Overview

- **Rows:** 7,043
- **Columns:** 21
- **Target:** `Churn`
- **Target values:** `Yes`, `No`

### Feature Groups

#### Demographics

- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`

#### Account Information

- `tenure`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`

#### Subscribed Services

- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`

### Class Distribution

The dataset is moderately imbalanced:

- Approximately **26.5% churn**
- Approximately **73.5% no churn**

Because of this imbalance, model evaluation considers more than accuracy, with particular attention to **recall, F1-score, and ROC-AUC**.

### Data Quality Consideration

The dataset contains blank `TotalCharges` values for some customers. These records are associated with customers having `tenure == 0`.

The project handles this condition explicitly during data validation and preprocessing.

---

## 🏗️ Architecture

```text
Raw CSV
   │
   ▼
Data Validation
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Preprocessing
(ColumnTransformer + Pipeline)
   │
   ▼
Train / Test Split
   │
   ├───────────────┬───────────────┐
   ▼               ▼               ▼
Logistic        Decision        Random
Regression        Tree           Forest
   │               │               │
   └───────────────┴───────────────┘
                   │
                   ▼
              Evaluation
                   │
                   ▼
             Model Selection
                   │
                   ▼
          Saved ML Pipeline
                   │
             ┌─────┴─────┐
             ▼           ▼
      Python Interface  Streamlit
```

The preprocessing steps and final model are stored together in a single `sklearn.Pipeline`.

This ensures that the same transformations learned during training are applied during inference.

---

## 🛠️ Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Visualization

- Matplotlib
- Seaborn

### Engineering

- Pytest
- Python Logging

### Application

- Streamlit

---

## 📁 Project Structure

```text
Telecom-Customer-Churn-Prediction/
│
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv
│   └── processed/
│
├── models/
│   ├── best_model.joblib
│   ├── metrics.json
│   └── model_metadata.json
│
├── notebooks/
│   └── churn_exploration.ipynb
│
├── visualizations/
│   ├── class_distribution.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curve.png
│   └── model_comparison.png
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_prediction.py
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── PROJECT_SUMMARY.md
├── DATASET.md
├── .gitignore
├── .env.example
└── LICENSE
```

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand:

- Dataset shape and structure
- Data types
- Missing values
- Duplicate records
- Churn distribution
- Numerical feature distributions
- Categorical feature relationships
- Correlations between numerical variables

Generated visualizations are saved in:

```text
visualizations/
```

---

## ⚙️ Data Preprocessing

The preprocessing pipeline handles:

- Feature and target separation
- Train/test splitting
- Stratified sampling
- Missing values
- Numerical features
- Categorical features
- Encoding
- Feature scaling

Scikit-learn's `Pipeline` and `ColumnTransformer` are used to keep preprocessing consistent between training and inference.

### Data Leakage Prevention

The preprocessing pipeline is fitted only on the training data.

The test set remains unseen until evaluation.

This prevents information from the test set from influencing the training process.

---

## 🤖 Machine Learning Models

Three supervised classification algorithms are trained and compared.

### 1. Logistic Regression

Used as an interpretable classification baseline.

### 2. Decision Tree

A tree-based model that learns decision rules from feature values.

### 3. Random Forest

An ensemble of decision trees designed to improve generalization and robustness.

---

## 📏 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Additional evaluation includes:

- Confusion matrices
- ROC curves
- Model comparison

### Why Accuracy Is Not Enough

The dataset contains more customers who do not churn than customers who do churn.

Therefore, a model can achieve relatively high accuracy while still performing poorly at identifying customers who actually churn.

For this use case, **recall is particularly important** because missing a potential churner may be more costly than contacting a customer who ultimately stays.

---

## 📊 Model Performance

The following results were obtained from an actual run of `main.py` on a held-out test set of **1,409 customers** using `random_state=42`.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.738 | 0.504 | **0.783** | 0.614 | **0.842** |
| Random Forest | **0.767** | **0.547** | 0.714 | **0.620** | 0.840 |
| Decision Tree | 0.740 | 0.507 | **0.789** | 0.617 | 0.828 |

### 5-Fold Cross-Validation

Cross-validated ROC-AUC on the training set:

| Model | CV ROC-AUC |
|---|---:|
| Logistic Regression | 0.845 |
| Random Forest | 0.844 |
| Decision Tree | 0.813 |

Cross-validation was used as an additional sanity check against overfitting to the train/test split.

---

## 🏆 Model Selection

### Selected Model: Logistic Regression

Logistic Regression was selected as the final model.

Although Random Forest achieved higher accuracy and precision, Logistic Regression provided:

- Strong recall
- The highest ROC-AUC among the three models
- Good cross-validation consistency
- Greater interpretability

For a churn early-warning use case, identifying as many potential churners as possible is important.

The final Logistic Regression model achieved:

- **Recall:** 0.783
- **ROC-AUC:** 0.842

Random Forest achieved higher accuracy and precision, but had lower recall.

Decision Tree produced the lowest ROC-AUC among the three models.

---

## 💾 Model Persistence

The selected model is saved using Joblib:

```text
models/best_model.joblib
```

The saved pipeline contains the preprocessing steps together with the trained model.

Additional metadata is stored in:

```text
models/metrics.json
models/model_metadata.json
```

---

## 🔮 Prediction Interface

A reusable prediction interface is provided through:

```text
src/predict.py
```

Example:

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor()

customer = {
    "tenure": 2,
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}

result = predictor.predict(customer)

print(result)
```

The prediction interface returns the predicted churn class and model probability.

It also supports batch predictions through:

```python
predictor.predict_batch(dataframe)
```

---

## 🖥️ Streamlit Application

A Streamlit interface is included for interactive predictions.

Run:

```bash
streamlit run app.py
```

The application allows users to:

- Enter customer information
- Submit the data
- Receive a churn prediction
- View the predicted probability

The application uses the same saved pipeline used by the prediction interface.

---

## 🧪 Testing

The project includes automated tests using Pytest.

Run:

```bash
pytest
```

The test suite covers:

- Data loading and validation
- Preprocessing
- Train/test splitting
- Leakage-related checks
- Model training
- Model reproducibility
- Required model availability
- Model loading
- Single predictions
- Batch predictions
- Missing input handling
- Missing model handling

**19 tests** are included in the test suite.

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Harshitraj123/Telecom-Customer-Churn-Prediction.git
cd Telecom-Customer-Churn-Prediction
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The dataset is included in:

```text
data/raw/telco_customer_churn.csv
```

No separate download step is required.

---

## ▶️ Train the Models

Run:

```bash
python main.py
```

The pipeline performs:

```text
Data Validation
      ↓
EDA
      ↓
Preprocessing
      ↓
Train/Test Split
      ↓
Model Training
      ↓
Evaluation
      ↓
Model Selection
      ↓
Model Persistence
```

The process generates:

```text
models/best_model.joblib
models/metrics.json
models/model_metadata.json
```

and saves visualizations under:

```text
visualizations/
```

---

## 📓 Notebook

The project includes:

```text
notebooks/churn_exploration.ipynb
```

The notebook demonstrates:

- Dataset inspection
- Exploratory analysis
- Preprocessing
- Model training
- Evaluation
- Model comparison
- Prediction

Reusable project logic is maintained in `src/`.

---

## 📁 Generated Artifacts

After running the pipeline, the project generates:

```text
models/
├── best_model.joblib
├── metrics.json
└── model_metadata.json
```

and visualizations under:

```text
visualizations/
```

---

## ⚠️ Limitations

This is a **portfolio-grade end-to-end ML project with production-oriented engineering practices**, not a fully deployed production churn platform.

Current limitations include:

- Static dataset snapshot
- No temporal validation against future customer behavior
- No live feature pipeline
- Limited hyperparameter tuning
- Streamlit-based manual prediction interface
- No production REST API
- No model monitoring
- No automated retraining pipeline

A production system would require additional infrastructure for serving, monitoring, data freshness, retraining, and operational management.

---

## 🔮 Future Improvements

Potential improvements include:

- Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
- SHAP-based model explainability
- Batch CSV prediction through Streamlit
- FastAPI model serving
- Docker containerization
- Cloud deployment
- Data drift monitoring
- Model performance monitoring
- Automated retraining
- Temporal validation
- CI/CD integration
- Model versioning

---

## 💡 Key Learning Outcomes

This project provided practical experience with:

- Supervised machine learning
- Binary classification
- Feature preprocessing
- Categorical encoding
- Feature scaling
- Train/test splitting
- Data leakage prevention
- Model comparison
- Classification metrics
- Confusion matrices
- ROC curves
- ROC-AUC
- Model persistence
- Reusable inference pipelines
- Streamlit application development
- Automated testing

---

## 🎓 Interview Topics

This project provides practical discussion points around:

- Why is churn prediction a classification problem?
- Why can accuracy be misleading?
- What is class imbalance?
- What are precision and recall?
- What is F1-score?
- What is ROC-AUC?
- What does a confusion matrix represent?
- Why use Logistic Regression?
- Why use Decision Tree?
- Why use Random Forest?
- What is overfitting?
- What is data leakage?
- Why use `Pipeline`?
- Why use `ColumnTransformer`?
- Why use stratified splitting?
- How was the final model selected?
- How would this model be deployed?
- How would model performance be monitored in production?

---

## 📌 Project Status

**Completed**

Current implementation includes:

- ✅ Data validation
- ✅ Exploratory Data Analysis
- ✅ Preprocessing
- ✅ Logistic Regression
- ✅ Decision Tree
- ✅ Random Forest
- ✅ Model evaluation
- ✅ Confusion matrices
- ✅ ROC curves
- ✅ Model selection
- ✅ Model persistence
- ✅ Prediction interface
- ✅ Streamlit application
- ✅ Automated tests

---

## 👤 Author

**Harshit Raj**

Computer Science & Engineering

GitHub: [Harshitraj123](https://github.com/Harshitraj123)

LinkedIn: [Harshit Raj](https://www.linkedin.com/in/harshitraj010204/)

---

## 📜 License

This project is intended for educational and portfolio purposes.

Please refer to the original dataset source for the applicable terms and usage conditions.