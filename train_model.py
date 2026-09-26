"""
train_model.py
---------------
Trains and compares three algorithms on the loan applicant dataset:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest

Workflow (matches the official synopsis):
    Loan Applicant Dataset
        -> Data Cleaning & Preprocessing
        -> EDA & Feature Engineering
        -> Train/Test Split (stratified)
        -> Logistic Regression / Decision Tree / Random Forest
        -> Model Evaluation
        -> Best Model Selection
        -> Save Model using joblib

Run:
    python train_model.py
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import preprocessing as prep

sns.set_theme(style="whitegrid")

DATASET_PATH = os.path.join("dataset", "loan_data.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
EDA_DIR = os.path.join("static", "images", "eda")
DASHBOARD_DIR = os.path.join("static", "images", "dashboard")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# EDA visualizations
# ---------------------------------------------------------------------------
def generate_eda_plots(df: pd.DataFrame) -> None:
    os.makedirs(EDA_DIR, exist_ok=True)

    # 1. Loan approval distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Loan_Status", hue="Loan_Status",
                  palette="Set2", legend=False)
    plt.title("Loan Approval Distribution")
    plt.xlabel("Loan Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "loan_status_distribution.png"), dpi=110)
    plt.close()

    # 2. Applicant income distribution
    plt.figure(figsize=(5, 4))
    sns.histplot(df["ApplicantIncome"], kde=True, color="#4c72b0")
    plt.title("Applicant Income Distribution")
    plt.xlabel("Applicant Income")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "income_distribution.png"), dpi=110)
    plt.close()

    # 3. Loan amount distribution
    plt.figure(figsize=(5, 4))
    sns.histplot(df["LoanAmount"], kde=True, color="#55a868")
    plt.title("Loan Amount Distribution")
    plt.xlabel("Loan Amount (in thousands)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "loan_amount_distribution.png"), dpi=110)
    plt.close()

    # 4. Credit history vs loan status
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Credit_History", hue="Loan_Status", palette="Set1")
    plt.title("Credit History vs Loan Status")
    plt.xlabel("Credit History")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "credit_history_vs_status.png"), dpi=110)
    plt.close()

    # 5. Education vs loan status
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Education", hue="Loan_Status", palette="Set1")
    plt.title("Education vs Loan Status")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "education_vs_status.png"), dpi=110)
    plt.close()

    # 6. Property area vs loan status
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Property_Area", hue="Loan_Status", palette="Set1")
    plt.title("Property Area vs Loan Status")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "property_area_vs_status.png"), dpi=110)
    plt.close()

    # 7. Correlation heatmap (numeric features)
    numeric_df = df[prep.BASE_NUMERIC_FEATURES + ["TotalIncome", "DebtToIncomeRatio"]]
    plt.figure(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=110)
    plt.close()


# ---------------------------------------------------------------------------
# Model comparison visualizations
# ---------------------------------------------------------------------------
def generate_dashboard_plots(results: dict, confusions: dict) -> None:
    os.makedirs(DASHBOARD_DIR, exist_ok=True)

    models = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1_score"]

    # Grouped bar chart comparing metrics across models
    x = np.arange(len(metric_names))
    width = 0.25

    plt.figure(figsize=(7, 5))
    for i, model_name in enumerate(models):
        values = [results[model_name][m] for m in metric_names]
        plt.bar(x + i * width, values, width, label=model_name)

    plt.xticks(x + width, ["Accuracy", "Precision", "Recall", "F1-score"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DASHBOARD_DIR, "metrics_comparison.png"), dpi=110)
    plt.close()

    # Confusion matrices, one per model
    for model_name, cm in confusions.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm),
                                       display_labels=["Rejected", "Approved"])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        plt.savefig(os.path.join(DASHBOARD_DIR, f"confusion_{safe_name}.png"), dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate():
    raw_df = prep.load_dataset(DATASET_PATH)

    # Cleaning + feature engineering pass used for EDA and for X/y prep
    eda_df = prep.clean_data(raw_df)
    eda_df = prep.engineer_features(eda_df)
    generate_eda_plots(eda_df)

    X, y = prep.prepare_features_and_target(raw_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, max_depth=8
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    confusions = {}
    fitted_pipelines = {}

    for name, clf in candidate_models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", prep.build_preprocessor()),
            ("classifier", clf),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
        }
        confusions[name] = confusion_matrix(y_test, y_pred).tolist()
        fitted_pipelines[name] = pipeline

        print(f"[{name}] " + ", ".join(f"{k}={v}" for k, v in results[name].items()))

    # Best model selection based on actual test F1-score (handles the
    # class imbalance in the approved/rejected labels better than raw
    # accuracy would).
    best_model_name = max(results, key=lambda n: results[n]["f1_score"])
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"\nSelected best model: {best_model_name}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    metrics_payload = {
        "results": results,
        "confusion_matrices": confusions,
        "best_model": best_model_name,
        "selection_metric": "f1_score",
        "test_set_size": int(len(y_test)),
        "train_set_size": int(len(y_train)),
        "class_distribution": {
            "Approved (Y)": int((y == 1).sum()),
            "Rejected (N)": int((y == 0).sum()),
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    generate_dashboard_plots(results, confusions)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"EDA plots saved to: {EDA_DIR}")
    print(f"Dashboard plots saved to: {DASHBOARD_DIR}")


if __name__ == "__main__":
    train_and_evaluate()
