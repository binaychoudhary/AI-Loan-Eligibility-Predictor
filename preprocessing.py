"""
preprocessing.py
-----------------
Data cleaning, feature engineering, and preprocessing-pipeline construction
for the Loan Eligibility Prediction project.

This module is imported by both train_model.py (training time) and app.py
(inference time) so that raw applicant data is always transformed the same
way, regardless of where it is used.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column definitions (kept identical to the official synopsis feature list)
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

BASE_NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Engineered numeric features (added on top of the base numeric features)
ENGINEERED_NUMERIC_FEATURES = [
    "TotalIncome",
    "DebtToIncomeRatio",
]

ALL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES

TARGET_COLUMN = "Loan_Status"

# Numeric columns that are typically right-skewed in this dataset and
# benefit from a log1p transform before scaling.
SKEWED_COLUMNS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome"]


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw applicant dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using the methodology described in the synopsis:
      - Mode imputation for categorical columns
      - Median imputation for numerical columns

    This is a defensive/explicit cleaning pass done on the raw dataframe
    (in addition to the imputers inside the sklearn pipeline), so the
    dataset is also clean for EDA and reporting purposes.
    """
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if len(mode_val):
                df[col] = df[col].fillna(mode_val.iloc[0])

    for col in BASE_NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the engineered features described in the synopsis:
      - Combined applicant income (TotalIncome)
      - Debt-to-income ratio (DebtToIncomeRatio)

    LoanAmount in this dataset is expressed in thousands, so it is scaled
    by 1000 before comparing against monthly income when forming the ratio.
    A small epsilon avoids division-by-zero for zero-income edge cases.
    """
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    epsilon = 1.0
    df["DebtToIncomeRatio"] = (df["LoanAmount"] * 1000.0) / (df["TotalIncome"] + epsilon)

    return df


def transform_skewed(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Apply a log1p transform to numeric columns whose skewness exceeds a
    threshold, reducing the influence of long-tailed income/loan values.
    """
    df = df.copy()
    columns = columns if columns is not None else SKEWED_COLUMNS

    for col in columns:
        if col not in df.columns:
            continue
        skew = df[col].skew()
        if abs(skew) > 0.75:
            df[col] = np.log1p(df[col].clip(lower=0))

    return df


def prepare_features_and_target(df: pd.DataFrame):
    """
    Run the full cleaning -> feature engineering -> skew transform pipeline
    on a raw dataframe and split it into X (features) and y (target),
    when the target column is present. Returns (X, y_or_None).
    """
    df = clean_data(df)
    df = engineer_features(df)
    df = transform_skewed(df)

    y = None
    if TARGET_COLUMN in df.columns:
        y = df[TARGET_COLUMN].map({"Y": 1, "N": 0})
        if y.isnull().any():
            # Already numeric / unexpected labels — leave as-is
            y = df[TARGET_COLUMN]

    feature_cols = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES
    X = df[feature_cols]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer used inside every model pipeline:
      - Numeric: median imputation + standard scaling
      - Categorical: most-frequent imputation + one-hot encoding
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def prepare_single_applicant(raw_input: dict) -> pd.DataFrame:
    """
    Build a single-row dataframe from raw form input (as collected by the
    Flask app) and apply the same cleaning / feature-engineering / skew
    transforms used at training time, so inference is consistent with
    training.
    """
    df = pd.DataFrame([raw_input])
    df = engineer_features(df)   # no missing values expected from a validated form
    df = transform_skewed(df)
    feature_cols = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES
    return df[feature_cols]
