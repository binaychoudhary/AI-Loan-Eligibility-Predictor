"""
preprocessing.py
----------------
Data cleaning, feature engineering, and preprocessing-pipeline construction
for the Loan Eligibility Prediction project.

Important consistency rule:
- Do not calculate a transformation from the current dataframe's skewness.
- Training and inference must construct the same raw feature columns.
- Imputation, scaling, and one-hot encoding are fitted inside the sklearn
  Pipeline in train_model.py and reused automatically during prediction.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Column definitions
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

ENGINEERED_NUMERIC_FEATURES = [
    "TotalIncome",
    "DebtToIncomeRatio",
]

ALL_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
TARGET_COLUMN = "Loan_Status"

# Retained for compatibility/documentation. These columns are NOT transformed
# based on per-dataframe skewness, because that would produce inconsistent
# training-time and single-applicant inference-time representations.
SKEWED_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "TotalIncome",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw applicant dataset from CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a defensive copy and fill missing values for EDA/reporting.
    Model training uses the imputers fitted inside the sklearn Pipeline.
    """
    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])

    for col in BASE_NUMERIC_FEATURES:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add TotalIncome and DebtToIncomeRatio.

    LoanAmount is expressed in thousands in the source dataset. The ratio
    below is loan principal divided by total income; it is a proxy feature,
    not a true monthly debt-to-income ratio because income periodicity and
    monthly debt obligations are not available in this dataset.
    """
    df = df.copy()

    applicant_income = pd.to_numeric(df["ApplicantIncome"], errors="coerce")
    coapplicant_income = pd.to_numeric(df["CoapplicantIncome"], errors="coerce")
    loan_amount = pd.to_numeric(df["LoanAmount"], errors="coerce")

    df["TotalIncome"] = applicant_income + coapplicant_income

    # Preserve missing values for the pipeline's SimpleImputer.
    denominator = df["TotalIncome"] + 1.0
    df["DebtToIncomeRatio"] = (loan_amount * 1000.0) / denominator
    df.loc[denominator.isna() | loan_amount.isna(), "DebtToIncomeRatio"] = np.nan

    return df


def transform_skewed(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Compatibility no-op.

    The old implementation decided whether to apply log1p based on the
    skewness of the dataframe passed in. That decision differed for the full
    training dataset and a single applicant. No skew-based transformation is
    now applied. If a log transform is added later, it should be a fixed,
    explicit transformer fitted as part of the sklearn Pipeline.
    """
    return df.copy()


def prepare_features_and_target(df: pd.DataFrame):
    """
    Clean and engineer features, then return X and y.

    Numeric transformations are not selected dynamically from the dataset.
    Missing values are imputed by the fitted pipeline, avoiding leakage from
    computing imputation/scaling statistics before the train/test split.
    """
    df = df.copy()
    df = engineer_features(df)

    y = None
    if TARGET_COLUMN in df.columns:
        mapped_target = df[TARGET_COLUMN].map({"Y": 1, "N": 0})
        if mapped_target.isna().any():
            # Permit already-numeric labels, but fail clearly for unknown text.
            numeric_target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
            if numeric_target.isna().any():
                bad_values = df.loc[numeric_target.isna(), TARGET_COLUMN].unique()
                raise ValueError(
                    f"Unexpected values in {TARGET_COLUMN}: {bad_values!r}"
                )
            y = numeric_target
        else:
            y = mapped_target.astype(int)

    feature_cols = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required feature columns: {missing_cols}")

    X = df[feature_cols].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn preprocessing stage:
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

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, ALL_NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def prepare_single_applicant(raw_input: dict) -> pd.DataFrame:
    """
    Build a single-row dataframe using the exact feature construction used
    during training. The fitted Pipeline performs imputation, scaling and
    encoding during model.predict().
    """
    df = pd.DataFrame([raw_input])
    df = engineer_features(df)

    feature_cols = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Applicant input is missing required features: {missing_cols}")

    return df[feature_cols].copy()
