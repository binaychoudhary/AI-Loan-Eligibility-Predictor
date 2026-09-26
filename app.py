"""
app.py
------
Flask web application for the Loan Eligibility Prediction project.

Routes:
    /            Applicant input form
    /predict     Handles form submission, runs prediction, shows result
    /dashboard   Model performance dashboard (accuracy/precision/recall/F1,
                 confusion matrices, EDA visualizations)

The saved model is a full scikit-learn Pipeline (preprocessing + classifier),
so the exact same preprocessing used at training time is applied
automatically at inference time -- there is no separate preprocessing step
to keep in sync here.

NEW in this version:
    - Prediction confidence score (from model.predict_proba, when available)
    - A simple loan repayment schedule (EMI + remaining balance) shown at
      month 1, 5, 10, 12, and every 12 months thereafter up to the full term.
      NOTE: the dataset/form doesn't include an interest rate, so a default
      ASSUMED_ANNUAL_INTEREST_RATE is used below. Replace with the real rate
      if your dataset/synopsis specifies one.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

import preprocessing as prep

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "loan_model.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")

# Assumed annual interest rate (%) used only for the illustrative repayment
# schedule -- the classifier itself does not use or need this value.
ASSUMED_ANNUAL_INTEREST_RATE = 10.0

VALID_OPTIONS = {
    "Gender": {"Male", "Female"},
    "Married": {"Yes", "No"},
    "Dependents": {"0", "1", "2", "3+"},
    "Education": {"Graduate", "Not Graduate"},
    "Self_Employed": {"Yes", "No"},
    "Property_Area": {"Urban", "Semiurban", "Rural"},
}

_model = None
_model_error = None


def get_model():
    """Lazily load the trained model pipeline, caching it after first load."""
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a friendly message
            _model_error = str(exc)
    return _model, _model_error


def load_metrics():
    """Load the saved evaluation metrics produced by train_model.py."""
    if not os.path.exists(METRICS_PATH):
        return None
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def validate_and_parse_form(form) -> tuple[dict, list]:
    """
    Validate raw form input. Returns (parsed_values, error_messages).
    If error_messages is non-empty, parsed_values should not be trusted.
    """
    errors = []
    parsed = {}

    for field, allowed in VALID_OPTIONS.items():
        value = form.get(field, "").strip()
        if value not in allowed:
            errors.append(f"Please select a valid value for '{field.replace('_', ' ')}'.")
        else:
            parsed[field] = value

    numeric_fields = {
        "ApplicantIncome": (0, None),
        "CoapplicantIncome": (0, None),
        "LoanAmount": (0.01, None),
        "Loan_Amount_Term": (1, None),
    }
    for field, (min_val, _max_val) in numeric_fields.items():
        raw_value = form.get(field, "").strip()
        try:
            value = float(raw_value)
            if value < min_val:
                errors.append(f"'{field.replace('_', ' ')}' must be at least {min_val}.")
            else:
                parsed[field] = value
        except (TypeError, ValueError):
            errors.append(f"'{field.replace('_', ' ')}' must be a valid number.")

    credit_history_raw = form.get("Credit_History", "").strip()
    if credit_history_raw not in {"1", "0"}:
        errors.append("Please select a valid credit history option.")
    else:
        parsed["Credit_History"] = float(credit_history_raw)

    return parsed, errors


def get_prediction_confidence(model, input_df, predicted_class):
    """
    Return the model's confidence (%) in the predicted class, using
    predict_proba if the underlying estimator supports it. Returns None
    if the model can't produce probabilities (e.g. some SVM configs).
    """
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        idx = classes.index(predicted_class)
        return round(float(proba[idx]) * 100, 1)
    except Exception:  # noqa: BLE001 - confidence is a nice-to-have, never fatal
        return None


def build_repayment_schedule(loan_amount_thousands, term_months, annual_rate=ASSUMED_ANNUAL_INTEREST_RATE):
    """
    Build an illustrative amortization schedule for an approved loan.

    loan_amount_thousands: LoanAmount as entered on the form (dataset convention
                            is loan amount in thousands of rupees).
    term_months:            Loan_Amount_Term in months.
    annual_rate:            Assumed annual interest rate (%), since the dataset
                            doesn't provide one.

    Returns (schedule, monthly_emi) where schedule is a list of dicts:
        {"month": int, "emi": float, "principal_paid": float,
         "interest_paid": float, "balance": float}
    for month 1, 5, 10, 12, and every 12 months after that, plus the final month.
    """
    principal = float(loan_amount_thousands) * 1000
    n = max(int(term_months), 1)
    monthly_rate = (annual_rate / 100) / 12

    if monthly_rate == 0:
        emi = principal / n
    else:
        factor = (1 + monthly_rate) ** n
        emi = principal * monthly_rate * factor / (factor - 1)
    emi = round(emi, 2)

    # Checkpoints: 1, 5, 10, 12, then every 12 months, always including the last month.
    checkpoints = sorted(set([1, 5, 10, 12] + list(range(12, n + 1, 12)) + [n]))
    checkpoints = [m for m in checkpoints if 1 <= m <= n]

    schedule = []
    balance = principal
    for month in range(1, n + 1):
        interest_paid = balance * monthly_rate
        principal_paid = emi - interest_paid
        balance = max(balance - principal_paid, 0)
        if month in checkpoints:
            schedule.append({
                "month": month,
                "emi": emi,
                "principal_paid": round(principal_paid, 2),
                "interest_paid": round(interest_paid, 2),
                "balance": round(balance, 2),
            })

    return schedule, emi


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    parsed, errors = validate_and_parse_form(request.form)

    if errors:
        return render_template("index.html", errors=errors, form_data=request.form), 400

    model, model_error = get_model()
    if model is None:
        errors = [f"The prediction model could not be loaded: {model_error}"]
        return render_template("index.html", errors=errors, form_data=request.form), 500

    try:
        input_df = prep.prepare_single_applicant(parsed)
        prediction = model.predict(input_df)[0]
        confidence = get_prediction_confidence(model, input_df, prediction)
    except Exception as exc:  # noqa: BLE001
        errors = [f"Something went wrong while generating the prediction: {exc}"]
        return render_template("index.html", errors=errors, form_data=request.form), 500

    metrics = load_metrics()
    model_name = metrics.get("best_model", "Trained Model") if metrics else "Trained Model"

    is_approved = int(prediction) == 1

    repayment_schedule = None
    monthly_emi = None
    if is_approved:
        repayment_schedule, monthly_emi = build_repayment_schedule(
            parsed["LoanAmount"], parsed["Loan_Amount_Term"]
        )

    applicant_summary = {
        "Gender": parsed["Gender"],
        "Married": parsed["Married"],
        "Dependents": parsed["Dependents"],
        "Education": parsed["Education"],
        "Self Employed": parsed["Self_Employed"],
        "Applicant Income": f"₹{parsed['ApplicantIncome']:,.0f}",
        "Co-applicant Income": f"₹{parsed['CoapplicantIncome']:,.0f}",
        "Loan Amount": f"₹{parsed['LoanAmount']:,.0f} (thousands)",
        "Loan Term": f"{parsed['Loan_Amount_Term']:.0f} months",
        "Credit History": "Meets guidelines" if parsed["Credit_History"] == 1.0 else "Does not meet guidelines",
        "Property Area": parsed["Property_Area"],
    }

    return render_template(
        "result.html",
        is_approved=is_approved,
        model_name=model_name,
        confidence=confidence,
        applicant_summary=applicant_summary,
        repayment_schedule=repayment_schedule,
        monthly_emi=monthly_emi,
        assumed_interest_rate=ASSUMED_ANNUAL_INTEREST_RATE,
    )


@app.route("/dashboard")
def dashboard():
    metrics = load_metrics()
    if metrics is None:
        return render_template(
            "dashboard.html",
            metrics=None,
            errors=["Model evaluation results are not available yet. Run train_model.py first."],
        )
    return render_template("dashboard.html", metrics=metrics, errors=None)


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=True)