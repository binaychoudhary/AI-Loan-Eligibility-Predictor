# Loan Eligibility Prediction Using Artificial Intelligence and Machine Learning

A B.Tech CSE minor project that predicts whether a loan applicant is likely
to be approved, based on personal and financial information such as income,
credit history, employment status, education, and property area.

## Project Objective

To build a machine learning-based decision-support system that assists in
loan eligibility screening by learning patterns from historical loan
application data, and to present that model through a usable web
application.

## Motivation

Manual loan eligibility screening is time-consuming and can be inconsistent
across reviewers. A data-driven model can provide a fast, consistent, and
explainable first estimate to support (not replace) human underwriting
decisions.

## Scope

- Predicts a binary outcome: loan **approved** or **rejected**.
- Trained on structured applicant data (no external credit bureau or banking
  integrations).
- Intended as an academic decision-support prototype, not a production
  banking system.

## Technology Stack

- Python 3
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- joblib
- Flask, HTML, CSS
- Jupyter Notebook (for exploratory analysis)
- Visual Studio Code

## Dataset

`dataset/loan_data.csv` — 614 loan applications with the following fields:

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Married | Applicant's marital status |
| Dependents | Number of dependents (0, 1, 2, 3+) |
| Education | Graduate / Not Graduate |
| Self_Employed | Yes / No |
| ApplicantIncome | Applicant's monthly income |
| CoapplicantIncome | Co-applicant's monthly income |
| LoanAmount | Requested loan amount (in thousands) |
| Loan_Amount_Term | Loan repayment term, in months |
| Credit_History | Whether the applicant meets credit guidelines (1) or not (0) |
| Property_Area | Urban / Semiurban / Rural |
| Loan_Status | Target: Y (approved) / N (rejected) |

## Preprocessing

Implemented in `preprocessing.py`, shared by both training and inference so
the same transformations are applied consistently everywhere:

- **Missing values:** mode imputation for categorical columns, median
  imputation for numerical columns (also re-applied inside the model
  pipeline via `SimpleImputer`, as a safeguard).
- **Encoding:** one-hot encoding of categorical variables.
- **Scaling:** standard scaling of numeric variables.
- **Train/test split:** stratified 80/20 split on the target label, to
  preserve the approved/rejected class ratio in both sets.
- **Skew correction:** log1p transform applied to numeric columns whose
  skewness exceeds 0.75 (typically income and loan amount).

## Feature Engineering

- **Total Income** — combined applicant and co-applicant income.
- **Debt-to-Income Ratio** — requested loan amount relative to total
  income, a standard indicator of repayment capacity.

## Algorithms

Three classifiers are trained and compared on identical train/test splits:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Evaluation Metrics

Computed from an actual run against the dataset (see
`model/metrics.json` for the current numbers, and the `/dashboard` page
for a live view):

| Model | Accuracy | Precision | Recall | F1-score | 5-fold CV Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 99.19% | 100.00% | 99.06% | 99.53% | 98.37% ± 1.39% |
| Decision Tree | 100.00% | 100.00% | 100.00% | 100.00% | 97.96% ± 2.14% |
| Random Forest | 99.19% | 99.07% | 100.00% | 99.53% | 98.57% ± 1.04% |

Confusion matrices for all three models are generated in
`static/images/dashboard/`.

> These figures come from this project's own supplied dataset, which
> follows fairly clean, rule-consistent approval patterns; results on a
> noisier real-world dataset would typically be lower. Re-run
> `train_model.py` any time the dataset changes to refresh these numbers.

## Model Selection

The best-performing model is chosen automatically in `train_model.py`
based on the highest **F1-score** on the held-out test set (F1 is used
rather than raw accuracy because the dataset's approved/rejected classes
are imbalanced). The selected model is not hardcoded — whichever model
wins on that run's metrics is the one saved and deployed.

## Project Workflow

```
Loan Applicant Dataset
        |
Data Cleaning & Preprocessing
        |
EDA & Feature Engineering
        |
Train/Test Split (stratified)
        |
Logistic Regression | Decision Tree | Random Forest
        |
Model Evaluation
        |
Best Model Selection
        |
Save Model using joblib
        |
Flask Web Application
        |
Loan Eligibility Prediction
```

## Project Structure

```
loan-eligibility-predictor/
├── app.py                  # Flask web application
├── train_model.py          # Training, evaluation, EDA & dashboard chart generation
├── preprocessing.py        # Shared cleaning / feature engineering / pipeline logic
├── model/
│   ├── loan_model.pkl      # Saved best-performing pipeline (joblib)
│   └── metrics.json        # Saved evaluation results for the dashboard
├── dataset/
│   └── loan_data.csv
├── templates/
│   ├── base.html, index.html, result.html, dashboard.html, 404.html, 500.html
├── static/
│   ├── css/style.css
│   └── images/eda/, images/dashboard/   # Generated plots
├── notebooks/
│   └── EDA_and_Model_Training.ipynb
├── requirements.txt
└── README.md
```

## How to Run the Project

1. Create a virtual environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. (Re-)train the models and regenerate metrics/plots:
   ```
   python train_model.py
   ```
   This produces `model/loan_model.pkl`, `model/metrics.json`, and the EDA
   / dashboard charts under `static/images/`.
3. Run the Flask app:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000/` in a browser.

## How to Use the Flask Application

- **Check Eligibility** (`/`): fill in the applicant form and submit to get
  a prediction, an applicant summary, and the model used.
- **Model Dashboard** (`/dashboard`): view accuracy/precision/recall/F1
  comparisons, confusion matrices, and the EDA visualizations.

Invalid input (missing fields, non-numeric values, out-of-range values) is
caught by server-side validation and shown back to the user with a clear
message instead of a raw error page. Model-loading or prediction failures
are also caught and shown as a friendly message rather than a stack trace.

## Limitations

- Trained on a limited, structured dataset — does not incorporate real
  credit bureau data or broader financial history.
- A prediction is an estimate from learned patterns, not a guaranteed
  underwriting decision.
- Class imbalance (far more approvals than rejections in the dataset)
  means rejection cases have less data to learn from.

## Future Scope

- Incorporate a larger and more diverse applicant dataset.
- Add explainability (e.g. feature-importance breakdown per prediction).
- Add authentication if this were extended into a multi-user tool.
- Periodic model retraining as new labelled data becomes available.

---

This project is a decision-support system intended to assist, not replace,
human judgement in loan eligibility assessment.
