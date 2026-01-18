from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
model = joblib.load("model.joblib")

# Features used in training
NUMERICAL_FEATURES = ["loan_amnt", "annual_inc", "dti", "open_acc", "revol_bal", "credit_history_length"]
CATEGORICAL_FEATURES = ["home_ownership"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Default values for missing features
NUMERIC_DEFAULTS = {col: 0 for col in NUMERICAL_FEATURES}
CATEGORICAL_DEFAULTS = {col: "RENT" for col in CATEGORICAL_FEATURES}  # adjust as needed

@app.post("/predict")
def predict(payload: dict):
    # Convert payload to DataFrame
    df = pd.DataFrame([payload])

    # Ensure all numerical columns exist
    for col in NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = NUMERIC_DEFAULTS[col]

    # Ensure all categorical columns exist and have correct dtype
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = CATEGORICAL_DEFAULTS[col]
        df[col] = df[col].astype("category")

    # Reorder columns to match training
    df = df[ALL_FEATURES]

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    # Decision logic
    decision = "APPROVE"
    if prob > 0.5:
        decision = "REJECT"
    elif prob > 0.3:
        decision = "REVIEW"

    return {
        "probability_of_default": round(prob, 4),
        "decision": decision,
    }
