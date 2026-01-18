from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    prob = model.predict_proba(df)[0][1]

    decision = "APPROVE"
    if prob > 0.5:
        decision = "REJECT"
    elif prob > 0.3:
        decision = "REVIEW"

    return {
        "probability_of_default": round(prob, 4),
        "decision": decision,
    }
