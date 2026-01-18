import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split

from data_validation.expectations import validate_dataframe
from features.build_features import build_features
from models.train_lgbm import train_lgbm
from models.calibrate import calibrate_model
from evaluation.metrics import evaluate
from experiments.mlflow_tracking import (
    start_experiment, log_params, log_metrics
)

def main():
    # Load config
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)

    # Load data
    df = pd.read_csv("data/samples/sample_loans.csv")

    # Validate data
    validation = validate_dataframe(df)
    if not validation["success"]:
        raise ValueError("Data validation failed")

    # Target mapping
    df["target"] = df["loan_status"].apply(
        lambda x: 1 if x == "Default" else 0
    )

    # Feature engineering
    df = build_features(df)

    X = df.drop(columns=["loan_status", "target"])
    y = df["target"]

    # Train / valid / test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Start MLflow
    start_experiment("credit-risk-model")

    # Train model
    model = train_lgbm(
        X_train, y_train,
        X_valid, y_valid,
        config["model"]
    )

    # Calibrate
    calibrated_model = calibrate_model(model, X_valid, y_valid)

    # Evaluate
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_prob)

    log_params(config["model"])
    log_metrics(metrics)

    # Save model
    joblib.dump(calibrated_model, "model.joblib")

    print("Training complete:", metrics)

if __name__ == "__main__":
    main()
