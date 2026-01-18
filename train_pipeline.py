import os
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split

from data_validation.expectations import validate_dataframe
from features.build_features import build_features
from models.train_lgbm import train_lgbm
from models.calibrate import calibrate_model
from evaluation.metrics import evaluate
from experiments.mlflow_tracking import start_experiment, log_params, log_metrics

# Import drift modules
from monitoring.data_drift import population_stability_index
from monitoring.model_drift import prediction_drift


def main():
    # ----------------------------
    # Load config
    # ----------------------------
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)

    # ----------------------------
    # Paths
    # ----------------------------
    raw_path = "data/raw/loans.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # ----------------------------
    # Load raw data
    # ----------------------------
    df = pd.read_csv(raw_path)

    # ----------------------------
    # Validate data
    # ----------------------------
    validation = validate_dataframe(df)
    if not validation["success"]:
        raise ValueError("Data validation failed")

    # ----------------------------
    # Target mapping
    # ----------------------------
    df["target"] = df["loan_status"].apply(lambda x: 1 if x.lower() == "default" else 0)

    # ----------------------------
    # Feature engineering
    # ----------------------------
    df = build_features(df)

    # ----------------------------
    # Split features and target
    # ----------------------------
    X = df.drop(columns=["loan_status", "target"])
    y = df["target"]

    # Train / temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - config["data"]["train_size"]),
        random_state=config["data"]["random_state"], stratify=y
    )

    # Valid / test split
    valid_size_relative = config["data"]["valid_size"] / (
        config["data"]["valid_size"] + config["data"]["test_size"]
    )

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - valid_size_relative),
        random_state=config["data"]["random_state"], stratify=y_temp
    )

    # ----------------------------
    # Ensure categorical columns are consistent
    # ----------------------------
    for col in config["features"]["categorical"]:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")
            X_test[col] = X_test[col].astype("category")

    # ----------------------------
    # Save splits
    # ----------------------------
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_valid.to_csv(os.path.join(processed_dir, "X_valid.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_valid.to_csv(os.path.join(processed_dir, "y_valid.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
    print(f"Train/valid/test splits saved to {processed_dir}")

    # ----------------------------
    # Start MLflow experiment
    # ----------------------------
    start_experiment("credit-risk-model")

    # ----------------------------
    # Train model
    # ----------------------------
    model = train_lgbm(
        X_train, y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        **config["model"]
    )

    # ----------------------------
    # Calibrate model
    # ----------------------------
    calibrated_model = calibrate_model(model, X_valid, y_valid)

    # ----------------------------
    # Predict & evaluate
    # ----------------------------
    y_prob_train = calibrated_model.predict_proba(X_train)[:, 1]
    y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_prob_test)

    # ----------------------------
    # Calculate Data Drift (PSI) for numeric columns
    # ----------------------------
    psi_metrics = {}
    numeric_cols = config["features"]["numerical"]
    for col in numeric_cols:
        psi = population_stability_index(X_train[col], X_test[col])
        psi_metrics[col] = psi

    # ----------------------------
    # Calculate Model Drift
    # ----------------------------
    pred_drift = prediction_drift(y_prob_train, y_prob_test)

    # ----------------------------
    # Log everything to MLflow
    # ----------------------------
    log_params(config["model"])
    log_metrics(metrics)
    log_metrics({"prediction_drift": pred_drift})
    for col, psi in psi_metrics.items():
        log_metrics({f"psi_{col}": psi})

    # ----------------------------
    # Save model
    # ----------------------------
    joblib.dump(calibrated_model, "model.joblib")

    print("Training complete. Metrics:", metrics)
    print("PSI metrics:", psi_metrics)
    print("Prediction drift:", pred_drift)


if __name__ == "__main__":
    main()
