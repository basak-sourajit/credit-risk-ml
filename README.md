# Credit Risk Modeling Platform
## Overview

This project implements an end-to-end credit risk machine learning platform designed to estimate the Probability of Default (PD) for loan applicants.
It mirrors how credit risk models are built, validated, explained, and monitored in regulated financial institutions such as banks.

The system covers the full ML lifecycle:

- Data validation
- Feature engineering
- Model training & calibration
- Explainability (SHAP)
- Drift monitoring
- Experiment tracking
- Production-ready inference API

## Business Problem

Financial institutions must assess credit risk to:
- Approve or reject loan applications
- Price loans based on risk
- Manage portfolio-level exposure
- Meet regulatory and model risk requirements

The objective is not just high accuracy, but:
- Stable and explainable predictions
- Cost-sensitive decision making
- Strong governance and monitoring

## Solution Architecture
Raw Data
   ↓
Data Validation (Great Expectations)
   ↓
Feature Engineering
   ↓
Model Training (Baseline + LightGBM)
   ↓
Probability Calibration
   ↓
Explainability (SHAP)
   ↓
Model & Data Drift Monitoring
   ↓
FastAPI Inference Service

## Dataset

- Source: LendingClub public loan dataset
- Target: Loan default (binary classification)
- Why this dataset:
    -Realistic credit features
    -Noisy, imperfect data (like real banking data)
    -Commonly used in financial risk modeling

## Repository Structure
credit-risk-ml/
├── data/                 # Raw and processed data
├── data_validation/      # Data quality & schema checks
├── features/             # Feature engineering logic
├── models/               # Model training & calibration
├── evaluation/           # Risk-focused evaluation metrics
├── explainability/       # SHAP-based explainability
├── monitoring/           # Data & model drift detection
├── experiments/          # MLflow experiment tracking
├── api/                  # FastAPI inference service
├── tests/                # Unit tests
├── configs/              # Config-driven modeling
├── Dockerfile
├── requirements.txt
└── README.md

## Modeling Approach
### Baseline Model
- Logistic Regression
- Class-weighted
- Used as a transparent benchmark

### Production Model
- LightGBM Gradient Boosting
- Handles non-linearities and feature interactions
- Class imbalance handled via class weighting
- Early stopping to prevent overfitting

### Probability Calibration
- Isotonic calibration
- Ensures predicted PDs are well-aligned with observed default rates
- Critical for downstream risk decisions

## Evaluation Strategy

Traditional accuracy is not sufficient for credit risk.

Metrics used:
- ROC-AUC
- Precision-Recall AUC
- Cost-sensitive decision thresholds
- Score distribution stability

### Decision logic reflects business tradeoffs:
- False negatives (missed defaulters) are more costly than false positives

## Explainability & Governance

Explainability is mandatory in regulated environments.

Implemented using SHAP:
- Global feature importance
- Local explanations per prediction
- Top contributing risk drivers (“reason codes”)

These explanations can be used by:

- Risk analysts
- Credit officers
- Model risk management teams

## Data & Model Monitoring
### Data Drift
- Population Stability Index (PSI)
- Detects shifts in input feature distributions

### Model Drift
- Monitoring prediction score distributions
- Mean PD shift tracking over time

These checks help identify when:
- Data quality degrades
- Customer behavior changes
- Model retraining is required

## Experiment Tracking

All experiments are tracked using MLflow:
- Model parameters
- Evaluation metrics
- Artifacts (models, plots)

This enables:
- Reproducibility
- Auditability
- Controlled model iteration

## Inference API

A production-ready FastAPI service exposes real-time predictions.

### Endpoint

POST /predict

### Example Response
{
  "probability_of_default": 0.27,
  "decision": "REVIEW"
}


Decision thresholds are configurable and reflect risk appetite.

## Deployment
- Fully Dockerized
- Reproducible environment
- Suitable for internal deployment or cloud environments

### Run locally:

docker build -t credit-risk-ml .
docker run -p 8000:8000 credit-risk-ml

## Fairness & Bias
Basic demographic parity and equal opportunity analysis are implemented to assess model behavior across customer segments. This reflects responsible ML practices in regulated environments.

### Example Usage:

bias = demographic_parity(
    y_pred=(y_prob > 0.3).astype(int),
    sensitive_feature=df["home_ownership"]
)

## Limitations & Future Work
- Batch retraining automation not implemented
- Limited macroeconomic features

Planned improvements:
- Feature store integration
- Automated retraining triggers
- Portfolio-level risk aggregation