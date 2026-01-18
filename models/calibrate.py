from sklearn.calibration import CalibratedClassifierCV

def calibrate_model(model, X_valid, y_valid):
    """
    Calibrate a pre-trained classifier using validation data.
    Compatible with scikit-learn >=1.2
    """
    # cv=None now, estimator is already fitted
    calibrated = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=None)

    # Fit on validation data
    calibrated.fit(X_valid, y_valid)

    return calibrated
