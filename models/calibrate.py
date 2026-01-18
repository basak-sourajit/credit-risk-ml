from sklearn.calibration import CalibratedClassifierCV

def calibrate_model(model, X_valid, y_valid):
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_valid, y_valid)
    return calibrated
