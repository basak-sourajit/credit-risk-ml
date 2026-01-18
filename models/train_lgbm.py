from lightgbm import LGBMClassifier

def train_lgbm(X_train, y_train, X_valid=None, y_valid=None, **params):
    """
    Train a LightGBM classifier with proper categorical handling and early stopping.
    Compatible with LightGBM >= 4.6.0
    """
    # Instantiate model
    model = LGBMClassifier(**params)

    fit_params = {}
    
    # If validation data is provided
    if X_valid is not None and y_valid is not None:
        fit_params["eval_set"] = [(X_valid, y_valid)]
        # Use early stopping with LightGBM callback
        from lightgbm import early_stopping
        fit_params["callbacks"] = [early_stopping(stopping_rounds=50)]

    # Fit model
    model.fit(X_train, y_train, **fit_params)

    return model
