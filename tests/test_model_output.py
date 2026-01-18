import numpy as np

def test_prediction_bounds(model, X_sample):
    probs = model.predict_proba(X_sample)[:, 1]
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
