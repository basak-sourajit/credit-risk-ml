import numpy as np

def prediction_drift(reference_scores, current_scores):
    return abs(
        np.mean(reference_scores) - np.mean(current_scores)
    )
