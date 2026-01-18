import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate(y_true, y_prob):
    metrics = {}
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    metrics["pr_auc"] = auc(recall, precision)

    return metrics
