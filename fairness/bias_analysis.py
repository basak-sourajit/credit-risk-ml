import pandas as pd
import numpy as np

def demographic_parity(y_pred, sensitive_feature):
    groups = sensitive_feature.unique()
    rates = {}

    for g in groups:
        rates[g] = y_pred[sensitive_feature == g].mean()

    return rates

def equal_opportunity(y_true, y_pred, sensitive_feature):
    groups = sensitive_feature.unique()
    tpr = {}

    for g in groups:
        idx = sensitive_feature == g
        positives = (y_true[idx] == 1)
        if positives.sum() == 0:
            tpr[g] = None
        else:
            tpr[g] = (y_pred[idx][positives] == 1).mean()

    return tpr
