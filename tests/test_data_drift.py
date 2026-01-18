import numpy as np
from monitoring.data_drift import population_stability_index

def test_psi_no_drift():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(0, 1, 1000)
    psi = population_stability_index(ref, cur)
    assert psi < 0.1

def test_psi_with_drift():
    ref = np.random.normal(0, 1, 1000)
    cur = np.random.normal(2, 1, 1000)
    psi = population_stability_index(ref, cur)
    assert psi > 0.1
