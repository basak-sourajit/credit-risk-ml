import numpy as np

def population_stability_index(expected, actual, buckets=10):
    def scale_range(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

    breakpoints = np.linspace(0, 1, buckets + 1)

    expected_scaled = scale_range(expected, expected.min(), expected.max())
    actual_scaled = scale_range(actual, expected.min(), expected.max())

    psi = 0
    for i in range(buckets):
        expected_pct = (
            (expected_scaled >= breakpoints[i]) & (expected_scaled < breakpoints[i + 1])
        ).mean()
        actual_pct = (
            (actual_scaled >= breakpoints[i]) & (actual_scaled < breakpoints[i + 1])
        ).mean()

        if actual_pct > 0 and expected_pct > 0:
            psi += (expected_pct - actual_pct) * np.log(expected_pct / actual_pct)

    return psi
