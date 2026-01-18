import pandas as pd
from features.build_features import build_features

def test_build_features_output():
    df = pd.DataFrame({
        "annual_inc": [50000, 80000],
        "earliest_cr_line": ["2010-01-01", "2015-01-01"]
    })

    out = build_features(df)

    assert "credit_history_length" in out.columns
    assert "log_annual_inc" in out.columns
    assert out.isnull().sum().sum() == 0
