import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Credit history length
    df["earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"])
    df["credit_history_length"] = (
        pd.Timestamp("today") - df["earliest_cr_line"]
    ).dt.days // 365

    # Log transforms
    # df["log_annual_inc"] = df["annual_inc"].apply(lambda x: 0 if x <= 0 else pd.np.log(x))
    df["log_annual_inc"] = df["annual_inc"].apply(lambda x: 0 if x <= 0 else np.log(x))

    # Drop raw date
    df.drop(columns=["earliest_cr_line"], inplace=True)

    return df
