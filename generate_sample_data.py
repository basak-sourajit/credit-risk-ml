# generate_sample_data.py

import os
import pandas as pd
import numpy as np

# ----------------------------
# Settings
# ----------------------------
np.random.seed(42)
n_rows = 1000
raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)
output_file = os.path.join(raw_dir, "loans.csv")

# ----------------------------
# Generate random data
# ----------------------------
loan_amnt = np.random.randint(5000, 35000, size=n_rows)
annual_inc = np.random.randint(30000, 120000, size=n_rows)
dti = np.round(np.random.uniform(5, 30, size=n_rows), 1)
open_acc = np.random.randint(1, 15, size=n_rows)
earliest_cr_line = pd.to_datetime(
    np.random.randint(
        pd.Timestamp("1990-01-01").value // 10**9,
        pd.Timestamp("2020-12-31").value // 10**9,
        size=n_rows
    ),
    unit='s'
)
loan_status = np.random.choice(["Fully Paid", "Default"], size=n_rows, p=[0.7, 0.3])
home_ownership = np.random.choice(["RENT", "MORTGAGE", "OWN"], size=n_rows, p=[0.5, 0.4, 0.1])

# ----------------------------
# Create DataFrame
# ----------------------------
df = pd.DataFrame({
    "loan_amnt": loan_amnt,
    "annual_inc": annual_inc,
    "dti": dti,
    "open_acc": open_acc,
    "earliest_cr_line": earliest_cr_line,
    "loan_status": loan_status,
    "home_ownership": home_ownership
})

# ----------------------------
# Save to CSV in raw folder
# ----------------------------
df.to_csv(output_file, index=False)
print(f"Sample data saved to {output_file}")
