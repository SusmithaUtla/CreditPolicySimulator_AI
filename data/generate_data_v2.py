import pandas as pd
import numpy as np

np.random.seed(42)

N = 10000

df = pd.DataFrame()

# ----------------------------------
# Customer Information
# ----------------------------------

df["customer_id"] = range(1, N + 1)

df["age"] = np.random.randint(
    21,
    65,
    N
)

df["income"] = np.random.randint(
    300000,  
    2500000,
    N
)

df["employment_years"] = np.random.randint(
    0,
    30,
    N
)

df["credit_score"] = np.random.randint(
    300,
    850,
    N
)

# ----------------------------------
# Loan Information
# ----------------------------------

df["loan_amount"] = (
    df["income"]
    * np.random.uniform(0.1, 0.8, N)
)

df["debt_to_income_ratio"] = (
    df["loan_amount"]
    / df["income"]
)

# ----------------------------------
# Additional Risk Signals
# ----------------------------------

df["existing_loans_count"] = np.random.randint(
    0,
    8,
    N
)

df["missed_payments_last_year"] = np.random.randint(
    0,
    6,
    N
)

# ----------------------------------
# Risk Score Calculation
# ----------------------------------

def calculate_risk_score(row):

    score_factor = (
        850 - row["credit_score"]
    ) / 550

    dti_factor = row[
        "debt_to_income_ratio"
    ]

    missed_payment_factor = (
        row["missed_payments_last_year"]
        / 5
    )

    loan_factor = (
        row["existing_loans_count"]
        / 7
    )

    employment_factor = (
        1 /
        (row["employment_years"] + 1)
    )

    income_factor = (
        1 - row["income"]/2500000
    )
    risk_score = (
        0.35 * score_factor
        + 0.20 * dti_factor
        + 0.15 * missed_payment_factor
        + 0.10 * loan_factor
        + 0.10 * employment_factor
        + 0.10 * income_factor
    )

    return min(
        max(risk_score, 0),
        1
    )

df["risk_score"] = df.apply(
    calculate_risk_score,
    axis=1
)

# ----------------------------------
# Default Logic
# ----------------------------------

def assign_default(risk):

    if risk >= 0.65:
        return 1

    elif risk <= 0.35:
        return 0

    else:
        probability = np.clip(
            (risk - 0.35) / 0.3,
            0,
            1
        )
        return int(np.random.rand() < probability)

df["default"] = df[
    "risk_score"
].apply(assign_default)

# ----------------------------------
# Risk Category
# ----------------------------------

df["risk_category"] = pd.cut(
    df["risk_score"],
    bins=[0, 0.3, 0.6, 1],
    labels=[
        "Low",
        "Medium",
        "High"
    ]
)

# ----------------------------------
# Save
# ----------------------------------

df.to_csv(
    "datasets/credit_data_v2.csv",
    index=False
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nDefault Distribution:")
print(
    df["default"]
    .value_counts()
)

print("\nRisk Category Distribution:")
print(
    df["risk_category"]
    .value_counts()
)

print(
    "\nDataset Generated Successfully!"
)