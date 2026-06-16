import pandas as pd

# -----------------------
# Load scored dataset
# -----------------------

df = pd.read_csv(
    "datasets/credit_data_scored.csv"
)

# -----------------------
# Policy Threshold
# -----------------------

RISK_THRESHOLD = 0.40

# -----------------------
# Approved Customers
# -----------------------

approved = df[
    df["predicted_default_probability"]
    < RISK_THRESHOLD
]

# -----------------------
# Metrics
# -----------------------

approval_rate = (
    len(approved)
    / len(df)
) * 100

avg_risk = approved[
    "predicted_default_probability"
].mean()

# Revenue

approved["revenue"] = (
    approved["loan_amount"]
    * 0.15
)

# Expected Loss

approved["expected_loss"] = (
    approved[
        "predicted_default_probability"
    ]
    * approved["loan_amount"]
    * 0.30
)

# Profit

approved["profit"] = (
    approved["revenue"]
    - approved["expected_loss"]
)

# -----------------------
# Results
# -----------------------

print("\nPolicy Threshold:")
print(RISK_THRESHOLD)

print("\nApproval Rate:")
print(round(approval_rate, 2))

print("\nAverage Risk:")
print(round(avg_risk, 4))

print("\nRevenue:")
print(
    round(
        approved["revenue"].sum(),
        2
    )
)

print("\nExpected Loss:")
print(
    round(
        approved["expected_loss"].sum(),
        2
    )
)

print("\nProfit:")
print(
    round(
        approved["profit"].sum(),
        2
    )
)