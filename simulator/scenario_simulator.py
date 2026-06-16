import pandas as pd

# -------------------------
# Load Data
# -------------------------

df = pd.read_csv(
    "datasets/credit_data_scored.csv"
)

results = []

# -------------------------
# Business Assumptions
# -------------------------

INTEREST_RATE = 0.15
LOSS_GIVEN_DEFAULT = 0.30

# -------------------------
# Simulate Thresholds
# -------------------------

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

for threshold in thresholds:

    approved = df[
        df["predicted_default_probability"]
        < threshold
    ].copy()

    approval_rate = (
        len(approved)
        / len(df)
    ) * 100

    avg_risk = approved[
        "predicted_default_probability"
    ].mean()

    revenue = (
        approved["loan_amount"]
        * INTEREST_RATE
    ).sum()

    expected_loss = (
        approved[
            "predicted_default_probability"
        ]
        * approved["loan_amount"]
        * LOSS_GIVEN_DEFAULT
    ).sum()

    profit = (
        revenue
        - expected_loss
    )

    results.append({
        "threshold": threshold,
        "approval_rate": round(
            approval_rate,
            2
        ),
        "avg_risk": round(
            avg_risk,
            4
        ),
        "revenue": round(
            revenue,
            2
        ),
        "expected_loss": round(
            expected_loss,
            2
        ),
        "profit": round(
            profit,
            2
        )
    })

# -------------------------
# Results Table
# -------------------------

results_df = pd.DataFrame(
    results
)

print("\nScenario Results\n")

print(results_df)

# -------------------------
# Best Policy
# -------------------------

best_policy = results_df.loc[
    results_df["profit"].idxmax()
]

print("\nBest Policy Found\n")

print(best_policy)