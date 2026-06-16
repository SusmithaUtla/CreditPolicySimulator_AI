import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("datasets/credit_data_v2.csv")

print(df.describe())

df["score_bucket"] = pd.cut(
    df["credit_score"],
    bins=[300,400,500,600,700,850]
)
print(
    df.groupby("score_bucket")["default"]
      .mean()
)

df["dti_bucket"] = pd.cut(
    df["debt_to_income_ratio"],
    bins=[0,0.2,0.4,0.6,0.8]
)

print(
    df.groupby("dti_bucket")["default"]
      .mean()
)

df["income_bucket"] = pd.qcut(
    df["income"],
    q=4
)

print(
    df.groupby("income_bucket")["default"]
      .mean()
)

print(
    df.groupby(
        "missed_payments_last_year"
    )["default"]
    .mean()
)

print(
    df.groupby(
        "existing_loans_count"
    )["default"]
    .mean()
)