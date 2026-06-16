import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

import numpy as np
from xgboost import XGBClassifier

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("datasets/credit_data_v2.csv")

# -------------------------
# Features
# -------------------------

features = [
    "credit_score",
    "age",
    "income",
    "employment_years",
    "loan_amount",
    "debt_to_income_ratio",
    # additional signals available in credit_data_v2
    "existing_loans_count",
    "missed_payments_last_year",
]

X = df[features]

# Target
y = df["default"]

# -------------------------
# Train Test Split
# -------------------------

# initial train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# create a small validation split from the training set for threshold tuning / model selection
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42
)

# -------------------------
# Train Model (handle class imbalance + hyperparam search)
# -------------------------

# balance weight: ratio of negative to positive samples in the training set
scale_pos_weight = len(y_tr[y_tr == 0]) / max(1, len(y_tr[y_tr == 1]))

# randomized search over reasonable XGBoost hyperparameters
param_dist = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0]
}

base = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42 
)

search = RandomizedSearchCV(
    estimator=base,
    param_distributions=param_dist,
    n_iter=30,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

search.fit(X_tr, y_tr)

model = search.best_estimator_

# fit final model on the full training set (tr + val) for final predictions
model.fit(pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]))

# -------------------------
# Predictions with threshold tuning
# -------------------------

# get validation probabilities to pick a decision threshold maximizing accuracy
y_prob_val = model.predict_proba(X_val)[:, 1]

best_thresh = 0.5
best_acc = 0.0
for t in np.linspace(0.0, 1.0, 101):
    y_val_pred = (y_prob_val >= t).astype(int)
    acc = accuracy_score(y_val, y_val_pred)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t

print(f"\nSelected threshold from validation: {best_thresh:.2f} (val acc={best_acc:.4f})")

# apply to test set
y_prob_test = model.predict_proba(X_test)[:, 1]
# use tuned threshold for final class predictions
y_pred_test = (y_prob_test >= best_thresh).astype(int)

# -------------------------
# Evaluation
# -------------------------

accuracy = accuracy_score(y_test, y_pred_test)
auc = roc_auc_score(y_test, y_prob_test)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nROC AUC:")
print(round(auc, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))


# add predicted probabilities for all rows and save scored dataset
df["predicted_default_probability"] = model.predict_proba(X)[:, 1]

df.to_csv(
    "datasets/credit_data_scored.csv",
    index=False
)

print(
    df[
        [
            "customer_id",
            "default",
            "predicted_default_probability"
        ]
    ].head()
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

print("\nDefault Distribution (proportions):")
print(df["default"].value_counts(normalize=True))

print("\nRisk Score Summary:")
print(df["risk_score"].describe())

# -------------------------
# Save Model and threshold
# -------------------------

# persist model
joblib.dump(
    model,
    "models/xgb_model.pkl"
)
# persist chosen threshold
with open("models/threshold.txt", "w") as f:
    f.write(str(best_thresh))

print("\nModel and threshold saved to models/")


