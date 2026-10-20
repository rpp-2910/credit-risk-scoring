

import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

from predict_preprocessing import preprocess_new_data

# ==================================================
# STEP 6.2: LOAD DATA
# ==================================================

df = pd.read_csv("data/application_train.csv")
splits = pd.read_csv("data/split_assignments.csv")

df = df.merge(
    splits,
    on="SK_ID_CURR",
    how="inner",
    validate="one_to_one"
)

assert len(df) == 307511
assert df["split"].notna().all()

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

# ==================================================
# APPLY SAVED PREPROCESSING
# ==================================================

X_train = preprocess_new_data(train_df)
X_val = preprocess_new_data(val_df)

y_train = train_df["TARGET"]
y_val = val_df["TARGET"]

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

# ==================================================
# TRAIN BASELINE XGBOOST
# ==================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("\nXGBoost training completed!")

# ==================================================
# VALIDATION PREDICTIONS
# ==================================================

y_val_proba = model.predict_proba(X_val)[:, 1]

roc_auc = roc_auc_score(y_val, y_val_proba)
pr_auc = average_precision_score(y_val, y_val_proba)

print("\nVALIDATION RESULTS")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision (PR-AUC): {pr_auc:.4f}")
print(f"Positive class prevalence: {y_val.mean():.4f}")

# ==================================================
# SAVE MODEL
# ==================================================

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model.save_model(
    str(model_dir / "xgboost_model_a.json")
)

print("\nModel saved successfully!")



# ==================================================
# STEP 6.3: THRESHOLD EVALUATION
# ==================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

thresholds = [0.10, 0.20, 0.30, 0.50]

print("\nTHRESHOLD COMPARISON")

for threshold in thresholds:

    # Convert predicted probabilities into 0/1 labels
    y_pred = (y_val_proba >= threshold).astype(int)

    precision = precision_score(
        y_val, y_pred, zero_division=0
    )

    recall = recall_score(
        y_val, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_val, y_pred, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("True Positives:", tp)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Negatives:", tn)


# ==================================================
# STEP 6.4: PREDICTED PROBABILITY ANALYSIS
# ==================================================

import numpy as np

print("\nPREDICTED PROBABILITY SUMMARY")

print("Minimum:", np.min(y_val_proba))
print("Maximum:", np.max(y_val_proba))
print("Mean:", np.mean(y_val_proba))
print("Median:", np.median(y_val_proba))

print("\nPREDICTED PROBABILITY PERCENTILES")

for p in [50, 75, 90, 95, 99, 99.9]:
    print(
        f"{p}th percentile:",
        round(np.percentile(y_val_proba, p), 4)
    )

# Compare predicted and observed event rates
print("\nMEAN PREDICTED RISK VS ACTUAL EVENT RATE")
print("Mean predicted probability:", round(y_val_proba.mean(), 4))
print("Actual positive rate:", round(y_val.mean(), 4))

# Group predictions into risk ranges
risk_bins = pd.cut(
    y_val_proba,
    bins=[0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0],
    include_lowest=True
)

risk_analysis = pd.DataFrame({
    "Risk Range": risk_bins,
    "Actual Target": y_val.to_numpy(),
    "Predicted Probability": y_val_proba
})

risk_summary = risk_analysis.groupby(
    "Risk Range",
    observed=False
).agg(
    Applicants=("Actual Target", "size"),
    Actual_Event_Rate=("Actual Target", "mean"),
    Mean_Predicted_Risk=("Predicted Probability", "mean")
)

print("\nRISK GROUP ANALYSIS")
print(risk_summary.to_string())
