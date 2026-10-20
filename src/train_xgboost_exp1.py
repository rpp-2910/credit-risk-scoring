from pathlib import Path
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from predict_preprocessing_exp1 import preprocess_new_data_exp1

# ==================================================
# 1. LOAD DATA (TRAIN & VALIDATION ONLY)
# ==================================================
print("Loading data for Experiment 1...")
df = pd.read_csv("data/application_train.csv")
splits = pd.read_csv("data/split_assignments.csv")

df = df.merge(
    splits,
    on="SK_ID_CURR",
    how="inner",
    validate="one_to_one"
)

# STRICT RULE: Train and Validation sets only.
# Test set must NOT be loaded or evaluated.
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

print(f"Training set: {train_df.shape[0]} rows")
print(f"Validation set: {val_df.shape[0]} rows")

# ==================================================
# 2. PREPROCESS USING EXPERIMENT 1 PIPELINE
# ==================================================
print("\nApplying Experiment 1 preprocessing (128 features)...")
X_train = preprocess_new_data_exp1(train_df)
X_val = preprocess_new_data_exp1(val_df)

y_train = train_df["TARGET"].to_numpy()
y_val = val_df["TARGET"].to_numpy()

assert X_train.shape == (215257, 128)
assert X_val.shape == (46127, 128)

# ==================================================
# 3. TRAIN XGBOOST WITH EXACT BASELINE HYPERPARAMETERS
# ==================================================
# STRICT RULE: No early stopping, no scale_pos_weight,
# no hyperparameter tuning. Identical to baseline model A.
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

print("\nTraining XGBoost Experiment 1 model...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("Training completed!")

# ==================================================
# 4. SAVE EXPERIMENT 1 MODEL ARTIFACT
# ==================================================
model_dir = Path("models/exp1_financial_ratios")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "xgboost_model_exp1.json"

model.save_model(str(model_path))
print(f"Saved Experiment 1 model to: {model_path}")

# ==================================================
# 5. VALIDATION EVALUATION (ONLY VALIDATION SET)
# ==================================================
y_val_proba = model.predict_proba(X_val)[:, 1]

roc_auc_exp1 = roc_auc_score(y_val, y_val_proba)
pr_auc_exp1 = average_precision_score(y_val, y_val_proba)
brier_exp1 = brier_score_loss(y_val, y_val_proba)

# Baseline metrics (from baseline verification on identical validation set)
roc_auc_base = 0.674503
pr_auc_base = 0.160046
brier_base = 0.071711

print("\n" + "=" * 60)
print("EXPERIMENT 1 vs BASELINE VALIDATION PERFORMANCE")
print("=" * 60)
print(f"{'Metric':<30} {'Baseline':<14} {'Exp 1 (Ratios)':<16} {'Delta':<12}")
print("-" * 72)
print(f"{'ROC-AUC':<30} {roc_auc_base:<14.6f} {roc_auc_exp1:<16.6f} {roc_auc_exp1 - roc_auc_base:+.6f}")
print(f"{'Average Precision (PR-AUC)':<30} {pr_auc_base:<14.6f} {pr_auc_exp1:<16.6f} {pr_auc_exp1 - pr_auc_base:+.6f}")
print(f"{'Brier Score (lower is better)':<30} {brier_base:<14.6f} {brier_exp1:<16.6f} {brier_exp1 - brier_base:+.6f}")

# Threshold evaluation for 0.10 and 0.20
print("\n" + "=" * 60)
print("THRESHOLD EVALUATION (0.10 and 0.20)")
print("=" * 60)

for t in [0.10, 0.20]:
    y_pred = (y_val_proba >= t).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

    print(f"\nThreshold {t:.2f}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Predicted Positives: {tp + fp} ({(tp + fp) / len(y_val) * 100:.2f}%)")

# ==================================================
# 6. FEATURE IMPORTANCE ANALYSIS FOR NEW RATIOS
# ==================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE OF THE 5 FINANCIAL RATIOS")
print("=" * 60)

booster = model.get_booster()
gain_dict = booster.get_score(importance_type="gain")
weight_dict = booster.get_score(importance_type="weight")

# Rank all features by gain
sorted_by_gain = sorted(gain_dict.items(), key=lambda x: x[1], reverse=True)
gain_rank = {feat: rank + 1 for rank, (feat, _) in enumerate(sorted_by_gain)}

# Rank all features by weight (split frequency)
sorted_by_weight = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)
weight_rank = {feat: rank + 1 for rank, (feat, _) in enumerate(sorted_by_weight)}

ratios = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "INCOME_PER_FAMILY_MEMBER"
]

print(f"{'Feature Name':<28} {'Gain':<12} {'Gain Rank':<12} {'Weight':<10} {'Weight Rank':<12}")
print("-" * 74)
for r in ratios:
    gain = gain_dict.get(r, 0.0)
    grank = gain_rank.get(r, "N/A")
    weight = weight_dict.get(r, 0)
    wrank = weight_rank.get(r, "N/A")
    print(f"{r:<28} {gain:<12.4f} {str(grank) + '/128':<12} {weight:<10} {str(wrank) + '/128':<12}")

# Save detailed evaluation summary as JSON artifact
eval_results = {
    "experiment": "Experiment 1: Financial Ratio Feature Engineering",
    "baseline_metrics": {
        "roc_auc": roc_auc_base,
        "pr_auc": pr_auc_base,
        "brier_score": brier_base
    },
    "exp1_metrics": {
        "roc_auc": float(roc_auc_exp1),
        "pr_auc": float(pr_auc_exp1),
        "brier_score": float(brier_exp1)
    },
    "delta": {
        "roc_auc": float(roc_auc_exp1 - roc_auc_base),
        "pr_auc": float(pr_auc_exp1 - pr_auc_base),
        "brier_score": float(brier_exp1 - brier_base)
    },
    "ratio_importances": {
        r: {
            "gain": float(gain_dict.get(r, 0.0)),
            "gain_rank": gain_rank.get(r, None),
            "weight": int(weight_dict.get(r, 0)),
            "weight_rank": weight_rank.get(r, None)
        } for r in ratios
    }
}

with open(model_dir / "eval_summary_exp1.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\nEvaluation summary saved to: {model_dir / 'eval_summary_exp1.json'}")
