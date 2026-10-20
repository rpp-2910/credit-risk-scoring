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
print("Loading data for Experiment 2: XGBoost Early Stopping...")
df = pd.read_csv("data/application_train.csv")
splits = pd.read_csv("data/split_assignments.csv")

df = df.merge(
    splits,
    on="SK_ID_CURR",
    how="inner",
    validate="one_to_one"
)

# STRICT RULE: Train and Validation sets only.
# Test set must NOT be loaded, transformed, or evaluated.
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

print(f"Training set: {train_df.shape[0]} rows")
print(f"Validation set: {val_df.shape[0]} rows")

# ==================================================
# 2. PREPROCESS USING EXISTING EXPERIMENT 1 ARTIFACTS
# ==================================================
# STRICT RULE: Use existing Experiment 1 preprocessing artifacts
# without refitting or modifying them.
print("\nApplying Experiment 1 preprocessing (128 features)...")
X_train = preprocess_new_data_exp1(train_df)
X_val = preprocess_new_data_exp1(val_df)

y_train = train_df["TARGET"].to_numpy()
y_val = val_df["TARGET"].to_numpy()

assert X_train.shape == (215257, 128)
assert X_val.shape == (46127, 128)
assert not X_train.isna().any().any()
assert not X_val.isna().any().any()
assert np.isfinite(X_train.to_numpy(dtype=float)).all()
assert np.isfinite(X_val.to_numpy(dtype=float)).all()

# ==================================================
# 3. TRAIN XGBOOST WITH EARLY STOPPING ONLY
# ==================================================
# STRICT RULE: Keep all hyperparameters unchanged from baseline/Exp 1 except:
# - n_estimators increased from 300 to 1000
# - early_stopping_rounds=50
# - monitor validation ROC-AUC
model = XGBClassifier(
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

print("\nTraining XGBoost Experiment 2 model (n_estimators=1000, early_stopping_rounds=50)...")
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("Training completed!")

# Extract early stopping iteration metrics
best_iteration = int(model.best_iteration)
best_score = float(model.best_score)
evals_result = model.evals_result()
val_auc_history = evals_result["validation_0"]["auc"]
total_rounds_trained = len(val_auc_history)
optimal_trees = best_iteration + 1

print(f"\nEARLY STOPPING SUMMARY:")
print(f"  Best Iteration (0-indexed): {best_iteration}")
print(f"  Optimal Number of Trees:    {optimal_trees}")
print(f"  Total Rounds Trained:       {total_rounds_trained} (stopped 50 rounds after peak)")
print(f"  Best Validation AUC:        {best_score:.6f}")

# ==================================================
# 4. SAVE EXPERIMENT 2 MODEL ARTIFACT SEPARATELY
# ==================================================
model_dir = Path("models/exp2_early_stopping")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "xgboost_model_exp2.json"

model.save_model(str(model_path))
print(f"\nSaved Experiment 2 model to: {model_path}")

# ==================================================
# 5. VALIDATION EVALUATION (ONLY VALIDATION SET)
# ==================================================
# modern XGBoost predict_proba automatically uses best_iteration
y_val_proba = model.predict_proba(X_val)[:, 1]

roc_auc_exp2 = roc_auc_score(y_val, y_val_proba)
pr_auc_exp2 = average_precision_score(y_val, y_val_proba)
brier_exp2 = brier_score_loss(y_val, y_val_proba)

# Prior milestones
roc_auc_base = 0.674503
pr_auc_base = 0.160046
brier_base = 0.071711

roc_auc_exp1 = 0.700623
pr_auc_exp1 = 0.177912
brier_exp1 = 0.070891

print("\n" + "=" * 76)
print("DIRECT COMPARISON: EXPERIMENT 2 vs EXPERIMENT 1 vs BASELINE")
print("=" * 76)
print(f"{'Metric':<28} {'Baseline':<12} {'Exp 1 (Ratios)':<16} {'Exp 2 (EarlyStop)':<18} {'Delta vs Exp 1':<14}")
print("-" * 88)
print(f"{'ROC-AUC':<28} {roc_auc_base:<12.6f} {roc_auc_exp1:<16.6f} {roc_auc_exp2:<18.6f} {roc_auc_exp2 - roc_auc_exp1:+.6f}")
print(f"{'Average Precision':<28} {pr_auc_base:<12.6f} {pr_auc_exp1:<16.6f} {pr_auc_exp2:<18.6f} {pr_auc_exp2 - pr_auc_exp1:+.6f}")
print(f"{'Brier Score (lower better)':<28} {brier_base:<12.6f} {brier_exp1:<16.6f} {brier_exp2:<18.6f} {brier_exp2 - brier_exp1:+.6f}")

# Threshold evaluation for 0.10 and 0.20
threshold_results = {}
print("\n" + "=" * 60)
print("EXPERIMENT 2 THRESHOLD EVALUATION (0.10 and 0.20)")
print("=" * 60)

for t in [0.10, 0.20]:
    y_pred = (y_val_proba >= t).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()

    threshold_results[str(t)] = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "predicted_positives": int(tp + fp)
    }

    print(f"\nThreshold {t:.2f}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Predicted Positives: {tp + fp} ({(tp + fp) / len(y_val) * 100:.2f}%)")

# Save detailed evaluation summary as JSON artifact
eval_results = {
    "experiment": "Experiment 2: XGBoost Early Stopping ONLY",
    "early_stopping_details": {
        "n_estimators_max": 1000,
        "early_stopping_rounds": 50,
        "best_iteration": best_iteration,
        "optimal_number_of_trees": optimal_trees,
        "total_rounds_trained": total_rounds_trained,
        "best_validation_auc": best_score
    },
    "metrics_comparison": {
        "baseline": {
            "roc_auc": roc_auc_base,
            "pr_auc": pr_auc_base,
            "brier_score": brier_base
        },
        "experiment_1": {
            "roc_auc": roc_auc_exp1,
            "pr_auc": pr_auc_exp1,
            "brier_score": brier_exp1
        },
        "experiment_2": {
            "roc_auc": float(roc_auc_exp2),
            "pr_auc": float(pr_auc_exp2),
            "brier_score": float(brier_exp2)
        },
        "delta_vs_exp1": {
            "roc_auc": float(roc_auc_exp2 - roc_auc_exp1),
            "pr_auc": float(pr_auc_exp2 - pr_auc_exp1),
            "brier_score": float(brier_exp2 - brier_exp1)
        },
        "delta_vs_baseline": {
            "roc_auc": float(roc_auc_exp2 - roc_auc_base),
            "pr_auc": float(pr_auc_exp2 - pr_auc_base),
            "brier_score": float(brier_exp2 - brier_base)
        }
    },
    "threshold_metrics": threshold_results
}

with open(model_dir / "eval_summary_exp2.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print(f"\nExperiment 2 summary saved to: {model_dir / 'eval_summary_exp2.json'}")
