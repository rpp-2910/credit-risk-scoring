from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib

import lightgbm as lgb
from lightgbm import LGBMClassifier, early_stopping
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

from predict_preprocessing_exp4 import preprocess_new_data_exp4

warnings.filterwarnings("ignore")

# ==================================================
# 1. LOAD DATA (TRAIN & VALIDATION ONLY)
# ==================================================
print("=" * 70)
print("EXPERIMENT 4: LIGHTGBM BENCHMARK ON IDENTICAL DATA")
print("=" * 70)
print("\n[1/5] Loading train and validation splits...")
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

print(f"Training set:   {train_df.shape[0]} records")
print(f"Validation set: {val_df.shape[0]} records")

# ==================================================
# 2. PREPROCESS USING EXISTING EXPERIMENT 1 ARTIFACTS
# ==================================================
print("\n[2/5] Preprocessing data (128 features with LightGBM column sanitation)...")
t0 = time.time()
X_train = preprocess_new_data_exp4(train_df)
X_val = preprocess_new_data_exp4(val_df)
print(f"Preprocessing completed in {time.time() - t0:.1f}s")

y_train = train_df["TARGET"].to_numpy()
y_val = val_df["TARGET"].to_numpy()

assert X_train.shape == (215257, 128)
assert X_val.shape == (46127, 128)
assert not X_train.isna().any().any()
assert not X_val.isna().any().any()
assert np.isfinite(X_train.to_numpy(dtype=float)).all()
assert np.isfinite(X_val.to_numpy(dtype=float)).all()

# ==================================================
# 3. TRAIN BASELINE LIGHTGBM CONFIGURATION
# ==================================================
# Reasonable baseline: n_estimators=1000, lr=0.05, num_leaves=31,
# subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42.
# Unweighted binary classification. Early stopping on validation ROC-AUC.
print("\n[3/5] Training LightGBM baseline model with early stopping (patience=50)...")

lgbm_model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    objective="binary",
    n_jobs=-1,
    verbose=-1
)

callbacks = [early_stopping(stopping_rounds=50, verbose=False)]

t_train_start = time.time()
lgbm_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=callbacks
)
train_time_sec = time.time() - t_train_start

best_iteration = int(lgbm_model.best_iteration_)
total_rounds = lgbm_model.booster_.num_trees()

print(f"LightGBM training completed in {train_time_sec:.1f}s")
print(f"  Best Iteration:     {best_iteration}")
print(f"  Optimal Trees:      {best_iteration} (LightGBM 1-indexed best_iteration_)")
print(f"  Total Trees Built:  {total_rounds}")

# ==================================================
# 4. SAVE LIGHTGBM MODEL ARTIFACTS
# ==================================================
out_dir = Path("models/exp4_lightgbm")
out_dir.mkdir(parents=True, exist_ok=True)

model_txt_path = out_dir / "lgbm_model_exp4.txt"
model_joblib_path = out_dir / "lgbm_model_exp4.joblib"

lgbm_model.booster_.save_model(str(model_txt_path))
joblib.dump(lgbm_model, model_joblib_path)

print(f"\nSaved LightGBM model to:")
print(f"  {model_txt_path}")
print(f"  {model_joblib_path}")

# ==================================================
# 5. VALIDATION EVALUATION (ONLY VALIDATION SET)
# ==================================================
print("\n[4/5] Evaluating LightGBM on validation set...")
y_val_proba = lgbm_model.predict_proba(X_val)[:, 1]

roc_auc_lgb = float(roc_auc_score(y_val, y_val_proba))
pr_auc_lgb = float(average_precision_score(y_val, y_val_proba))
brier_lgb = float(brier_score_loss(y_val, y_val_proba))

# Historical benchmarks
roc_auc_base = 0.674503
pr_auc_base = 0.160046
brier_base = 0.071711

roc_auc_exp2 = 0.704638
pr_auc_exp2 = 0.181076
brier_exp2 = 0.070720

roc_auc_exp3 = 0.705085
pr_auc_exp3 = 0.181261
brier_exp3 = 0.070702

print("\n" + "=" * 80)
print("BENCHMARK COMPARISON: LIGHTGBM vs XGBOOST MODELS")
print("=" * 80)
print(f"{'Metric':<26} {'Baseline XGB':<14} {'Exp 2 XGB':<14} {'Exp 3 Tuned XGB':<16} {'Exp 4 LightGBM':<16}")
print("-" * 88)
print(f"{'ROC-AUC':<26} {roc_auc_base:<14.6f} {roc_auc_exp2:<14.6f} {roc_auc_exp3:<16.6f} {roc_auc_lgb:<16.6f}")
print(f"{'Average Precision':<26} {pr_auc_base:<14.6f} {pr_auc_exp2:<14.6f} {pr_auc_exp3:<16.6f} {pr_auc_lgb:<16.6f}")
print(f"{'Brier Score (lower better)':<26} {brier_base:<14.6f} {brier_exp2:<14.6f} {brier_exp3:<16.6f} {brier_lgb:<16.6f}")
print("-" * 88)
print(f"Delta vs Exp 2 XGB: ROC-AUC: {roc_auc_lgb - roc_auc_exp2:+.6f} | PR-AUC: {pr_auc_lgb - pr_auc_exp2:+.6f} | Brier: {brier_lgb - brier_exp2:+.6f}")
print(f"Delta vs Exp 3 XGB: ROC-AUC: {roc_auc_lgb - roc_auc_exp3:+.6f} | PR-AUC: {pr_auc_lgb - pr_auc_exp3:+.6f} | Brier: {brier_lgb - brier_exp3:+.6f}")

# Threshold evaluation for 0.10 and 0.20
threshold_results = {}
print("\n" + "=" * 60)
print("LIGHTGBM THRESHOLD EVALUATION (0.10 and 0.20)")
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

# ==================================================
# 6. FEATURE IMPORTANCE SUMMARY
# ==================================================
print("\n[5/5] Extracting LightGBM feature importances...")
feature_names = lgbm_model.booster_.feature_name()
gains = lgbm_model.booster_.feature_importance(importance_type="gain")
splits_count = lgbm_model.booster_.feature_importance(importance_type="split")

importance_df = pd.DataFrame({
    "feature": feature_names,
    "gain": gains,
    "split_count": splits_count
}).sort_values(by="gain", ascending=False).reset_index(drop=True)

importance_df["gain_rank"] = importance_df.index + 1

# Check ranking of the 5 financial ratios
ratios = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "INCOME_PER_FAMILY_MEMBER"
]

ratio_importance_summary = {}
print("\n" + "=" * 70)
print("FINANCIAL RATIOS IMPORTANCE IN LIGHTGBM (out of 128 features)")
print("=" * 70)
print(f"{'Feature Name':<28} {'Gain':<14} {'Gain Rank':<12} {'Split Count':<14}")
print("-" * 70)
for r in ratios:
    match = importance_df[importance_df["feature"] == r]
    if not match.empty:
        g = float(match["gain"].values[0])
        grank = int(match["gain_rank"].values[0])
        sp = int(match["split_count"].values[0])
        ratio_importance_summary[r] = {
            "gain": g,
            "gain_rank": grank,
            "split_count": sp
        }
        print(f"{r:<28} {g:<14.2f} {str(grank) + '/128':<12} {sp:<14}")

# Top 10 overall features in LightGBM by gain
print("\nTOP 10 FEATURES IN LIGHTGBM BY GAIN:")
for i, row in importance_df.head(10).iterrows():
    print(f"  #{row['gain_rank']:<2} {row['feature']:<35} Gain: {row['gain']:>10.2f} | Splits: {row['split_count']}")

# Save full evaluation summary JSON
eval_summary = {
    "experiment": "Experiment 4: LightGBM Benchmark",
    "algorithm": "LightGBM (LGBMClassifier 4.7.0)",
    "training_time_seconds": round(train_time_sec, 2),
    "hyperparameters": {
        "n_estimators_max": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "early_stopping_rounds": 50,
        "random_state": 42
    },
    "early_stopping_results": {
        "best_iteration": best_iteration,
        "total_trees": total_rounds
    },
    "metrics": {
        "roc_auc": roc_auc_lgb,
        "average_precision": pr_auc_lgb,
        "brier_score": brier_lgb
    },
    "comparisons": {
        "vs_exp2_xgboost": {
            "roc_auc_delta": roc_auc_lgb - roc_auc_exp2,
            "pr_auc_delta": pr_auc_lgb - pr_auc_exp2,
            "brier_score_delta": brier_lgb - brier_exp2
        },
        "vs_exp3_tuned_xgboost": {
            "roc_auc_delta": roc_auc_lgb - roc_auc_exp3,
            "pr_auc_delta": pr_auc_lgb - pr_auc_exp3,
            "brier_score_delta": brier_lgb - brier_exp3
        },
        "vs_baseline": {
            "roc_auc_delta": roc_auc_lgb - roc_auc_base,
            "pr_auc_delta": pr_auc_lgb - pr_auc_base,
            "brier_score_delta": brier_lgb - brier_base
        }
    },
    "threshold_metrics": threshold_results,
    "ratio_importance": ratio_importance_summary,
    "top_10_features_by_gain": importance_df.head(10).to_dict(orient="records")
}

with open(out_dir / "eval_summary_exp4.json", "w") as f:
    json.dump(eval_summary, f, indent=2)

print(f"\nExperiment 4 summary saved to: {out_dir / 'eval_summary_exp4.json'}")
