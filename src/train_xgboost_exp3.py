from pathlib import Path
import json
import time
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
print("=" * 70)
print("EXPERIMENT 3: CONTROLLED XGBOOST HYPERPARAMETER OPTIMIZATION")
print("=" * 70)
print("\n[1/4] Loading train and validation splits...")
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
print("\n[2/4] Preprocessing data using existing Experiment 1 artifacts (128 features)...")
t0 = time.time()
X_train = preprocess_new_data_exp1(train_df)
X_val = preprocess_new_data_exp1(val_df)
print(f"Preprocessing completed in {time.time() - t0:.1f}s")

y_train = train_df["TARGET"].to_numpy()
y_val = val_df["TARGET"].to_numpy()

assert X_train.shape == (215257, 128)
assert X_val.shape == (46127, 128)

# ==================================================
# 3. DEFINE SEARCH SPACE (14 SYSTEMATIC CONFIGURATIONS)
# ==================================================
# Investigating max_depth (3, 4, 5), min_child_weight (1, 5, 10),
# subsample (0.7-0.9), colsample_bytree (0.7-0.9), reg_lambda (1, 5, 10).
# Keeping lr=0.05, hist, scale_pos_weight=1.0, random_state=42.

candidate_configs = [
    # 1. Baseline / Exp 2 anchor
    {"name": "Trial 01 (Exp2 Anchor)", "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1},
    # 2. Tree depth exploration
    {"name": "Trial 02 (Depth 3)",      "max_depth": 3, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1},
    {"name": "Trial 03 (Depth 5)",      "max_depth": 5, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1},
    # 3. Leaf weight regularization
    {"name": "Trial 04 (MCW 5)",        "max_depth": 4, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1},
    {"name": "Trial 05 (MCW 10)",       "max_depth": 4, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1},
    {"name": "Trial 06 (Depth 5 + MCW 10)", "max_depth": 5, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1},
    # 4. L2 regularization (reg_lambda)
    {"name": "Trial 07 (Lambda 5)",     "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5},
    {"name": "Trial 08 (Lambda 10)",    "max_depth": 4, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 10},
    {"name": "Trial 09 (Depth 5 + L5)", "max_depth": 5, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5},
    # 5. Stochastic sampling variation
    {"name": "Trial 10 (Subsample 0.7)", "max_depth": 4, "min_child_weight": 5, "subsample": 0.7, "colsample_bytree": 0.8, "reg_lambda": 1},
    {"name": "Trial 11 (Subsample 0.9)", "max_depth": 4, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1},
    {"name": "Trial 12 (Colsample 0.7)", "max_depth": 4, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1},
    {"name": "Trial 13 (Colsample 0.9)", "max_depth": 4, "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1},
    # 6. Combined regularized capacity
    {"name": "Trial 14 (Synergy)",       "max_depth": 5, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 5},
]

print(f"\n[3/4] Running systematic search across {len(candidate_configs)} candidate configurations...")

trial_results = []
best_model_obj = None
best_auc = -1.0
best_config_name = ""

for i, cfg in enumerate(candidate_configs, 1):
    t_start = time.time()
    name = cfg["name"]
    params = {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": cfg["max_depth"],
        "min_child_weight": cfg["min_child_weight"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "reg_lambda": cfg["reg_lambda"],
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 50,
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist"
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    t_elapsed = time.time() - t_start

    best_iter = int(model.best_iteration)
    optimal_trees = best_iter + 1
    total_rounds = len(model.evals_result()["validation_0"]["auc"])

    # Predictions on validation set
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, y_val_proba))
    val_pr_auc = float(average_precision_score(y_val, y_val_proba))
    val_brier = float(brier_score_loss(y_val, y_val_proba))

    res = {
        "trial_id": i,
        "name": name,
        "max_depth": cfg["max_depth"],
        "min_child_weight": cfg["min_child_weight"],
        "subsample": cfg["subsample"],
        "colsample_bytree": cfg["colsample_bytree"],
        "reg_lambda": cfg["reg_lambda"],
        "best_iteration": best_iter,
        "optimal_trees": optimal_trees,
        "total_rounds_trained": total_rounds,
        "roc_auc": val_auc,
        "average_precision": val_pr_auc,
        "brier_score": val_brier,
        "train_time_sec": round(t_elapsed, 1)
    }
    trial_results.append(res)

    print(f"[{i:02d}/14] {name:<26} | Trees: {optimal_trees:>4} | AUC: {val_auc:.6f} | PR-AUC: {val_pr_auc:.6f} | Brier: {val_brier:.6f} ({t_elapsed:.1f}s)")

    if val_auc > best_auc:
        best_auc = val_auc
        best_model_obj = model
        best_config_name = name

# ==================================================
# 4. RANK CANDIDATES & SELECT BEST
# ==================================================
# Sort primarily by ROC-AUC (descending), secondary Average Precision (descending), tertiary Brier (ascending)
ranked_trials = sorted(
    trial_results,
    key=lambda x: (x["roc_auc"], x["average_precision"], -x["brier_score"]),
    reverse=True
)

for rank, trial in enumerate(ranked_trials, 1):
    trial["rank"] = rank

best_trial = ranked_trials[0]

print("\n" + "=" * 70)
print(f"TOP CANDIDATE SELECTED: {best_trial['name']}")
print("=" * 70)
print(f"  max_depth:         {best_trial['max_depth']}")
print(f"  min_child_weight:  {best_trial['min_child_weight']}")
print(f"  subsample:         {best_trial['subsample']}")
print(f"  colsample_bytree:  {best_trial['colsample_bytree']}")
print(f"  reg_lambda:        {best_trial['reg_lambda']}")
print(f"  Optimal Trees:     {best_trial['optimal_trees']} (best_iteration={best_trial['best_iteration']})")
print(f"  Validation ROC-AUC:{best_trial['roc_auc']:.6f}")
print(f"  Average Precision: {best_trial['average_precision']:.6f}")
print(f"  Brier Score:       {best_trial['brier_score']:.6f}")

# ==================================================
# 5. SAVE EXPERIMENT 3 ARTIFACTS
# ==================================================
out_dir = Path("models/exp3_tuning")
out_dir.mkdir(parents=True, exist_ok=True)

# Save best model
best_model_path = out_dir / "xgboost_model_exp3_best.json"
best_model_obj.save_model(str(best_model_path))
print(f"\nSaved best model artifact to: {best_model_path}")

# Evaluate best model at thresholds 0.10 and 0.20
y_best_proba = best_model_obj.predict_proba(X_val)[:, 1]
threshold_metrics = {}
for t in [0.10, 0.20]:
    y_pred = (y_best_proba >= t).astype(int)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
    threshold_metrics[str(t)] = {
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

# Save complete trials and summary
exp2_roc_auc = 0.704638
exp2_pr_auc = 0.181076
exp2_brier = 0.070720

summary_payload = {
    "experiment": "Experiment 3: Controlled XGBoost Hyperparameter Optimization",
    "search_space_explored": {
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_lambda": [1, 5, 10]
    },
    "fixed_parameters": {
        "learning_rate": 0.05,
        "n_estimators_max": 1000,
        "early_stopping_rounds": 50,
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "tree_method": "hist",
        "scale_pos_weight": 1.0,
        "random_state": 42
    },
    "best_candidate": best_trial,
    "top_5_candidates": ranked_trials[:5],
    "comparison_vs_exp2": {
        "exp2_metrics": {
            "roc_auc": exp2_roc_auc,
            "pr_auc": exp2_pr_auc,
            "brier_score": exp2_brier
        },
        "exp3_best_metrics": {
            "roc_auc": best_trial["roc_auc"],
            "pr_auc": best_trial["average_precision"],
            "brier_score": best_trial["brier_score"]
        },
        "delta": {
            "roc_auc": best_trial["roc_auc"] - exp2_roc_auc,
            "pr_auc": best_trial["average_precision"] - exp2_pr_auc,
            "brier_score": best_trial["brier_score"] - exp2_brier
        }
    },
    "threshold_metrics": threshold_metrics,
    "all_trials": ranked_trials
}

with open(out_dir / "all_trials_exp3.json", "w") as f:
    json.dump(ranked_trials, f, indent=2)

with open(out_dir / "eval_summary_exp3.json", "w") as f:
    json.dump(summary_payload, f, indent=2)

print(f"Saved full trials log to: {out_dir / 'all_trials_exp3.json'}")
print(f"Saved experiment summary to: {out_dir / 'eval_summary_exp3.json'}")
