
import pandas as pd
from sklearn.model_selection import train_test_split

# ==================================================
# STEP 4.1: TRAIN / VALIDATION / TEST SPLIT
# ==================================================

df = pd.read_csv("data/application_train.csv")

# Separate target
y = df["TARGET"]

# Keep application IDs separately for audit/tracking
ids = df["SK_ID_CURR"]

# First split: 70% training, 30% temporary
train_idx, temp_idx = train_test_split(
    df.index,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% validation, 15% testing
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y.loc[temp_idx]
)

# Validate that splits don't overlap
assert set(train_idx).isdisjoint(val_idx)
assert set(train_idx).isdisjoint(test_idx)
assert set(val_idx).isdisjoint(test_idx)

# Ensure all rows are accounted for
assert len(train_idx) + len(val_idx) + len(test_idx) == len(df)

# Save the split assignments
split_df = pd.DataFrame({
    "SK_ID_CURR": ids,
    "split": "unassigned"
})

split_df.loc[train_idx, "split"] = "train"
split_df.loc[val_idx, "split"] = "validation"
split_df.loc[test_idx, "split"] = "test"

assert not (split_df["split"] == "unassigned").any()

split_df.to_csv("data/split_assignments.csv", index=False)

# Display split statistics
for name, idx in [
    ("Training", train_idx),
    ("Validation", val_idx),
    ("Testing", test_idx)
]:
    print(f"\n{name} Set")
    print("Records:", len(idx))
    print("Target Distribution (%):")
    print(
        y.loc[idx].value_counts(normalize=True).mul(100).round(2)
    )

print("\nSplit assignments saved successfully!")
