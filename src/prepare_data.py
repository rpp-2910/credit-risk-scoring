
import pandas as pd

from preprocess import clean_special_values

# ==================================================
# STEP 5.2: LOAD SAVED DATA SPLITS
# ==================================================

df = pd.read_csv("data/application_train.csv")

splits = pd.read_csv("data/split_assignments.csv")

# Validate IDs
assert df["SK_ID_CURR"].is_unique
assert splits["SK_ID_CURR"].is_unique

assert set(df["SK_ID_CURR"]) == set(splits["SK_ID_CURR"])

assert set(splits["split"].unique()) == {
    "train", "validation", "test"
}

# Join original data with saved split assignments
df = df.merge(
    splits,
    on="SK_ID_CURR",
    how="left",
    validate="one_to_one"
)

assert df["split"].notna().all()

# Apply fixed cleaning rules
df = clean_special_values(df)

# Separate the datasets
train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

# ==================================================
# VALIDATE RESULTS
# ==================================================

print("\nDATASET SPLIT SIZES")

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

print("\nMISSING EMPLOYMENT VALUES")

print(
    "Training:",
    train_df["DAYS_EMPLOYED"].isna().sum()
)

print(
    "Validation:",
    val_df["DAYS_EMPLOYED"].isna().sum()
)

print(
    "Testing:",
    test_df["DAYS_EMPLOYED"].isna().sum()
)

assert (
    len(train_df) + len(val_df) + len(test_df)
    == len(df)
)

print("\nData preparation checks passed!")


# ==================================================
# STEP 5.2A: CAR AGE AND EMPLOYMENT CORRECTIONS
# ==================================================

# Preserve whether employment duration was missing
for dataset in [train_df, val_df, test_df]:
    dataset["EMPLOYMENT_DURATION_MISSING"] = (
        dataset["DAYS_EMPLOYED"].isna().astype(int)
    )

# Calculate median car age among car owners
# using training data only
car_owner_median = train_df.loc[
    train_df["FLAG_OWN_CAR"] == "Y",
    "OWN_CAR_AGE"
].median()

assert pd.notna(car_owner_median)

# Fill missing car ages among car owners
for dataset in [train_df, val_df, test_df]:
    mask = (
        (dataset["FLAG_OWN_CAR"] == "Y")
        & dataset["OWN_CAR_AGE"].isna()
    )

    dataset.loc[mask, "OWN_CAR_AGE"] = car_owner_median

print("\nCAR OWNER MEDIAN AGE:", car_owner_median)

print("\nEMPLOYMENT MISSING INDICATOR COUNTS")

for name, dataset in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    print(
        f"{name}:",
        dataset["EMPLOYMENT_DURATION_MISSING"].sum()
    )


# ==================================================
# STEP 5.3: NUMERICAL MISSING VALUE IMPUTATION
# ==================================================

from sklearn.impute import SimpleImputer

numerical_features = [
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "AMT_INCOME_TOTAL",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE"
]

# Create median imputer
num_imputer = SimpleImputer(strategy="median")

# Learn median values from training data only
num_imputer.fit(train_df[numerical_features])

# Apply training medians to all three datasets
train_df[numerical_features] = num_imputer.transform(
    train_df[numerical_features]
)

val_df[numerical_features] = num_imputer.transform(
    val_df[numerical_features]
)

test_df[numerical_features] = num_imputer.transform(
    test_df[numerical_features]
)

# Verify no numerical missing values remain
for name, dataset in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:
    missing_count = dataset[numerical_features].isna().sum().sum()

    print(f"{name} numerical missing values: {missing_count}")

    assert missing_count == 0

# Display learned median values
print("\nTRAINING DATA MEDIANS")

for feature, median in zip(
    numerical_features,
    num_imputer.statistics_
):
    print(f"{feature}: {median}")

print("\nNumerical imputation completed!")



# ==================================================
# STEP 5.4: CATEGORICAL ENCODING
# ==================================================

from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "NAME_CONTRACT_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_HOUSING_TYPE"
]

# Create encoder
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Learn categories from training data only
encoder.fit(train_df[categorical_features])

# Transform all three datasets
train_encoded = encoder.transform(
    train_df[categorical_features]
)

val_encoded = encoder.transform(
    val_df[categorical_features]
)

test_encoded = encoder.transform(
    test_df[categorical_features]
)

# Verify shapes
print("\nCATEGORICAL ENCODING RESULTS")

print("Training:", train_encoded.shape)
print("Validation:", val_encoded.shape)
print("Testing:", test_encoded.shape)

# Check that the encoded feature count matches
assert train_encoded.shape[1] == val_encoded.shape[1]
assert train_encoded.shape[1] == test_encoded.shape[1]

print("\nEncoded feature names:")
print(encoder.get_feature_names_out(categorical_features))

print("\nCategorical encoding completed successfully!")



# ==================================================
# STEP 5.5: COMBINE PROCESSED FEATURES - MODEL A
# ==================================================

binary_features = [
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL",
    "EMPLOYMENT_DURATION_MISSING"
]

# Get one-hot encoded feature names
encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
).tolist()

# Final feature names
final_feature_names = (
    numerical_features
    + encoded_feature_names
    + binary_features
)

# Function to combine features
def combine_features(dataset, encoded_array):

    numerical_data = dataset[numerical_features].reset_index(
        drop=True
    )

    categorical_data = pd.DataFrame(
        encoded_array,
        columns=encoded_feature_names
    )

    binary_data = dataset[binary_features].reset_index(
        drop=True
    )

    return pd.concat(
        [numerical_data, categorical_data, binary_data],
        axis=1
    )

# Combine train, validation, and test features
X_train = combine_features(train_df, train_encoded)
X_val = combine_features(val_df, val_encoded)
X_test = combine_features(test_df, test_encoded)

# Extract target values
y_train = train_df["TARGET"].reset_index(drop=True)
y_val = val_df["TARGET"].reset_index(drop=True)
y_test = test_df["TARGET"].reset_index(drop=True)

# ==================================================
# VALIDATION
# ==================================================

assert X_train.shape[1] == len(final_feature_names)
assert X_val.shape[1] == len(final_feature_names)
assert X_test.shape[1] == len(final_feature_names)

assert X_train.columns.equals(X_val.columns)
assert X_train.columns.equals(X_test.columns)

assert not X_train.isnull().any().any()
assert not X_val.isnull().any().any()
assert not X_test.isnull().any().any()

assert len(X_train) == len(y_train)
assert len(X_val) == len(y_val)
assert len(X_test) == len(y_test)

print("\nFINAL MODEL A FEATURE MATRICES")

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nTarget shapes")
print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)

print("\nFinal number of model features:", len(final_feature_names))

print("\nModel A feature preparation completed!")



# ==================================================
# STEP 5.6: SAVE PREPROCESSING ARTIFACTS
# ==================================================

from pathlib import Path
import joblib

# Create artifacts directory
artifact_dir = Path("models")
artifact_dir.mkdir(parents=True, exist_ok=True)

# Save fitted numerical imputer
joblib.dump(
    num_imputer,
    artifact_dir / "num_imputer.joblib"
)

# Save fitted categorical encoder
joblib.dump(
    encoder,
    artifact_dir / "encoder.joblib"
)

# Save preprocessing configuration
preprocessing_config = {
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "binary_features": binary_features,
    "encoded_feature_names": encoded_feature_names,
    "final_feature_names": final_feature_names,
    "car_owner_median": float(car_owner_median)
}

joblib.dump(
    preprocessing_config,
    artifact_dir / "preprocessing_config.joblib"
)

# Verify saved files exist
for filename in [
    "num_imputer.joblib",
    "encoder.joblib",
    "preprocessing_config.joblib"
]:
    assert (artifact_dir / filename).exists(), (
        f"{filename} was not saved!"
    )

print("\nPREPROCESSING ARTIFACTS SAVED")

print("Numerical imputer saved")
print("Categorical encoder saved")
print("Preprocessing configuration saved")

print("\nStep 5.6 completed successfully!")



# ==================================================
# STEP 5.8: VERIFY PREPROCESSING CONSISTENCY
# ==================================================

from predict_preprocessing import preprocess_new_data
import numpy as np

# Select the first applicant from the training split
applicant_id = train_df.iloc[0]["SK_ID_CURR"]

# Retrieve the original, unprocessed application
raw_df = pd.read_csv("data/application_train.csv")

raw_applicant = raw_df[
    raw_df["SK_ID_CURR"] == applicant_id
].copy()

# Process using saved preprocessing artifacts
new_processed = preprocess_new_data(raw_applicant)

# Retrieve corresponding processed training row
original_processed = X_train.iloc[[0]]

# Compare feature names
assert new_processed.columns.tolist() == (
    original_processed.columns.tolist()
)

# Compare actual feature values
np.testing.assert_allclose(
    new_processed.to_numpy(dtype=float),
    original_processed.to_numpy(dtype=float),
    rtol=1e-7,
    atol=1e-9
)

print("\nPREPROCESSING CONSISTENCY CHECK")
print("Applicant ID:", applicant_id)
print("Features compared:", new_processed.shape[1])
print("All feature values match!")

print("\nStep 5.8 completed successfully!")
