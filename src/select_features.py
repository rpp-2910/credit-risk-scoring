
import pandas as pd

# Load dataset
df = pd.read_csv("data/application_train.csv")

# ==================================================
# MODEL A: Applicant-provided information (22)
# ==================================================

model_a_features = [
    # Loan
    "NAME_CONTRACT_TYPE",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",

    # Family
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS",
    "NAME_EDUCATION_TYPE",

    # Income and employment
    "AMT_INCOME_TOTAL",
    "NAME_INCOME_TYPE",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "DAYS_EMPLOYED",

    # Assets and housing
    "FLAG_OWN_CAR",
    "OWN_CAR_AGE",
    "FLAG_OWN_REALTY",
    "NAME_HOUSING_TYPE",

    # Contact information
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL"
]

# ==================================================
# MODEL A + AGE: 23 features
# ==================================================

model_a_age_features = model_a_features + [
    "DAYS_BIRTH"
]

# ==================================================
# MODEL B: Applicant data + external scores (25)
# ==================================================

model_b_features = model_a_features + [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

# ==================================================
# FEATURE VALIDATION
# ==================================================

models = {
    "Model A": model_a_features,
    "Model A + Age": model_a_age_features,
    "Model B": model_b_features
}

for name, features in models.items():

    # Check for duplicate feature names
    assert len(features) == len(set(features)), (
        f"{name}: Duplicate feature names found!"
    )

    # Check whether all selected columns exist
    missing = set(features) - set(df.columns)

    assert not missing, (
        f"{name}: Missing columns: {missing}"
    )

    # Check expected feature count
    expected_counts = {
        "Model A": 22,
        "Model A + Age": 23,
        "Model B": 25
    }

    assert len(features) == expected_counts[name], (
        f"{name}: Incorrect feature count!"
    )

    print(f"\n{name}")
    print("Number of features:", len(features))
    print("Feature matrix shape:", df[features].shape)

# ==================================================
# TARGET VALIDATION
# ==================================================

assert "TARGET" in df.columns, "TARGET column missing!"

assert df["TARGET"].notna().all(), (
    "TARGET contains missing values!"
)

assert set(df["TARGET"].unique()) <= {0, 1}, (
    "TARGET must contain only 0 and 1!"
)

print("\nTarget validation passed!")

# ==================================================
# FEATURE DATA TYPES
# ==================================================

print("\nModel A Feature Data Types:")
print(df[model_a_features].dtypes)

print("\nFeature selection validation completed!")


# ==================================================
# FEATURE TYPE CLASSIFICATION - MODEL A
# ==================================================

X = df[model_a_features]

# Binary features (0 or 1)
binary_features = [
    "FLAG_MOBIL",
    "FLAG_EMP_PHONE",
    "FLAG_WORK_PHONE",
    "FLAG_PHONE",
    "FLAG_EMAIL"
]

# Categorical features
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Numerical features (excluding binary features)
numerical_features = [
    col
    for col in X.select_dtypes(include=["number"]).columns
    if col not in binary_features
]

# ==================================================
# FEATURE TYPE VALIDATION
# ==================================================

all_features = (
    numerical_features
    + categorical_features
    + binary_features
)

# Check for duplicate classifications
assert len(all_features) == len(set(all_features)), (
    "Duplicate feature classification detected!"
)

# Check that all 22 features are classified
assert set(all_features) == set(model_a_features), (
    "Some features are missing or incorrectly classified!"
)

# Verify binary features contain only 0 and 1
for col in binary_features:
    assert X[col].dropna().isin([0, 1]).all(), (
        f"{col} contains values other than 0 and 1!"
    )

# ==================================================
# DISPLAY FEATURE TYPES
# ==================================================

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nBinary Features:")
print(binary_features)

print("\nFEATURE COUNTS")
print("Numerical:", len(numerical_features))
print("Categorical:", len(categorical_features))
print("Binary:", len(binary_features))
print("Total:", len(all_features))

print("\nFeature classification completed successfully!")


# ==================================================
# STEP 2.3: MISSING VALUE ANALYSIS
# ==================================================

# Missing values in Model A
missing_report = pd.DataFrame({
    "Missing Count": X.isnull().sum(),
    "Missing Percentage": (X.isnull().mean() * 100).round(2)
})

missing_report = missing_report.sort_values(
    by="Missing Percentage",
    ascending=False
)

print("\nMODEL A - MISSING VALUE REPORT")
print(missing_report.to_string())

# Missing values in Model B's external scores
external_features = [
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3"
]

print("\nEXTERNAL SCORE MISSING VALUES")

for col in external_features:
    missing_pct = df[col].isnull().mean() * 100
    print(f"{col}: {missing_pct:.2f}% missing")

# Special employment placeholder
print("\nDAYS_EMPLOYED PLACEHOLDER CHECK")

placeholder_count = (df["DAYS_EMPLOYED"] == 365243).sum()

placeholder_pct = placeholder_count / len(df) * 100

print("Placeholder count:", placeholder_count)
print(f"Placeholder percentage: {placeholder_pct:.2f}%")

# ==================================================
# STEP 2.4: INVESTIGATE MISSING VALUES
# ==================================================

# 1. Does missing car age depend on car ownership?
print("\nCAR OWNERSHIP VS MISSING CAR AGE")

print(pd.crosstab(
    df["FLAG_OWN_CAR"],
    df["OWN_CAR_AGE"].isnull(),
    rownames=["Owns Car"],
    colnames=["Car Age Missing"]
))

# 2. Which income types have missing occupations?
print("\nINCOME TYPE VS MISSING OCCUPATION")

print(pd.crosstab(
    df["NAME_INCOME_TYPE"],
    df["OCCUPATION_TYPE"].isnull(),
    rownames=["Income Type"],
    colnames=["Occupation Missing"]
))

# 3. Which income types have employment placeholders?
print("\nINCOME TYPE VS EMPLOYMENT PLACEHOLDER")

print(pd.crosstab(
    df["NAME_INCOME_TYPE"],
    df["DAYS_EMPLOYED"] == 365243,
    rownames=["Income Type"],
    colnames=["Employment Placeholder"]
))