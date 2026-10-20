
import pandas as pd

# ==================================================
# STEP 3.1: NUMERICAL FEATURE ANALYSIS
# ==================================================

df = pd.read_csv("data/application_train.csv")

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

# Descriptive statistics
print("\nNUMERICAL FEATURE SUMMARY")

summary = df[numerical_features].describe().T

print(summary.to_string())

# Check negative values
print("\nNEGATIVE VALUE COUNTS")

for col in numerical_features:
    negative_count = (df[col] < 0).sum()
    print(f"{col}: {negative_count}")

# Check unique values
print("\nUNIQUE VALUE COUNTS")

for col in numerical_features:
    print(f"{col}: {df[col].nunique()}")

# Check special employment placeholder
print("\nEMPLOYMENT PLACEHOLDER")

print(
    "DAYS_EMPLOYED = 365243:",
    (df["DAYS_EMPLOYED"] == 365243).sum()
)


# ==================================================
# STEP 3.2: INVESTIGATE EXTREME NUMERICAL VALUES
# ==================================================

# 1. Income distribution at high percentiles
print("\nINCOME PERCENTILES")
print(
    df["AMT_INCOME_TOTAL"].quantile(
        [0.50, 0.90, 0.95, 0.99, 0.999, 1.00]
    )
)

# 2. Applicants with unusually high income
print("\nAPPLICANTS WITH INCOME ABOVE 10 MILLION")
print(
    (df["AMT_INCOME_TOTAL"] > 10_000_000).sum()
)

# 3. Children count distribution
print("\nCHILDREN COUNT DISTRIBUTION")
print(
    df["CNT_CHILDREN"].value_counts().sort_index()
)

# 4. Family member count distribution
print("\nFAMILY MEMBER COUNT DISTRIBUTION")
print(
    df["CNT_FAM_MEMBERS"].value_counts().sort_index()
)

# 5. Car age extreme values
print("\nCAR AGE ABOVE 50 YEARS")
print(
    (df["OWN_CAR_AGE"] > 50).sum()
)



# ==================================================
# STEP 3.3: CATEGORICAL FEATURE ANALYSIS
# ==================================================

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

print("\nCATEGORICAL FEATURE SUMMARY")

for col in categorical_features:

    print("\n" + "=" * 50)
    print("FEATURE:", col)

    print("Unique categories:", df[col].nunique())

    print("Missing values:", df[col].isnull().sum())

    print("\nCategory Distribution:")
    print(
        df[col].value_counts(
            dropna=False
        ).head(15).to_string()
    )
