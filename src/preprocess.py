
import pandas as pd
import numpy as np

# ==================================================
# STEP 5.1: CLEAN SPECIAL VALUES
# ==================================================

def clean_special_values(df):
    df = df.copy()

    # 1. Employment duration
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        365243, np.nan
    )

    # Convert negative employment days into positive years
    df["DAYS_EMPLOYED"] = (
        -df["DAYS_EMPLOYED"] / 365.25
    )

    # 2. Car age
    # No car -> car age not applicable, represented as 0
    no_car = df["FLAG_OWN_CAR"] == "N"

    df.loc[no_car, "OWN_CAR_AGE"] = 0

    # 3. Occupation type
    not_applicable = df["NAME_INCOME_TYPE"].isin([
        "Pensioner",
        "Unemployed"
    ])

    df.loc[
        not_applicable & df["OCCUPATION_TYPE"].isna(),
        "OCCUPATION_TYPE"
    ] = "Not Applicable"

    df["OCCUPATION_TYPE"] = (
        df["OCCUPATION_TYPE"].fillna("Unknown")
    )

    # 4. Organization type
    df["ORGANIZATION_TYPE"] = (
        df["ORGANIZATION_TYPE"].replace(
            "XNA", "Not Applicable"
        )
    )

    return df


# ==================================================
# TEST THE CLEANING FUNCTION
# ==================================================

if __name__ == "__main__":

    df = pd.read_csv("data/application_train.csv")

    cleaned_df = clean_special_values(df)

    print("\nEMPLOYMENT PLACEHOLDER CHECK")
    print(
        "Remaining placeholders:",
        (cleaned_df["DAYS_EMPLOYED"] == 365243).sum()
    )

    print("\nEMPLOYMENT YEARS SUMMARY")
    print(cleaned_df["DAYS_EMPLOYED"].describe())

    print("\nCAR AGE MISSING VALUES")
    print(cleaned_df["OWN_CAR_AGE"].isna().sum())

    print("\nOCCUPATION MISSING VALUES")
    print(cleaned_df["OCCUPATION_TYPE"].isna().sum())

    print("\nORGANIZATION TYPE XNA COUNT")
    print(
        (cleaned_df["ORGANIZATION_TYPE"] == "XNA").sum()
    )

    print("\nCleaning completed successfully!")
