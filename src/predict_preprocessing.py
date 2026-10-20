
import pandas as pd
import joblib

from pathlib import Path
from preprocess import clean_special_values

# ==================================================
# LOAD SAVED PREPROCESSING ARTIFACTS
# ==================================================

artifact_dir = Path("models")

num_imputer = joblib.load(
    artifact_dir / "num_imputer.joblib"
)

encoder = joblib.load(
    artifact_dir / "encoder.joblib"
)

config = joblib.load(
    artifact_dir / "preprocessing_config.joblib"
)


# ==================================================
# TRANSFORM NEW APPLICANT DATA
# ==================================================

def preprocess_new_data(applicant_df):

    # Validate input columns
    required_features = (
        config["numerical_features"]
        + config["categorical_features"]
        + [
            col for col in config["binary_features"]
            if col != "EMPLOYMENT_DURATION_MISSING"
        ]
    )

    missing = set(required_features) - set(applicant_df.columns)

    if missing:
        raise ValueError(
            f"Missing applicant fields: {missing}"
        )

    # Apply the same cleaning rules as training
    df = clean_special_values(applicant_df)

    # Create employment missing indicator
    df["EMPLOYMENT_DURATION_MISSING"] = (
        df["DAYS_EMPLOYED"].isna().astype(int)
    )

    # Impute car age for car owners
    car_mask = (
        (df["FLAG_OWN_CAR"] == "Y")
        & df["OWN_CAR_AGE"].isna()
    )

    df.loc[car_mask, "OWN_CAR_AGE"] = (
        config["car_owner_median"]
    )

    # Numerical transformation
    numerical_data = pd.DataFrame(
        num_imputer.transform(
            df[config["numerical_features"]]
        ),
        columns=config["numerical_features"],
        index=df.index
    )

    # Categorical transformation
    categorical_data = pd.DataFrame(
        encoder.transform(
            df[config["categorical_features"]]
        ),
        columns=config["encoded_feature_names"],
        index=df.index
    )

    # Binary features
    binary_data = df[config["binary_features"]].copy()

    # Combine everything
    processed = pd.concat(
        [
            numerical_data,
            categorical_data,
            binary_data
        ],
        axis=1
    )

    # Match training feature order
    processed = processed[
        config["final_feature_names"]
    ]

    assert not processed.isnull().any().any(), (
        "Missing values remain after preprocessing!"
    )

    return processed


# ==================================================
# TEST USING ONE HISTORICAL APPLICANT
# ==================================================

if __name__ == "__main__":

    original_df = pd.read_csv(
        "data/application_train.csv"
    )

    applicant = original_df.iloc[[0]].copy()

    processed = preprocess_new_data(applicant)

    print("\nORIGINAL INPUT SHAPE")
    print(applicant.shape)

    print("\nPROCESSED INPUT SHAPE")
    print(processed.shape)

    print("\nMISSING VALUES")
    print(processed.isnull().sum().sum())

    print("\nPREPROCESSING TEST PASSED!")
