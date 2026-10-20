from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from preprocess import clean_special_values
from feature_engineering import compute_financial_ratios

# Load Experiment 1 preprocessing artifacts
artifact_dir = Path("models/exp1_financial_ratios")

num_imputer = joblib.load(artifact_dir / "num_imputer.joblib")
encoder = joblib.load(artifact_dir / "encoder.joblib")
config = joblib.load(artifact_dir / "preprocessing_config.joblib")


def preprocess_new_data_exp1(applicant_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw applicant data using Experiment 1 preprocessing pipeline.

    Computes the 5 financial ratios, applies median imputation and one-hot encoding,
    and returns a DataFrame with the exact 128 features in the identical training order.
    """
    # Validate required raw input features
    required_features = (
        config["base_numerical_features"]
        + config["categorical_features"]
        + [
            col for col in config["binary_features"]
            if col != "EMPLOYMENT_DURATION_MISSING"
        ]
    )

    missing = set(required_features) - set(applicant_df.columns)
    if missing:
        raise ValueError(f"Missing applicant fields: {missing}")

    # 1. Clean special domain values
    df = clean_special_values(applicant_df)

    # 2. Compute 5 financial domain ratios safely
    df = compute_financial_ratios(df)

    # 3. Create employment missing indicator
    df["EMPLOYMENT_DURATION_MISSING"] = (
        df["DAYS_EMPLOYED"].isna().astype(int)
    )

    # 4. Impute car age for car owners using training median
    car_mask = (
        (df["FLAG_OWN_CAR"] == "Y")
        & df["OWN_CAR_AGE"].isna()
    )
    df.loc[car_mask, "OWN_CAR_AGE"] = config["car_owner_median"]

    # 5. Transform all 13 numerical features (8 original + 5 ratios)
    numerical_data = pd.DataFrame(
        num_imputer.transform(df[config["numerical_features"]]),
        columns=config["numerical_features"],
        index=df.index
    )

    # 6. Transform categorical features (109 columns)
    categorical_data = pd.DataFrame(
        encoder.transform(df[config["categorical_features"]]),
        columns=config["encoded_feature_names"],
        index=df.index
    )

    # 7. Binary features
    binary_data = df[config["binary_features"]].copy()

    # 8. Combine in identical training order
    processed = pd.concat(
        [numerical_data, categorical_data, binary_data],
        axis=1
    )
    processed = processed[config["final_feature_names"]]

    assert not processed.isna().any().any(), (
        "Missing values remain after Experiment 1 preprocessing!"
    )
    assert np.isfinite(processed.to_numpy(dtype=float)).all(), (
        "Non-finite values found in preprocessed data!"
    )

    return processed
