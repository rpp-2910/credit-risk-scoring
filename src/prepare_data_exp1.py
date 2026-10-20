from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from preprocess import clean_special_values
from feature_engineering import compute_financial_ratios, FINANCIAL_RATIO_NAMES


def prepare_experiment1_data():
    print("Loading dataset and split assignments...")
    df = pd.read_csv("data/application_train.csv")
    splits = pd.read_csv("data/split_assignments.csv")

    assert df["SK_ID_CURR"].is_unique
    assert splits["SK_ID_CURR"].is_unique

    df = df.merge(
        splits,
        on="SK_ID_CURR",
        how="inner",
        validate="one_to_one"
    )

    # RULE: Use existing training and validation splits.
    # Do NOT load, transform, or evaluate the test partition.
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    print(f"Training set records: {len(train_df)}")
    print(f"Validation set records: {len(val_df)}")

    # 1. Clean special domain values
    train_df = clean_special_values(train_df)
    val_df = clean_special_values(val_df)

    # 2. Add the 5 financial domain ratios safely
    train_df = compute_financial_ratios(train_df)
    val_df = compute_financial_ratios(val_df)

    # 3. Preserve employment missing indicator
    for dataset in [train_df, val_df]:
        dataset["EMPLOYMENT_DURATION_MISSING"] = (
            dataset["DAYS_EMPLOYED"].isna().astype(int)
        )

    # 4. Calculate car owner median age using TRAINING data only (no data leakage)
    car_owner_median = train_df.loc[
        train_df["FLAG_OWN_CAR"] == "Y",
        "OWN_CAR_AGE"
    ].median()
    assert pd.notna(car_owner_median)

    for dataset in [train_df, val_df]:
        mask = (
            (dataset["FLAG_OWN_CAR"] == "Y")
            & dataset["OWN_CAR_AGE"].isna()
        )
        dataset.loc[mask, "OWN_CAR_AGE"] = car_owner_median

    # 5. Define numerical features (original 8 + 5 financial ratios = 13)
    base_numerical_features = [
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "AMT_INCOME_TOTAL",
        "DAYS_EMPLOYED",
        "OWN_CAR_AGE"
    ]
    numerical_features = base_numerical_features + FINANCIAL_RATIO_NAMES
    assert len(numerical_features) == 13

    # Fit median imputer on training data only
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(train_df[numerical_features])

    train_df[numerical_features] = num_imputer.transform(
        train_df[numerical_features]
    )
    val_df[numerical_features] = num_imputer.transform(
        val_df[numerical_features]
    )

    # Verify no missing or infinite values in numerical features
    assert not train_df[numerical_features].isna().any().any()
    assert not val_df[numerical_features].isna().any().any()
    assert np.isfinite(train_df[numerical_features].to_numpy()).all()
    assert np.isfinite(val_df[numerical_features].to_numpy()).all()

    # 6. Categorical encoding (identical 9 categorical features)
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

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    encoder.fit(train_df[categorical_features])

    train_encoded = encoder.transform(train_df[categorical_features])
    val_encoded = encoder.transform(val_df[categorical_features])

    encoded_feature_names = encoder.get_feature_names_out(
        categorical_features
    ).tolist()
    assert len(encoded_feature_names) == 109

    # 7. Binary features
    binary_features = [
        "FLAG_MOBIL",
        "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "EMPLOYMENT_DURATION_MISSING"
    ]

    final_feature_names = (
        numerical_features
        + encoded_feature_names
        + binary_features
    )
    assert len(final_feature_names) == 128

    def combine_features(dataset, encoded_array):
        num_part = dataset[numerical_features].reset_index(drop=True)
        cat_part = pd.DataFrame(
            encoded_array,
            columns=encoded_feature_names
        )
        bin_part = dataset[binary_features].reset_index(drop=True)
        return pd.concat([num_part, cat_part, bin_part], axis=1)

    X_train = combine_features(train_df, train_encoded)
    X_val = combine_features(val_df, val_encoded)

    y_train = train_df["TARGET"].reset_index(drop=True)
    y_val = val_df["TARGET"].reset_index(drop=True)

    # Integrity assertions
    assert X_train.shape == (215257, 128)
    assert X_val.shape == (46127, 128)
    assert X_train.columns.equals(X_val.columns)
    assert not X_train.isna().any().any()
    assert not X_val.isna().any().any()
    assert np.isfinite(X_train.to_numpy(dtype=float)).all()
    assert np.isfinite(X_val.to_numpy(dtype=float)).all()

    # 8. Save artifacts to isolated directory models/exp1_financial_ratios/
    artifact_dir = Path("models/exp1_financial_ratios")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(num_imputer, artifact_dir / "num_imputer.joblib")
    joblib.dump(encoder, artifact_dir / "encoder.joblib")

    config = {
        "base_numerical_features": base_numerical_features,
        "financial_ratio_names": FINANCIAL_RATIO_NAMES,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "binary_features": binary_features,
        "encoded_feature_names": encoded_feature_names,
        "final_feature_names": final_feature_names,
        "car_owner_median": float(car_owner_median)
    }
    joblib.dump(config, artifact_dir / "preprocessing_config.joblib")

    print("\nExperiment 1 preprocessing artifacts saved successfully to:")
    print(f"  {artifact_dir}")
    print(f"Total features: {len(final_feature_names)} (123 baseline + 5 financial ratios)")
    return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    prepare_experiment1_data()
