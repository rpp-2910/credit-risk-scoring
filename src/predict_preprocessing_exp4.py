import pandas as pd
from predict_preprocessing_exp1 import preprocess_new_data_exp1


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes column names for LightGBM compatibility.

    Replaces special JSON characters (like colons in one-hot encoded organization types)
    with underscores.
    """
    df = df.copy()
    df.columns = [c.replace(":", "_") for c in df.columns]
    return df


def preprocess_new_data_exp4(applicant_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw applicant data using Experiment 1 preprocessing

    and applies LightGBM-compatible column name sanitation.
    Produces identical 128 features in identical order.
    """
    processed = preprocess_new_data_exp1(applicant_df)
    return clean_feature_names(processed)
