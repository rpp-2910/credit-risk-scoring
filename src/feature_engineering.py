import numpy as np
import pandas as pd


def safe_ratio(numerator_series: pd.Series, denominator_series: pd.Series) -> pd.Series:
    """Safely divide two series, converting non-positive, missing,

    or non-finite denominators to NaN before division to prevent
    zero-division and infinity.
    """
    # Replace non-positive, zero, or non-finite denominators with NaN before division
    valid_denominator = denominator_series.where(
        (denominator_series > 0) & np.isfinite(denominator_series),
        np.nan
    )
    valid_numerator = numerator_series.where(
        np.isfinite(numerator_series),
        np.nan
    )
    ratio = valid_numerator / valid_denominator
    # Ensure any non-finite result (e.g. accidental overflow) is mapped to NaN
    return ratio.mask(~np.isfinite(ratio), np.nan)


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Computes five financial domain ratios from raw applicant features.

    1. CREDIT_INCOME_RATIO: Credit amount relative to total income (credit-to-income).
    2. ANNUITY_INCOME_RATIO: Annuity payment relative to total income (annuity-to-income).
    3. CREDIT_GOODS_RATIO: Credit amount relative to goods price (credit-to-goods-price).
    4. ANNUITY_CREDIT_RATIO: Annuity payment relative to credit amount (annuity-to-credit).
    5. INCOME_PER_FAMILY_MEMBER: Household total income divided by family size (household income per member).
    """
    df = df.copy()

    # 1. Credit-to-Income Ratio
    df["CREDIT_INCOME_RATIO"] = safe_ratio(
        df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"]
    )

    # 2. Annuity-to-Income Ratio
    df["ANNUITY_INCOME_RATIO"] = safe_ratio(
        df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"]
    )

    # 3. Credit-to-Goods-Price Ratio
    df["CREDIT_GOODS_RATIO"] = safe_ratio(
        df["AMT_CREDIT"], df["AMT_GOODS_PRICE"]
    )

    # 4. Annuity-to-Credit Ratio
    df["ANNUITY_CREDIT_RATIO"] = safe_ratio(
        df["AMT_ANNUITY"], df["AMT_CREDIT"]
    )

    # 5. Household Income Per Family Member
    df["INCOME_PER_FAMILY_MEMBER"] = safe_ratio(
        df["AMT_INCOME_TOTAL"], df["CNT_FAM_MEMBERS"]
    )

    return df


FINANCIAL_RATIO_NAMES = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_GOODS_RATIO",
    "ANNUITY_CREDIT_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
]
