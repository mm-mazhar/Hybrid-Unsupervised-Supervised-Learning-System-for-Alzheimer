import pandas as pd
from typing import Any


def impute_missing_values(
    df: pd.DataFrame, num_strategy: str = "median", cat_strategy: str = "mode"
) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame using specified strategies (safe for pandas 3.0+).

    Args:
        df (pd.DataFrame): Input DataFrame.
        num_strategy (str): Strategy for numerical columns: 'mean', 'median', or 'mode'.
        cat_strategy (str): Strategy for categorical columns: 'mode' or a string (e.g., 'missing').

    Returns:
        pd.DataFrame: A DataFrame with missing values imputed.
    """
    df = df.copy()

    # Validate numerical strategy
    if num_strategy not in {"mean", "median", "mode"}:
        raise ValueError("num_strategy must be 'mean', 'median', or 'mode'")

    # Get column types
    numerical_cols: pd.Index[str] = df.select_dtypes(include=["number"]).columns
    categorical_cols: pd.Index[str] = df.select_dtypes(exclude=["number"]).columns

    # Impute numerical columns
    for col in numerical_cols:
        if df[col].isnull().any():
            if num_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif num_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif num_strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])

    # Impute categorical columns
    for col in categorical_cols:
        if df[col].isnull().any():
            if cat_strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(cat_strategy)

    # Report total missing values
    missing_total: Any = df.isnull().sum().sum()
    print(f"Number of missing values after imputation: {missing_total}")

    return df
