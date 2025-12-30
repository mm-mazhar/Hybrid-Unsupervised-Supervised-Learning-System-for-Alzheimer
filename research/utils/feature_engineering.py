import pandas as pd
import numpy as np


def engineer_temporal_features(df, drop_originals=True):
    """
    Engineers new features based on the temporal nature of the data (_03 and _12 suffixes).

    Args:
        df (pd.DataFrame): The input dataframe.
        drop_originals (bool): If True, drops the original _03 and _12 columns after creating new features.

    Returns:
        pd.DataFrame: The dataframe with new engineered features.
    """
    print("Engineering temporal features...")
    df_engineered = df.copy()

    # Identify columns with _03 and _12 suffixes
    cols_03 = [col for col in df.columns if col.endswith("_03")]
    cols_12 = [col for col in df.columns if col.endswith("_12")]

    # Create a mapping from _03 to _12 columns based on the base name
    base_names_03 = {col[:-3] for col in cols_03}
    base_names_12 = {col[:-3] for col in cols_12}
    common_base_names = list(base_names_03.intersection(base_names_12))

    print(f"Found {len(common_base_names)} feature pairs between 2003 and 2012.")

    new_features = pd.DataFrame(index=df_engineered.index)

    for base_name in common_base_names:
        col_03 = base_name + "_03"
        col_12 = base_name + "_12"

        # --- Handle Numeric Features: Create "delta" features ---
        # Ensure your custom pipeline has already converted numerics to float/int
        if pd.api.types.is_numeric_dtype(df[col_03]) and pd.api.types.is_numeric_dtype(
            df[col_12]
        ):
            new_features[f"delta_{base_name}"] = df[col_12] - df[col_03]

        # --- Handle Categorical Features: Create "changed" features ---
        elif pd.api.types.is_categorical_dtype(
            df[col_03]
        ) or pd.api.types.is_object_dtype(df[col_03]):
            # Convert to string to ensure consistent comparison, even for mixed types
            new_features[f"changed_{base_name}"] = (
                df[col_03].astype(str) != df[col_12].astype(str)
            ).astype(int)

    df_engineered = pd.concat([df_engineered, new_features], axis=1)

    if drop_originals:
        print(f"Dropping {len(cols_03) + len(cols_12)} original temporal columns...")
        df_engineered = df_engineered.drop(columns=cols_03 + cols_12)

    print(
        f"Feature engineering complete. New feature count: {len(df_engineered.columns)}"
    )
    return df_engineered
