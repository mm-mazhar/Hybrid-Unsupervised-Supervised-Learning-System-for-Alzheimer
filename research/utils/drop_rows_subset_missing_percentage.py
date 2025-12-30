import pandas as pd


def drop_rows_by_subset_missing_percentage(
    df: pd.DataFrame, cols_subset: list[str], threshold_percent: float
) -> pd.DataFrame:
    """
    Drops rows from a DataFrame where the percentage of missing values
    within a specified subset of columns meets or exceeds a threshold.

    Args:
        df: The input pandas DataFrame.
        cols_subset: A list of column names to consider for the missing percentage calculation.
        threshold_percent: The percentage threshold (e.g., 80) for dropping a row.
                           A row is dropped if >= threshold_percent of values
                           in cols_subset are missing for that row.

    Returns:
        A new DataFrame with rows dropped based on the criteria.
    """
    # Ensure the subset columns actually exist in the DataFrame
    # Filter the input list to only include columns present in the DataFrame
    actual_cols_subset: list[str] = [col for col in cols_subset if col in df.columns]

    # Handle edge case: no columns in the subset or no columns in the original DataFrame
    if not actual_cols_subset or df.empty:
        if df.empty:
            print("Warning: DataFrame is empty. No rows dropped.")
        elif not actual_cols_subset:
            print(
                "Warning: Subset of columns is empty or none of the specified columns are in the DataFrame. No rows dropped."
            )
        return df.copy()  # Return a copy to maintain function signature consistency

    num_cols_subset: int = len(actual_cols_subset)

    # Select the subset of the DataFrame
    df_subset: pd.DataFrame = df[actual_cols_subset]

    # Calculate the number of missing values per row within the subset (axis=1 for row-wise sum)
    missing_counts_per_row: pd.Series = df_subset.isnull().sum(axis=1)

    # Calculate the percentage of missing values per row within the subset
    # Use .astype(float) for robustness in division
    percentage_missing_per_row = (
        missing_counts_per_row.astype(float) / num_cols_subset
    ) * 100

    # Identify rows where the percentage is >= the threshold
    # This creates a boolean Series with the same index as the DataFrame
    rows_to_drop_mask: pd.Series[bool] = percentage_missing_per_row >= threshold_percent

    # Drop the identified rows - select rows where the mask is False (~)
    df_cleaned: pd.DataFrame = df[
        ~rows_to_drop_mask
    ].copy()  # Use .copy() to ensure a new DataFrame

    print(
        f"Criteria: Drop rows with >= {threshold_percent}% missing in {num_cols_subset} columns ({', '.join(actual_cols_subset[:5])}{'...' if len(actual_cols_subset)>5 else ''})"
    )
    print(f"Original rows: {len(df)}")
    print(
        f"Rows dropped: {rows_to_drop_mask.sum()}"
    )  # Summing the boolean mask gives the count of True values
    print(f"Remaining rows: {len(df_cleaned)}")

    return df_cleaned
