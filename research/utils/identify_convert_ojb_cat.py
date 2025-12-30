from typing import Literal
import pandas as pd


def identify_and_convert_object_to_category(
    df: pd.DataFrame,
    threshold_ratio: float = 0.1,  # Max unique values / total rows
    max_unique: int = 50,  # Absolute max unique values allowed
) -> tuple[pd.DataFrame, list[str]]:
    """
    Identifies object columns with low cardinality and converts them to the
    pandas category dtype.

    Args:
        df: The input pandas DataFrame.
        threshold_ratio: Maximum ratio of unique non-missing values to total rows
                         to consider a column as low cardinality. Default is 0.1 (10%).
        max_unique: Absolute maximum number of unique non-missing values to
                    consider a column as low cardinality. Default is 50.

    Returns:
        A tuple containing:
        - The DataFrame with converted columns.
        - A list of column names that were converted.

    Note: Columns with high cardinality (like unique identifiers or free text)
          should generally NOT be converted to category. Adjust thresholds
          as needed for your specific data.
    """
    # Create a copy to avoid modifying the original DataFrame in place
    df_converted: pd.DataFrame = df.copy()
    converted_cols: list[str] = []
    total_rows: int = len(df_converted)

    # Handle empty DataFrame edge case
    if total_rows == 0:
        print("Warning: DataFrame is empty. No columns processed.")
        return df_converted, converted_cols

    # Iterate through each column
    for col in df_converted.columns:
        # Check if the column's dtype is 'object'
        if df_converted[col].dtype == "object":
            # Get the number of unique non-missing values
            unique_count: int = df_converted[col].dropna().nunique()

            # Get the number of non-missing values (count)
            non_null_count: int = df_converted[col].count()

            # Calculate the ratio of unique values to total rows (if not all missing)
            unique_ratio: float | Literal[0] = (
                unique_count / total_rows if total_rows > 0 else 0
            )

            # Determine if the column has low cardinality based on thresholds
            # Consider a column low cardinality if:
            # 1. The number of unique non-missing values is below max_unique AND
            # 2. The ratio of unique non-missing values to total rows is below threshold_ratio
            # This prevents converting columns that might have few uniques but are very long
            # or columns that have many uniques but the ratio is low (less common).
            # It also handles columns that are all NaNs correctly (unique_count = 0).

            if unique_count <= max_unique and unique_ratio <= threshold_ratio:
                # Also, maybe add a check that it's not a date/time string that needs parsing
                # For simplicity here, we rely on the cardinality check.
                # Real-world might need more sophisticated checks or exclusion lists.

                try:
                    # Attempt the conversion to category
                    df_converted[col] = df_converted[col].astype("category")
                    converted_cols.append(col)
                    # print(f"Converted '{col}' from object to category (unique: {unique_count}, ratio: {unique_ratio:.4f})") # Optional: print converted columns
                except Exception as e:
                    print(f"Could not convert '{col}' to category: {e}")
                    pass  # Do not add to converted_cols if conversion failed
            else:
                print(
                    f"Skipped '{col}': High cardinality (unique: {unique_count}, ratio: {unique_ratio:.4f})"
                )  # Optional: print skipped columns

    return df_converted, converted_cols
