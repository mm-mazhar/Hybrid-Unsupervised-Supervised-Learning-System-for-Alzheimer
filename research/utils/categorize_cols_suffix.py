import pandas as pd


def categorize_columns_by_suffix(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Categorizes DataFrame columns into three lists based on suffixes '_03' and '_12'.

    Args:
        df: The input pandas DataFrame.

    Returns:
        A dictionary with three keys:
        - 'cols_03': List of column names ending with '_03'.
        - 'cols_12': List of column names ending with '_12'.
        - 'cols_rest': List of remaining column names.
    """
    all_cols: pd.Index[str] = df.columns  # Get all column names

    # Use list comprehensions for concise filtering
    cols_03: list[str] = [col for col in all_cols if col.endswith("_03")]
    cols_12: list[str] = [col for col in all_cols if col.endswith("_12")]

    # Columns not in cols_03 AND not in cols_12 belong to the rest
    cols_rest: list[str] = [
        col for col in all_cols if col not in cols_03 and col not in cols_12
    ]

    # Return the lists in a dictionary
    return {"cols_03": cols_03, "cols_12": cols_12, "cols_rest": cols_rest}
