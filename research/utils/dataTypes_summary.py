import pprint
from typing import Any

import pandas as pd


def get_dtype_summary(df: pd.DataFrame) -> dict:
    """
    Generates a summary of data types in a DataFrame using pandas groupby.

    Returns a dictionary where keys are string representations of data types
    and values are dictionaries containing the count of columns and a list
    of column names for that data type.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        print("Warning: DataFrame is empty. Returning an empty summary.")
        return {}

    # Group column names by their data type (as a string)
    grouped_by_dtype_str: dict[Any, str] = (
        df.dtypes.astype(str)
        .groupby(df.dtypes.astype(str))
        .apply(lambda x: x.index.tolist())
        .to_dict()
    )

    # Build the final summary dictionary with counts
    dtype_summary: dict = {}
    for dtype_str, columns in grouped_by_dtype_str.items():
        dtype_summary[dtype_str] = {"count": len(columns), "columns": columns}
    return dtype_summary


# # --- Example Usage of the function ---
# dtype_summary_result = get_dtype_summary(dfTrain)

# print("\nDetailed Dtype Summary (from function):")
# pp = pprint.PrettyPrinter(indent=2)
# pp.pprint(dtype_summary_result)
