import pandas as pd


def convert_boolean_to_category(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Converts all pandas 'boolean' columns in a DataFrame to 'category' dtype.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.Index]: Updated DataFrame and converted column names.
    """
    df = df.copy()
    bool_cols: list[str] = df.select_dtypes(include="boolean").columns.to_list()

    for col in bool_cols:
        df[col] = df[col].astype("category")

    print(
        f"Converted {len(bool_cols)} boolean columns to category: {list(bool_cols)}\n"
    )
    return df, bool_cols
