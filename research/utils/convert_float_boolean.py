import pandas as pd


def convert_float_to_bool(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    converted_cols: list = []
    for col in df.select_dtypes(include="float").columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0.0, 1.0}):
            df[col] = df[col].astype("boolean")
            converted_cols.append(col)
    print(f"Converted {len(converted_cols)} columns to boolean: {converted_cols}\n")
    return df, converted_cols
