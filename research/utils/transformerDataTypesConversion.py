from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# --- Custom Transformers ---
class SpecificColumnCategorizer(BaseEstimator, TransformerMixin):
    """Converts a predefined list of columns to 'category' dtype."""

    def __init__(self, columns_to_categorize) -> None:
        self.columns_to_categorize: Any = columns_to_categorize
        self.fitted_columns_: list[Any] = []

    def fit(self, X, y=None):
        # Identify which of the specified columns actually exist in X
        self.fitted_columns_ = [
            col for col in self.columns_to_categorize if col in X.columns
        ]
        if not self.fitted_columns_:
            print(
                f"Warning: None of the specified columns {self.columns_to_categorize} found in DataFrame."
            )
        return self

    def transform(self, X) -> None | pd.DataFrame:
        X_copy: pd.DataFrame = X.copy()
        for col in self.fitted_columns_:
            X_copy[col] = X_copy[col].astype("category")
            print(f"SpecificColumnCategorizer: Converted '{col}' to category.")
        return X_copy

    def get_feature_names_out(self, input_features=None) -> Any | list[Any]:
        return input_features if input_features is not None else self.fitted_columns_


class ObjectToCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Identifies object columns and converts them to 'category' if they meet
    uniqueness criteria (low ratio of unique values or few absolute unique values).
    """

    def __init__(self, threshold_ratio=0.1, max_unique=50) -> None:
        self.threshold_ratio: float = threshold_ratio
        self.max_unique: int = max_unique
        self.object_cols_to_convert_: list[Any] = []

    def fit(self, X, y=None):
        self.object_cols_to_convert_ = []
        object_columns: Any = X.select_dtypes(include=["object"]).columns
        for col in object_columns:
            n_unique: Any = X[col].nunique()
            ratio: Any = n_unique / len(X[col])
            if (
                ratio < self.threshold_ratio and n_unique <= self.max_unique
            ):  # Changed OR to AND
                self.object_cols_to_convert_.append(col)
        return self

    def transform(self, X) -> None | pd.DataFrame:
        X_copy: pd.DataFrame = X.copy()
        if not self.object_cols_to_convert_:
            print(
                "ObjectToCategoryTransformer: No object columns met criteria for conversion."
            )
        for col in self.object_cols_to_convert_:
            X_copy[col] = X_copy[col].astype("category")
            print(f"ObjectToCategoryTransformer: Converted '{col}' to category.")
        return X_copy

    def get_feature_names_out(self, input_features=None) -> None | Any:
        return input_features  # Dtypes change, names don't


class FloatToCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Converts float columns that only contain 0.0 and 1.0 (and NaNs) to 'category'.
    """

    def __init__(self) -> None:
        self.float_cols_to_convert_: list[Any] = []

    def fit(self, X, y=None):
        self.float_cols_to_convert_ = []
        float_columns: Any = X.select_dtypes(
            include=[np.float64, np.float32]
        ).columns  # or np.floating
        for col in float_columns:
            unique_values: Any = X[col].dropna().unique()
            # Check if unique values are a subset of {0.0, 1.0}
            if (
                all(val in [0.0, 1.0] for val in unique_values)
                and len(unique_values) > 0
            ):
                # ensure it's not an empty column after dropna, and has at most 0 and 1
                if len(unique_values) <= 2:
                    self.float_cols_to_convert_.append(col)
        return self

    def transform(self, X) -> None | pd.DataFrame:
        X_copy: pd.DataFrame = X.copy()
        if not self.float_cols_to_convert_:
            print(
                "FloatToCategoryTransformer: No float columns met criteria for conversion to category (0/1)."
            )
        for col in self.float_cols_to_convert_:
            X_copy[col] = X_copy[col].astype("category")
            print(
                f"FloatToCategoryTransformer: Converted float column '{col}' to category."
            )
        return X_copy

    def get_feature_names_out(self, input_features=None) -> None | Any:
        return input_features


class BooleanToCategoryTransformer(BaseEstimator, TransformerMixin):
    """Converts boolean columns to 'category' dtype."""

    def __init__(self) -> None:
        self.bool_cols_to_convert_ = []

    def fit(self, X, y=None):
        self.bool_cols_to_convert_: Any = X.select_dtypes(
            include=["bool"]
        ).columns.tolist()
        return self

    def transform(self, X) -> None | pd.DataFrame:
        X_copy: pd.DataFrame = X.copy()
        if not self.bool_cols_to_convert_:
            print(
                "BooleanToCategoryTransformer: No boolean columns found for conversion."
            )
        for col in self.bool_cols_to_convert_:
            X_copy[col] = X_copy[col].astype("category")
            print(
                f"BooleanToCategoryTransformer: Converted boolean column '{col}' to category."
            )
        return X_copy

    def get_feature_names_out(self, input_features=None) -> None | Any:
        return input_features
