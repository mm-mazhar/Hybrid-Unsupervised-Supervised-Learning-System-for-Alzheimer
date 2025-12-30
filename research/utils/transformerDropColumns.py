from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    A Scikit-learn transformer to drop specified columns from a DataFrame.
    Halts with an error if any specified column does not exist during fit.
    """

    def __init__(self, columns_to_drop: list[str]) -> None:
        """
        Args:
            columns_to_drop (List[str]): A list of column names to be dropped.
        """
        if not isinstance(columns_to_drop, list):
            raise TypeError("columns_to_drop must be a list of strings.")
        if not all(isinstance(col, str) for col in columns_to_drop):
            raise TypeError("All elements in columns_to_drop must be strings.")

        self.columns_to_drop: list[str] = columns_to_drop
        self._fitted_columns_to_drop: list[str] = []  # Columns validated during fit

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Checks if all specified columns exist in the DataFrame.
        If any column is missing, it raises a ValueError.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Any, optional): Ignored. Defaults to None.

        Returns:
            self: The fitted transformer.

        Raises:
            ValueError: If any column in `columns_to_drop` is not found in X.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        missing_columns: list[str] = [
            col for col in self.columns_to_drop if col not in X.columns
        ]

        if missing_columns:
            message: str = (
                f"ColumnDropper Error: The following specified columns for dropping "
                f"were not found in the DataFrame: {missing_columns}. "
                "Halting process."
            )
            print(message)  # Also print for visibility if error is caught elsewhere
            raise ValueError(message)

        # All columns exist, store them for transform
        self._fitted_columns_to_drop = self.columns_to_drop
        # print(
        #     f"ColumnDropper: All specified columns to drop found: {self._fitted_columns_to_drop}"
        # )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the specified (and validated) columns from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with specified columns removed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Check if fit has been called (i.e., _fitted_columns_to_drop is populated)
        if (
            not hasattr(self, "_fitted_columns_to_drop")
            or not self._fitted_columns_to_drop
        ):
            if not self.columns_to_drop:  # No columns were ever specified
                print(
                    "ColumnDropper: No columns were specified to drop. Returning original DataFrame."
                )
                return X.copy()  # Or just X if in-place modification is okay for caller
            # This state should ideally not be reached if fit is always called before transform in a pipeline
            raise RuntimeError(
                "Transform called before fit, or fit did not identify any columns to drop (though it should have raised an error if columns_to_drop was non-empty and columns were missing)."
            )

        X_copy: pd.DataFrame = X.copy()
        # Only drop columns that were confirmed during fit AND are still in X (robustness for transform calls on slightly different DFs)
        cols_to_actually_drop_now: list[str] = [
            col for col in self._fitted_columns_to_drop if col in X_copy.columns
        ]

        if not cols_to_actually_drop_now and self._fitted_columns_to_drop:
            missing_at_transform: list[str] = [
                col for col in self._fitted_columns_to_drop if col not in X_copy.columns
            ]
            print(
                f"ColumnDropper Warning: Columns {missing_at_transform} were present during fit but are missing now during transform. Dropping remaining."
            )
            # This warning indicates an inconsistency between fit and transform data, but we proceed if some are still there.

        if cols_to_actually_drop_now:
            X_copy = X_copy.drop(columns=cols_to_actually_drop_now)
            print(f"ColumnDropper: Dropped columns: {cols_to_actually_drop_now}")
        elif (
            self._fitted_columns_to_drop
        ):  # Columns were specified and fit, but none are in X now
            print(
                f"ColumnDropper Warning: All columns ({self._fitted_columns_to_drop}) intended for dropping are missing from the DataFrame provided to transform."
            )
        # If self._fitted_columns_to_drop is empty (because self.columns_to_drop was empty), do nothing.
        return X_copy

    def get_feature_names_out(
        self, input_features: Optional[list[str]] = None
    ) -> list[str]:
        """
        Returns the feature names after transformation.

        Args:
            input_features (List[str], optional): Column names of the input DataFrame.

        Returns:
            List[str]: List of column names after dropping specified columns.
        """
        if input_features is None:
            raise ValueError(
                "input_features must be provided to get_feature_names_out."
            )

        if not hasattr(self, "_fitted_columns_to_drop"):
            # Fit hasn't been called, assume no columns are dropped yet for get_feature_names_out
            return list(input_features)

        return [
            col for col in input_features if col not in self._fitted_columns_to_drop
        ]
