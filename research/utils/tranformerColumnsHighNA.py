from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DropColumnsHighNA(BaseEstimator, TransformerMixin):
    """
    A Scikit-learn transformer to drop columns from a DataFrame if their
    percentage of missing values exceeds a specified threshold.
    """

    def __init__(self, threshold: float = 70.0) -> None:
        """
        Args:
            threshold (float): The percentage threshold (0-100). Columns with missing
                               values greater than or equal to this threshold will be dropped.
                               Defaults to 70.0.
        """
        if not (0 <= threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100.")
        self.threshold: float = threshold
        self.columns_to_drop_: list[Any] = (
            []
        )  # Stores names of columns to drop, learned during fit

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Identifies columns with a high percentage of missing values.

        Args:
            X (pd.DataFrame): The input DataFrame.
            y (Any, optional): Ignored. Defaults to None.

        Returns:
            self: The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if X.empty:
            print(
                "DropColumnsHighNA: DataFrame is empty. No columns to analyze or drop."
            )
            self.columns_to_drop_ = []
            return self

        missing_percentage = X.isnull().sum() * 100 / len(X)
        self.columns_to_drop_ = missing_percentage[
            missing_percentage >= self.threshold
        ].index.tolist()

        if self.columns_to_drop_:
            # print(
            #     f"DropColumnsHighNA: Identified columns to drop (missing >= {self.threshold}%): {self.columns_to_drop_}"
            # )
            pass
        else:
            print(
                f"DropColumnsHighNA: No columns found with missing values >= {self.threshold}%."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified columns from the DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with high-missing-value columns removed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Check if fit has been called
        if not hasattr(self, "columns_to_drop_"):
            raise RuntimeError("Transform called before fit. Call fit first.")

        X_copy = X.copy()

        # It's safer to only try to drop columns that are actually present in the current X
        # and were identified during fit.
        cols_to_actually_drop_now: list[Any] = [
            col for col in self.columns_to_drop_ if col in X_copy.columns
        ]

        if cols_to_actually_drop_now:
            X_copy: pd.DataFrame = X_copy.drop(
                columns=cols_to_actually_drop_now, errors="ignore"
            )  # errors='ignore' as safeguard
            print(f"DropColumnsHighNA: Dropped columns: {cols_to_actually_drop_now}")
        elif (
            self.columns_to_drop_
        ):  # Columns were identified in fit, but none exist in current X
            print(
                f"DropColumnsHighNA: Columns {self.columns_to_drop_} were targeted for dropping based on 'fit', but none are present in the DataFrame provided to 'transform'."
            )
        # If self.columns_to_drop_ was empty, do nothing.

        return X_copy

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """
        Returns the feature names after transformation.

        Args:
            input_features (list[str], optional): Column names of the input DataFrame.
                                                  If None, an attempt might be made to infer,
                                                  but it's best if provided.

        Returns:
            list[str]: List of column names after dropping high-missing columns.
        """
        if input_features is None:
            # This is a fallback. Ideally, you'd ensure `fit` has been called and `X` had columns.
            # For robust behavior, it's better if input_features is always provided when get_feature_names_out is called.
            if hasattr(self, "columns_to_drop_"):  # check if fit was called
                print(
                    "DropColumnsHighNA Warning: input_features was None in get_feature_names_out. Returning an empty list or based on limited info."
                )
                # This part is tricky without input_features; it assumes columns_to_drop_ was derived from *some* features.
                # A more robust approach might require X to be passed or stored in fit for this scenario.
                return []  # Simplistic fallback
            else:
                raise RuntimeError(
                    "get_feature_names_out called before fit or on an uninitialized transformer."
                )

        if not hasattr(self, "columns_to_drop_"):
            # Fit hasn't been called
            return list(input_features)

        return [col for col in input_features if col not in self.columns_to_drop_]
