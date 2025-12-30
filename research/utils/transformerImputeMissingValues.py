from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    A Scikit-learn transformer to impute missing values in a DataFrame.
    Numerical columns can be imputed using 'mean', 'median', or 'mode'.
    Categorical/Object columns can be imputed using 'mode' or a specified constant string.
    """

    def __init__(
        self, num_strategy: str = "median", cat_strategy: str | Any = "mode"
    ) -> None:
        """
        Args:
            num_strategy (str): Strategy for numerical columns: 'mean', 'median', or 'mode'.
                                Defaults to "median".
            cat_strategy (Union[str, Any]): Strategy for categorical/object columns.
                                          Can be 'mode' or a specific constant value
                                          (e.g., "Unknown", "Missing"). Defaults to "mode".
        """
        if num_strategy not in {"mean", "median", "mode"}:
            raise ValueError("num_strategy must be 'mean', 'median', or 'mode'")
        # For cat_strategy, 'mode' is a special keyword, others are direct fill values.
        self.num_strategy: str = num_strategy
        self.cat_strategy: str | Any = cat_strategy

        self.num_imputers_: Dict[str, Any] = {}
        self.cat_imputers_: Dict[str, Any] = {}
        self.numerical_cols_: list[str] = []
        self.categorical_cols_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Learns the imputation values from the training data X.

        Args:
            X (pd.DataFrame): The input training DataFrame.
            y (Any, optional): Ignored. Defaults to None.

        Returns:
            self: The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        X_fit: pd.DataFrame = X.copy()  # Work on a copy

        self.numerical_cols_ = X_fit.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols_ = X_fit.select_dtypes(exclude=np.number).columns.tolist()

        # Learn numerical imputers
        for col in self.numerical_cols_:
            if X_fit[col].isnull().any():  # Only learn if there are NaNs to learn from
                if self.num_strategy == "mean":
                    self.num_imputers_[col] = X_fit[col].mean()
                elif self.num_strategy == "median":
                    self.num_imputers_[col] = X_fit[col].median()
                elif self.num_strategy == "mode":
                    mode_val: pd.Series = X_fit[col].mode()
                    # Handle empty mode (e.g., if column is all NaN or no clear mode)
                    self.num_imputers_[col] = (
                        mode_val.iloc[0] if not mode_val.empty else 0
                    )  # Fallback to 0 if mode is empty
            else:  # If no NaNs, store a placeholder or skip; fillna won't act anyway
                self.num_imputers_[col] = (
                    None  # Or a value like X_fit[col].median() just in case test has NaNs
                )

        # Learn categorical imputers
        for col in self.categorical_cols_:
            if X_fit[col].isnull().any():
                if self.cat_strategy == "mode":
                    mode_val = X_fit[col].mode()
                    # Handle empty mode
                    self.cat_imputers_[col] = (
                        mode_val.iloc[0] if not mode_val.empty else "Unknown"
                    )  # Fallback if mode is empty
                else:
                    self.cat_imputers_[col] = (
                        self.cat_strategy
                    )  # Use the provided constant
            else:
                self.cat_imputers_[col] = (
                    None  # Or a value like X_fit[col].mode().iloc[0] if not X_fit[col].mode().empty else self.cat_strategy
                )

        print(
            f"MissingValueImputer: Fitted. Numerical imputers: {self.num_imputers_}, Categorical imputers: {self.cat_imputers_}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in X using the learned imputation values.

        Args:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if (
            not self.num_imputers_
            and not self.cat_imputers_
            and (self.numerical_cols_ or self.categorical_cols_)
        ):
            # This condition means fit might have found columns but no NaNs to learn imputers for,
            # OR fit was called on an empty dataframe. For robustness, we can re-check if _cols attribute exists.
            if not hasattr(self, "numerical_cols_"):
                raise RuntimeError("Transform called before fit. Call fit first.")

        X_transformed: pd.DataFrame = X.copy()

        # Impute numerical columns
        for col in self.numerical_cols_:
            if (
                col in X_transformed.columns
                and col in self.num_imputers_
                and self.num_imputers_[col] is not None
            ):
                if X_transformed[col].isnull().any():
                    X_transformed[col] = X_transformed[col].fillna(
                        self.num_imputers_[col]
                    )
            elif col in X_transformed.columns and X_transformed[col].isnull().any():
                print(
                    f"Warning: Column '{col}' has NaNs in transform data but no imputer was learned (or column was not in fit data). NaNs will remain."
                )

        # Impute categorical columns
        for col in self.categorical_cols_:
            if (
                col in X_transformed.columns
                and col in self.cat_imputers_
                and self.cat_imputers_[col] is not None
            ):
                if X_transformed[col].isnull().any():
                    X_transformed[col] = X_transformed[col].fillna(
                        self.cat_imputers_[col]
                    )
            elif col in X_transformed.columns and X_transformed[col].isnull().any():
                print(
                    f"Warning: Column '{col}' has NaNs in transform data but no imputer was learned (or column was not in fit data). NaNs will remain."
                )

        missing_total_after: Any = X_transformed.isnull().sum().sum()
        print(
            f"MissingValueImputer: Number of missing values after imputation: {missing_total_after}"
        )

        return X_transformed

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """
        Returns feature names, which are unchanged by this transformer.
        """
        if input_features is None:
            # Try to use columns seen during fit if available, otherwise raise error or return default
            if hasattr(self, "numerical_cols_") and hasattr(self, "categorical_cols_"):
                # This assumes the order and set of columns from fit is what's expected
                # It's more robust if input_features is provided.
                return self.numerical_cols_ + self.categorical_cols_
            raise ValueError(
                "input_features must be provided to get_feature_names_out if fit hasn't been called with a non-empty DataFrame."
            )
        return list(input_features)
