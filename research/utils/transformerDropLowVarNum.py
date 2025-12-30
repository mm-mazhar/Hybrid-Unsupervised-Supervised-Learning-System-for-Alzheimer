from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler  # For temporary scaling


class IdentifyAndDropLowVarNum(BaseEstimator, TransformerMixin):
    """
    A Scikit-learn transformer to identify and drop numeric columns based on
    their variance after a temporary internal scaling to [0, 1].
    The output DataFrame is NOT scaled by this transformer.

    Steps during fit:
    1. Identify numeric columns.
    2. Temporarily scale these numeric columns to the [0, 1] range.
    3. Identify scaled numeric columns with variance exactly 0 (constant features).
    4. Identify remaining scaled numeric columns with variance below a
       specified quasi_constant_threshold.
    5. Store the names of these original columns for dropping.
    Non-numeric columns are ignored for variance checks.
    """

    def __init__(self, quasi_constant_threshold: float = 0.01) -> None:
        """
        Args:
            quasi_constant_threshold (float): The variance threshold for quasi-constant
                                              features, applied to temporarily scaled data.
                                              Numeric columns with scaled variance strictly
                                              less than this (and not zero) will be identified
                                              for dropping. Defaults to 0.01.
                                              Zero-variance columns (on scaled data) are
                                              always identified for dropping.
        """
        if quasi_constant_threshold < 0:
            raise ValueError("quasi_constant_threshold must be non-negative.")
        self.quasi_constant_threshold: float = quasi_constant_threshold

        # Scaler is now an internal detail of fit, not stored as self.scaler_
        self.numeric_cols_considered_: list[str] = []  # Columns identified as numeric
        self.columns_to_drop_: list[str] = (
            []
        )  # Final list of original column names to drop
        self._input_features_during_fit: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Identifies numeric columns for dropping based on their variance after
        temporary internal scaling.

        Args:
            X (pd.DataFrame): The input training DataFrame.
            y (Any, optional): Ignored. Defaults to None.

        Returns:
            self: The fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self._input_features_during_fit = list(X.columns)
        self.columns_to_drop_ = []  # Reset for each fit

        self.numeric_cols_considered_ = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        if not self.numeric_cols_considered_:
            print(
                "IdentifyAndDropLowVarNum: No numeric columns found to analyze for variance."
            )
            return self

        # Work on a copy of the numeric part of X for temporary scaling
        X_numeric_fit: pd.DataFrame = X[self.numeric_cols_considered_].copy()

        valid_numeric_cols_for_scaling: list[str] = []
        for col in self.numeric_cols_considered_:
            if X_numeric_fit[col].nunique(dropna=True) > 0:
                valid_numeric_cols_for_scaling.append(col)
            else:
                print(
                    f"IdentifyAndDropLowVarNum: Numeric column '{col}' has no unique non-NaN values. Checking its original variance."
                )

        if not valid_numeric_cols_for_scaling:
            print(
                "IdentifyAndDropLowVarNum: No valid numeric columns for temporary scaling (e.g., all are entirely NaN or have no variance)."
            )
            # Check for zero variance on original data if no scaling possible
            if self.numeric_cols_considered_:
                original_variances: pd.Series = X[self.numeric_cols_considered_].var(
                    ddof=0
                )
                self.columns_to_drop_ = original_variances[
                    original_variances == 0.0
                ].index.tolist()
                if self.columns_to_drop_:
                    print(
                        f"IdentifyAndDropLowVarNum: Dropping originally zero-variance columns (no scaling performed): {self.columns_to_drop_}"
                    )
            return self

        # Initialize and fit a temporary scaler
        temp_scaler = MinMaxScaler()
        temp_scaler.fit(X_numeric_fit[valid_numeric_cols_for_scaling])

        # Temporarily transform these valid numeric columns for variance calculation
        X_scaled_numeric_array: np.ndarray = temp_scaler.transform(
            X_numeric_fit[valid_numeric_cols_for_scaling]
        )
        X_scaled_numeric_df = pd.DataFrame(
            X_scaled_numeric_array,
            columns=valid_numeric_cols_for_scaling,
            index=X_numeric_fit.index,
        )

        # Calculate variance on this temporary scaled data
        scaled_variances: pd.Series = X_scaled_numeric_df.var(ddof=0)

        # 1. Identify zero-variance (constant) columns from the scaled data
        zero_var_cols: list[str] = scaled_variances[
            scaled_variances == 0.0
        ].index.tolist()
        if zero_var_cols:
            print(
                f"IdentifyAndDropLowVarNum: Identified columns as CONSTANT after temporary scaling: {zero_var_cols}"
            )
            self.columns_to_drop_.extend(zero_var_cols)
        else:
            print(
                "IdentifyAndDropLowVarNum: No columns identified as CONSTANT after temporary scaling."
            )

        # 2. Identify quasi-constant columns from the scaled data
        quasi_constant_candidates: pd.Series = scaled_variances[
            (scaled_variances > 0.0)
            & (scaled_variances < self.quasi_constant_threshold)
        ]
        quasi_var_cols: list[str] = quasi_constant_candidates.index.tolist()

        if quasi_var_cols:
            print(
                f"IdentifyAndDropLowVarNum: Number of Identified columns as QUASI-CONSTANT after temporary scaling (0 < scaled_var < {self.quasi_constant_threshold}): {len(quasi_var_cols)}"
            )
            self.columns_to_drop_.extend(quasi_var_cols)

        # Handle original numeric columns that were not scaled (e.g., all NaN or single unique value)
        # Their original variance determines if they are constant.
        unscaled_numeric_cols: list[str] = [
            col
            for col in self.numeric_cols_considered_
            if col not in valid_numeric_cols_for_scaling
        ]
        if unscaled_numeric_cols:
            original_unscaled_variances: pd.Series = X[unscaled_numeric_cols].var(
                ddof=0
            )
            # Columns that are all NaNs will have NaN variance.
            # Columns that have one unique value (e.g., all 7s, or [7, nan, 7]) will have 0 variance.
            zero_var_unscaled_cols: list[str] = original_unscaled_variances[
                original_unscaled_variances == 0.0
            ].index.tolist()
            if zero_var_unscaled_cols:
                print(
                    f"IdentifyAndDropLowVarNum: Identified originally constant UNSCALED numeric columns: {zero_var_unscaled_cols}"
                )
                for col in zero_var_unscaled_cols:
                    if col not in self.columns_to_drop_:
                        self.columns_to_drop_.append(col)

        # Ensure unique columns to drop
        self.columns_to_drop_ = list(set(self.columns_to_drop_))

        if not self.columns_to_drop_ and self.numeric_cols_considered_:
            print(
                f"IdentifyAndDropLowVarNum: No numeric columns met criteria for dropping (based on temporarily scaled variance) among {self.numeric_cols_considered_}."
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drops the identified zero/low-variance numeric columns from the
        original (unscaled) DataFrame.

        Args:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            pd.DataFrame: The DataFrame with identified columns removed.
                          The remaining columns are NOT scaled by this transformer.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        if not hasattr(self, "columns_to_drop_"):
            raise RuntimeError("Transform called before fit. Call fit first.")

        X_transformed: pd.DataFrame = X.copy()  # Work on a copy

        # Drop columns identified during fit from the original unscaled data
        cols_to_actually_drop_now: list[str] = [
            col for col in self.columns_to_drop_ if col in X_transformed.columns
        ]
        if cols_to_actually_drop_now:
            X_transformed = X_transformed.drop(
                columns=cols_to_actually_drop_now, errors="ignore"
            )
            print(
                f"IdentifyAndDropLowVarNum: Dropped columns from original data: {cols_to_actually_drop_now}"
            )
        elif self.columns_to_drop_:
            print(
                f"IdentifyAndDropLowVarNum: Columns {self.columns_to_drop_} were targeted for dropping, but none are present in the current DataFrame."
            )

        return X_transformed

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """
        Returns feature names after transformation.
        """
        if input_features is None:
            if (
                hasattr(self, "_input_features_during_fit")
                and self._input_features_during_fit
            ):
                input_features = self._input_features_during_fit
            else:
                raise ValueError(
                    "input_features must be provided if fit hasn't been called or was called on an empty DataFrame."
                )

        if not hasattr(self, "columns_to_drop_"):  # Fit not called
            return list(input_features)

        return [col for col in input_features if col not in self.columns_to_drop_]
