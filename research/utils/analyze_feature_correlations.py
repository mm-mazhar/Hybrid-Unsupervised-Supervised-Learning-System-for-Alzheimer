import numbers
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def analyze_feature_correlations(
    df: pd.DataFrame, thresholds: Dict[str, Tuple[float, float]] = {}
) -> Dict[str, list[Tuple[str, str, float]]]:
    """
    Identifies pairs of features based on their correlation strength,
    categorized into levels (e.g., low, medium, high).

    Args:
        - df (pd.DataFrame): A DataFrame
        - thresholds (Dict[str, Tuple[float, float]], optional):
            A dictionary defining the correlation levels and their absolute
            correlation value ranges (inclusive lower bound, exclusive upper bound).
            Keys are level names (e.g., "low", "medium", "high").
            Values are tuples (min_abs_corr, max_abs_corr).
            Example:
            {
                "low": (0.3, 0.5),      # |corr| >= 0.3 and |corr| < 0.5
                "medium": (0.5, 0.7),   # |corr| >= 0.5 and |corr| < 0.7
                "high": (0.7, 1.01)     # |corr| >= 0.7 (1.01 to include 1.0)
            }
            If None, a default set of thresholds will be used:
            {
                "low_positive": (0.3, 0.5), "low_negative": (-0.5, -0.3),
                "medium_positive": (0.5, 0.7), "medium_negative": (-0.7, -0.5),
                "high_positive": (0.7, 1.01), "high_negative": (-1.01, -0.7) # 1.01 to include 1.0
            }
            Note: For absolute correlation, ranges should be positive.
                    For directional correlation, define positive and negative ranges.

    Returns:
        Dict[str, List[Tuple[str, str, float]]]:
            A dictionary where keys are the correlation level names (from thresholds)
            and values are lists of tuples. Each tuple contains:
            (feature1_name, feature2_name, correlation_value).
    """

    df = df.copy()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    # # Calculate the correlation matrix
    correlation_matrix: pd.DataFrame = df.corr(numeric_only=True)

    if not isinstance(correlation_matrix, pd.DataFrame):
        raise TypeError("correlation_matrix must be a pandas DataFrame.")
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix must be a square matrix.")
    if not all(correlation_matrix.columns == correlation_matrix.index):
        raise ValueError("correlation_matrix columns and index must be identical.")

    if thresholds is None:
        # Default thresholds (directional)
        thresholds = {
            "very_low_positive": (0.1, 0.3),
            "very_low_negative": (-0.3, -0.1),
            "low_positive": (0.3, 0.5),
            "low_negative": (-0.5, -0.3),
            "medium_positive": (0.5, 0.7),
            "medium_negative": (-0.7, -0.5),
            "high_positive": (0.7, 1.000001),
            "high_negative": (-1.000001, -0.7),  # Use 1.000001 to include 1.0
            "perfect_positive": (0.999999, 1.000001),
            "perfect_negative": (-1.000001, -0.999999),  # For near perfect
        }
        print("Using default directional thresholds.")

    # Alternative: Default thresholds based on absolute values
    # if thresholds is None:
    #     thresholds = {
    #         "low_abs": (0.3, 0.5),
    #         "medium_abs": (0.5, 0.7),
    #         "high_abs": (0.7, 1.01) # 1.01 to include 1.0
    #     }
    #     print("Using default absolute value thresholds.")

    # Initialize dictionary to store results
    categorized_correlations: Dict[str, list[Tuple[str, str, float]]] = {
        level: [] for level in thresholds.keys()
    }

    columns: pd.Index[str] = correlation_matrix.columns
    n_cols: int = len(columns)

    for i in range(n_cols):
        for j in range(i):
            col1: str = columns[i]
            col2: str = columns[j]
            correlation_value_raw = correlation_matrix.iloc[i, j]
            # Only process int, float, or complex values (not e.g. datetime, timedelta)
            if isinstance(correlation_value_raw, (int, float, complex)):
                if isinstance(correlation_value_raw, complex):
                    correlation_value = float(correlation_value_raw.real)
                else:
                    correlation_value = float(correlation_value_raw)
                abs_correlation_value: float = abs(correlation_value)
            else:
                print(
                    f"Warning: Non-numeric correlation value for ({col1}, {col2}): {correlation_value_raw}"
                )
                continue

            for level, (min_corr, max_corr) in thresholds.items():
                is_directional_range: bool = min_corr < 0 or max_corr <= 0
                if is_directional_range:
                    if min_corr <= correlation_value < max_corr:
                        categorized_correlations[level].append(
                            (col1, col2, correlation_value)
                        )
                        break
                else:
                    if min_corr <= abs_correlation_value < max_corr:
                        categorized_correlations[level].append(
                            (col1, col2, correlation_value)
                        )
                        break
    return categorized_correlations


# # Example Usage
# df_sample = pd.DataFrame(data)
# correlation_matrix = df_sample.corr(numeric_only = True)

# # Scenario 1: Using default directional thresholds
# print("\n--- Scenario 1: Default Directional Thresholds ---")
# correlation_results_default = analyze_feature_correlations(correlation_matrix)
# for level, pairs in correlation_results_default.items():
#     if pairs:
#         print(f"\n{level.replace('_', ' ').title()} Correlations:")
#         for col1, col2, corr_val in pairs:
#             print(f"- `{col1}` and `{col2}`: {corr_val:.2f}")
#     # else:
#     #     print(f"\nNo correlations found for level: {level}")

# # Scenario 2: Custom thresholds (absolute values)
# custom_abs_thresholds = {
#     "low_magnitude": (0.3, 0.5),  # |corr| in [0.3, 0.5)
#     "medium_magnitude": (0.5, 0.8),  # |corr| in [0.5, 0.8)
#     "high_magnitude": (0.8, 1.000001),  # |corr| in [0.8, 1.0]
# }
# print("\n--- Scenario 2: Custom Absolute Value Thresholds ---")
# correlation_results_custom_abs = analyze_feature_correlations(
#     correlation_matrix, thresholds=custom_abs_thresholds
# )
# for level, pairs in correlation_results_custom_abs.items():
#     if pairs:
#         print(f"\n{level.replace('_', ' ').title()} Correlations:")
#         for col1, col2, corr_val in pairs:
#             print(f"- `{col1}` and `{col2}`: {corr_val:.2f} (abs: {abs(corr_val):.2f})")

# # Scenario 3: Custom thresholds (directional)
# custom_dir_thresholds = {
#     "moderate_positive": (0.4, 0.7),
#     "strong_positive": (0.7, 1.000001),
#     "moderate_negative": (-0.7, -0.4),
#     "strong_negative": (-1.000001, -0.7),
# }
# print("\n--- Scenario 3: Custom Directional Thresholds ---")
# correlation_results_custom_dir = analyze_feature_correlations(
#     correlation_matrix, thresholds=custom_dir_thresholds
# )
# for level, pairs in correlation_results_custom_dir.items():
#     if pairs:
#         print(f"\n{level.replace('_', ' ').title()} Correlations:")
#         for col1, col2, corr_val in pairs:
#             print(f"- `{col1}` and `{col2}`: {corr_val:.2f}")

# # Scenario 4: No correlations meeting criteria (strict thresholds)
# strict_thresholds = {"extremely_high_abs": (0.9999999, 1.000001)}
# # To make this scenario work with the sample, let's create a near-perfect one
# df_strict_sample = df_sample.copy()
# df_strict_sample["B_almost_A"] = df_strict_sample["A"] * 1.00000001
# corr_matrix_strict = df_strict_sample.corr()

# print("\n--- Scenario 4: Strict Thresholds (Example with near perfect) ---")
# correlation_results_strict = analyze_feature_correlations(
#     corr_matrix_strict, thresholds=strict_thresholds
# )
# for level, pairs in correlation_results_strict.items():
#     if pairs:
#         print(f"\n{level.replace('_', ' ').title()} Correlations:")
#         for col1, col2, corr_val in pairs:
#             print(f"- `{col1}` and `{col2}`: {corr_val:.2f}")
#     else:
#         print(f"\nNo correlations found for level: {level}")
