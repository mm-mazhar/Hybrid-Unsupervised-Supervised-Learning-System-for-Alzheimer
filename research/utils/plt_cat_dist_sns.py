import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_categorical_distributions_seaborn(
    df: pd.DataFrame,
    categorical_cols: list[str],
    palette: str = "viridis",
    figsize: tuple = (10, 5),
) -> None:
    """
    Plots horizontal bar charts for each categorical column in the given list,
    sorted by descending frequency, with value labels and missing value handling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list[str]): List of column names to plot.
        palette (str): Seaborn palette name for bar coloring.
        figsize (tuple): Size of each plot (width, height).
    """
    # print("*" * 50)
    print(f"--- Plotting Categorical Distributions ---")
    print(f"Input columns: {categorical_cols}, Number of Cols: {len(categorical_cols)}")

    # Filter and validate columns
    valid_cols = [col for col in categorical_cols if col in df.columns]
    invalid_cols = list(set(categorical_cols) - set(valid_cols))
    if invalid_cols:
        print(f"⚠️ Warning: These columns are missing from DataFrame: {invalid_cols}")
    if not valid_cols:
        print("❌ No valid columns found. Exiting.")
        return

    # Extract relevant data and handle missing values
    df_cat = df[valid_cols].copy()
    df_cat = df_cat.astype("object").fillna("Missing").astype("category")

    # Plot each column
    for feature in valid_cols:
        plt.figure(figsize=figsize)
        order = df_cat[feature].value_counts().index  # Descending order
        ax = sns.countplot(
            y=feature,
            data=df_cat,
            hue=feature,
            order=order,
            palette=palette,
            legend=False,
        )

        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", label_type="edge", padding=3)

        plt.title(f'Distribution of "{feature}"', fontsize=13)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
