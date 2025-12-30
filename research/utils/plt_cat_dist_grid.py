import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_categorical_distributions_grid(
    df: pd.DataFrame,
    categorical_cols: list[str],
    palette: str = "viridis",
    n_cols: int = 3,
    figsize_per_plot: tuple = (5, 4),
    title_fontsize: int = 12,
    axis_labelsize: int = 10,
    tick_fontsize: int = 9,
) -> None:
    """
    Plots multiple categorical variable distributions in a grid layout using Seaborn.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list[str]): List of categorical column names to plot.
        palette (str): Seaborn palette for coloring bars.
        n_cols (int): Number of plots per row.
        figsize_per_plot (tuple): Size of each subplot (width, height).
        title_fontsize (int): Font size for subplot titles.
        axis_labelsize (int): Font size for axis labels.
        tick_fontsize (int): Font size for tick labels.
    """

    valid_cols: list[str] = [col for col in categorical_cols if col in df.columns]
    if not valid_cols:
        print("❌ No valid categorical columns found.")
        return

    print("--- Plotting Categorical Distributions (Grid Layout) ---")
    print(f"Number of valid columns: {len(valid_cols)}")

    df_cat: pd.DataFrame = df[valid_cols].copy()
    df_cat = df_cat.astype("object").fillna("Missing").astype("category")

    n_plots: int = len(valid_cols)
    n_rows: int = math.ceil(n_plots / n_cols)

    total_width: float = figsize_per_plot[0] * n_cols
    total_height: float = figsize_per_plot[1] * n_rows

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(total_width, total_height),
        constrained_layout=True,
    )
    axes: Any = axes.flatten()

    for i, col in enumerate(valid_cols):
        ax: Any = axes[i]
        order = df_cat[col].value_counts().index
        sns.countplot(
            y=col,
            data=df_cat,
            hue=col,
            order=order,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(col, fontsize=title_fontsize)
        ax.set_xlabel("Count", fontsize=axis_labelsize)
        ax.set_ylabel("", fontsize=axis_labelsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%d",
                label_type="edge",
                padding=2,
                fontsize=tick_fontsize,
            )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused axes

    plt.show()
