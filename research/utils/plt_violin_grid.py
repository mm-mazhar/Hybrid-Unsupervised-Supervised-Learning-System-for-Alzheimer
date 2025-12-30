import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_violin_grid(
    df: pd.DataFrame,
    numeric_cols: list[str],
    n_cols: int = 3,
    figsize_per_plot: tuple = (5, 4),
    palette: str = "muted",
    title_fontsize: int = 12,
    axis_labelsize: int = 10,
    tick_fontsize: int = 9,
    show_outliers: bool = True,
) -> None:
    """
    Plots violin plots for numeric columns in a grid layout with quantile lines and optional outliers.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (list[str]): List of numeric column names.
        n_cols (int): Number of plots per row.
        figsize_per_plot (tuple): Size of each subplot (width, height).
        palette (str): Seaborn color palette name.
        title_fontsize (int): Font size for titles.
        axis_labelsize (int): Font size for axis labels.
        tick_fontsize (int): Font size for tick labels.
        show_outliers (bool): Whether to show raw data points as outliers.
    """
    valid_cols: list[str] = [col for col in numeric_cols if col in df.columns]
    n_plots: int = len(valid_cols)
    n_rows: int = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1]),
    )
    axes = axes.flatten()

    print(f"\nPlotting Violin Plots...Number of Plots: {n_plots}\n")

    for i, col in enumerate(valid_cols):
        ax: Any = axes[i]

        sns.violinplot(
            x=col,
            data=df,
            # hue=col,
            # palette=palette,
            inner=None,  # Remove default box/points so we control what’s shown
            ax=ax,
            legend=False,
            # dodge=False,
        )

        # Overlay quantile lines
        q1: float = df[col].quantile(0.25)
        q2: float = df[col].quantile(0.50)
        q3: float = df[col].quantile(0.75)

        ax.axvline(q1, linestyle="--", color="black", linewidth=1)
        ax.axvline(q2, linestyle="-", color="black", linewidth=1.5, label="Median")
        ax.axvline(q3, linestyle="--", color="black", linewidth=1)

        # Overlay individual points as outliers if enabled
        if show_outliers:
            sns.stripplot(x=df[col], ax=ax, color="red", size=2, jitter=True, alpha=0.5)

        ax.set_title(col, fontsize=title_fontsize)
        ax.set_xlabel("", fontsize=axis_labelsize)
        ax.set_ylabel("", fontsize=axis_labelsize)
        ax.tick_params(axis="x", labelsize=tick_fontsize)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused axes

    plt.tight_layout()
    plt.show()
