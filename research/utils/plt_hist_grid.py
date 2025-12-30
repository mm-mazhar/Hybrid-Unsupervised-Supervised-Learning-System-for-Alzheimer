import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_hist_grid(
    df: pd.DataFrame,
    numeric_cols: list[str],
    bins: int = 30,
    n_cols: int = 3,
    figsize_per_plot: tuple = (5, 4),
    color: str = "skyblue",
    title_fontsize: int = 12,
    xlabel_fontsize: int = 10,
    ylabel_fontsize: int = 10,
    tick_fontsize: int = 9,
) -> None:
    """
    Plots histograms for multiple numeric columns using seaborn.histplot in a grid layout.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (list[str]): List of numerical column names.
        bins (int): Number of histogram bins.
        n_cols (int): Number of subplots per row.
        figsize_per_plot (tuple): Size of each subplot (width, height).
        color (str): Bar color.
        title_fontsize (int): Font size for subplot titles.
        xlabel_fontsize (int): Font size for x-axis labels.
        ylabel_fontsize (int): Font size for y-axis labels.
        tick_fontsize (int): Font size for tick labels.
    """

    sns.set_style(style="whitegrid")

    n_plots: int = len(numeric_cols)
    n_rows: int = math.ceil(n_plots / n_cols)
    total_width: Any = figsize_per_plot[0] * n_cols
    total_height: Any = figsize_per_plot[1] * n_rows

    print(f"\nPlotting histograms...Total plots: {n_plots}\n")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    axes: Any = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, bins=bins, kde=False, color=color, ax=axes[i])
        axes[i].set_title(f"Distribution of {col}", fontsize=title_fontsize)
        axes[i].set_xlabel(col, fontsize=xlabel_fontsize)
        axes[i].set_ylabel("Frequency", fontsize=ylabel_fontsize)
        axes[i].tick_params(axis="x", rotation=45, labelsize=tick_fontsize)
        axes[i].tick_params(axis="y", labelsize=tick_fontsize)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
