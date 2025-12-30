import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_box_grid(
    df: pd.DataFrame,
    numeric_cols: list[str],
    n_cols: int = 3,
    figsize_per_plot: tuple = (5, 4),
    box_color: str = "lightblue",
    outlier_color: str = "red",
    title_fontsize: int = 12,
) -> None:
    """
    Plots boxplots for numeric columns with outliers in red using Seaborn in grid layout.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numeric_cols (list[str]): List of numeric column names.
        n_cols (int): Number of plots per row.
        figsize_per_plot (tuple): Size of each subplot.
        box_color (str): Fill color of the box.
        outlier_color (str): Color for outlier points.
        title_fontsize (int): Font size for subplot titles.
    """
    valid_cols = [col for col in numeric_cols if col in df.columns]
    n_plots = len(valid_cols)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * figsize_per_plot[0], n_rows * figsize_per_plot[1]),
    )
    axes = axes.flatten()

    print(f"\nPlotting boxplots with outliers...Number of plots: {n_plots}\n")

    for i, col in enumerate(valid_cols):
        ax = axes[i]
        sns.boxplot(
            x=df[col],
            ax=ax,
            color=box_color,
            fliersize=5,
            flierprops=dict(
                marker="o", markerfacecolor=outlier_color, markeredgecolor=outlier_color
            ),
        )

        # Mark Q1, Median (Q2), Q3
        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)
        ax.axvline(q1, color="black", linestyle="--", linewidth=1)
        ax.axvline(q2, color="black", linestyle="-", linewidth=1.5, label="Median")
        ax.axvline(q3, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{col}", fontsize=title_fontsize)
        ax.set_xlabel("")  # Hide x-axis label text

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # Remove unused subplot axes

    plt.tight_layout()
    plt.show()
