import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram_and_boxplot(
    df: pd.DataFrame,
    variable: str,
    bins: int = 20,
    hist_color: str = "skyblue",
    box_color: str = "lightgreen",
    outlier_color: str = "red",
    width: int = 1000,
    height: int = 600,
) -> None:
    """
    Plots a histogram and box plot side-by-side using Plotly subplots (without KDE).

    Args:
        df (pd.DataFrame): DataFrame containing the variable.
        variable (str): Column name of the numeric variable to plot.
        bins (int): Number of bins for histogram.
        hist_color (str): Histogram bar color.
        box_color (str): Box fill color.
        outlier_color (str): Color for outlier points in boxplot.
        width (int): Plot width in pixels.
        height (int): Plot height in pixels.
    """
    if variable not in df.columns:
        print(f"Column '{variable}' not found in DataFrame.")
        return

    if not pd.api.types.is_numeric_dtype(df[variable]):
        print(f"Column '{variable}' is not numeric.")
        return

    data = df[variable].dropna()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(f"Histogram of '{variable}'", f"Box Plot of '{variable}'"),
        column_widths=[0.7, 0.3],
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bins,
            name="Histogram",
            marker_color=hist_color,
            opacity=0.8,
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Box Plot
    fig.add_trace(
        go.Box(
            x=data,
            name="",
            boxpoints="outliers",
            orientation="h",
            fillcolor=box_color,
            line=dict(color="black"),
            marker=dict(
                color=outlier_color,
                outliercolor=outlier_color,
                line=dict(outliercolor=outlier_color),
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        width=width,
        height=height,
        title_text=f"Distribution Summary for '{variable}'",
        template="plotly_white",
    )

    fig.show()


def plot_histogram_and_boxplot_sns(
    df: pd.DataFrame,
    variable: str,
    bins: int = 20,
    hist_color: str = "skyblue",
    box_color: str = "lightgreen",
    outlier_color: str = "red",
    figsize: tuple = (12, 6),
) -> None:
    """
    Plots a histogram and box plot side-by-side using Seaborn and Matplotlib.

    Args:
        df (pd.DataFrame): DataFrame containing the variable.
        variable (str): Column name of the numeric variable to plot.
        bins (int): Number of bins for the histogram.
        hist_color (str): Color for the histogram bars.
        box_color (str): Fill color for the box plot.
        outlier_color (str): Color for outlier points in the box plot.
        figsize (tuple): Figure size in inches (width, height).
    """
    # --- Input Validation ---
    if variable not in df.columns:
        print(f"Column '{variable}' not found in DataFrame.")
        return

    if not pd.api.types.is_numeric_dtype(df[variable]):
        print(f"Column '{variable}' is not numeric.")
        return

    # --- Plotting Setup ---
    # Create a figure with two subplots, specifying their relative widths
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [0.7, 0.3]}
    )
    fig.suptitle(f"Distribution Summary for '{variable}'", fontsize=16)

    # --- Histogram ---
    sns.histplot(
        data=df,
        x=variable,
        bins=bins,
        color=hist_color,
        ax=axes[0],  # Specify the first subplot
    )
    axes[0].set_title(f"Histogram of '{variable}'")
    axes[0].set_xlabel(variable)
    axes[0].set_ylabel("Frequency")

    # --- Box Plot ---
    sns.boxplot(
        data=df,
        x=variable,  # Use 'x' for a horizontal box plot
        color=box_color,
        ax=axes[1],  # Specify the second subplot
        flierprops=dict(
            markerfacecolor=outlier_color, marker="o"
        ),  # Style the outliers
    )
    axes[1].set_title(f"Box Plot of '{variable}'")
    axes[1].set_xlabel(variable)
    axes[1].set_ylabel("")  # Hide y-axis label for the box plot for a cleaner look
    axes[1].set_yticks([])  # Hide y-axis ticks

    # --- Final Touches ---
    # Adjust layout to prevent titles from overlapping and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for the suptitle
    plt.show()
