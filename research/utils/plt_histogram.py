import pandas as pd
import plotly.graph_objects as go


def plot_numeric_distribution_plotly(
    df: pd.DataFrame,
    variable: str,
    bins: int = 20,
    width: int = 800,
    height: int = 500,
    color: str = "skyblue",
) -> None:
    """
    Plots the histogram distribution of a numerical variable using Plotly.

    Args:
        df (pd.DataFrame): Input DataFrame.
        variable (str): Name of the numerical column to plot.
        bins (int): Number of histogram bins.
        width (int): Plot width in pixels.
        height (int): Plot height in pixels.
        color (str): Bar color.
    """
    if variable not in df.columns:
        print(f"Column '{variable}' not found in DataFrame.")
        return

    if not pd.api.types.is_numeric_dtype(df[variable]):
        print(f"Column '{variable}' is not numeric.")
        return

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(x=df[variable], nbinsx=bins, marker_color=color, opacity=0.85)
    )

    fig.update_layout(
        title=f"Distribution of '{variable}'",
        xaxis_title=variable,
        yaxis_title="Frequency",
        bargap=0.05,
        width=width,
        height=height,
        template="plotly_white",
    )

    fig.show()
