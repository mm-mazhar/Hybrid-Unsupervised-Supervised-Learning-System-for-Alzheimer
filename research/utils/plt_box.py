import plotly.graph_objects as go
import pandas as pd


def plot_box_plotly(
    df: pd.DataFrame,
    variable: str,
    width: int = 800,
    height: int = 400,
    show_outliers: bool = True,
    color: str = "lightgreen",
    box_line_color: str = "black",
    outlier_color: str = "red",
) -> None:
    """
    Plots a box plot of a numerical variable using Plotly.

    Args:
        df (pd.DataFrame): Input DataFrame.
        variable (str): Name of the numerical column to plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        show_outliers (bool): Whether to display outlier points.
        color (str): Fill color of the box.
        box_line_color (str): Outline color of the box and whiskers.
        outlier_color (str): Color of outlier points.
    """
    if variable not in df.columns:
        print(f"Column '{variable}' not found in DataFrame.")
        return

    if not pd.api.types.is_numeric_dtype(df[variable]):
        print(f"Column '{variable}' is not numeric.")
        return

    data = df[variable].dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            x=data,
            name=variable,
            boxpoints="outliers" if show_outliers else False,
            marker=dict(
                color=outlier_color if show_outliers else color,
                outliercolor=outlier_color,
                line=dict(outliercolor=outlier_color),
            ),
            line=dict(color=box_line_color),
            fillcolor=color,
            orientation="h",
        )
    )

    fig.update_layout(
        title=f"Box Plot of '{variable}'",
        xaxis_title=variable,
        yaxis=dict(showticklabels=False),
        width=width,
        height=height,
        template="plotly_white",
    )

    fig.show()
