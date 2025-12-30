from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from numpy._typing._array_like import NDArray


def plot_correlation_heatmap_plotly(
    df: pd.DataFrame,
    background_color: str = "white",
    text_color: str = "black",
    color_scale: str = "Viridis",  # Plotly named colorscale (e.g., 'Viridis', 'Plasma', 'Blues', 'Reds')
    title: str = "Heatmap",
    decimal_places_annot: int = 2,  # For annotation rounding
    width: int = 800,
    height: int = 700,
) -> None:
    """
    Calculates and plots a heatmap of the correlation of the correlation matrix
    (i.e., df.corr().corr()) using Plotly.

    Args:
        df (pd.DataFrame): The input DataFrame.
        background_color (str): Background color of the plot.
        text_color (str): Color for text elements (title, labels, annotations).
        color_scale (str): Plotly colorscale name for the heatmap.
        title (str): Title of the plot.
        width (int): Width of the plot.
        height (int): Height of the plot.
        decimal_places_annot (int): Number of decimal places for annotations.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame.")

    df = df.copy()

    numeric_df: pd.DataFrame = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        print(
            "Warning: DataFrame has less than 2 numeric columns. Cannot compute a meaningful correlation matrix."
        )
        fig = go.Figure()
        fig.update_layout(
            title="Not enough numeric data for correlation heatmap",
            width=width,
            height=height,
            paper_bgcolor=background_color,
            font_color=text_color,
        )
        fig.show()
        return

    try:
        # Step 1: Calculate the first correlation matrix (correlations between original features)
        correlation_matrix: pd.DataFrame = numeric_df.corr()

    except Exception as e:
        print(f"Error calculating correlation matrices: {e}")
        fig: Any = go.Figure()
        fig.update_layout(
            title=f"Error: {e}",
            width=width,
            height=height,
            paper_bgcolor=background_color,
            font_color=text_color,
        )
        fig.show()
        return

    # Prepare annotations (rounded text values)
    annotations_text: np.ndarray = np.around(
        correlation_matrix.values, decimals=decimal_places_annot
    )

    # Step 3: Create the Plotly heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale=color_scale,
            text=annotations_text,
            texttemplate="%{text}",  # Display the rounded text on cells
            hoverongaps=False,
            zmin=-1,  # Ensure colorscale covers full range for correlations
            zmax=1,
        )
    )

    # Step 4: Update layout with arguments
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color=text_color)),  # Center title
        width=width,
        height=height,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,  # Background of the plotting area
        font=dict(color=text_color),  # Default font color for axes, legend etc.
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",  # Common for matrix displays (origin top-left)
        xaxis=dict(
            tickangle=-45, tickfont=dict(color=text_color)
        ),  # Angle x-axis labels
        yaxis=dict(tickfont=dict(color=text_color)),
        # Adjust margins to prevent labels/title from being cut off
        margin=dict(
            l=100, r=50, t=100, b=120 if correlation_matrix.shape[1] > 5 else 80
        ),
    )

    fig.show()
