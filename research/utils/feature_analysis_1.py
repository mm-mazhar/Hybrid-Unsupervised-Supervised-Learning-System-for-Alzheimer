# -*- coding: utf-8 -*-
# """
# feature_analysis.py
# Created on Nov 07, 2024
# @ Author: Mazhar
# """

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs._figure import Figure


def visualize_bar_plot(
    df: pd.DataFrame,
    selected_numeric_feature,
    selected_method,
    selected_categorical_feature,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """
    Visualize the aggregation of a numeric feature across different categories using a bar plot.
    """
    df = df.copy()

    if selected_numeric_feature and selected_categorical_feature:
        # Clean the categorical feature to remove duplicates/inconsistencies
        df[selected_categorical_feature] = (
            df[selected_categorical_feature]
            .astype(str)  # Ensure it is string-type
            .str.strip()  # Remove leading/trailing spaces
            .str.lower()  # Convert to lowercase (optional, for consistency)
        )

        # Determine the aggregation based on the selected method
        if selected_method == "mean":
            aggregated_values: Any = (
                df.groupby(selected_categorical_feature)[selected_numeric_feature]
                .mean()
                .reset_index()
            )
        elif selected_method == "sum":
            aggregated_values = (
                df.groupby(selected_categorical_feature)[selected_numeric_feature]
                .sum()
                .reset_index()
            )
        elif selected_method == "count":
            aggregated_values = (
                df.groupby(selected_categorical_feature)[selected_numeric_feature]
                .count()
                .reset_index()
            )
        else:
            raise ValueError("Invalid method. Please use 'mean', 'sum', or 'count'")

        # Plot the bar plot
        fig: Figure = px.bar(
            aggregated_values,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Bar Plot | {selected_method.capitalize()} | {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,  # Optional: Remove this to avoid duplicates in the legend
            color_discrete_sequence=px.colors.qualitative.Set2,
        )

        # Update layout for background and text colors
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
        )

        # Display the Plotly chart in Streamlit
        # st.plotly_chart(fig)
        return fig


def visualize_comparison_box(
    df: pd.DataFrame,
    selected_numeric_feature: str,
    selected_categorical_feature: str,
    background_color: str = "white",
    text_color: str = "black",
    width: int = 1000,
    height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Box Plot", color="red")
    # st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    df = df.copy()

    # Plot the comparison
    if selected_numeric_feature and selected_categorical_feature:
        # st.subheader(
        #     f"Comparison of {selected_numeric_feature} by {selected_categorical_feature}"
        # )
        fig: Figure = px.box(
            df,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Box Plot | {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # Optional background color
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            width=width,
            height=height,
        )
        # st.plotly_chart(fig)
        return fig


def visualize_comparison_violin(
    df: pd.DataFrame,
    selected_numeric_feature: str,
    selected_categorical_feature: str,
    background_color: str = "white",
    text_color: str = "black",
    width: int = 1000,
    height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Violin Plot", color="red")
    # st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    df = df.copy()

    # Plot the comparison
    if selected_numeric_feature and selected_categorical_feature:
        # st.subheader(
        #     f"Comparison of {selected_numeric_feature} by {selected_categorical_feature}"
        # )
        fig: Figure = px.violin(
            df,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Violin Plot | {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            box=True,  # Adds a box plot inside the violin plot
            points="all",  # Shows all points
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            width=width,
            height=height,
        )
        # st.plotly_chart(fig)
        return fig


def visualize_comparison_strip(
    df: pd.DataFrame,
    selected_numeric_feature,
    selected_categorical_feature,
    background_color: str = "white",
    text_color: str = "black",
    width: int = 1000,
    height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Strip Plot", color="red")
    # st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    df = df.copy()

    # Plot the comparison
    if selected_numeric_feature and selected_categorical_feature:
        # st.subheader(
        #     f"Comparison of {selected_numeric_feature} by {selected_categorical_feature}"
        # )
        fig: Figure = px.strip(
            df,
            x=selected_categorical_feature,
            y=selected_numeric_feature,
            title=f"Strip Plot | {selected_numeric_feature} by {selected_categorical_feature}",
            color=selected_categorical_feature,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # Optional background and text color
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            xaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            yaxis=dict(
                title_font=dict(color=text_color), tickfont=dict(color=text_color)
            ),
            width=width,
            height=height,
        )
        # st.plotly_chart(fig)
        return fig
