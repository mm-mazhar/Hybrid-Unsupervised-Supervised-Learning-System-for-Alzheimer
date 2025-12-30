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
import streamlit as st
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
            st.error("Invalid method. Please use 'mean', 'sum', or 'count'.")
            return

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
        st.plotly_chart(fig)


def visualize_comparison_box(
    df: pd.DataFrame,
    selected_numeric_feature: str,
    selected_categorical_feature: str,
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Box Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

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
            # width=width,
            # height=height,
        )
        st.plotly_chart(fig)


def visualize_comparison_violin(
    df: pd.DataFrame,
    selected_numeric_feature: str,
    selected_categorical_feature: str,
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Violin Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

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
            # width=width,
            # height=height,
        )
        st.plotly_chart(fig)


def visualize_comparison_strip(
    df: pd.DataFrame,
    selected_numeric_feature,
    selected_categorical_feature,
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Visualize the comparison between selected numeric and categorical columns using Plotly."""
    # display_centered_title("Strip Plot", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

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
            # width=width,
            # height=height,
        )
        st.plotly_chart(fig)


def plot_stacked_bar_chart(
    df: pd.DataFrame,
    selected_features: list[str],
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Plot a stacked bar chart for selected categorical features with customizable colors."""
    # display_centered_title("Stacked Bar Chart", color="red")
    st.markdown("""---""")
    # # Ensure width and height are integers
    # width = int(width)
    # height = int(height)

    if len(selected_features) < 2:
        st.warning("Please select at least two categorical features.")
        return

    # Create a list to hold the traces
    traces: list[go.Bar] = []

    # Iterate over pairs of selected features
    for i in range(len(selected_features) - 1):
        feature1: str = selected_features[i]
        feature2: str = selected_features[i + 1]

        # Create a crosstab for the current pair of features
        crosstab: pd.DataFrame = pd.crosstab(df[feature1], df[feature2])

        # Add a trace for each category in feature2
        for category in crosstab.columns:
            traces.append(
                go.Bar(
                    x=crosstab.index,
                    y=crosstab[category],
                    name=f"{feature2}: {category}",
                )
            )

    # Create the figure
    fig = go.Figure(data=traces)

    # Update layout for stacked bars with custom colors
    fig.update_layout(
        barmode="stack",
        title="Stacked Bar Chart of Selected Categorical Features",
        xaxis_title="Categories",
        yaxis_title="Count",
        legend_title="Categories",
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        font=dict(color=text_color),
        title_font=dict(color=text_color),
        # width=width,
        # height=height,
    )

    # Display the chart
    st.plotly_chart(fig)


def plot_pie_chart(
    df: pd.DataFrame,
    selected_categorical_feature: str,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """Plot a pie chart for selected categorical features with customizable colors."""
    st.markdown("""---""")

    if selected_categorical_feature and selected_categorical_feature in df.columns:
        # Plot pie chart for the selected feature
        fig: Figure = px.pie(
            df,
            names=selected_categorical_feature,
            title=f"Pie Chart | {selected_categorical_feature}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Use a color sequence
        )
        fig.update_layout(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            title_font=dict(color=text_color),
            legend=dict(
                title=dict(
                    text=selected_categorical_feature, font=dict(color=text_color)
                ),
                font=dict(color=text_color),
            ),
        )
        st.plotly_chart(fig)
    else:
        st.error("Selected feature is not a valid column in the DataFrame.")


def plot_dist_chart(
    df: pd.DataFrame,
    selected_categorical_feature,
    background_color: str = "white",
    text_color: str = "black",
    # width: int = 1000,
    # height: int = 600,
) -> None:
    """Plot a distribution bar chart for selected categorical features with customizable colors."""
    st.markdown("""---""")

    if selected_categorical_feature and selected_categorical_feature in df.columns:
        # Plot distribution plot for the selected feature
        fig: Figure = px.histogram(
            df,
            x=selected_categorical_feature,
            color=selected_categorical_feature,  # Use the selected feature for coloring
            title=f"Distribution of {selected_categorical_feature}",
            color_discrete_sequence=px.colors.qualitative.Set2,  # Use a color sequence
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
            # legend=dict(
            #     title=dict(text=selected_categorical_feature, font=dict(color=text_color)),
            #     font=dict(color=text_color),
            # ),
        )
        st.plotly_chart(fig)
    else:
        st.error("Selected feature is not a valid column in the DataFrame.")


def scatter_plot(
    df: pd.DataFrame,
    selected_numeric_feature_x,
    selected_numeric_feature,
    selected_categorical_feature,
    background_color: str = "white",
    text_color: str = "black",
) -> None:
    """Visualize data with respect to categorical and numerical features."""
    st.markdown("""---""")

    # Ensure there is at least one categorical column
    if not selected_categorical_feature:
        st.error("No categorical columns available.")
        return

    # Scatter Plot
    if (
        selected_numeric_feature_x
        and selected_numeric_feature
        and selected_categorical_feature
    ):
        fig: Figure = px.scatter(
            df,
            x=selected_numeric_feature_x,
            y=selected_numeric_feature,
            color=selected_categorical_feature,
            title=f"Scatter Plot | {selected_numeric_feature_x} vs {selected_numeric_feature} by {selected_categorical_feature}",
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
        )
        st.plotly_chart(fig)


# Example usage
# scatter_plot(
#             data,
#             background_color=background_color,
#             text_color=text_color,
#         )

# # Usage
# categorical_features = ['Area', 'soil group', 'Land class', 'knit (surface)']
# plot_stacked_bar_chart(data, categorical_features)

# def scatter_plot(
#     df: pd.DataFrame,
#     key="scatter_plot",
#     background_color: str = "white",
#     text_color: str = "black",
# ) -> None:
#     """Visualize data with respect to 'Data Collection Year' and other features."""
#     st.markdown("""---""")

#     # Create two columns with different widths
#     col1, col2 = st.columns(
#         [3, 1]
#     )  # 3:1 ratio for larger left and smaller right column

#     with col2:
#         # Add vertical space to center the content
#         for _ in range(6):
#             st.write("")  # Add empty strings to create space

#         # Exclude 'Data Collection Year' from numerical features
#         numerical_features: list[str] = [
#             col
#             for col in df.select_dtypes(include=["number"]).columns
#             if col != "Data Collection Year"
#         ]
#         categorical_features: list[str] = df.select_dtypes(
#             include=["object", "category"]
#         ).columns.tolist()

#         # Ensure at least one categorical column is selected
#         if not categorical_features:
#             st.error("No categorical columns available.")
#             return

#         # Select time filter using a slider
#         min_year = int(df["Data Collection Year"].min())
#         max_year = int(df["Data Collection Year"].max())
#         selected_year: int = st.slider(
#             "Select Data Collection Year",
#             min_value=min_year,
#             max_value=max_year,
#             value=min_year,  # Default to the minimum year
#             step=1,
#             key=f"{key}_year",
#         )

#         # Select categorical and numerical features
#         selected_categorical_feature: str = st.selectbox(
#             "Select Categorical Feature",
#             categorical_features,
#             key=f"{key}_categorical",
#         )

#         selected_numerical_feature: str = st.selectbox(
#             "Select Numerical Feature",
#             numerical_features,
#             key=f"{key}_numerical",
#         )

#     with col1:
#         # Filter data by selected year
#         filtered_df: pd.DataFrame = df[df["Data Collection Year"] == selected_year]

#         # Plot with respect to 'Data Collection Year'
#         if selected_categorical_feature and selected_numerical_feature:
#             fig: Figure = px.scatter(
#                 filtered_df,
#                 x="Data Collection Year",
#                 y=selected_numerical_feature,
#                 color=selected_categorical_feature,
#                 title=f"{selected_numerical_feature} over Data Collection Year by {selected_categorical_feature}",
#                 color_discrete_sequence=px.colors.qualitative.Set2,
#             )
#             fig.update_layout(
#                 plot_bgcolor=background_color,
#                 paper_bgcolor=background_color,
#                 title_font=dict(color=text_color),
#                 xaxis=dict(
#                     title_font=dict(color=text_color), tickfont=dict(color=text_color)
#                 ),
#                 yaxis=dict(
#                     title_font=dict(color=text_color), tickfont=dict(color=text_color)
#                 ),
#                 # legend=dict(
#                 #     title=dict(
#                 #         text=selected_categorical_feature, font=dict(color=text_color)
#                 #     ),
#                 #     font=dict(color=text_color),
#                 # ),
#             )
#             st.plotly_chart(fig)
