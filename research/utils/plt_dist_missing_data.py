from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure


def plot_missing_value_distribution(
    df: pd.DataFrame,
    base_title: str = "Distribution of Missing Values per Column",
    show_plot: bool = True,
    width: int | None = 1000,
    height: int | None = 700,
) -> go.Figure | None:
    """
    Plots the distribution of missing values per column in a pandas DataFrame
    using Plotly Express, including the total count of columns with missing data
    in the title.

    Args:
        df: The input pandas DataFrame.
        base_title: The base title for the plot (default: 'Distribution of Missing Values per Column').
                    The count of columns with missing data will be appended to this.
        show_plot: If True (default), displays the plot. If False, returns the figure object.
        width: The width of the plot in pixels (optional).
        height: The height of the plot in pixels (optional).

    Returns:
        A plotly.graph_objects.Figure object if show_plot is False, otherwise None.
    """
    # 1. Count missing values per column
    missing_counts = df.isnull().sum()

    # 2. Filter out columns with no missing values
    missing_counts = missing_counts[missing_counts > 0]

    # 3. Get the count of columns with missing values
    num_cols_with_missing: int = len(missing_counts)

    # 4. Check if there are any missing values to plot
    if num_cols_with_missing == 0:
        print("No missing values found in the DataFrame.")
        return None if not show_plot else None

    # 5. Sort the counts (optional, but makes the plot easier to read)
    missing_counts: pd.Series = missing_counts.sort_values(ascending=False)

    # 6. Convert the Series to a DataFrame suitable for Plotly Express
    missing_counts_df: pd.DataFrame = missing_counts.reset_index()
    missing_counts_df.columns = ["Column_Name", "Missing_Count"]  # Rename columns

    # 7. Construct the final plot title with the count
    plot_title: str = (
        f"{base_title} ({num_cols_with_missing} columns with missing data)"
    )

    # 8. Create the Plotly bar chart
    fig: Any = px.bar(
        missing_counts_df,
        x="Missing_Count",
        y="Column_Name",
        orientation="h",
        title=plot_title,  # Use the constructed title
        labels={
            "Missing_Count": "Number of Missing Values",
            "Column_Name": "Column Name",
        },
        color="Missing_Count",
        color_continuous_scale=px.colors.sequential.Viridis,
        category_orders={"Column_Name": missing_counts_df["Column_Name"].tolist()},
    )

    # 9. Update layout including width and height
    fig.update_layout(
        xaxis_title="Number of Missing Values",
        yaxis_title="Column Name",
        width=width,
        height=height,
    )

    # 10. Show or return the plot
    if show_plot:
        fig.show()
        return None
    else:
        return fig


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_missing_value_distribution_sns(
    df: pd.DataFrame,
    base_title: str = "Distribution of Missing Values",
    top_n: int | None = None,
    show_plot: bool = True,
    figsize: tuple = (10, 8),
    color: str = "steelblue",
) -> Figure | None:
    """
    Plots the distribution of missing values per column using Seaborn.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.
        base_title (str): The base title for the plot.
        top_n (int | None): The "flag" to control display.
                            - If None (default), displays all columns with missing values.
                            - If an integer is provided (e.g., 25), it displays only
                              the top N columns with the most missing values.
        show_plot (bool): If True, displays the plot. If False, returns the figure object.
        figsize (tuple): The size of the plot in inches (width, height).
        color (str): The color for the bars.

    Returns:
        A matplotlib.figure.Figure object if show_plot is False, otherwise None.
    """
    # 1. Calculate, filter, and sort missing value counts
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    # 2. Check if there is anything to plot
    if missing_counts.empty:
        print("No missing values found in the DataFrame.")
        return None

    # 3. Apply the 'top_n' flag logic
    if top_n is not None and top_n < len(missing_counts):
        # If the flag is set, take the top N and update the title
        missing_counts = missing_counts.head(top_n)
        plot_title = f"Top {top_n} {base_title} per Column"
    else:
        # Default behavior: use all columns and update title accordingly
        num_cols_with_missing = len(missing_counts)
        plot_title = f"{base_title} per Column ({num_cols_with_missing} columns)"

    # 4. Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=missing_counts.values, y=missing_counts.index, color=color, ax=ax)

    # 5. Set titles and labels
    ax.set_title(plot_title, fontsize=16, pad=20)  # Add padding to title
    ax.set_xlabel("Number of Missing Values")
    ax.set_ylabel("Column Name")

    # 6. Finalize layout and show/return plot
    plt.tight_layout()

    if show_plot:
        plt.show()
        return None
    else:
        return fig
