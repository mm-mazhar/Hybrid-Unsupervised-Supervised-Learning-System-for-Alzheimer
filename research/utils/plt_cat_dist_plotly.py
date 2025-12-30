import pandas as pd
import plotly.graph_objects as go


def plot_categorical_distributions_plotly(
    df: pd.DataFrame,
    categorical_cols: list[str],
    width: int = 900,
    height: int = 450,
    colorscale: str = "Viridis",  # 'Plasma' or 'Cividis' or 'Turbo' or 'Viridis'
) -> None:
    """
    Plots horizontal bar charts for categorical variables using Plotly,
    sorted in descending order with colorful bars and value annotations.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (list[str]): List of categorical column names to plot.
        width (int): Base plot width.
        height (int): Base plot height.
        colorscale (str): Plotly colorscale name for colorful bars.
    """
    # print("*" * 50)
    print("--- Plotting Categorical Distributions (Plotly) ---")
    print(f"Columns to plot: {len(categorical_cols)}")

    # Validate columns
    valid_cols: list[str] = [col for col in categorical_cols if col in df.columns]
    invalid_cols = list(set(categorical_cols) - set(valid_cols))
    if invalid_cols:
        print(f"⚠️ Warning: These columns were not found in DataFrame: {invalid_cols}")
    if not valid_cols:
        print("❌ No valid columns to plot.")
        return

    # Preprocess categorical values
    df_cat: pd.DataFrame = df[valid_cols].copy()
    df_cat = df_cat.astype("object").fillna("Missing")
    df_cat = df_cat.astype("category")

    for col in valid_cols:
        counts: pd.Series[int] = df_cat[col].value_counts()
        categories: list[str] = counts.index.tolist()
        values: list[int] = counts.values.tolist()

        # Create color range using normalized values
        normalized_vals: list[float] = [
            (v - min(values)) / (max(values) - min(values) + 1e-5) for v in values
        ]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=categories,
                orientation="h",
                marker=dict(
                    color=normalized_vals,
                    colorscale=colorscale,
                    line=dict(color="black", width=0.5),
                ),
                text=values,
                textposition="outside",
                hovertemplate="%{y}: %{x}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f'Distribution of "{col}" (Descending)',
            xaxis_title="Count",
            yaxis_title=col,
            yaxis=dict(categoryorder="total ascending"),  # For descending bars
            width=width,
            height=max(height, 40 * len(categories)),
            template="plotly_white",
            margin=dict(t=50, b=50, l=140, r=50),
        )

        fig.show()
