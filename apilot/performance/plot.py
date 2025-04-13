"""
Plotting Module

Provides chart visualization for backtest results, focusing on three core charts:
- Equity Curve
- Drawdown Curve
- Return Distribution
"""

import plotly.graph_objects as go


# Chart element generation functions
def get_equity_trace(df):
    """Get equity curve chart element"""
    return go.Scatter(
        x=df.index,
        y=df["balance"],
        mode="lines",
        name="Equity",
        line={"color": "rgb(161, 201, 14)", "width": 2},
    )


def get_drawdown_trace(df):
    """Get drawdown curve chart element"""
    return go.Scatter(
        x=df.index,
        y=df["ddpercent"],
        fill="tozeroy",
        mode="lines",
        name="Drawdown",
        line={"color": "rgb(216, 67, 67)", "width": 2},
        fillcolor="rgba(220, 20, 60, 0.3)",
    )


def get_return_dist_trace(df):
    """Get return distribution chart element"""
    # Convert to percentage for better readability
    returns_pct = (df["return"] * 100).dropna()

    return go.Histogram(
        x=returns_pct,
        nbinsx=30,
        name="Daily Returns",
        marker_color="rgb(255, 211, 109)",
        # opacity=0.7
    )


# Independent chart functions
def create_equity_curve(df):
    """Create equity curve chart"""
    if df is None or df.empty or "balance" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(get_equity_trace(df))

    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Capital",
        height=400,
        width=800,
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
        template="plotly_dark",
    )

    return fig


def create_drawdown_curve(df):
    """Create drawdown curve chart"""
    if df is None or df.empty or "ddpercent" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(get_drawdown_trace(df))

    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        height=400,
        width=800,
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
        template="plotly_dark",
    )

    return fig


def create_return_distribution(df):
    """Create return distribution chart"""
    if df is None or df.empty or "return" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(get_return_dist_trace(df))

    fig.update_layout(
        title="Daily Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400,
        width=800,
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
        template="plotly_dark",
    )

    return fig
