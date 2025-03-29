"""
绘图模块

提供回测结果和交易数据的可视化功能
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_backtest_results(df: pd.DataFrame) -> None:
    """
    绘制回测结果的综合图表

    包括:
    - 资金曲线
    - 回撤百分比
    - 每日盈亏
    - 盈亏分布

    参数:
        df: 回测结果数据框,需包含 balance、ddpercent、net_pnl 列
    """
    if df is None or df.empty:
        return

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=["Balance", "Drawdown", "Pnl", "Pnl Distribution"],
        vertical_spacing=0.06,
    )

    # 添加资金曲线
    fig.add_trace(
        go.Scatter(x=df.index, y=df["balance"], mode="lines", name="Balance"),
        row=1,
        col=1,
    )

    # 添加回撤图
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ddpercent"],
            fillcolor="red",
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
        ),
        row=2,
        col=1,
    )

    # 添加盈亏柱状图
    fig.add_trace(go.Bar(y=df["net_pnl"], name="Pnl"), row=3, col=1)

    # 添加盈亏分布直方图
    fig.add_trace(go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days"), row=4, col=1)

    fig.update_layout(height=1000, width=1000)
    fig.show()


def plot_equity_curve(df: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    """
    绘制资金曲线

    参数:
        df: 回测结果数据框,需包含 balance 列
        title: 图表标题

    返回:
        plotly图表对象
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df["balance"], mode="lines", name="Balance"))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Balance",
        height=500,
        width=800,
    )

    return fig


def plot_drawdown(df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """
    绘制回撤图

    参数:
        df: 回测结果数据框,需包含 ddpercent 列
        title: 图表标题

    返回:
        plotly图表对象
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["ddpercent"],
            fillcolor="red",
            fill="tozeroy",
            mode="lines",
            name="Drawdown %",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        height=500,
        width=800,
    )

    return fig


def plot_pnl(df: pd.DataFrame, title: str = "Daily PnL") -> go.Figure:
    """
    绘制每日盈亏柱状图

    参数:
        df: 回测结果数据框,需包含 net_pnl 列
        title: 图表标题

    返回:
        plotly图表对象
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(x=df.index, y=df["net_pnl"], name="PnL", marker_color="lightblue")
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Profit/Loss",
        height=500,
        width=800,
    )

    return fig


def plot_pnl_distribution(
    df: pd.DataFrame, title: str = "PnL Distribution"
) -> go.Figure:
    """
    绘制盈亏分布直方图

    参数:
        df: 回测结果数据框,需包含 net_pnl 列
        title: 图表标题

    返回:
        plotly图表对象
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["net_pnl"],
            nbinsx=50,
            marker_color="lightgreen",
            name="PnL Distribution",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Profit/Loss",
        yaxis_title="Frequency",
        height=500,
        width=800,
    )

    return fig
