"""
绘图模块

提供回测结果的图表可视化功能,专注于三种核心图表:
- 资金曲线 (Equity Curve)
- 回撤曲线 (Drawdown Curve)
- 收益分布 (Return Distribution)
"""

import plotly.graph_objects as go


# 图表元素生成函数
def get_equity_trace(df):
    """获取资金曲线图表元素"""
    return go.Scatter(
        x=df.index,
        y=df["balance"],
        mode="lines",
        name="Equity",
        line={"color": "rgb(161, 201, 14)", "width": 2},
    )


def get_drawdown_trace(df):
    """获取回撤曲线图表元素"""
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
    """获取收益分布图表元素"""
    # 转换为百分比以提高可读性
    returns_pct = (df["return"] * 100).dropna()

    return go.Histogram(
        x=returns_pct,
        nbinsx=30,
        name="Daily Returns",
        marker_color="rgb(255, 211, 109)",
        # opacity=0.7
    )


# 独立图表函数
def create_equity_curve(df):
    """创建资金曲线图"""
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
    """创建回撤曲线图"""
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
    """创建收益分布图"""
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
