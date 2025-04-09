"""
绘图模块

提供回测结果的图表可视化功能，专注于三种核心图表：
- 资金曲线 (Equity Curve)
- 回撤曲线 (Drawdown Curve)
- 收益分布 (Return Distribution)
"""

import pandas as pd
import plotly.graph_objects as go


def create_equity_curve(df: pd.DataFrame) -> go.Figure:
    """
    创建资金曲线图
    
    Args:
        df: 包含balance列的DataFrame
        
    Returns:
        plotly图表对象
    """
    if df is None or df.empty or "balance" not in df:
        return go.Figure()
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df["balance"], 
            mode="lines", 
            name="Equity",
            line=dict(color="rgb(0, 153, 76)", width=2)
        )
    )
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Capital",
        height=400,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"
    )
    
    return fig


def create_drawdown_curve(df: pd.DataFrame) -> go.Figure:
    """
    创建回撤曲线图
    
    Args:
        df: 包含ddpercent列的DataFrame
        
    Returns:
        plotly图表对象
    """
    if df is None or df.empty or "ddpercent" not in df:
        return go.Figure()
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df["ddpercent"], 
            fill="tozeroy",
            mode="lines", 
            name="Drawdown",
            line=dict(color="rgb(220, 20, 60)", width=2),
            fillcolor="rgba(220, 20, 60, 0.3)"
        )
    )
    
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        height=400,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"
    )
    
    return fig


def create_return_distribution(df: pd.DataFrame) -> go.Figure:
    """
    创建收益分布图
    
    Args:
        df: 包含return列的DataFrame
        
    Returns:
        plotly图表对象
    """
    if df is None or df.empty or "return" not in df:
        return go.Figure()
        
    # 转换为百分比以提高可读性
    returns_pct = (df["return"] * 100).dropna()
        
    fig = go.Figure()
    
    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=40,
            name="Daily Returns",
            marker_color="rgb(83, 143, 255)",
            opacity=0.7
        )
    )
    
    fig.update_layout(
        title="Daily Return Distribution",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=400,
        width=800,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"
    )
    
    return fig