"""
性能报告模块

整合计算、图表和AI分析，生成完整的策略性能报告
"""

from typing import Dict, List, Optional, Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apilot.utils import get_logger
from apilot.performance.calculator import calculate_statistics
from apilot.performance.plot import create_equity_curve, create_drawdown_curve, create_return_distribution
from apilot.performance.aisummary import generate_strategy_assessment

logger = get_logger("PerformanceReport")


class PerformanceReport:
    """性能报告类"""
    
    def __init__(self, df: Optional[pd.DataFrame] = None, 
                 trades: Optional[List] = None, 
                 capital: float = 0,
                 annual_days: int = 240):
        """
        初始化性能报告对象
        
        Args:
            df: 包含每日结果的DataFrame
            trades: 交易列表
            capital: 初始资金
            annual_days: 年交易日数
        """
        self.df = df
        self.trades = trades or []
        self.capital = capital
        self.annual_days = annual_days
        self.stats = None
    
    def generate(self) -> "PerformanceReport":
        """
        生成性能报告数据
        
        Returns:
            返回自身以支持链式调用
        """
        self.stats = calculate_statistics(
            df=self.df,
            trades=self.trades,
            capital=self.capital,
            annual_days=self.annual_days
        )
        return self
    
    def create_dashboard(self) -> go.Figure:
        """
        创建性能仪表板
        
        Returns:
            包含所有性能信息的图表对象
        """
        if not self.stats:
            self.generate()
        
        # 创建具有子图的Figure对象
        fig = make_subplots(
            rows=4, 
            cols=1, 
            row_heights=[0.2, 0.4, 0.2, 0.2],  # 20% 用于表格, 40% 用于资金曲线, 20% 用于回撤, 20% 用于收益分布
            vertical_spacing=0.02,
            specs=[
                [{"type": "table"}],    # 第1行 - 用于表格
                [{"type": "scatter"}],  # 第2行 - 用于资金曲线
                [{"type": "scatter"}],  # 第3行 - 用于回撤曲线
                [{"type": "histogram"}] # 第4行 - 用于收益分布
            ],
            subplot_titles=["", "Equity Curve", "Drawdown", "Daily Return Distribution"]
        )
        
        # 创建性能指标表格数据
        overview_headers = ["Period", "Initial", "Final", "Return", "Win Rate", "P/L Ratio"]
        overview_values = [
            f"{self.stats.get('start_date', '')} - {self.stats.get('end_date', '')}",
            f"${self.stats.get('initial_capital', 0):,.2f}",
            f"${self.stats.get('final_capital', 0):,.2f}",
            f"{self.stats.get('total_return', 0):.2f}%",
            f"{self.stats.get('win_rate', 0):.2f}%",
            f"{self.stats.get('profit_factor', 0):.2f}"
        ]
        
        metrics_headers = ["Annual Return", "Max Drawdown", "Sharpe", "Turnover"]
        metrics_values = [
            f"{self.stats.get('annual_return', 0):.2f}%",
            f"{self.stats.get('max_drawdown', 0):.2f}%",
            f"{self.stats.get('sharpe_ratio', 0):.2f}",
            f"{self.stats.get('turnover_ratio', 0):.2f}"
        ]
        
        # 获取AI分析
        assessment = generate_strategy_assessment(self.stats)
        ai_summary = assessment[0] if assessment else "AI分析功能准备中..."
        
        # 合并所有指标到一个表格中
        table_headers = ["Overview"] + overview_headers + ["<br>"] + ["Key Metrics"] + metrics_headers + ["<br>"] + ["AI Summary"]
        table_values = [""] + overview_values + [""] + [""] + metrics_values + [""] + [ai_summary]
        
        # 添加表格 (所有指标)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    line_color="white",
                    fill_color="rgb(0, 102, 204)",
                    align="left",
                    font=dict(color="white", size=14),
                    height=30
                ),
                cells=dict(
                    values=[table_headers, table_values],
                    line_color="white",
                    fill_color=["rgb(242, 242, 242)", "white"],
                    align="left",
                    font=dict(color="black", size=13),
                    height=25
                ),
                columnwidth=[300, 700]
            ),
            row=1, col=1
        )
        
        # 添加资金曲线 (第2行)
        if self.df is not None and not self.df.empty and "balance" in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index, 
                    y=self.df["balance"], 
                    mode="lines", 
                    name="Equity",
                    line=dict(color="rgb(0, 153, 76)", width=2)
                ),
                row=2, col=1
            )
            
            # 添加基准线
            fig.add_shape(
                type="line",
                x0=self.df.index[0],
                x1=self.df.index[-1],
                y0=self.capital,
                y1=self.capital,
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
                row=2, col=1
            )
        
        # 添加回撤曲线 (第3行)
        if self.df is not None and not self.df.empty and "ddpercent" in self.df.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.df.index, 
                    y=self.df["ddpercent"], 
                    fill="tozeroy",
                    mode="lines", 
                    name="Drawdown",
                    line=dict(color="rgb(220, 20, 60)", width=2),
                    fillcolor="rgba(220, 20, 60, 0.3)"
                ),
                row=3, col=1
            )
        
        # 添加收益分布 (第4行)
        if self.df is not None and not self.df.empty and "return" in self.df.columns:
            returns_pct = (self.df["return"] * 100).dropna()
            fig.add_trace(
                go.Histogram(
                    x=returns_pct,
                    nbinsx=30,
                    name="Daily Returns",
                    marker_color="rgb(83, 143, 255)",
                    opacity=0.7
                ),
                row=4, col=1
            )
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text="Performance Dashboard",
                font=dict(size=18),
            ),
            height=900,  # 整体高度
            width=1000,  # 整体宽度
            template="plotly_white",
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=False,
            hovermode="x unified"
        )
        
        # 更新各子图属性
        # 资金曲线 Y轴
        fig.update_yaxes(title_text="Capital", row=2, col=1, gridcolor="lightgray")
        # 回撤 Y轴
        fig.update_yaxes(title_text="Drawdown %", row=3, col=1, gridcolor="lightgray")
        # 分布 Y轴
        fig.update_yaxes(title_text="Frequency", row=4, col=1, gridcolor="lightgray")
        
        # 所有 X轴
        fig.update_xaxes(gridcolor="lightgray", row=2, col=1)
        fig.update_xaxes(gridcolor="lightgray", row=3, col=1)
        fig.update_xaxes(title_text="Daily Return (%)", gridcolor="lightgray", row=4, col=1)
        
        return fig
    
    def show(self):
        """显示性能报告"""
        fig = self.create_dashboard()
        fig.show()
        
    def print_summary(self):
        """打印性能报告文本摘要"""
        if not self.stats:
            self.generate()
            
        logger.info("=" * 50)
        logger.info("Performance Summary")
        logger.info("=" * 50)
        
        # Overview
        logger.info("\n== Overview ==")
        logger.info(
            f"Period: {self.stats.get('start_date', '')} - {self.stats.get('end_date', '')} "
            f"({self.stats.get('total_days', 0)} days)"
        )
        logger.info(f"Initial Capital: ${self.stats.get('initial_capital', 0):,.2f}")
        logger.info(f"Final Capital: ${self.stats.get('final_capital', 0):,.2f}")
        logger.info(f"Total Return: {self.stats.get('total_return', 0):.2f}%")
        logger.info(f"Win Rate: {self.stats.get('win_rate', 0):.2f}%")
        logger.info(f"Profit/Loss Ratio: {self.stats.get('profit_factor', 0):.2f}")
        
        # Key Metrics
        logger.info("\n== Key Metrics ==")
        logger.info(f"Annualized Return: {self.stats.get('annual_return', 0):.2f}%")
        logger.info(f"Max Drawdown: {self.stats.get('max_drawdown', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {self.stats.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Turnover Ratio: {self.stats.get('turnover_ratio', 0):.2f}")
        logger.info(f"Total Trades: {self.stats.get('total_trade_count', 0)}")
        
        # AI Summary
        assessment = generate_strategy_assessment(self.stats)
        if assessment:
            logger.info("\n== Strategy Assessment ==")
            for insight in assessment:
                if insight:  # Skip empty strings
                    logger.info(insight)
                    
        logger.info("=" * 50)


def create_performance_report(df=None, trades=None, capital=0, annual_days=240):
    """
    创建并显示性能报告
    
    Args:
        df: 包含每日结果的DataFrame
        trades: 交易列表
        capital: 初始资金
        annual_days: 年交易日数
    """
    report = PerformanceReport(df, trades, capital, annual_days)
    report.generate()
    report.show()
    return report