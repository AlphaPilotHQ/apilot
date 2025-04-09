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
            
        # 创建多子图仪表板
        fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=[
                "Equity Curve", "Drawdown",
                "Return Distribution", "Overview",
                "Key Metrics", "Strategy Assessment"
            ],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}],
                [{"type": "table"}, {"type": "table"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # 添加图表
        
        # 1. 资金曲线
        if self.df is not None and not self.df.empty:
            equity_fig = create_equity_curve(self.df)
            for trace in equity_fig.data:
                fig.add_trace(trace, row=1, col=1)
                
        # 2. 回撤曲线
        if self.df is not None and not self.df.empty:
            drawdown_fig = create_drawdown_curve(self.df)
            for trace in drawdown_fig.data:
                fig.add_trace(trace, row=1, col=2)
                
        # 3. 收益分布
        if self.df is not None and not self.df.empty:
            returns_fig = create_return_distribution(self.df)
            for trace in returns_fig.data:
                fig.add_trace(trace, row=2, col=1)
                
        # 4. Overview表格
        overview_data = [
            ["Backtest Period", f"{self.stats.get('start_date', '')} - {self.stats.get('end_date', '')}"],
            ["Initial Capital", f"${self.stats.get('initial_capital', 0):,.2f}"],
            ["Final Capital", f"${self.stats.get('final_capital', 0):,.2f}"],
            ["Total Return", f"{self.stats.get('total_return', 0):.2f}%"],
            ["Win Rate", f"{self.stats.get('win_rate', 0):.2f}%"],
            ["Profit/Loss Ratio", f"{self.stats.get('profit_factor', 0):.2f}"],
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    font=dict(size=14, color="white"),
                    fill_color="rgb(0, 102, 204)",
                    align="left"
                ),
                cells=dict(
                    values=list(zip(*overview_data)),
                    font=dict(size=12),
                    fill_color="rgb(242, 242, 242)",
                    align="left"
                )
            ),
            row=2, col=2
        )
        
        # 5. Key Metrics表格
        metrics_data = [
            ["Annualized Return", f"{self.stats.get('annual_return', 0):.2f}%"],
            ["Max Drawdown", f"{self.stats.get('max_drawdown', 0):.2f}%"],
            ["Sharpe Ratio", f"{self.stats.get('sharpe_ratio', 0):.2f}"],
            ["Return/Drawdown", f"{self.stats.get('return_drawdown_ratio', 0):.2f}"],
            ["Total Trades", f"{self.stats.get('total_trade_count', 0)}"],
            ["Total Turnover", f"${self.stats.get('total_turnover', 0):,.2f}"],
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    font=dict(size=14, color="white"),
                    fill_color="rgb(0, 102, 204)",
                    align="left"
                ),
                cells=dict(
                    values=list(zip(*metrics_data)),
                    font=dict(size=12),
                    fill_color="rgb(242, 242, 242)",
                    align="left"
                )
            ),
            row=3, col=1
        )
        
        # 6. AI Strategy Assessment
        assessment = generate_strategy_assessment(self.stats)
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Strategy Assessment"],
                    font=dict(size=14, color="white"),
                    fill_color="rgb(0, 102, 204)",
                    align="left"
                ),
                cells=dict(
                    values=[assessment],
                    font=dict(size=12),
                    fill_color="rgb(242, 242, 242)",
                    align="left",
                    height=30
                )
            ),
            row=3, col=2
        )
        
        # 更新布局
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Strategy Performance Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
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
        logger.info(f"Return/Drawdown: {self.stats.get('return_drawdown_ratio', 0):.2f}")
        logger.info(f"Total Turnover: ${self.stats.get('total_turnover', 0):,.2f}")
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