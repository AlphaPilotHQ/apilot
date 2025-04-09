"""
Performance reporting module.

Implements the new performance dashboard with organized sections:
- Overview
- Key Metrics
- Plots
- AI Summary
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apilot.utils import get_logger

logger = get_logger("PerformanceReporter")


class PerformanceReporter:
    """
    Performance reporting class for strategy backtesting results
    
    Provides comprehensive visualization and analysis of backtest results
    with an organized dashboard approach.
    """
    
    def __init__(self, stats=None, df=None, trades=None):
        """
        Initialize the performance reporter
        
        Args:
            stats: Dictionary of calculated performance statistics
            df: DataFrame with daily performance data
            trades: List of trade objects
        """
        self.stats = stats or {}
        self.df = df
        self.trades = trades or []
        
    def generate_dashboard(self) -> go.Figure:
        """
        Generate a comprehensive performance dashboard with four sections.
        
        Returns:
            Plotly figure object containing the dashboard
        """
        # Create the main figure with subplots
        fig = make_subplots(
            rows=3, 
            cols=2,
            subplot_titles=[
                "Equity Curve", "Drawdown",
                "Return Distribution", "Overview",
                "Key Metrics", "AI Summary"
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
        
        # Add equity curve
        self._add_equity_curve(fig, row=1, col=1)
        
        # Add drawdown curve
        self._add_drawdown_curve(fig, row=1, col=2)
        
        # Add return distribution
        self._add_return_distribution(fig, row=2, col=1)
        
        # Add overview table
        self._add_overview_table(fig, row=2, col=2)
        
        # Add key metrics table
        self._add_metrics_table(fig, row=3, col=1)
        
        # Add AI summary table
        self._add_ai_summary(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Strategy Performance Dashboard",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def _add_equity_curve(self, fig, row=1, col=1):
        """
        Add equity curve to the dashboard
        """
        if self.df is None or "balance" not in self.df:
            return
            
        fig.add_trace(
            go.Scatter(
                x=self.df.index, 
                y=self.df["balance"], 
                mode="lines", 
                name="Equity",
                line=dict(color="rgb(0, 153, 76)", width=2)
            ),
            row=row, 
            col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Capital", row=row, col=col)
    
    def _add_drawdown_curve(self, fig, row=1, col=2):
        """
        Add drawdown curve to the dashboard
        """
        if self.df is None or "ddpercent" not in self.df:
            return
            
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
            row=row, 
            col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Drawdown %", row=row, col=col)
    
    def _add_return_distribution(self, fig, row=2, col=1):
        """
        Add daily return distribution to the dashboard
        """
        if self.df is None or "return" not in self.df:
            return
            
        # Convert log returns to percentage for better readability
        returns_pct = (self.df["return"] * 100).dropna()
            
        fig.add_trace(
            go.Histogram(
                x=returns_pct,
                nbinsx=40,
                name="Daily Returns",
                marker_color="rgb(83, 143, 255)",
                opacity=0.7
            ),
            row=row, 
            col=col
        )
        
        fig.update_xaxes(title_text="Daily Return (%)", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)
    
    def _add_overview_table(self, fig, row=2, col=2):
        """
        Add overview summary table to the dashboard
        """
        if not self.stats:
            return
            
        # Prepare table data
        overview_data = [
            ["Backtest Period", f"{self.stats.get('start_date', '')} - {self.stats.get('end_date', '')}"],
            ["Initial Capital", f"${self.stats.get('capital', 0):,.2f}"],
            ["Final Capital", f"${self.stats.get('end_balance', 0):,.2f}"],
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
            row=row, 
            col=col
        )
    
    def _add_metrics_table(self, fig, row=3, col=1):
        """
        Add detailed metrics table to the dashboard
        """
        if not self.stats:
            return
            
        # Prepare table data
        metrics_data = [
            ["Annualized Return", f"{self.stats.get('annual_return', 0):.2f}%"],
            ["Max Drawdown", f"{self.stats.get('max_ddpercent', 0):.2f}%"],
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
            row=row, 
            col=col
        )
    
    def _add_ai_summary(self, fig, row=3, col=2):
        """
        Add AI summary analysis to the dashboard
        """
        # TODO: Implement AI analysis later
        summary = self._generate_summary()
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["AI Strategy Assessment"],
                    font=dict(size=14, color="white"),
                    fill_color="rgb(0, 102, 204)",
                    align="left"
                ),
                cells=dict(
                    values=[summary],
                    font=dict(size=12),
                    fill_color="rgb(242, 242, 242)",
                    align="left",
                    height=30
                )
            ),
            row=row, 
            col=col
        )
    
    def _generate_summary(self) -> list:
        """
        Generate a simple summary based on performance statistics
        
        Returns:
            List of summary points for display in the table
        """
        # Simple placeholder for AI analysis
        insights = []
        
        # Assess return
        if self.stats.get("total_return", 0) > 20:
            insights.append("✓ Strong absolute returns")
        elif self.stats.get("total_return", 0) > 0:
            insights.append("✓ Positive returns but room for improvement")
        else:
            insights.append("✗ Strategy showing negative returns")
            
        # Assess risk-adjusted return
        if self.stats.get("sharpe_ratio", 0) > 2:
            insights.append("✓ Excellent risk-adjusted performance")
        elif self.stats.get("sharpe_ratio", 0) > 1:
            insights.append("✓ Good risk-adjusted performance")
        else:
            insights.append("✗ Poor risk-adjusted performance")
            
        # Assess drawdown
        if abs(self.stats.get("max_ddpercent", 0)) < 10:
            insights.append("✓ Well-controlled risk with minimal drawdown")
        elif abs(self.stats.get("max_ddpercent", 0)) < 20:
            insights.append("⚠ Moderate drawdown")
        else:
            insights.append("✗ Significant drawdown - review risk management")
            
        # Assess win rate
        if self.stats.get("win_rate", 0) > 60:
            insights.append("✓ High win rate")
        elif self.stats.get("win_rate", 0) > 45:
            insights.append("⚠ Average win rate")
        else:
            insights.append("✗ Low win rate - review entry/exit criteria")
            
        # Add summary and recommendations
        insights.append("")
        if self.stats.get("return_drawdown_ratio", 0) > 3 and self.stats.get("sharpe_ratio", 0) > 1:
            insights.append("Overall assessment: Strong strategy")
        elif self.stats.get("total_return", 0) > 0 and self.stats.get("sharpe_ratio", 0) > 0.5:
            insights.append("Overall assessment: Promising but needs refinement")
        else:
            insights.append("Overall assessment: Strategy requires significant improvements")
            
        return insights
    
    def show(self):
        """
        Generate and display the performance dashboard
        """
        fig = self.generate_dashboard()
        fig.show()
        
    def print_summary(self):
        """
        Print a text summary of performance to the log
        """
        if not self.stats:
            logger.warning("No statistics available for printing summary")
            return
            
        # Print header
        logger.info("=" * 50)
        logger.info("Performance Summary")
        logger.info("=" * 50)
        
        # Overview section
        logger.info("\n== Overview ==")
        logger.info(
            f"Period: {self.stats.get('start_date', '')} - {self.stats.get('end_date', '')} "
            f"({self.stats.get('total_days', 0)} days)"
        )
        logger.info(f"Initial Capital: ${self.stats.get('capital', 0):,.2f}")
        logger.info(f"Final Capital: ${self.stats.get('end_balance', 0):,.2f}")
        logger.info(f"Total Return: {self.stats.get('total_return', 0):.2f}%")
        logger.info(f"Win Rate: {self.stats.get('win_rate', 0):.2f}%")
        logger.info(f"Profit/Loss Ratio: {self.stats.get('profit_factor', 0):.2f}")
        
        # Key Metrics section
        logger.info("\n== Key Metrics ==")
        logger.info(f"Annualized Return: {self.stats.get('annual_return', 0):.2f}%")
        logger.info(f"Max Drawdown: {self.stats.get('max_ddpercent', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {self.stats.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Return/Drawdown: {self.stats.get('return_drawdown_ratio', 0):.2f}")
        logger.info(f"Total Turnover: ${self.stats.get('total_turnover', 0):,.2f}")
        logger.info(f"Total Trades: {self.stats.get('total_trade_count', 0)}")
        
        # AI Summary
        insights = self._generate_summary()
        if insights:
            logger.info("\n== Strategy Assessment ==")
            for insight in insights:
                if insight:  # Skip empty strings
                    logger.info(insight)
        
        logger.info("=" * 50)


def create_performance_dashboard(stats=None, df=None, trades=None):
    """
    Create and show a performance dashboard.
    
    Args:
        stats: Dictionary of calculated performance statistics
        df: DataFrame with daily performance data
        trades: List of trade objects
    """
    reporter = PerformanceReporter(stats, df, trades)
    reporter.show()
    return reporter
