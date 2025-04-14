"""
Performance Report Module

Integrates calculations, charts and AI analysis to generate complete strategy performance reports
"""

import logging

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apilot.performance.aisummary import generate_strategy_assessment
from apilot.performance.calculator import calculate_statistics
from apilot.performance.plot import (
    get_drawdown_trace,
    get_equity_trace,
    get_return_dist_trace,
)

logger = logging.getLogger(__name__)


class PerformanceReport:
    """Performance Report Class"""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        trades: list | None = None,
        capital: float = 0,
        annual_days: int = 240,
    ):  # TODO: Redundant
        self.df = df
        self.trades = trades or []
        self.capital = capital
        self.annual_days = annual_days
        self.stats = None

    def generate(self) -> "PerformanceReport":
        self.stats = calculate_statistics(
            df=self.df,
            trades=self.trades,
            capital=self.capital,
            annual_days=self.annual_days,
        )
        return self

    def create_dashboard(self) -> go.Figure:
        if not self.stats:
            self.generate()

        # Create figure with subplots
        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[1 / 3, 1 / 3, 1 / 3],
            vertical_spacing=0.1,
            specs=[
                [{"type": "scatter"}],
                [{"type": "scatter"}],
                [{"type": "histogram"}],
            ],
            subplot_titles=["Equity Curve", "Drawdown", "Daily Return Distribution"],
        )

        # Add charts if data is available
        if self.df is not None and not self.df.empty:
            # Add equity curve (1st row)
            if "balance" in self.df.columns:
                fig.add_trace(get_equity_trace(self.df), row=1, col=1)

                # Add baseline
                if self.capital > 0:
                    fig.add_shape(
                        type="line",
                        x0=self.df.index[0],
                        x1=self.df.index[-1],
                        y0=self.capital,
                        y1=self.capital,
                        line={"color": "rgba(0,0,0,0.3)", "width": 1, "dash": "dash"},
                        row=1,
                        col=1,
                    )

            # Add drawdown curve (2nd row)
            if "ddpercent" in self.df.columns:
                fig.add_trace(get_drawdown_trace(self.df), row=2, col=1)

            # Add return distribution (3rd row)
            if "return" in self.df.columns:
                fig.add_trace(get_return_dist_trace(self.df), row=3, col=1)

        # Update layout
        fig.update_layout(
            title={
                "text": "AlphaPilot Performance Charts",
                "font": {"size": 24},
                "x": 0.5,  # Center title
                "xanchor": "center",
            },
            height=1200,  # Overall height
            width=1000,  # Overall width
            template="plotly_white",
            margin={"l": 50, "r": 50, "t": 100, "b": 100},
            showlegend=False,
            hovermode="x unified",
        )

        # Update axes properties
        # Equity curve Y-axis
        fig.update_yaxes(title_text="Capital", row=1, col=1, gridcolor="lightgray")
        # Drawdown Y-axis
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1, gridcolor="lightgray")
        # Distribution Y-axis
        fig.update_yaxes(title_text="Frequency", row=3, col=1, gridcolor="lightgray")

        # All X-axes
        fig.update_xaxes(gridcolor="lightgray", row=1, col=1)
        fig.update_xaxes(gridcolor="lightgray", row=2, col=1)
        fig.update_xaxes(title_text="Daily Return", gridcolor="lightgray", row=3, col=1)

        return fig

    def get_metrics_text(self) -> str:
        """
        Get text description of performance metrics

        Returns:
            Formatted text containing all performance metrics
        """
        if not self.stats:
            self.generate()

        # Build detailed text report
        lines = []
        lines.append("         Performance Metrics Summary")
        lines.append("=" * 50)

        # Overview section
        lines.append("\nOverview:")
        lines.append(
            f"Backtest Period: {self.stats.get('start_date', '')} - {self.stats.get('end_date', '')}"
        )
        lines.append(f"Initial Capital: ${self.stats.get('initial_capital', 0):,.2f}")
        lines.append(f"Final Capital: ${self.stats.get('final_capital', 0):,.2f}")
        lines.append(f"Total Return: {self.stats.get('total_return', 0):.2f}%")
        lines.append(f"Win Rate: {self.stats.get('win_rate', 0):.2f}%")
        lines.append(f"Profit/Loss Ratio: {self.stats.get('profit_loss_ratio', 0):.2f}")

        # Key Metrics section
        lines.append("\nKey Metrics:")
        lines.append(f"Annual Return: {self.stats.get('annual_return', 0):.2f}%")
        lines.append(f"Max Drawdown: {self.stats.get('max_drawdown', 0):.2f}%")
        lines.append(f"Sharpe Ratio: {self.stats.get('sharpe_ratio', 0):.2f}")
        lines.append(f"Turnover Ratio: {self.stats.get('turnover_ratio', 0):.2f}")

        # AI Summary section
        assessment = generate_strategy_assessment(self.stats)
        if assessment:
            lines.append("\nStrategy Summary:")
            for insight in assessment:
                if insight:  # Skip empty strings
                    lines.append(insight)
        lines.append("=" * 50)

        return "\n".join(lines)

    def show(self):
        # First print text metrics
        metrics_text = self.get_metrics_text()
        print(metrics_text)

        # Then display charts
        fig = self.create_dashboard()
        fig.show()
