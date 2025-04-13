"""
Performance Analysis and Reporting Module

Provides performance calculation, visualization and reporting functions, including:
- Overview: Backtest Period, Initial Capital, Final Capital, Total Return, Win Rate, Profit/Loss Ratio
- Key Metrics: Annualized Return, Max Drawdown, Sharpe Ratio, Turnover
- Plot: Equity Curve, Drawdown Curve, Daily Return Distribution
- AI Summary: Strategy Intelligent Assessment
"""

# Export performance calculation functions and tools
# Export AI analysis functions
from apilot.performance.aisummary import (
    generate_strategy_assessment,
)
from apilot.performance.calculator import (
    calculate_statistics,
    calculate_trade_metrics,
)

# Export chart functions
from apilot.performance.plot import (
    create_drawdown_curve,
    create_equity_curve,
    create_return_distribution,
    get_drawdown_trace,
    get_equity_trace,
    get_return_dist_trace,
)

# Export reporting functions
from apilot.performance.report import (
    PerformanceReport,
)

__all__ = [
    # Alphabetically sorted export list
    "calculate_statistics",
    "calculate_trade_metrics",
    "create_drawdown_curve",
    "create_equity_curve",
    "create_return_distribution",
    "generate_strategy_assessment",
    "get_drawdown_trace",
    "get_equity_trace",
    "get_return_dist_trace",
    "PerformanceReport",
]
