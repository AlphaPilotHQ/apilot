"""
Performance module for strategy backtesting analysis and visualization.
"""

from apilot.performance.calculator import (
    DailyResult,
    PerformanceCalculator,
    TradeAnalyzer,
    calculate_statistics,
)

from apilot.performance.chart import (
    plot_backtest_results,
    plot_drawdown,
    plot_equity_curve,
    plot_pnl_distribution,
)

from apilot.performance.reporter import (
    PerformanceReporter,
    create_performance_dashboard,
)

__all__ = [
    # Performance calculation
    "DailyResult",
    "PerformanceCalculator",
    "TradeAnalyzer",
    "calculate_statistics",
    
    # Visualization
    "plot_backtest_results",
    "plot_drawdown",
    "plot_equity_curve",
    "plot_pnl_distribution",
    
    # Reporting
    "PerformanceReporter",
    "create_performance_dashboard",
]
