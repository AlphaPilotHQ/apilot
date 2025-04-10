"""
性能分析和报告模块

提供性能计算、可视化和报告功能，包括:
- Overview: Backtest Period, Initial Capital, Final Capital, Total Return, Win Rate, Profit/Loss Ratio
- Key Metrics: Annualized Return, Max Drawdown, Sharpe Ratio, Turnover
- Plot: Equity Curve, Drawdown Curve, Daily Return Distribution
- AI Summary: 策略智能评估
"""

# 导出性能计算函数和工具
# 导出AI分析函数
from apilot.performance.aisummary import (
    generate_strategy_assessment,
)
from apilot.performance.calculator import (
    calculate_statistics,
    calculate_trade_metrics,
)

# 导出图表函数
from apilot.performance.plot import (
    create_drawdown_curve,
    create_equity_curve,
    create_return_distribution,
    get_drawdown_trace,
    get_equity_trace,
    get_return_dist_trace,
)

# 导出报告功能
from apilot.performance.report import (
    PerformanceReport,
)

__all__ = [
    # 按字母排序的导出列表
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
