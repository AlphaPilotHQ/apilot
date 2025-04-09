"""
性能分析和报告模块

提供性能计算、可视化和报告功能，包括:
- Overview: Backtest Period, Initial Capital, Final Capital, Total Return, Win Rate, Profit/Loss Ratio
- Key Metrics: Annualized Return, Max Drawdown, Sharpe Ratio, Turnover
- Plot: Equity Curve, Drawdown Curve, Daily Return Distribution
- AI Summary: 策略智能评估
"""

# 导出性能计算函数和工具
from apilot.performance.calculator import (
    calculate_statistics,
    calculate_trade_metrics,
)

# 导出图表函数
from apilot.performance.plot import (
    create_equity_curve,
    create_drawdown_curve,
    create_return_distribution,
)

# 导出AI分析函数
from apilot.performance.aisummary import (
    generate_strategy_assessment,
)

# 导出报告功能
from apilot.performance.report import (
    PerformanceReport,
    create_performance_report,
)

__all__ = [
    # 计算部分
    "calculate_statistics",
    "calculate_trade_metrics",
    
    # 图表部分
    "create_equity_curve",
    "create_drawdown_curve",
    "create_return_distribution",
    
    # AI分析部分
    "generate_strategy_assessment",
    
    # 报告部分
    "PerformanceReport",
    "create_performance_report",
]