"""
AlphaPilot (apilot) - 量化交易平台

这个包提供了一套完整的量化交易工具,包括数据获取、策略开发、回测和实盘交易功能.

推荐导入方式:
    import apilot as ap                     # 一般用法
    from apilot import BarData, TickData    # 导入特定组件
    import apilot.core as apcore            # 大量使用某模块

版本: 0.1.0
"""

__version__ = "0.1.0"

# 定义公共API
__all__ = [
    "AccountData",
    "ArrayManager",
    "BacktestingEngine",
    "BarData",
    "BarGenerator",
    "ContractData",
    "Direction",
    "Exchange",
    "Interval",
    "LocalOrderManager",
    "Offset",
    "OptimizationSetting",
    "OrderData",
    "OrderType",
    "PATemplate",
    "PositionData",
    "Product",
    "Status",
    "TickData",
    "TradeData",
    "core",
    "create_csv_data",
    "create_mongodb_data",
    "datafeed",
    "engine",
    "execution",
    "get_logger",
    "log_exceptions",
    "optimizer",
    "risk",
    "run_ga_optimization",
    "set_level",
    "strategy",
    "utils",
]

# 导出子模块 (包级别)
from . import core, datafeed, engine, execution, optimizer, strategy, utils

# 导出常量
from .core.constant import (
    Direction,
    Exchange,
    Interval,
    Offset,
    OrderType,
    Product,
    Status,
)

# 导出核心数据对象
from .core.object import (
    AccountData,
    BarData,
    ContractData,
    OrderData,
    PositionData,
    TickData,
    TradeData,
)

# 导出工具类
from .core.utility import ArrayManager, BarGenerator

# 导出回测和优化组件
from .engine.backtest import BacktestingEngine
from .optimizer import OptimizationSetting, run_ga_optimization

# 导出策略模板
from .strategy.template import PATemplate

# 导出日志系统
from .utils.logger import get_logger, log_exceptions, set_level
from .utils.order_manager import LocalOrderManager
