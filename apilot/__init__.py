"""
AlphaPilot (apilot) - 量化交易平台

这个包提供了一套完整的量化交易工具，包括数据获取、策略开发、回测和实盘交易功能。

推荐导入方式:
    import apilot as ap                     # 一般用法
    from apilot import BarData, TickData    # 导入特定组件
    import apilot.core as apcore            # 大量使用某模块

版本: 0.1.0
"""

__version__ = "0.1.0"

# 定义公共API
__all__ = [
    # 核心数据对象
    "BarData", "TickData", "OrderData", "TradeData", "AccountData", "PositionData", "ContractData",

    # 工具类
    "BarGenerator", "ArrayManager",

    # 策略模板
    "CtaTemplate", "TargetPosTemplate",

    # 回测和优化
    "BacktestingEngine", "OptimizationSetting", "run_ga_optimization",

    # 常量
    "Direction", "Offset", "Exchange", "Interval", "Status", "Product", "OrderType",

    # 日志系统
    "get_logger", "set_level", "log_exceptions",

    # 模块包
    "core", "engine", "execution", "optimizer", "risk", "strategy", "utils", "datafeed"
]

# 导出子模块 (包级别)
from . import core
from . import engine
from . import execution
from . import optimizer
from . import risk
from . import strategy
from . import utils
from . import datafeed

# 导出常量
from .core.constant import (
    Direction,
    Offset,
    Exchange,
    Interval,
    Status,
    Product,
    OrderType
)

# 导出核心数据对象
from .core.object import (
    BarData,
    TickData,
    OrderData,
    TradeData,
    AccountData,
    PositionData,
    ContractData
)

# 导出工具类
from .core.utility import BarGenerator, ArrayManager

# 导出策略模板
from .strategy.template import CtaTemplate, TargetPosTemplate

# 导出回测和优化组件
from .engine.backtest import BacktestingEngine
from .optimizer import OptimizationSetting, run_ga_optimization

# 导出日志系统
from .utils.logger import get_logger, set_level, log_exceptions
