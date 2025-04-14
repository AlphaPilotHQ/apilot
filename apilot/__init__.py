"""
AlphaPilot (apilot) - AI-driven quant, open to all.

This package provides a complete set of quantitative trading tools, including data acquisition, strategy development, backtesting and live trading.

Recommended import methods:
    import apilot as ap                     # General usage
    from apilot import BarData              # Import specific components
    import apilot.core as apcore            # Extensive use of a module

"""

from .version import __version__

# Define public API
__all__ = [
    "AccountData",
    "ArrayManager",
    "BacktestingEngine",
    "BarData",
    "BarGenerator",
    "BinanceGateway",
    "ContractData",
    "Direction",
    "EventEngine",
    "Interval",
    "LocalOrderManager",
    "MainEngine",
    "OptimizationSetting",
    "OrderData",
    "OrderType",
    "PAEngine",
    "PATemplate",
    "PerformanceReport",
    "PositionData",
    "Product",
    "Status",
    "TradeData",
    "__version__",
    "core",
    "create_csv_data",
    "create_mongodb_data",
    "datafeed",
    "engine",
    "execution",
    "get_logger",
    "log_exceptions",
    "optimizer",
    "performance",
    "risk",
    "run_grid_search",
    "set_level",
    "strategy",
    "utils",
]

# Export submodules (package level)
from . import core, datafeed, engine, execution, optimizer, performance, strategy, utils

# Export constants
from .core.constant import (
    Direction,
    Interval,
    OrderType,
    Product,
    Status,
)
from .core.engine import MainEngine

# Export core engine components
from .core.event import EventEngine

# Export core data objects
from .core.object import (
    AccountData,
    BarData,
    ContractData,
    OrderData,
    PositionData,
    TradeData,
)

# Export utility classes
from .core.utility import BarGenerator

# Export backtesting and optimization components
from .engine.backtest import BacktestingEngine
from .engine.live import PAEngine

# Export gateway components
from .execution.gateway.binance import BinanceGateway
from .optimizer import OptimizationSetting, run_grid_search

# Export performance analysis components
from .performance.report import PerformanceReport

# Export strategy templates
from .strategy.template import PATemplate
from .utils.indicators import ArrayManager

# Export logging system
from .utils.logger import get_logger, log_exceptions, set_level
from .utils.order_manager import LocalOrderManager
