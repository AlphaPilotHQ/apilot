from pathlib import Path
import sys

# 针对Python 3.8及以上版本使用标准库中的importlib.metadata
if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

from apilot.trader.constant import Direction
from apilot.trader.object import TickData, BarData, TradeData, OrderData
from apilot.trader.utility import BarGenerator, ArrayManager

from .base import APP_NAME
from .engine import StrategyEngine
from .template import StrategyTemplate as PortfolioStrategyTemplate
from .backtesting import BacktestingEngine

try:
    __version__ = importlib_metadata.version("apilot_portfoliostrategy")
except importlib_metadata.PackageNotFoundError:
    __version__ = "dev"
