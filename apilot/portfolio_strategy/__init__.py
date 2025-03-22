from pathlib import Path
import sys

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
