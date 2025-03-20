from pathlib import Path
import sys

# 针对Python 3.8及以上版本使用标准库中的importlib.metadata
if sys.version_info >= (3, 8):
    import importlib.metadata as importlib_metadata
else:
    import importlib_metadata

from apilot.trader.app import BaseApp
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


class PortfolioStrategyApp(BaseApp):
    """"""

    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "组合策略"
    engine_class: StrategyEngine = StrategyEngine
    widget_name: str = "PortfolioStrategyManager"
    icon_name: str = str(app_path.joinpath("ui", "strategy.ico"))
