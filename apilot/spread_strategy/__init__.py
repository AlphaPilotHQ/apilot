from pathlib import Path

try:
    # Python 3.8及以上版本使用内置库
    import importlib.metadata as importlib_metadata
except ImportError:
    # 低于3.8版本回退到独立包
    import importlib_metadata

from apilot.trader.app import BaseApp
from apilot.trader.object import (
    OrderData,
    TradeData,
    TickData,
    BarData
)

from .engine import (
    SpreadEngine,
    APP_NAME,
    SpreadData,
    LegData,
    SpreadStrategyTemplate,
    SpreadAlgoTemplate
)



class SpreadTradingApp(BaseApp):
    """"""

    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "价差交易"
    engine_class: SpreadEngine = SpreadEngine
    widget_name: str = "SpreadManager"
    icon_name: str = str(app_path.joinpath("ui", "spread.ico"))
