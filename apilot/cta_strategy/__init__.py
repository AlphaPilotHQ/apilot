from pathlib import Path
from typing import Type

from apilot.trader.app import BaseApp
from apilot.trader.constant import Direction
from apilot.trader.object import TickData, BarData, TradeData, OrderData
from apilot.trader.utility import BarGenerator, ArrayManager

from .base import APP_NAME, StopOrder
from .engine import CtaEngine
from .template import CtaTemplate, TargetPosTemplate

from .backtesting import BacktestingEngine


class CtaStrategyApp(BaseApp):

    app_name: str = APP_NAME
    app_module: str = __module__
    app_path: Path = Path(__file__).parent
    display_name: str = "CTA策略"
    engine_class: Type[CtaEngine] = CtaEngine
    widget_name: str = "CtaManager"
    icon_name: str = str(app_path.joinpath("ui", "cta.ico"))

"""
__init__.py    模块入口
base.py        基础定义层
template.py    接口定义层
engine.py      业务逻辑层
backtesting.py 应用场景层
"""
