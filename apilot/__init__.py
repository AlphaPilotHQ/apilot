__version__ = "0.1.0"

# 导出主要模块
from . import core
from . import engine
from . import execution
from . import optimizer
from . import risk
from . import strategy
from . import utils
from . import datafeed

# 直接导出常用类，方便使用ap.类名的形式调用
from .core.object import BarData, TickData, OrderData, TradeData, AccountData, PositionData, ContractData
from .core.utility import BarGenerator, ArrayManager
from .strategy.template import CtaTemplate, TargetPosTemplate
from .engine.backtest import BacktestingEngine
from .optimizer import OptimizationSetting, run_ga_optimization
