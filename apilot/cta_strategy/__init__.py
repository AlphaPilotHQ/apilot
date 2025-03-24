from pathlib import Path

from apilot.trader.constant import Direction
from apilot.trader.object import TickData, BarData, TradeData, OrderData
from apilot.trader.utility import BarGenerator, ArrayManager

# 导入常量 (旧名称: base.py -> 新名称: constants.py)
from .constants import APP_NAME, EngineType
# 导入交易器 (旧名称: engine.py -> 新名称: trader.py)
from .trader import CtaEngine
# 导入策略基类 (旧名称: template.py -> 新名称: strategy_base.py)
from .strategy_base import CtaTemplate, TargetPosTemplate
# 导入回测引擎 (旧名称: backtesting.py -> 新名称: backtest.py)
from .backtest import BacktestingEngine

# 为了保持向后兼容性，需要有这些别名
from .constants import *  # 替代 base.py
from .strategy_base import *  # 替代 template.py
