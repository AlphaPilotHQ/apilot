"""
引擎模块

包含回测引擎、实盘引擎及其他交易核心引擎。
"""

# 导入回测相关引擎
from .backtest import BacktestingEngine, optimize

# 导入核心引擎组件
from .log_engine import LogEngine
from .oms_engine import OmsEngine
from .email_engine import EmailEngine

# 从策略模块导入策略模板 (为保持向后兼容)
from apilot.strategy.template import CtaTemplate, TargetPosTemplate
