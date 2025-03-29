"""
引擎模块

包含回测引擎、实盘引擎及其他交易核心引擎.

主要组件:
- BacktestingEngine: 回测引擎,用于策略历史表现回测
- OmsEngine: 订单管理系统引擎,处理订单生命周期

推荐用法:
    from apilot.engine import BacktestingEngine
    engine = BacktestingEngine()
"""

# 定义公共API
__all__ = [
    "EVENT_LOG",
    "EVENT_TIMER",
    "BacktestingEngine",
    "Event",
    "EventType",
    "OmsEngine",
]

# 导入回测相关引擎
from apilot.strategy.template import PATemplate, TargetPosTemplate

from .backtest import BacktestingEngine

# 导入核心引擎组件
from .oms_engine import OmsEngine
