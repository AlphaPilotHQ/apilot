"""
Engine Module

Contains backtesting engine, live trading engine and other core trading engines.

Main components:
- BacktestingEngine: Backtesting engine, used for historical strategy performance testing
- OmsEngine: Order Management System engine, handles order lifecycle

Recommended usage:
    from apilot.engine import BacktestingEngine
    engine = BacktestingEngine()
"""

# Define public API
__all__ = [
    "EVENT_LOG",
    "EVENT_TIMER",
    "BacktestingEngine",
    "Event",
    "EventType",
    "OmsEngine",
]

# Import backtesting related engines
from apilot.strategy.template import PATemplate

from .backtest import BacktestingEngine

# Import core engine components
from .oms_engine import OmsEngine
