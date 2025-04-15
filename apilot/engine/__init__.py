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
    "BacktestingEngine",
    "OmsEngine",
    "BaseEngine",
    "MainEngine",
    "LiveEngine",
]


from .backtest import BacktestingEngine
from .base_engine import BaseEngine
from .live import LiveEngine
from .main_engine import MainEngine

# Import core engine components
from .oms_engine import OmsEngine
