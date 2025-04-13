"""
Algorithmic Trading Module

Contains implementations of various execution algorithms for optimizing trade execution.
"""

# First import required components from apilot.core to resolve import issues in algorithm files
from apilot.core.constant import Direction, Exchange, OrderType
from apilot.core.engine import BaseEngine
from apilot.core.object import OrderData, OrderRequest, TickData, TradeData

# Then export algorithm engine and algorithm template
from .algo_engine import AlgoEngine
from .algo_template import AlgoTemplate

# Export specific algorithm implementations
from .best_limit_algo import BestLimitAlgo
from .twap_algo import TwapAlgo

# Define public API
__all__ = [
    "AlgoEngine",
    "AlgoTemplate",
    "BaseEngine",
    "BestLimitAlgo",
    "Direction",
    "Exchange",
    "OrderData",
    "OrderRequest",
    "OrderType",
    "TickData",
    "TradeData",
    "TwapAlgo",
]
