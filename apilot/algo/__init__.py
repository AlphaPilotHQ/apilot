"""
Algorithmic Trading Module

Contains implementations of various execution algorithms for optimizing trade execution.
"""

# First import required components from apilot.core to resolve import issues in algorithm files
from apilot.core.constant import Direction, OrderType
from apilot.core.engine import BaseEngine
from apilot.core.object import OrderData, OrderRequest, TradeData

# Then export algorithm engine and algorithm template
from .algo_engine import AlgoEngine
from .algo_template import AlgoTemplate

# Export specific algorithm implementations
from .best_limit_algo import BestLimitAlgo

# Define public API
__all__ = [
    "AlgoEngine",
    "AlgoTemplate",
    "BaseEngine",
    "BestLimitAlgo",
    "Direction",
    "OrderData",
    "OrderRequest",
    "OrderType",
    "TradeData",
]
