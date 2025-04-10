"""
算法交易模块

包含各种执行算法的实现,用于优化交易执行.
"""

# 首先导入apilot.core中需要的组件,以解决算法文件中的导入问题
from apilot.core.constant import Direction, Exchange, OrderType
from apilot.core.engine import BaseEngine
from apilot.core.object import OrderData, OrderRequest, TickData, TradeData

# 然后导出算法引擎和算法模板
from .algo_engine import AlgoEngine
from .algo_template import AlgoTemplate

# 导出具体算法实现
from .best_limit_algo import BestLimitAlgo
from .twap_algo import TwapAlgo

# 定义公共API
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
