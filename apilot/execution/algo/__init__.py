"""
算法交易模块

包含各种执行算法的实现，用于优化交易执行。
"""

# 首先导入apilot.core中需要的组件，以解决算法文件中的导入问题
from apilot.core.constant import Direction, Offset, OrderType, Exchange
from apilot.core.object import TradeData, OrderData, TickData, OrderRequest
from apilot.core.engine import BaseEngine

# 然后导出算法引擎和算法模板
from .algo_engine import AlgoEngine
from .algo_template import AlgoTemplate

# 导出具体算法实现
from .best_limit_algo import BestLimitAlgo
from .iceberg_algo import IcebergAlgo
from .sniper_algo import SniperAlgo
from .stop_algo import StopAlgo
from .twap_algo import TwapAlgo

# 定义公共API
__all__ = [
    # 算法引擎和模板
    "AlgoEngine", "AlgoTemplate",
    
    # 具体算法实现
    "BestLimitAlgo", "IcebergAlgo", "SniperAlgo", "StopAlgo", "TwapAlgo",
    
    # 核心组件
    "Direction", "Offset", "OrderType", "Exchange",
    "TradeData", "OrderData", "TickData", "OrderRequest", "BaseEngine"
]

# TODO: /Users/bobbyding/Documents/GitHub/apilot/apilot/algotrading/algo_base.py
# 可以挪到core里面去
