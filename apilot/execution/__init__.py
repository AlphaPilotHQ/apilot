"""
量化交易执行模块

提供各种交易算法和执行引擎，用于智能化订单执行。

主要组件:
- AlgoEngine: 算法交易引擎，管理各种执行算法
- AlgoTemplate: 算法交易模板，所有算法的基类
- 多种预设算法实现，如冰山算法、TWAP算法等

推荐用法:
    from apilot.execution import AlgoEngine
    algo_engine = AlgoEngine(main_engine)
"""

# 定义公共API
__all__ = [
    # 算法引擎
    "AlgoEngine",
    
    # 算法模板
    "AlgoTemplate",
    
    # 算法实现
    "BestLimitAlgo",
    "IcebergAlgo",
    "SniperAlgo",
    "StopAlgo",
    "TwapAlgo"
]

# 导入算法引擎
from .algo.algo_engine import AlgoEngine

# 导入算法模板
from .algo.algo_template import AlgoTemplate

# 导入算法实现
from .algo.best_limit_algo import BestLimitAlgo
from .algo.iceberg_algo import IcebergAlgo
from .algo.sniper_algo import SniperAlgo
from .algo.stop_algo import StopAlgo
from .algo.twap_algo import TwapAlgo
