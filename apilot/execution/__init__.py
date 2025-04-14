"""
Quantitative Trading Execution Module

Provides various trading algorithms and execution engines for intelligent order execution.

Main components:
- AlgoEngine: Algorithm trading engine, manages various execution algorithms
- AlgoTemplate: Algorithm trading template, base class for all algorithms
- Multiple preset algorithm implementations, such as Iceberg algorithm, TWAP algorithm, etc.

Recommended usage:
    from apilot.execution import AlgoEngine
    algo_engine = AlgoEngine(main_engine)
"""

# Define public API
__all__ = [
    # Algorithm engine
    "AlgoEngine",
    # Algorithm template
    "AlgoTemplate",
    # Algorithm implementations
    "BestLimitAlgo",
    "TwapAlgo",
]

# Import algorithm engine
from .algo.algo_engine import AlgoEngine

# Import algorithm template
from .algo.algo_template import AlgoTemplate

# Import algorithm implementations
from .algo.best_limit_algo import BestLimitAlgo
