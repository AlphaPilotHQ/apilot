"""
优化模块

提供策略参数优化功能，包括网格搜索和遗传算法优化。
"""

from .optimizer import (
    OptimizationSetting,
    check_optimization_setting,
    run_ga_optimization,
    ga_evaluate
)
