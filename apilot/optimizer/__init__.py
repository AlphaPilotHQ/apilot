"""
优化模块

提供策略参数优化功能,使用网格搜索优化.

主要组件:
- OptimizationSetting: 优化配置类,用于设置参数范围和目标
- run_grid_search: 运行网格搜索优化函数

推荐用法:
    from apilot.optimizer import OptimizationSetting, run_grid_search

    # 创建优化设置
    setting = OptimizationSetting()
    setting.add_parameter("atr_length", 10, 30, 5)
    setting.add_parameter("stop_multiplier", 2.0, 5.0, 1.0)
    setting.set_target("total_return")  # 优化总回报率

    # 运行优化
    results = run_grid_search(strategy_class, setting, key_func)
"""

# 定义公共API
__all__ = [
    "OptimizationSetting",
    "run_grid_search",
]

from .gridoptimizer import run_grid_search
from .settings import OptimizationSetting
