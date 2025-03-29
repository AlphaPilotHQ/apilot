"""
优化模块

提供策略参数优化功能,包括网格搜索和遗传算法优化.

主要组件:
- OptimizationSetting: 优化配置类,用于设置参数范围和目标
- run_ga_optimization: 运行遗传算法优化函数

推荐用法:
    from apilot.optimizer import OptimizationSetting, run_ga_optimization

    # 创建优化设置
    setting = OptimizationSetting()
    setting.add_parameter("atr_length", 10, 30, 5)
    setting.add_parameter("stop_multiplier", 2.0, 5.0, 1.0)
    setting.set_target("sharpe_ratio")  # 优化夏普比率

    # 运行优化
    results = run_ga_optimization(strategy_class, setting, ...)
"""

# 定义公共API
__all__ = [
    "OptimizationSetting",
    "check_optimization_setting",
    "ga_evaluate",
    "run_ga_optimization",
]

from .optimizer import (
    OptimizationSetting,
    check_optimization_setting,
    ga_evaluate,
    run_ga_optimization,
)
