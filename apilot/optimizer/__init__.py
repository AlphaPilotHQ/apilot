"""
Optimization Module

Provides strategy parameter optimization using grid search.

Main components:
- OptimizationSetting: Optimization configuration class for setting parameter ranges and targets
- run_grid_search: Function to run grid search optimization

Recommended usage:
    from apilot.optimizer import OptimizationSetting, run_grid_search

    # Create optimization settings
    setting = OptimizationSetting()
    setting.add_parameter("atr_length", 10, 30, 5)
    setting.add_parameter("stop_multiplier", 2.0, 5.0, 1.0)
    setting.set_target("total_return")  # Optimize total return

    # Run optimization
    results = run_grid_search(strategy_class, setting, key_func)
"""

# Define public API
__all__ = [
    "OptimizationSetting",
    "run_grid_search",
]

from .gridoptimizer import run_grid_search
from .settings import OptimizationSetting
