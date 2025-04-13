"""
Parameter optimization settings module

Provides parameter space definition and management for strategy optimization.
"""

import logging
from itertools import product

# Get logger
logger = logging.getLogger("Optimizer")


class OptimizationSetting:
    """
    Optimization parameter settings class

    Utility class for defining parameter space and generating parameter combinations.
    Supports both discrete and continuous parameters.
    """

    def __init__(self) -> None:
        """Initialize optimization settings"""
        self.params: dict[str, list] = {}
        self.target_name: str = ""

    def add_parameter(
        self,
        name: str,
        start: float,
        end: float | None = None,
        step: float | None = None,
    ) -> tuple[bool, str]:
        """
        Add optimization parameter

        Two types of parameters can be added:
        1. Discrete: Directly add a specific value list [1, 2, 3, 4, 5]
        2. Continuous: Given start/end/step, generate uniformly distributed values

        Args:
            name: Parameter name
            start: Parameter start value or discrete list
            end: Parameter end value (for continuous)
            step: Parameter step (for continuous)

        Returns:
            (success flag, error message)
        """
        try:
            if end is None or step is None:
                if isinstance(start, list):
                    self.params[name] = start
                else:
                    self.params[name] = [start]
            else:
                value = start
                value_list = []

                while value <= end:
                    value_list.append(value)
                    value += step

                self.params[name] = value_list

            return True, ""
        except Exception as e:
            return False, str(e)

    def set_target(self, target_name: str) -> None:
        """
        Set optimization target

        Args:
            target_name: Name of the optimization target metric (e.g. 'total_return', 'sharpe_ratio', etc.)
        """
        self.target_name = target_name

    def generate_setting(self) -> list[dict]:
        """
        Generate all parameter combinations

        Returns:
            List of dictionaries containing all parameter combinations
        """
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p, strict=False))
            settings.append(setting)

        return settings

    def check_setting(self) -> tuple[bool, str]:
        """
        Check if optimization parameters are valid

        Returns:
            (valid flag, error message)
        """
        if not self.params:
            return False, "Optimization parameters are empty"

        params_range = False
        for value in self.params.values():
            if len(value) > 1:
                params_range = True
                break

        if not params_range:
            return (
                False,
                "All parameters are fixed values, optimization is not possible",
            )

        return True, ""
