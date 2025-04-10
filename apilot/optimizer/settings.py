"""
参数优化设置模块

提供参数空间的定义和管理功能，用于策略参数优化。
"""

import logging
from itertools import product

# 获取日志记录器
logger = logging.getLogger("Optimizer")


class OptimizationSetting:
    """
    优化参数设置类

    用于定义参数空间和生成参数组合的工具类。支持离散型参数和连续型参数。
    """

    def __init__(self) -> None:
        """初始化优化设置"""
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
        添加优化参数

        可以添加两种类型的参数:
        1. 离散型: 直接添加具体数值列表 [1, 2, 3, 4, 5]
        2. 连续型: 给定 start/end/step, 生成均匀分布的数值

        Args:
            name: 参数名称
            start: 参数起始值或离散列表
            end: 参数结束值（连续型）
            step: 参数步长（连续型）

        Returns:
            (成功标志, 错误信息)
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
        设置优化目标

        Args:
            target_name: 优化目标指标名称（如'total_return', 'sharpe_ratio'等）
        """
        self.target_name = target_name

    def generate_setting(self) -> list[dict]:
        """
        生成所有参数组合

        Returns:
            包含所有参数组合的字典列表
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
        检查优化参数是否有效

        Returns:
            (有效标志, 错误信息)
        """
        if not self.params:
            return False, "优化参数为空"

        params_range = False
        for value in self.params.values():
            if len(value) > 1:
                params_range = True
                break

        if not params_range:
            return False, "所有参数都是固定值,无法进行优化"

        return True, ""
