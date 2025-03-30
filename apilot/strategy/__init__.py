"""
策略模块

包含各种交易策略的模板和实现,支持PA策略和目标持仓策略.

主要组件:
- PATemplate: PA策略基类,提供标准的策略框架

"""

# 从模板导入基类
from .template import PATemplate

# 定义公共API
__all__ = [
    "PATemplate",
]
