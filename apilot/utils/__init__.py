"""
工具函数模块

包含通用辅助函数，用于数据处理、计算和其他常用操作。

本模块将包含:
- 数学计算工具: 如夏普比率、最大回撤等指标计算
- 日期处理工具: 时间转换、回测区间划分等
- 数据处理工具: 数据格式转换、滤波等
- 可视化工具: 绘制K线、收益曲线等
- 订单管理工具: 本地订单ID管理等

推荐用法:
    from apilot.utils import 特定工具函数
"""

# 定义公共API
__all__ = [
    # 订单管理
    "LocalOrderManager",
    
    # 其他工具函数会逐步添加
]

# 订单管理工具
from .order_manager import LocalOrderManager
