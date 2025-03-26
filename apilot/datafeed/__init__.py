"""
数据源模块

包含数据库接口和CSV数据源实现，用于存储和加载行情数据。

主要组件:
- CsvDatabase: 基于CSV文件的数据库实现
- get_database: 获取已配置的数据库实例

推荐用法:
    from apilot.datafeed import get_database
    db = get_database()
    bars = db.load_bar_data(...)
"""

from __future__ import annotations

# 定义公共API
__all__ = [
    # 数据库工厂函数
    "get_database",
    
    # 具体数据库实现
    "CsvDatabase"
]

# 从核心模块导入数据库工厂函数
from apilot.core.database import get_database

# 导入具体数据库实现
from .csv_database import CsvDatabase
