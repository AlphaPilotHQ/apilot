"""
数据源模块

包含数据库接口和CSV数据源实现, 用于存储和加载行情数据.

主要组件:
- CsvDatabase: 基于CSV文件的数据库实现
- DataManager: 数据管理类, 负责数据加载操作
- get_database: 获取已配置的数据库实例
"""

from apilot.core.database import (
    BaseDatabase,
    database_registry,
    get_database,
    register_database,
)
from apilot.core.object import BarData, TickData

# 定义公共API
__all__ = [
    "CsvDatabase",
    "DataManager",
    "get_database",
]

# 从核心模块导入数据库工厂函数
from .data_manager import DataManager

# 导入CSV数据库提供者
from .providers.csv_provider import CsvDatabase

# 尝试导入MongoDB数据库提供者(可选依赖)
try:
    from .providers.mongodb_provider import MongoDBDatabase

    __all__.append("MongoDBDatabase")
except ImportError:
    pass
