"""
数据提供者模块

包含所有数据库提供者的实现, 负责数据存储和加载.
"""

# 导入CSV数据库提供者
from .csv_provider import CsvDatabase

# 尝试导入MongoDB数据库提供者(可选依赖)
try:
    from .mongodb_provider import MongoDBDatabase

    _HAS_MONGODB = True
except ImportError:
    _HAS_MONGODB = False

# 根据可用情况注册数据库提供者
from apilot.core.database import register_database

# 默认注册CSV提供者
register_database("csv", CsvDatabase)

__all__ = ["CsvDatabase"]

# 如果MongoDB可用, 则添加到公开API中
if _HAS_MONGODB:
    __all__.append("MongoDBDatabase")
    register_database("mongodb", MongoDBDatabase)
