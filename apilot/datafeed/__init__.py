"""
数据源模块

包含数据库接口和数据源实现, 用于存储和加载行情数据.
"""

from apilot.core.database import BaseDatabase

from .providers.csv_provider import CsvDatabase

# 数据提供者注册表
DATA_PROVIDERS = {}


def register_provider(name, provider_class):
    """注册数据提供者"""
    DATA_PROVIDERS[name] = provider_class


register_provider("csv", CsvDatabase)

# 尝试导入MongoDB数据库提供者(可选依赖)
try:
    from .providers.mongodb_provider import MongoDBDatabase

    register_provider("mongodb", MongoDBDatabase)
except ImportError:
    pass

# 定义公共API
__all__ = [
    "DATA_PROVIDERS",
    "CsvDatabase",
    "register_provider",
]
