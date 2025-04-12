"""
数据源模块

包含数据库接口和数据源实现, 用于存储和加载行情数据.
"""

from apilot.core.database import BaseDatabase

from .providers.csv_provider import CsvDatabase

DATA_PROVIDERS = {}


def register_provider(name, provider_class):
    DATA_PROVIDERS[name] = provider_class


try:
    from .providers.csv_provider import CsvDatabase

    register_provider("csv", CsvDatabase)
except ImportError:
    pass

# MongoDB提供者已移至examples目录，不再自动注册
# 如需使用，请参考examples/data_providers/mongodb_provider.py

# 定义公共API
__all__ = [
    "DATA_PROVIDERS",
    "CsvDatabase",
    "register_provider",
]
