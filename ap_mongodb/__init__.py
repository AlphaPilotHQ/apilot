"""
MongoDB数据库插件包，用于APilot量化交易框架。
"""

from .mongodb_database import MongodbDatabase
from apilot.trader.database import register_database

# 自动注册MongoDB数据库实现
register_database("mongodb", MongodbDatabase)