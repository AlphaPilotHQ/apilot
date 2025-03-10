"""
数据库模块
"""

import importlib

# 导出MongoDB数据库接口
try:
    from .mongodb import Database as MongodbDatabase
except ImportError:
    MongodbDatabase = None
