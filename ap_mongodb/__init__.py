from vnpy.trader.database import register_database
from .mongodb_database import MongodbDatabase

# 自动注册MongoDB数据库实现
register_database("mongodb", MongodbDatabase)