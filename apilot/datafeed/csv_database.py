"""
CSV文件数据库兼容模块

这是一个向后兼容的模块,重定向到新的providers.csv_provider模块
"""

# 从新位置导入
from apilot.core.database import register_database
from apilot.datafeed.providers.csv_provider import CsvDatabase

# 重新注册CSV数据库(确保向后兼容)
register_database("csv", CsvDatabase)
