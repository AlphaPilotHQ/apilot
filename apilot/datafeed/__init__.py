"""
数据源模块

包含数据库接口和CSV数据源实现，用于存储和加载行情数据。

主要组件:
- BaseDatabase: 数据库接口抽象基类
- CsvDatabase: 基于CSV文件的数据库实现
- get_database: 获取已配置的数据库实例
- register_database: 注册自定义数据库实现

推荐用法:
    from apilot.datafeed import get_database
    db = get_database()
    bars = db.load_bar_data(...)
"""

from __future__ import annotations

# 定义公共API
__all__ = [
    # 数据库基类和工厂函数
    "BaseDatabase",
    "register_database",
    "get_database",
    
    # 数据概览类
    "BarOverview",
    "TickOverview",
    
    # 具体数据库实现
    "CsvDatabase"
]

# 从模块导入组件
from .database import (
    BaseDatabase,
    register_database, 
    get_database,
    BarOverview,
    TickOverview
)
from .csv_database import CsvDatabase
