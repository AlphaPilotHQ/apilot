"""
数据库管理器(DatabaseManager)单元测试
"""
import pytest
from datetime import datetime, timedelta

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import BaseDatabase, get_database_manager
from vnpy.trader.object import BarData, TickData

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
