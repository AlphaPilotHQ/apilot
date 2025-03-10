"""
数据服务(Datafeed)单元测试
"""
import pytest
from typing import List, Optional
from datetime import datetime, timedelta

from vnpy.trader.object import HistoryRequest, BarData, TickData
from vnpy.trader.datafeed import BaseDatafeed, get_datafeed

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
