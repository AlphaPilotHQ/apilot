"""
多网关并行工作集成测试
测试多个交易接口并行工作的场景
"""
import pytest
from datetime import datetime

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import SubscribeRequest, OrderRequest
from vnpy.trader.constant import Direction, Offset, OrderType, Exchange

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
