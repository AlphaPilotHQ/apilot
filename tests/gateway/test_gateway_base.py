"""
网关基类(BaseGateway)单元测试
"""
import pytest

from vnpy.trader.gateway import BaseGateway
from vnpy.event import EventEngine
from vnpy.trader.object import SubscribeRequest, OrderRequest, CancelRequest

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
