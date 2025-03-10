"""
CTA策略引擎单元测试
"""
import pytest
from datetime import datetime

from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy_ctastrategy import CtaStrategyApp
from vnpy_ctastrategy.engine import CtaEngine

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
