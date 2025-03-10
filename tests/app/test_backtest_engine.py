"""
回测引擎(BacktestingEngine)单元测试
"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from vnpy.trader.constant import Interval, Direction, Offset, Exchange
from vnpy.trader.object import BarData
from vnpy_ctastrategy.backtesting import BacktestingEngine

# 测试函数将在此处实现

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
