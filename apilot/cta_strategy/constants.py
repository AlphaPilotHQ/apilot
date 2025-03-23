from datetime import datetime, timedelta
from enum import Enum
from typing import Dict

from apilot.trader.constant import Interval, Direction, Offset


# 引擎类型
class EngineType(Enum):
    LIVE = "实盘"
    BACKTESTING = "回测"


# 回测模式
class BacktestingMode(Enum):
    BAR = 1
    TICK = 2


# CTA策略事件
EVENT_CTA_LOG: str = "EVENT_CTA_LOG"
EVENT_CTA_STRATEGY = "EVENT_CTA_STRATEGY"

# 时间间隔映射
INTERVAL_DELTA_MAP: Dict[Interval, timedelta] = {
    Interval.TICK: timedelta(milliseconds=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


# 常量名称
APP_NAME = "CtaStrategy"
