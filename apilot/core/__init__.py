"""
核心模块

包含量化交易平台的基础组件和数据结构.

推荐导入:
from apilot.core import BarData, OrderData  # 常规使用(推荐)
import apilot.core as apcore  # 大量组件使用
"""

# 定义公共API
__all__ = [
    "EVENT_ACCOUNT",
    "EVENT_CONTRACT",
    "EVENT_ORDER",
    "EVENT_POSITION",
    "EVENT_QUOTE",
    "EVENT_TICK",
    "EVENT_TIMER",
    "EVENT_TRADE",
    "AccountData",
    "ArrayManager",
    "BarData",
    "BarGenerator",
    "BarOverview",
    "BaseDatabase",
    "BaseEngine",
    "BaseGateway",
    "CancelRequest",
    "ContractData",
    "Direction",
    "Event",
    "EventEngine",
    "Exchange",
    "Interval",
    "LogData",
    "MainEngine",
    "OrderData",
    "OrderRequest",
    "OrderType",
    "PositionData",
    "Product",
    "QuoteData",
    "Status",
    "SubscribeRequest",
    "TickData",
    "TickOverview",
    "TradeData",
    "get_database",
    "use_database",
]

# 导入常量定义
from apilot.utils.indicators import ArrayManager

from .constant import (
    Direction,  # type: Enum
    Exchange,  # type: Enum
    Interval,  # type: Enum
    OrderType,  # type: Enum
    Product,  # type: Enum
    Status,  # type: Enum
)

# 导入数据库接口
from .database import (
    BarOverview,  # type: class
    BaseDatabase,  # type: class
    TickOverview,  # type: class
    use_database,  # type: function
)

# 导入引擎组件
from .engine import BaseEngine, MainEngine  # type: class, class

# 导入事件相关组件
from .event import (
    EVENT_ACCOUNT,  # type: str
    EVENT_CONTRACT,  # type: str
    EVENT_ORDER,  # type: str
    EVENT_POSITION,  # type: str
    EVENT_QUOTE,
    EVENT_TICK,  # type: str
    EVENT_TIMER,  # type: str
    EVENT_TRADE,  # type: str
    Event,  # type: class
    EventEngine,  # type: class
)

# 导入网关接口
from .gateway import BaseGateway  # type: class

# 导入核心数据对象
from .object import (
    AccountData,  # type: class
    BarData,  # type: class
    CancelRequest,  # type: class
    ContractData,  # type: class
    LogData,  # type: class
    OrderData,  # type: class
    OrderRequest,  # type: class
    PositionData,  # type: class
    QuoteData,
    SubscribeRequest,  # type: class
    TickData,  # type: class
    TradeData,  # type: class
)

# 导入配置和工具函数
from .utility import BarGenerator
