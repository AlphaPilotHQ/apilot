"""
Core Module

Contains basic components and data structures for the quantitative trading platform.

Recommended imports:
from apilot.core import BarData, OrderData  # Regular use (recommended)
import apilot.core as apcore  # For using many components
"""

# Define public API
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

# Import constant definitions
from apilot.utils.indicators import ArrayManager

from .constant import (
    Direction,  # type: Enum
    Exchange,  # type: Enum
    Interval,  # type: Enum
    OrderType,  # type: Enum
    Product,  # type: Enum
    Status,  # type: Enum
)

# Import database interfaces
from .database import (
    BarOverview,  # type: class
    BaseDatabase,  # type: class
    TickOverview,  # type: class
    use_database,  # type: function
)

# Import engine components
from .engine import BaseEngine, MainEngine  # type: class, class

# Import event-related components
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

# Import gateway interfaces
from .gateway import BaseGateway  # type: class

# Import core data objects
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

# Import configuration and utility functions
from .utility import BarGenerator
