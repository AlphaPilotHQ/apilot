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
    "EngineType",
    "Event",
    "EventEngine",
    "Interval",
    "LogData",
    "MainEngine",
    "OrderData",
    "OrderRequest",
    "OrderType",
    "PositionData",
    "Product",
    "QuoteData",
    "round_to",
    "Status",
    "SubscribeRequest",
    "TradeData",
    "get_database",
]


from .constant import (
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_QUOTE,
    EVENT_TRADE,
    Direction,
    EngineType,
    Interval,
    OrderType,
    Product,
    Status,
)

# Import event-related components
from .event import (
    Event,
    EventEngine,
)

# Import core data objects
from .models import (
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    HistoryRequest,
    OrderData,
    OrderRequest,
    PositionData,
    QuoteData,
    SubscribeRequest,
    TradeData,
)
