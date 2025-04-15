"""
APilot quantitative trading platform constant definitions.

This module contains all enumerations used throughout the platform.
"""

from enum import Enum


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Status(Enum):
    SUBMITTING = "SUBMITTING"
    NOTTRADED = "NOT_TRADED"
    PARTTRADED = "PARTIALLY_TRADED"
    ALLTRADED = "FULLY_TRADED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Product(Enum):
    SPOT = "SPOT"
    FUTURES = "FUTURES"
    MARGIN = "MARGIN"
    OPTION = "OPTION"


class OrderType(Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    BRACKET = "BRACKET"  # market stop order SL, limit stop order TP


class Interval(Enum):
    MINUTE = "1m"
    MINUTE5 = "5m"
    MINUTE15 = "15m"
    HOUR = "1h"
    HOUR4 = "4h"
    DAILY = "d"
    WEEKLY = "w"


class EngineType(Enum):
    LIVE = "LIVE"
    BACKTESTING = "BACKTESTING"


# Event types
EVENT_TIMER = "EVENT_TIMER"
EVENT_TRADE = "EVENT_TRADE"
EVENT_ORDER = "EVENT_ORDER"
EVENT_POSITION = "EVENT_POSITION"
EVENT_ACCOUNT = "EVENT_ACCOUNT"
EVENT_QUOTE = "EVENT_QUOTE"
EVENT_CONTRACT = "EVENT_CONTRACT"
