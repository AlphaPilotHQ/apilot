"""
Database abstract interface and implementation

Defines common interfaces and factory methods for market data storage
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .constant import Interval
from .object import BarData, TickData


@dataclass
class BarOverview:
    """
    Overview of bar data stored in database.
    """

    symbol: str = ""
    interval: Interval = None
    count: int = 0
    start: int = None
    end: int = None


@dataclass
class TickOverview:
    """
    Overview of tick data stored in database.
    """

    symbol: str = ""
    count: int = 0
    start: int = None
    end: int = None


class BaseDatabase(ABC):
    """
    Abstract base class defining standard methods for database interface
    """

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        """Abstract method to load bar data"""
        pass

    def load_tick_data(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[TickData]:
        """Load tick data (optional)"""
        return []

    def delete_bar_data(self, symbol: str, interval: Interval) -> int:
        """Delete bar data (optional)"""
        return 0

    def delete_tick_data(self, symbol: str) -> int:
        """Delete tick data (optional)"""
        return 0

    def get_bar_overview(self) -> list[BarOverview]:
        """Get bar data overview (optional)"""
        return []

    def get_tick_overview(self) -> list[TickOverview]:
        """Get tick data overview (optional)"""
        return []


# Internal database implementation registry
_DATABASE_REGISTRY: dict[str, type[BaseDatabase]] = {}

# Configure database
DATABASE_CONFIG: dict[str, Any] = {"name": "", "params": {}}


def register_database(name: str, database_class: type) -> None:
    """Register custom database implementation"""
    _DATABASE_REGISTRY[name] = database_class


def use_database(name: str, **kwargs) -> BaseDatabase:
    """Use specified database implementation"""
    if name in _DATABASE_REGISTRY:
        database_class = _DATABASE_REGISTRY[name]
        return database_class(**kwargs)
    else:
        raise ValueError(f"Database implementation not found: {name}")
