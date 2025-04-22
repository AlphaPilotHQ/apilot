"""
Database abstract interface and implementation

Defines common interfaces and factory methods for market data storage
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from apilot.core.constant import Interval
from apilot.core.models import BarData


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

    def delete_bar_data(self, symbol: str, interval: Interval) -> int:
        """Delete bar data (optional)"""
        return 0

    def get_bar_overview(self) -> list[BarOverview]:
        """Get bar data overview (optional)"""
        return []


# Internal database implementation registry
_DATABASE_REGISTRY: dict[str, type[BaseDatabase]] = {}

# Configure database
DATABASE_CONFIG: dict[str, Any] = {"name": "", "params": {}}


def register_database(name: str, database_class: type) -> None:
    """Register custom database implementation"""
    _DATABASE_REGISTRY[name] = database_class
