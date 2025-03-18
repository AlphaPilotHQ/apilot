from abc import ABC, abstractmethod
from datetime import datetime
from types import ModuleType
from typing import List, Optional, Dict, Any, Type
from dataclasses import dataclass
from importlib import import_module
import sys
import os

from .constant import Exchange, Interval
from .object import BarData, TickData
from .setting import SETTINGS


@dataclass
class BarOverview:
    """
    Overview of bar data stored in database.
    """

    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    start: int = None  # 毫秒级时间戳
    end: int = None    # 毫秒级时间戳


@dataclass
class TickOverview:
    """
    Overview of tick data stored in database.
    """

    symbol: str = ""
    exchange: Exchange = None
    count: int = 0
    start: int = None  # 毫秒级时间戳
    end: int = None    # 毫秒级时间戳


class BaseDatabase(ABC):
    """
    Abstract database class for connecting to different database.
    """

    @abstractmethod
    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        pass

    @abstractmethod
    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        pass

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        pass

    @abstractmethod
    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        pass

    @abstractmethod
    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        pass

    @abstractmethod
    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        pass

    @abstractmethod
    def get_bar_overview(self) -> List[BarOverview]:
        """
        Return bar data avaible in database.
        """
        pass

    @abstractmethod
    def get_tick_overview(self) -> List[TickOverview]:
        """
        Return tick data avaible in database.
        """
        pass


database: BaseDatabase = None


def get_database() -> BaseDatabase:
    """
    返回数据库对象。
    默认使用CSV数据库，其他数据库实现需要用户自行开发。
    """
    # 如果数据库已初始化，直接返回
    global database
    if database:
        return database

    # 默认使用CSV数据库
    from vnpy.database.csv.csv_database import CsvDatabase
    database = CsvDatabase()
    return database
