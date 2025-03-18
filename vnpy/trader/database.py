from abc import ABC, abstractmethod
from datetime import datetime
from types import ModuleType
from typing import List, Optional
from dataclasses import dataclass
from importlib import import_module
import sys
import os

from .constant import Exchange, Interval
from .object import BarData, TickData
from .setting import SETTINGS
from .utility import ZoneInfo


# Get local timezone
LOCAL_TZ = ZoneInfo("Asia/Shanghai")

# DB_TZ is the timezone setting in database
DB_TZ = ZoneInfo(SETTINGS["database.timezone"])


def convert_tz(dt: datetime) -> datetime:
    """
    Convert timezone of datetime object to DB_TZ.
    """
    dt: datetime = dt.astimezone(DB_TZ)
    return dt.replace(tzinfo=None)


@dataclass
class BarOverview:
    """
    Overview of bar data stored in database.
    """

    symbol: str = ""
    exchange: Exchange = None
    interval: Interval = None
    count: int = 0
    start: datetime = None
    end: datetime = None


@dataclass
class TickOverview:
    """
    Overview of tick data stored in database.
    """

    symbol: str = ""
    exchange: Exchange = None
    count: int = 0
    start: datetime = None
    end: datetime = None


class BaseDatabase(ABC):
    """
    Abstract database class for connecting to different database.
    """

    @abstractmethod
    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """
        Save bar data into database.
        """
        pass

    @abstractmethod
    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """
        Save tick data into database.
        """
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
        """
        Load bar data from database.
        """
        pass

    @abstractmethod
    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """
        Load tick data from database.
        """
        pass

    @abstractmethod
    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """
        Delete all bar data with given symbol + exchange + interval.
        """
        pass

    @abstractmethod
    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """
        Delete all tick data with given symbol + exchange.
        """
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
    Return database object if inited.
    """
    # Return database object if already inited
    global database
    if database:
        return database

    # Read database related global setting
    database_name: str = SETTINGS["database.name"]

    # Use CSV database by default
    if database_name == "csv":
        from vnpy.database.csv.csv_database import CsvDatabase
        database = CsvDatabase()
        return database
    # Use MongoDB if selected
    elif database_name == "mongodb":
        from vnpy.database.mongodb.mongodb_database import MongodbDatabase
        database = MongodbDatabase()
        return database
    # Add other database implementations here
    else:
        # 尝试导入其他数据库模块
        try:
            module_name: str = f"vnpy_{database_name}"
            from importlib import import_module
            module = import_module(module_name)
            database = module.Database()
            return database
        except ModuleNotFoundError:
            print(f"找不到数据库驱动{module_name}，使用默认的CSV数据库")
            from vnpy.database.csv.csv_database import CsvDatabase
            database = CsvDatabase()
            return database
