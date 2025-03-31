"""
数据库抽象接口和实现

定义了行情数据存储的通用接口和工厂方法
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
    抽象基类,定义了数据库接口的标准方法
    """

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        pass

    @abstractmethod
    def load_tick_data(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[TickData]:
        pass

    @abstractmethod
    def delete_bar_data(self, symbol: str, interval: Interval) -> int:
        pass

    @abstractmethod
    def delete_tick_data(self, symbol: str) -> int:
        pass

    @abstractmethod
    def get_bar_overview(self) -> list[BarOverview]:
        pass

    @abstractmethod
    def get_tick_overview(self) -> list[TickOverview]:
        pass


# 内部使用的数据库实现注册表
_DATABASE_REGISTRY: dict[str, type[BaseDatabase]] = {}

# 配置数据库
DATABASE_CONFIG: dict[str, Any] = {"name": "", "params": {}}


def register_database(name: str, database_class: type) -> None:
    """注册自定义数据库实现"""
    _DATABASE_REGISTRY[name] = database_class


def use_database(name: str, **kwargs) -> BaseDatabase:
    """使用指定的数据库实现"""
    if name in _DATABASE_REGISTRY:
        database_class = _DATABASE_REGISTRY[name]
        return database_class(**kwargs)
    else:
        raise ValueError(f"未找到数据库实现: {name}")
