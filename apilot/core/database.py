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
    start: int = None
    end: int = None


@dataclass
class TickOverview:
    """
    Overview of tick data stored in database.
    """
    symbol: str = ""
    exchange: Exchange = None
    count: int = 0
    start: int = None
    end: int = None


class BaseDatabase(ABC):

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
        pass

    @abstractmethod
    def get_tick_overview(self) -> List[TickOverview]:

        pass

# 全局数据库实例
database: BaseDatabase = None

# 数据库插件注册表
_DATABASE_REGISTRY: Dict[str, Type[BaseDatabase]] = {}

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

def get_database() -> BaseDatabase:
    """
    获取数据库对象，如果未初始化则进行初始化。
    """
    global database

    if database:
        return database

    # 从设置中读取数据库类型
    database_name = SETTINGS.get("database.name", "")

    # 提取对应数据库类型的参数
    database_params = {}
    prefix = f"database.{database_name}."
    for key, value in SETTINGS.items():
        if key.startswith(prefix):
            param_name = key[len(prefix):]
            database_params[param_name] = value

    try:
        database = use_database(database_name, **database_params)
        return database
    except Exception as e:
        # 默认使用已注册的数据库，或者抛出异常
        if _DATABASE_REGISTRY:
            first_db_name = next(iter(_DATABASE_REGISTRY))
            database_class = _DATABASE_REGISTRY[first_db_name]
            database = database_class()
            return database
        else:
            raise ValueError("没有可用的数据库实现")
