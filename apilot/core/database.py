"""
数据库抽象接口和实现

定义了行情数据存储的通用接口和工厂方法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from .constant import Exchange, Interval
from .object import BarData, TickData

# Database default configuration
DATABASE_CONFIG = {
    "name": "csv",  # Default to CSV database for direct file reading
    "data_path": "data",  # CSV data file storage path
}


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
    """
    抽象基类,定义了数据库接口的标准方法
    """

    @abstractmethod
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        pass

    @abstractmethod
    def load_tick_data(
        self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> list[TickData]:
        pass

    @abstractmethod
    def delete_bar_data(
        self, symbol: str, exchange: Exchange, interval: Interval
    ) -> int:
        pass

    @abstractmethod
    def delete_tick_data(self, symbol: str, exchange: Exchange) -> int:
        pass

    @abstractmethod
    def get_bar_overview(self) -> list[BarOverview]:
        pass

    @abstractmethod
    def get_tick_overview(self) -> list[TickOverview]:
        pass


# 全局数据库实例
database: BaseDatabase | None = None

# 数据库插件注册表
_DATABASE_REGISTRY: dict[str, type[BaseDatabase]] = {}


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
    获取数据库对象,如果未初始化则进行初始化.
    """
    global database

    if database:
        return database

    # 从设置中读取数据库类型
    database_name = DATABASE_CONFIG.get("name", "")
    database_params = DATABASE_CONFIG.get("params", {})

    # 如果配置了数据库,则初始化
    if database_name:
        try:
            database = use_database(database_name, **database_params)
            return database
        except Exception as err:
            raise ValueError(f"初始化数据库 {database_name} 失败") from err
    else:
        raise ValueError("没有可用的数据库实现") from None
