"""
数据源模块

包含数据库接口和CSV数据源实现, 用于存储和加载行情数据.

主要组件:
- CsvDatabase: 基于CSV文件的数据库实现
- DataManager: 数据管理类, 负责数据加载操作
- get_database: 获取已配置的数据库实例
"""

from datetime import datetime
from typing import Any, Optional

from apilot.core.database import (
    BaseDatabase,
    get_database,
    register_database,
)
from apilot.core.object import BarData, TickData

# 从配置模块导入
from .config import CsvSourceConfig, MongoSourceConfig

# 定义公共API
__all__ = [
    "CsvDatabase",
    "DataManager",
    "create_csv_data",
    "create_mongodb_data",
    "get_database",
]

# 从核心模块导入数据库工厂函数
from .data_manager import DataManager

# 导入CSV数据库提供者
from .providers.csv_provider import CsvDatabase

# 尝试导入MongoDB数据库提供者(可选依赖)
try:
    from .providers.mongodb_provider import MongoDBDatabase

    __all__.append("MongoDBDatabase")
except ImportError:
    pass


def create_csv_data(
    symbol: str,
    dataname: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    datetime_index: int = 0,
    open_index: int = 1,
    high_index: int = 2,
    low_index: int = 3,
    close_index: int = 4,
    volume_index: int = 5,
    openinterest_index: int = -1,
    dtformat: str = "%Y-%m-%d %H:%M:%S",
    use_column_names: bool = False,
    datetime_field: str = "datetime",
    open_field: str = "open",
    high_field: str = "high",
    low_field: str = "low",
    close_field: str = "close",
    volume_field: str = "volume",
    extra_params: dict[str, Any] | None = None,
) -> CsvSourceConfig:
    """
    Create CSV data source configuration

    Args:
        symbol: Trading symbol (e.g. "BTC-USDT.BINANCE")
        dataname: Path to CSV file
        start_date: Start date for data filtering
        end_date: End date for data filtering
        datetime_index: Column index for datetime (when using index mode)
        open_index: Column index for open price (when using index mode)
        high_index: Column index for high price (when using index mode)
        low_index: Column index for low price (when using index mode)
        close_index: Column index for close price (when using index mode)
        volume_index: Column index for volume (when using index mode)
        openinterest_index: Column index for open interest (when using index mode)
        dtformat: Datetime format string
        use_column_names: Whether to use column names instead of indices
        datetime_field: Column name for datetime (when using column name mode)
        open_field: Column name for open price (when using column name mode)
        high_field: Column name for high price (when using column name mode)
        low_field: Column name for low price (when using column name mode)
        close_field: Column name for close price (when using column name mode)
        volume_field: Column name for volume (when using column name mode)
        extra_params: Additional parameters for the data source

    Returns:
        CsvSourceConfig: Configuration object for CSV data source
    """
    if extra_params is None:
        extra_params = {}

    return CsvSourceConfig(
        symbol=symbol,
        dataname=dataname,
        start_date=start_date,
        end_date=end_date,
        datetime_index=datetime_index,
        open_index=open_index,
        high_index=high_index,
        low_index=low_index,
        close_index=close_index,
        volume_index=volume_index,
        openinterest_index=openinterest_index,
        dtformat=dtformat,
        use_column_names=use_column_names,
        datetime_field=datetime_field,
        open_field=open_field,
        high_field=high_field,
        low_field=low_field,
        close_field=close_field,
        volume_field=volume_field,
        extra_params=extra_params,
    )


def create_mongodb_data(
    symbol: str,
    database: str,
    collection: str,
    host: str = "localhost",
    port: int = 27017,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    query_filter: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
) -> MongoSourceConfig:
    """
    Create MongoDB data source configuration

    Args:
        symbol: Trading symbol (e.g. "BTC-USDT.BINANCE")
        database: MongoDB database name
        collection: MongoDB collection name
        host: MongoDB server hostname
        port: MongoDB server port
        start_date: Start date for data filtering
        end_date: End date for data filtering
        query_filter: Additional query filters for MongoDB
        extra_params: Additional parameters for the data source

    Returns:
        MongoSourceConfig: Configuration object for MongoDB data source
    """
    if query_filter is None:
        query_filter = {}

    if extra_params is None:
        extra_params = {}

    return MongoSourceConfig(
        symbol=symbol,
        database=database,
        collection=collection,
        host=host,
        port=port,
        start_date=start_date,
        end_date=end_date,
        query_filter=query_filter,
        extra_params=extra_params,
    )
