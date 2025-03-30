"""
数据源配置类

定义不同数据源的配置类, 作为用户API和底层实现之间的桥梁。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DataSourceConfig(Protocol):
    """
    数据源配置接口协议

    所有数据源配置类应该实现这个协议, 提供基本的数据源属性。
    该协议是隐式实现的, 任何具有这些属性的类都可以被视为DataSourceConfig。
    """

    symbol: str  # 交易符号标识符
    start_date: datetime | None  # 开始日期（可选）
    end_date: datetime | None  # 结束日期（可选）
    extra_params: dict[str, Any]  # 额外参数


@dataclass
class CsvSourceConfig:
    """CSV数据源配置"""

    # 所有必需参数放在前面
    symbol: str  # 交易符号标识符
    dataname: str  # CSV文件路径
    # 可选参数放在必需参数之后
    start_date: datetime | None = None
    end_date: datetime | None = None
    datetime_index: int = 0
    open_index: int = 1
    high_index: int = 2
    low_index: int = 3
    close_index: int = 4
    volume_index: int = 5
    openinterest_index: int = -1
    dtformat: str = "%Y-%m-%d %H:%M:%S"
    use_column_names: bool = False
    datetime_field: str = "datetime"
    open_field: str = "open"
    high_field: str = "high"
    low_field: str = "low"
    close_field: str = "close"
    volume_field: str = "volume"
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MongoSourceConfig:
    """MongoDB数据源配置"""

    # 所有必需参数放在前面
    symbol: str  # 交易符号标识符
    database: str
    collection: str
    # 可选参数放在必需参数之后
    start_date: datetime | None = None
    end_date: datetime | None = None
    host: str = "localhost"
    port: int = 27017
    query_filter: dict[str, Any] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)
