"""
MongoDB数据库提供者

实现了BaseDatabase接口的MongoDB数据提供者, 用于从MongoDB数据库加载和管理量化交易数据.
"""

from datetime import datetime

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database as MongoDatabase

from apilot.core.constant import Exchange, Interval
from apilot.core.database import BarOverview, BaseDatabase, TickOverview
from apilot.core.object import BarData, TickData
from apilot.utils.logger import get_logger

# 获取日志记录器
logger = get_logger("MongoDB")


class MongoDBDatabase(BaseDatabase):
    """
    MongoDB数据提供者

    实现了BaseDatabase接口, 支持从MongoDB读取和写入量化交易数据.

    特性:
    1. 自动管理数据库连接
    2. 支持K线和Tick数据的存储与查询
    3. 针对交易数据的时间序列优化

    用法示例:
    ```python
    # 创建MongoDB数据库实例
    db = MongoDBDatabase(host="localhost", port=27017, database="apilot")

    # 加载K线数据
    bars = db.load_bar_data(
        symbol="BTCUSDT",
        exchange="BINANCE",
        interval="1m",
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 31)
    )

    # 保存K线数据
    db.save_bar_data(bars)
    ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        username: str = "",
        password: str = "",
        database: str = "apilot",
        **kwargs,
    ):
        """
        初始化MongoDB数据库连接

        参数:
            host: MongoDB服务器地址
            port: MongoDB服务器端口
            username: 用户名(如需身份验证)
            password: 密码(如需身份验证)
            database: 数据库名称
        """
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password

        # 创建MongoDB客户端
        if username and password:
            self.client = MongoClient(
                host=host, port=port, username=username, password=password, **kwargs
            )
        else:
            self.client = MongoClient(host=host, port=port, **kwargs)

        # 获取数据库
        self.db: MongoDatabase = self.client[database]

        # 确保创建必要的索引
        self._create_indexes()

    def _create_indexes(self) -> None:
        """创建必要的索引以提高查询性能"""
        # 为K线数据集合创建索引
        bar_collection = self.db["bar_data"]
        bar_collection.create_index(
            [
                ("symbol", ASCENDING),
                ("exchange", ASCENDING),
                ("interval", ASCENDING),
                ("datetime", ASCENDING),
            ],
            unique=True,
        )

        # 为Tick数据集合创建索引
        tick_collection = self.db["tick_data"]
        tick_collection.create_index(
            [
                ("symbol", ASCENDING),
                ("exchange", ASCENDING),
                ("datetime", ASCENDING),
            ],
            unique=True,
        )

    def _get_bar_collection(self) -> Collection:
        """获取K线数据集合"""
        return self.db["bar_data"]

    def _get_tick_collection(self) -> Collection:
        """获取Tick数据集合"""
        return self.db["tick_data"]

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange | str,
        interval: Interval | str,
        start: datetime,
        end: datetime,
        **kwargs,
    ) -> list[BarData]:
        """
        从MongoDB加载K线数据

        参数:
            symbol: 交易对名称
            exchange: 交易所, 可以是Exchange枚举或字符串
            interval: K线周期, 可以是Interval枚举或字符串
            start: 开始时间
            end: 结束时间

        返回:
            BarData对象列表
        """
        # 转换参数类型
        if isinstance(exchange, str):
            exchange = Exchange(exchange)
        if isinstance(interval, str):
            interval = Interval(interval)

        # 获取集合
        collection = self._get_bar_collection()

        # 构建查询条件
        filter_dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
            "datetime": {
                "$gte": start,
                "$lte": end,
            },
        }

        # 执行查询
        cursor = collection.find(filter_dict).sort("datetime", ASCENDING)

        # 将文档转换为BarData对象
        bars: list[BarData] = []
        for doc in cursor:
            # 创建K线数据对象
            bar = BarData(
                symbol=doc["symbol"],
                exchange=Exchange(doc["exchange"]),
                datetime=doc["datetime"],
                interval=Interval(doc["interval"]),
                open_price=doc["open_price"],
                high_price=doc["high_price"],
                low_price=doc["low_price"],
                close_price=doc["close_price"],
                volume=doc["volume"],
                open_interest=doc.get("open_interest", 0.0),
                gateway_name=doc.get("gateway_name", "MongoDB"),
            )
            bars.append(bar)

        return bars

    def load_tick_data(
        self, symbol: str, exchange: Exchange | str, start: datetime, end: datetime
    ) -> list[TickData]:
        """
        从MongoDB加载Tick数据

        参数:
            symbol: 交易对名称
            exchange: 交易所, 可以是Exchange枚举或字符串
            start: 开始时间
            end: 结束时间

        返回:
            TickData对象列表
        """
        # 转换参数类型
        if isinstance(exchange, str):
            exchange = Exchange(exchange)

        # 获取集合
        collection = self._get_tick_collection()

        # 构建查询条件
        filter_dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "datetime": {
                "$gte": start,
                "$lte": end,
            },
        }

        # 执行查询
        cursor = collection.find(filter_dict).sort("datetime", ASCENDING)

        # 将文档转换为TickData对象
        ticks: list[TickData] = []
        for doc in cursor:
            # 创建Tick数据对象
            tick = TickData(
                symbol=doc["symbol"],
                exchange=Exchange(doc["exchange"]),
                datetime=doc["datetime"],
                name=doc.get("name", ""),
                volume=doc.get("volume", 0.0),
                last_price=doc.get("last_price", 0.0),
                last_volume=doc.get("last_volume", 0.0),
                limit_up=doc.get("limit_up", 0.0),
                limit_down=doc.get("limit_down", 0.0),
                open_price=doc.get("open_price", 0.0),
                high_price=doc.get("high_price", 0.0),
                low_price=doc.get("low_price", 0.0),
                pre_close=doc.get("pre_close", 0.0),
                bid_price_1=doc.get("bid_price_1", 0.0),
                bid_price_2=doc.get("bid_price_2", 0.0),
                bid_price_3=doc.get("bid_price_3", 0.0),
                bid_price_4=doc.get("bid_price_4", 0.0),
                bid_price_5=doc.get("bid_price_5", 0.0),
                ask_price_1=doc.get("ask_price_1", 0.0),
                ask_price_2=doc.get("ask_price_2", 0.0),
                ask_price_3=doc.get("ask_price_3", 0.0),
                ask_price_4=doc.get("ask_price_4", 0.0),
                ask_price_5=doc.get("ask_price_5", 0.0),
                bid_volume_1=doc.get("bid_volume_1", 0.0),
                bid_volume_2=doc.get("bid_volume_2", 0.0),
                bid_volume_3=doc.get("bid_volume_3", 0.0),
                bid_volume_4=doc.get("bid_volume_4", 0.0),
                bid_volume_5=doc.get("bid_volume_5", 0.0),
                ask_volume_1=doc.get("ask_volume_1", 0.0),
                ask_volume_2=doc.get("ask_volume_2", 0.0),
                ask_volume_3=doc.get("ask_volume_3", 0.0),
                ask_volume_4=doc.get("ask_volume_4", 0.0),
                ask_volume_5=doc.get("ask_volume_5", 0.0),
                gateway_name=doc.get("gateway_name", "MongoDB"),
            )
            ticks.append(tick)

        return ticks

    def save_bar_data(self, bars: list[BarData], overwrite: bool = False) -> int:
        """
        保存K线数据到MongoDB

        参数:
            bars: BarData列表
            overwrite: 是否覆盖已有数据

        返回:
            保存的数据条数
        """
        if not bars:
            return 0

        # 获取集合
        collection = self._get_bar_collection()
        count = 0

        for bar in bars:
            # 转换为文档
            doc = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "interval": bar.interval.value,
                "datetime": bar.datetime,
                "open_price": bar.open_price,
                "high_price": bar.high_price,
                "low_price": bar.low_price,
                "close_price": bar.close_price,
                "volume": bar.volume,
                "open_interest": bar.open_interest,
                "gateway_name": bar.gateway_name,
            }

            # 构建查询过滤器
            filter_dict = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "interval": bar.interval.value,
                "datetime": bar.datetime,
            }

            if overwrite:
                # 使用upsert模式, 存在则更新, 不存在则插入
                result = collection.replace_one(filter_dict, doc, upsert=True)
                if result.upserted_id or result.modified_count:
                    count += 1
            else:
                # 只在不存在时插入
                if not collection.find_one(filter_dict):
                    collection.insert_one(doc)
                    count += 1

        return count

    def save_tick_data(self, ticks: list[TickData], overwrite: bool = False) -> int:
        """
        保存Tick数据到MongoDB

        参数:
            ticks: TickData列表
            overwrite: 是否覆盖已有数据

        返回:
            保存的数据条数
        """
        if not ticks:
            return 0

        # 获取集合
        collection = self._get_tick_collection()
        count = 0

        for tick in ticks:
            # 转换为文档
            doc = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": tick.datetime,
                "name": tick.name,
                "volume": tick.volume,
                "last_price": tick.last_price,
                "last_volume": tick.last_volume,
                "limit_up": tick.limit_up,
                "limit_down": tick.limit_down,
                "open_price": tick.open_price,
                "high_price": tick.high_price,
                "low_price": tick.low_price,
                "pre_close": tick.pre_close,
                "bid_price_1": tick.bid_price_1,
                "bid_price_2": tick.bid_price_2,
                "bid_price_3": tick.bid_price_3,
                "bid_price_4": tick.bid_price_4,
                "bid_price_5": tick.bid_price_5,
                "ask_price_1": tick.ask_price_1,
                "ask_price_2": tick.ask_price_2,
                "ask_price_3": tick.ask_price_3,
                "ask_price_4": tick.ask_price_4,
                "ask_price_5": tick.ask_price_5,
                "bid_volume_1": tick.bid_volume_1,
                "bid_volume_2": tick.bid_volume_2,
                "bid_volume_3": tick.bid_volume_3,
                "bid_volume_4": tick.bid_volume_4,
                "bid_volume_5": tick.bid_volume_5,
                "ask_volume_1": tick.ask_volume_1,
                "ask_volume_2": tick.ask_volume_2,
                "ask_volume_3": tick.ask_volume_3,
                "ask_volume_4": tick.ask_volume_4,
                "ask_volume_5": tick.ask_volume_5,
                "gateway_name": tick.gateway_name,
            }

            # 构建查询过滤器
            filter_dict = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": tick.datetime,
            }

            if overwrite:
                # 使用upsert模式, 存在则更新, 不存在则插入
                result = collection.replace_one(filter_dict, doc, upsert=True)
                if result.upserted_id or result.modified_count:
                    count += 1
            else:
                # 只在不存在时插入
                if not collection.find_one(filter_dict):
                    collection.insert_one(doc)
                    count += 1

        return count

    def delete_bar_data(
        self, symbol: str, exchange: Exchange | str, interval: Interval | str
    ) -> int:
        """
        删除K线数据

        参数:
            symbol: 交易对名称
            exchange: 交易所
            interval: K线周期

        返回:
            删除的数据条数
        """
        # 转换参数类型
        if isinstance(exchange, str):
            exchange = Exchange(exchange)
        if isinstance(interval, str):
            interval = Interval(interval)

        # 获取集合
        collection = self._get_bar_collection()

        # 构建过滤条件
        filter_dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
        }

        # 执行删除
        result = collection.delete_many(filter_dict)
        return result.deleted_count

    def delete_tick_data(self, symbol: str, exchange: Exchange | str) -> int:
        """
        删除Tick数据

        参数:
            symbol: 交易对名称
            exchange: 交易所

        返回:
            删除的数据条数
        """
        # 转换参数类型
        if isinstance(exchange, str):
            exchange = Exchange(exchange)

        # 获取集合
        collection = self._get_tick_collection()

        # 构建过滤条件
        filter_dict = {"symbol": symbol, "exchange": exchange.value}

        # 执行删除
        result = collection.delete_many(filter_dict)
        return result.deleted_count

    def get_bar_overview(self) -> list[BarOverview]:
        """
        获取K线数据概览

        返回:
            BarOverview列表, 包含每个交易对-周期组合的数据统计
        """
        # 获取K线数据集合
        collection = self._get_bar_collection()

        # 使用聚合管道获取概览数据
        pipeline = [
            {
                "$group": {
                    "_id": {
                        "symbol": "$symbol",
                        "exchange": "$exchange",
                        "interval": "$interval",
                    },
                    "count": {"$sum": 1},
                    "start": {"$min": "$datetime"},
                    "end": {"$max": "$datetime"},
                }
            }
        ]

        # 执行聚合查询
        cursor = collection.aggregate(pipeline)

        # 转换结果
        overviews = []
        for doc in cursor:
            overview = BarOverview(
                symbol=doc["_id"]["symbol"],
                exchange=Exchange(doc["_id"]["exchange"]),
                interval=Interval(doc["_id"]["interval"]),
                count=doc["count"],
                start=doc["start"],
                end=doc["end"],
            )
            overviews.append(overview)

        return overviews

    def get_tick_overview(self) -> list[TickOverview]:
        """
        获取Tick数据概览

        返回:
            TickOverview列表, 包含每个交易对的数据统计
        """
        # 获取Tick数据集合
        collection = self._get_tick_collection()

        # 使用聚合管道获取概览数据
        pipeline = [
            {
                "$group": {
                    "_id": {"symbol": "$symbol", "exchange": "$exchange"},
                    "count": {"$sum": 1},
                    "start": {"$min": "$datetime"},
                    "end": {"$max": "$datetime"},
                }
            }
        ]

        # 执行聚合查询
        cursor = collection.aggregate(pipeline)

        # 转换结果
        overviews = []
        for doc in cursor:
            overview = TickOverview(
                symbol=doc["_id"]["symbol"],
                exchange=Exchange(doc["_id"]["exchange"]),
                count=doc["count"],
                start=doc["start"],
                end=doc["end"],
            )
            overviews.append(overview)

        return overviews
