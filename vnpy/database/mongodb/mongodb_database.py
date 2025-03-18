""""""
from datetime import datetime
from typing import List
import traceback

from pymongo import ASCENDING, MongoClient, ReplaceOne
from pymongo.database import Database
from pymongo.cursor import Cursor
from pymongo.collection import Collection
from pymongo.results import DeleteResult

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import BaseDatabase, BarOverview, TickOverview, DB_TZ
from vnpy.trader.setting import SETTINGS


class MongodbDatabase(BaseDatabase):
    """MongoDB数据库接口"""

    def __init__(self) -> None:
        """"""
        # 读取配置
        self.database: str = SETTINGS.get("database.mongodb.database", "vnpy")
        self.host: str = SETTINGS.get("database.mongodb.host", "localhost")
        self.port: int = SETTINGS.get("database.mongodb.port", 27017)
        self.username: str = SETTINGS.get("database.mongodb.username", "")
        self.password: str = SETTINGS.get("database.mongodb.password", "")
        self.authentication_source: str = SETTINGS.get("database.mongodb.authentication_source", "admin")

        # 创建客户端
        if self.username and self.password:
            self.client: MongoClient = MongoClient(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                authSource=self.authentication_source
            )
        else:
            self.client: MongoClient = MongoClient(
                host=self.host,
                port=self.port
            )

        # 初始化数据库
        self.db = self.client[self.database]
        
        # 初始化集合
        self.bar_collection = self.db["bar_data"]
        self.tick_collection = self.db["tick_data"]

        # 初始化K线数据表
        self.bar_collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("interval", ASCENDING),
                ("datetime", ASCENDING),
            ],
            unique=True
        )

        # 初始化Tick数据表
        self.tick_collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("datetime", ASCENDING),
            ],
            unique=True
        )

        # 初始化K线概览表
        self.bar_overview_collection = self.db["bar_overview"]
        self.bar_overview_collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
                ("interval", ASCENDING),
            ],
            unique=True
        )

        # 初始化Tick概览表
        self.tick_overview_collection = self.db["tick_overview"]
        self.tick_overview_collection.create_index(
            [
                ("exchange", ASCENDING),
                ("symbol", ASCENDING),
            ],
            unique=True
        )

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """保存K线数据"""
        requests: List[ReplaceOne] = []

        for bar in bars:
            # 逐个插入
            filter: dict = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "datetime": bar.datetime,
                "interval": bar.interval.value,
            }

            d: dict = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "datetime": bar.datetime,
                "interval": bar.interval.value,
                "volume": bar.volume,
                "turnover": bar.turnover,
                "open_interest": bar.open_interest,
                "open_price": bar.open_price,
                "high_price": bar.high_price,
                "low_price": bar.low_price,
                "close_price": bar.close_price,
            }

            requests.append(ReplaceOne(filter, d, upsert=True))

        self.bar_collection.bulk_write(requests, ordered=False)

        # 更新汇总
        filter: dict = {
            "symbol": bar.symbol,
            "exchange": bar.exchange.value,
            "interval": bar.interval.value
        }

        overview: dict = self.bar_overview_collection.find_one(filter)

        if not overview:
            overview = {
                "symbol": bar.symbol,
                "exchange": bar.exchange.value,
                "interval": bar.interval.value,
                "count": len(bars),
                "start": bars[0].datetime,
                "end": bars[-1].datetime
            }
        elif stream:
            overview["end"] = bars[-1].datetime
            overview["count"] += len(bars)
        else:
            overview["start"] = min(bars[0].datetime, overview["start"])
            overview["end"] = max(bars[-1].datetime, overview["end"])
            overview["count"] = self.bar_collection.count_documents(filter)

        self.bar_overview_collection.update_one(filter, {"$set": overview}, upsert=True)

        return True

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """保存TICK数据"""
        requests: List[ReplaceOne] = []

        for tick in ticks:
            filter: dict = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": tick.datetime,
            }

            d: dict = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "datetime": tick.datetime,
                "name": tick.name,
                "volume": tick.volume,
                "turnover": tick.turnover,
                "open_interest": tick.open_interest,
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
                "localtime": tick.localtime,
            }

            requests.append(ReplaceOne(filter, d, upsert=True))

        self.tick_collection.bulk_write(requests, ordered=False)

        # 更新Tick汇总
        filter: dict = {
            "symbol": tick.symbol,
            "exchange": tick.exchange.value
        }

        overview: dict = self.tick_overview_collection.find_one(filter)

        if not overview:
            overview = {
                "symbol": tick.symbol,
                "exchange": tick.exchange.value,
                "count": len(ticks),
                "start": ticks[0].datetime,
                "end": ticks[-1].datetime
            }
        elif stream:
            overview["end"] = ticks[-1].datetime
            overview["count"] += len(ticks)
        else:
            overview["start"] = min(ticks[0].datetime, overview["start"])
            overview["end"] = max(ticks[-1].datetime, overview["end"])
            overview["count"] = self.tick_collection.count_documents(filter)

        self.tick_overview_collection.update_one(filter, {"$set": overview}, upsert=True)

        return True

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime,
        collection: str = None
    ) -> List[BarData]:
        """读取K线数据"""
        start = start.astimezone(DB_TZ)
        end = end.astimezone(DB_TZ)
        
        # 转换为毫秒级时间戳
        start_timestamp = int(start.timestamp() * 1000)
        end_timestamp = int(end.timestamp() * 1000)

        # 确定集合名称
        if collection:
            collection_name = collection
        else:
            collection_name = "bar_data"

        # 对于symbol_trade集合的特殊处理，使用不同的字段映射
        if collection_name == "symbol_trade":
            # 使用kline_st字段过滤时间，而非datetime
            filter_dict = {
                "kline_st": {
                    "$gte": start_timestamp,
                    "$lte": end_timestamp
                }
            }
            
            # 如果有提供symbol，添加到查询条件
            if symbol:
                # 移除可能的交易所后缀，如 "BTC-USDT.BINANCE" -> "BTCUSDT"
                pure_symbol = symbol.split(".")[0].replace("-", "")
                filter_dict["symbol"] = pure_symbol
                
            # 输出调试信息
            print(f"MongoDB查询(symbol_trade格式): 集合={collection_name}, 过滤条件={filter_dict}")
            
            # 使用指定的集合查询数据
            db_collection = self.db.get_collection(collection_name)
            
            # 获取查询到的文档数量
            count = db_collection.count_documents(filter_dict)
            print(f"MongoDB查询结果: {count}条记录")
            
            if count == 0:
                # 查看是否存在任何记录
                sample = db_collection.find_one()
                if sample:
                    print(f"MongoDB集合中的样本文档: {sample}")
                    
                # 尝试不用symbol过滤，看是否有任何数据
                if "symbol" in filter_dict:
                    alt_filter = {"kline_st": filter_dict["kline_st"]}
                    alt_count = db_collection.count_documents(alt_filter)
                    if alt_count > 0:
                        first_doc = db_collection.find_one(alt_filter)
                        print(f"不过滤symbol时找到{alt_count}条记录，第一条:{first_doc}")
            
            # 查询数据并排序
            cursor = db_collection.find(filter_dict)
            cursor.sort("kline_st", ASCENDING)
            
            # 将数据从MongoDB转换为BarData
            bars: List[BarData] = []
            for d in cursor:
                try:
                    # 从时间戳创建datetime对象
                    bar_datetime = datetime.fromtimestamp(d["kline_st"] / 1000, DB_TZ)
                    
                    # 打印调试信息
                    print(f"处理数据: exchange类型={type(exchange)}, interval类型={type(interval)}")
                    
                    # 不要转换枚举类型为字符串，保持原始的枚举类型
                    # 创建新的字典，映射字段
                    bar_dict = {
                        "symbol": symbol.split(".")[0] if "." in symbol else symbol,
                        "exchange": exchange,  # 保持为枚举类型
                        "datetime": bar_datetime,
                        "interval": interval,  # 保持为枚举类型
                        "volume": float(d.get("trade_volume", 0)),
                        "open_price": float(d.get("first_trade_price", 0)),
                        "high_price": float(d.get("high_price", 0)),
                        "low_price": float(d.get("low_price", 0)),
                        "close_price": float(d.get("last_trade_price", 0)),
                        "gateway_name": "DB"
                    }
                    
                    print(f"创建BarData字典: exchange类型={type(bar_dict['exchange'])}, interval类型={type(bar_dict['interval'])}")
                    
                    # 创建BarData对象
                    bar = BarData.from_dict(bar_dict)
                    bars.append(bar)
                except Exception as e:
                    print(f"处理K线数据时出错: {e}, 文档: {d}")
                    traceback.print_exc()  # 打印完整的错误栈
                    continue
                    
            return bars
        
        # 标准K线数据处理
        else:
            # 标准数据库查询逻辑
            filter_dict = {
                "datetime": {
                    "$gte": start,
                    "$lte": end
                }
            }
            
            # 添加symbol和exchange过滤
            if symbol and exchange:
                filter_dict["symbol"] = symbol
                filter_dict["exchange"] = exchange.value
                
            # 添加interval过滤
            if interval:
                filter_dict["interval"] = interval.value
                
            # 输出调试信息
            print(f"MongoDB查询(标准格式): 集合={collection_name}, 过滤条件={filter_dict}")
            
            # 使用指定的集合
            db_collection = self.bar_collection if collection_name == "bar_data" else self.db.get_collection(collection_name)
            
            # 查询结果
            cursor = db_collection.find(filter_dict)
            cursor.sort("datetime", ASCENDING)
            
            # 获取查询到的文档数量
            count = db_collection.count_documents(filter_dict)
            print(f"MongoDB查询结果: {count}条记录")
            
            # 标准BarData处理
            bars: List[BarData] = []
            for d in cursor:
                try:
                    d["exchange"] = Exchange(d["exchange"])
                    d["interval"] = Interval(d["interval"])
                    d["gateway_name"] = "DB"
                    
                    bar = BarData.from_dict(d)
                    bars.append(bar)
                except Exception as e:
                    print(f"处理K线数据时出错: {e}, 文档: {d}")
                    continue
                    
            return bars

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime,
        collection: str = None
    ) -> List[TickData]:
        """读取TICK数据"""
        start = start.astimezone(DB_TZ)
        end = end.astimezone(DB_TZ)
        
        # 转换为毫秒级时间戳
        start_timestamp = int(start.timestamp() * 1000)
        end_timestamp = int(end.timestamp() * 1000)

        # 确定集合名称
        if collection:
            collection_name = collection
        else:
            collection_name = "tick_data"

        # 对于symbol_trade集合的特殊处理
        if collection_name == "symbol_trade":
            # 这个集合实际上存储的是K线数据，我们可以尝试构建Tick数据
            # 但这不是理想的解决方案，仅用于演示
            print(f"警告: 从K线集合({collection_name})构建Tick数据可能不准确")
            
            filter_dict = {
                "kline_st": {
                    "$gte": start_timestamp,
                    "$lte": end_timestamp
                }
            }
            
            # 如果有提供symbol，添加到查询条件
            if symbol:
                pure_symbol = symbol.split(".")[0].replace("-", "")
                filter_dict["symbol"] = pure_symbol
                
            print(f"MongoDB Tick查询(symbol_trade格式): 集合={collection_name}, 过滤条件={filter_dict}")
            
            db_collection = self.db.get_collection(collection_name)
            
            # 查询数据并排序
            cursor = db_collection.find(filter_dict)
            cursor.sort("kline_st", ASCENDING)
            
            count = db_collection.count_documents(filter_dict)
            print(f"MongoDB Tick查询结果: {count}条记录")
            
            # 从K线数据构建Tick数据(近似)
            ticks: List[TickData] = []
            for d in cursor:
                try:
                    # 使用K线开始时间作为Tick时间
                    tick_datetime = datetime.fromtimestamp(d["kline_st"] / 1000, DB_TZ)
                    
                    # 创建Tick数据字典
                    tick_dict = {
                        "symbol": symbol.split(".")[0] if "." in symbol else symbol,
                        "exchange": exchange,  # 保持为枚举类型
                        "datetime": tick_datetime,
                        "name": symbol.split(".")[0] if "." in symbol else symbol,
                        
                        # 价格字段
                        "last_price": float(d.get("last_trade_price", 0)),
                        "high_price": float(d.get("high_price", 0)),
                        "low_price": float(d.get("low_price", 0)),
                        
                        # 成交量
                        "volume": float(d.get("trade_volume", 0)),
                        "turnover": float(d.get("transaction_volume", 0)),
                        
                        # 其他必要的字段设置默认值
                        "open_interest": 0,
                        "bid_price_1": float(d.get("last_trade_price", 0)),
                        "bid_volume_1": 0,
                        "ask_price_1": float(d.get("last_trade_price", 0)),
                        "ask_volume_1": 0,
                        
                        "gateway_name": "DB"
                    }
                    
                    # 创建TickData对象
                    tick = TickData.from_dict(tick_dict)
                    ticks.append(tick)
                except Exception as e:
                    print(f"处理Tick数据时出错: {e}, 文档: {d}")
                    traceback.print_exc()  # 打印完整的错误栈
                    continue
                    
            return ticks
            
        # 标准Tick数据处理
        else:
            filter_dict = {
                "datetime": {
                    "$gte": start,
                    "$lte": end
                }
            }
            
            if symbol and exchange:
                filter_dict["symbol"] = symbol
                filter_dict["exchange"] = exchange.value
                
            print(f"MongoDB Tick查询(标准格式): 集合={collection_name}, 过滤条件={filter_dict}")
            
            db_collection = self.tick_collection if collection_name == "tick_data" else self.db.get_collection(collection_name)
            
            cursor = db_collection.find(filter_dict)
            cursor.sort("datetime", ASCENDING)
            
            count = db_collection.count_documents(filter_dict)
            print(f"MongoDB Tick查询结果: {count}条记录")
            
            ticks: List[TickData] = []
            for d in cursor:
                try:
                    d["exchange"] = Exchange(d["exchange"])
                    d["gateway_name"] = "DB"
                    
                    tick = TickData.from_dict(d)
                    ticks.append(tick)
                except Exception as e:
                    print(f"处理Tick数据时出错: {e}, 文档: {d}")
                    continue
                    
            return ticks

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        filter: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
        }

        result: DeleteResult = self.bar_collection.delete_many(filter)
        self.bar_overview_collection.delete_one(filter)

        return result.deleted_count

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除TICK数据"""
        filter: dict = {
            "symbol": symbol,
            "exchange": exchange.value
        }

        result: DeleteResult = self.tick_collection.delete_many(filter)
        self.tick_overview_collection.delete_one(filter)

        return result.deleted_count

    def get_bar_overview(self) -> List[BarOverview]:
        """查询数据库中的K线汇总信息"""
        c: Cursor = self.bar_overview_collection.find()

        overviews: List[BarOverview] = []
        for d in c:
            d["exchange"] = Exchange(d["exchange"])
            d["interval"] = Interval(d["interval"])
            d.pop("_id")

            overview: BarOverview = BarOverview(**d)
            overviews.append(overview)

        return overviews

    def get_tick_overview(self) -> List[TickOverview]:
        """查询数据库中的Tick汇总信息"""
        c: Cursor = self.tick_overview_collection.find()

        overviews: List[TickOverview] = []
        for d in c:
            d["exchange"] = Exchange(d["exchange"])
            d.pop("_id")

            overview: TickOverview = TickOverview(**d)
            overviews.append(overview)

        return overviews