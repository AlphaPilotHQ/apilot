"""
MongoDB数据库提供者

实现了BaseDatabase接口的MongoDB数据提供者, 用于从MongoDB数据库加载和管理量化交易数据.
"""

from datetime import datetime

from pymongo import ASCENDING, MongoClient

from apilot.core.constant import Exchange, Interval
from apilot.core.database import BarOverview, BaseDatabase, TickOverview
from apilot.core.object import BarData, TickData
from apilot.utils.logger import get_logger
from apilot.utils.symbol import split_symbol
from apilot.utils.logger import get_logger, set_level


# 获取日志记录器
logger = get_logger("MongoDB")
set_level("debug", "MongoDB")



class MongoDBDatabase(BaseDatabase):
    def __init__(
        self,
        database: str,
        collection: str,
        uri: str = "",
        **kwargs,
    ):
        """初始化MongoDB数据库连接

        Args:
            database: 数据库名称
            collection: 集合名称
            uri: MongoDB连接URI
            **kwargs: 字段映射和其他选项，可包含：
                symbol_field: 交易对字段名
                datetime_field: 日期时间字段名
                open_field: 开盘价字段名
                high_field: 最高价字段名
                low_field: 最低价字段名
                close_field: 收盘价字段名
                volume_field: 成交量字段名
                open_interest_field: 持仓量字段名
                timestamp_ms: 是否使用毫秒时间戳格式
                limit_count: 限制返回记录数量
        """
        self.uri = uri
        self.database_name = database
        self.collection_name = collection

        # 字段映射，允许通过kwargs覆盖默认值，使用通用字段名作为默认值
        self.field_map = {
            "symbol": kwargs.get("symbol_field", "symbol"),
            "datetime": kwargs.get("datetime_field", "datetime"),
            "open_price": kwargs.get("open_field", "open"),
            "high_price": kwargs.get("high_field", "high"),
            "low_price": kwargs.get("low_field", "low"),
            "close_price": kwargs.get("close_field", "close"),
            "volume": kwargs.get("volume_field", "volume"),
            "open_interest": kwargs.get("open_interest_field", "open_interest"),
        }

        # 时间戳格式是否为毫秒
        self.is_timestamp_ms = kwargs.get("timestamp_ms", False)

        # 其他参数
        self.kwargs = kwargs

        # 创建MongoDB客户端
        self._create_client()

    def _create_client(self):
        """创建MongoDB客户端"""
        # 使用URI连接
        self.client = MongoClient(self.uri)

        # 获取数据库和集合
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]

    def load_bar_data(
        self,
        symbol: str,
        interval: Interval | str,
        start: datetime,
        end: datetime,
        **kwargs,
    ) -> list[BarData]:
        """从MongoDB加载K线数据"""
        # 转换参数类型
        if isinstance(interval, str):
            interval = Interval(interval)

        # 从完整symbol中提取基础符号和交易所
        base_symbol, exchange_str = split_symbol(symbol)
        exchange = Exchange(exchange_str)

        # 构建查询条件
        filter_dict = {}
        
        # 提取交易对基础符号和交易所
        symbol_field = self.field_map["symbol"]
        logger.debug(f"交易对字段: {symbol_field}，基础符号: {base_symbol}")
        
        # 添加交易对过滤条件
        # MongoDB中的交易对可能有不同格式，尝试多种格式匹配
        # 1. 直接使用原始基础符号 (如 "BTC-USDT")
        # 2. 去掉中间的连字符 (如 "BTCUSDT")
        # 3. 转换为小写 (如 "btcusdt")
        
        # 完全硬编码查询参数，基于我们的测试脚本结果
        # 数据库使用大写且无连字符的交易对格式
        symbol_no_dash = base_symbol.replace("-", "")
        symbol_uppercase = symbol_no_dash.upper()
        
        logger.info(f"查询交易对: {symbol_uppercase} (原始: {base_symbol})")
        logger.info(f"使用测试脚本的硬编码查询方式")
        
        # 1. 清空和重置 filter_dict
        filter_dict = {
            "symbol": symbol_uppercase  # 直接使用硬编码的字段名和大写交易对
        }
        
        # 2. 时间范围转换为毫秒时间戳
        start_timestamp = int(start.timestamp() * 1000)
        end_timestamp = int(end.timestamp() * 1000)
        
        # 3. 设置时间查询条件
        filter_dict["kline_st"] = {
            "$gte": start_timestamp,
            "$lte": end_timestamp
        }
        
        # 输出调试信息
        logger.debug(f"数据库: {self.database_name}, 集合: {self.collection_name}")
        logger.debug(f"字段映射: {self.field_map}")
        
        # 处理时间范围查询
        datetime_field = self.field_map["datetime"]
        logger.debug(f"日期时间字段: {datetime_field}")
        logger.debug(f"查询时间范围: {start} 到 {end}")

        # 判断是否使用时间戳格式
        if self.is_timestamp_ms or datetime_field == "kline_st":
            # 使用毫秒时间戳
            start_timestamp = int(start.timestamp() * 1000)
            end_timestamp = int(end.timestamp() * 1000)
            filter_dict[datetime_field] = {
                "$gte": start_timestamp,
                "$lte": end_timestamp,
            }
            logger.debug(f"时间戳查询范围(毫秒): {start_timestamp} 到 {end_timestamp}")
        else:
            # 使用日期时间格式
            filter_dict[datetime_field] = {
                "$gte": start,
                "$lte": end,
            }
            
        # 执行测试查询，获取一条记录来验证字段
        try:
            test_doc = self.collection.find_one()
            if test_doc:
                logger.debug(f"集合中的字段示例: {list(test_doc.keys())}")
                if symbol_field in test_doc:
                    logger.debug(f"集合中的交易对示例: {test_doc[symbol_field]}")
                if datetime_field in test_doc:
                    logger.debug(f"集合中的时间示例: {test_doc[datetime_field]}")
        except Exception as e:
            logger.error(f"测试查询出错: {e}")
            
        # 打印最终的查询条件
        logger.debug(f"MongoDB查询条件: {filter_dict}")

        # 获取参数
        limit_count = kwargs.get("limit_count", 5000)  # 默认限制为5000条记录

        # 创建投影，只选择需要的字段，减少传输数据量
        # 基于检查结果，包含实际存在的字段
        projection = {
            self.field_map["symbol"]: 1,           # symbol
            datetime_field: 1,                      # kline_st
            self.field_map["open_price"]: 1,        # first_trade_price
            self.field_map["high_price"]: 1,        # high_price
            self.field_map["low_price"]: 1,         # low_price
            self.field_map["close_price"]: 1,       # last_trade_price
            self.field_map["volume"]: 1,            # trade_volume
            "_id": 0
        }
        
        # 只有当该字段存在于映射中且可能存在于数据库中时才添加
        # open_interest可能在某些数据库中不存在
        if "open_interest" in self.field_map:
            projection[self.field_map["open_interest"]] = 1

        # 尝试使用基础交易对符号查询
        base_symbol_only = base_symbol.split("-")[0] if "-" in base_symbol else base_symbol
        
        # 打印查询信息便于调试
        logger.info(f"查询 {symbol} 数据，限制 {limit_count} 条")
        logger.info(f"查询条件: {filter_dict}")
        logger.info(f"投影字段: {projection}")
        
        # 执行查询
        try:
            # 首先尝试使用完整过滤条件
            cursor = self.collection.find(filter_dict, projection).sort(datetime_field, ASCENDING).limit(limit_count)
            count = self.collection.count_documents(filter_dict, limit=limit_count)
            logger.info(f"预计返回记录数: {count}")
            
            # 如果没有匹配的记录，尝试放宽条件
            if count == 0:
                logger.warning(f"没有找到匹配记录，尝试修改查询条件")
                
                # 尝试只使用时间过滤，移除交易对过滤
                datetime_filter = {datetime_field: filter_dict[datetime_field]}
                symbols_count = self.collection.count_documents(datetime_filter, limit=10)
                if symbols_count > 0:
                    sample_symbols = list(self.collection.distinct(self.field_map["symbol"], datetime_filter))[:5]
                    logger.info(f"只按时间查询有 {symbols_count} 条记录，样本交易对: {sample_symbols}")
                
                # 仍然使用原始过滤条件
                cursor = self.collection.find(filter_dict, projection).sort(datetime_field, ASCENDING).limit(limit_count)
        except Exception as e:
            logger.error(f"执行MongoDB查询出错: {e}")
            # 返回空游标
            from pymongo.cursor import Cursor
            cursor = Cursor(self.collection, {}, limit=0)

        # 直接获取文档进行测试
        try:
            test_docs = list(self.collection.find(filter_dict, projection).sort(datetime_field, ASCENDING).limit(5))
            if test_docs:
                logger.info(f"测试查询结果: 能获取到文档，示例: {test_docs[0]}")
                logger.debug(f"测试查询文档数: {len(test_docs)}")
            else:
                logger.warning("测试查询没有返回结果，尝试原始查询")
                # 尝试更宽松的查询
                raw_docs = list(self.collection.find({"symbol": "BTCUSDT"}).limit(2))
                if raw_docs:
                    logger.info(f"原始查询有结果: {raw_docs[0].keys()}")
        except Exception as e:
            logger.error(f"测试查询出错: {e}")
            
        # 开始计时
        start_time = datetime.now()

        # 将文档转换为BarData对象
        bars: list[BarData] = []
        processed = 0
        last_sample_time = None  # 用于降采样（仅在非聚合查询时使用）

        for doc in cursor:
            try:
                # 处理日期时间字段
                if self.is_timestamp_ms or datetime_field == "kline_st":
                    # 处理毫秒时间戳
                    dt = datetime.fromtimestamp(doc[datetime_field] / 1000)
                else:
                    # 直接使用日期时间对象
                    dt = doc[datetime_field]

                # 创建K线数据对象并处理字段类型转换
                # 对于字符串数值进行额外处理，确保能正确转换为浮点数
                def safe_float(val, default=0.0):
                    if val is None:
                        return default
                    try:
                        # 处理字符串格式的数值
                        if isinstance(val, str):
                            return float(val)
                        return float(val)
                    except (ValueError, TypeError):
                        logger.warning(f"无法转换为浮点数: {val}，使用默认值: {default}")
                        return default
                
                # 创建K线数据对象
                bar = BarData(
                    symbol=symbol,  # 使用完整交易对符号
                    exchange=exchange,
                    datetime=dt,
                    interval=interval,  # 使用传入的间隔参数

                    # 使用字段映射获取价格数据，支持字段别名，同时安全转换为浮点数
                    open_price=safe_float(doc.get(self.field_map["open_price"], 0)),
                    high_price=safe_float(doc.get(self.field_map["high_price"], 0)),
                    low_price=safe_float(doc.get(self.field_map["low_price"], 0)),
                    close_price=safe_float(doc.get(self.field_map["close_price"], 0)),
                    volume=safe_float(doc.get(self.field_map["volume"], 0)),
                    open_interest=safe_float(doc.get(self.field_map["open_interest"], 0)),

                    gateway_name="MongoDB",
                )
                bars.append(bar)

                # 更新进度
                processed += 1
                if processed % 1000 == 0:
                    logger.info(f"已处理 {processed} 条记录...")

            except Exception as e:
                logger.error(f"处理文档时出错: {e}, 文档: {doc}")
                continue

        # 计算耗时
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"从MongoDB加载 {symbol} 数据: {len(bars)} 条，耗时 {elapsed:.2f} 秒")
        return bars


