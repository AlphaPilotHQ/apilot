"""
数据管理类

负责管理数据源, 提供多种数据源配置方法, 包括CSV和MongoDB。
"""

import logging

from apilot.core.constant import Exchange, Interval
from apilot.core.object import BarData, TickData
from apilot.core.utility import extract_symbol
from apilot.datafeed.config import CsvSourceConfig, MongoSourceConfig

logger = logging.getLogger(__name__)


class DataManager:
    """
    数据管理类

    提供统一的数据访问接口, 支持多种数据源配置。
    """

    def __init__(self, engine=None):
        """
        初始化数据管理器

        参数:
            engine: 回测引擎实例, 可选
        """
        from apilot.core.database import get_database

        self.engine = engine
        self.database = get_database()

    def set_database(self, database):
        """
        设置当前使用的数据库

        参数:
            database: 数据库实例, 必须是BaseDatabase的子类
        """
        self.database = database

    def csv(
        self,
        data_path,
        symbol_name=None,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        dtformat="%Y-%m-%d %H:%M:%S",
        **kwargs,
    ):
        """
        使用CSV数据源配置

        参数:
            data_path: CSV文件路径
            symbol_name: 可选, 指定要加载的交易对名称
            datetime, open, high, low, close, volume: 列索引映射
            openinterest: 可选, 持仓量列索引
            dtformat: 日期时间格式
            **kwargs: 其他参数

        返回:
            如果在引擎模式下,返回引擎实例; 否则返回数据管理器实例
        """
        # 创建CSV数据库
        from apilot.datafeed.providers.csv_provider import CsvDatabase

        database = CsvDatabase(data_path)

        # 使用列索引模式
        database.set_index_mapping(
            datetime=datetime,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            openinterest=openinterest,
            dtformat=dtformat,
        )

        # 设置为当前数据库
        self.set_database(database)

        # 如果在引擎模式下且提供了symbol_name, 则加载数据
        if self.engine and symbol_name:
            self._load_data_from_source(symbol_name)

        return self.engine if self.engine else self

    def mongodb(
        self,
        host="localhost",
        port=27017,
        database_name="apilot",
        symbol_name=None,
        **kwargs,
    ):
        """
        使用MongoDB数据源配置(预留接口)

        参数:
            host: MongoDB主机地址
            port: MongoDB端口
            database_name: 数据库名称
            symbol_name: 可选,指定要加载的交易对名称
            **kwargs: 其他参数

        返回:
            如果在引擎模式下,返回引擎实例; 否则返回数据管理器实例
        """
        # 检查MongoDB提供者是否已注册
        from apilot.core.database import _DATABASE_REGISTRY

        if "mongodb" not in _DATABASE_REGISTRY:
            raise ValueError("MongoDB提供者未注册, 请先安装MongoDB提供者")

        # 获取MongoDB数据库实例
        database = _DATABASE_REGISTRY["mongodb"](
            host=host, port=port, database=database_name, **kwargs
        )

        # 设置为当前数据库
        self.set_database(database)

        # 如果在引擎模式下且提供了symbol_name, 则加载数据
        if self.engine and symbol_name:
            self._load_data_from_source(symbol_name)

        return self.engine if self.engine else self

    def add_data(self, config):
        """
        根据配置对象添加数据源

        参数:
            config: 数据源配置对象, 可以是CsvSourceConfig或MongoSourceConfig

        返回:
            如果在引擎模式下返回引擎实例, 否则返回数据管理器实例
        """
        if isinstance(config, CsvSourceConfig):
            return self._add_csv_data(config)
        elif isinstance(config, MongoSourceConfig):
            return self._add_mongodb_data(config)
        else:
            raise TypeError(f"不支持的数据源配置类型: {type(config)}")

    def _add_csv_data(self, config: CsvSourceConfig):
        """
        Add CSV data source from configuration

        Args:
            config: CSV source configuration object

        Returns:
            Engine instance if in engine mode, otherwise data manager instance
        """
        from apilot.datafeed.providers.csv_provider import CsvDatabase

        # Create CSV database instance from config
        database = CsvDatabase.from_config(config)
        self.set_database(database)

        # Load data if in engine mode
        if self.engine:
            self._load_data_from_source(config.symbol)

        return self.engine if self.engine else self

    def _add_mongodb_data(self, config: MongoSourceConfig):
        """
        Add MongoDB data source from configuration

        Args:
            config: MongoDB source configuration object

        Returns:
            Engine instance if in engine mode, otherwise data manager instance
        """
        # Check if MongoDB provider is registered
        from apilot.core.database import _DATABASE_REGISTRY

        if "mongodb" not in _DATABASE_REGISTRY:
            raise ValueError(
                "MongoDB provider not registered, please install MongoDB provider first"
            )

        # Get MongoDB database class and create instance from config
        mongodb_class = _DATABASE_REGISTRY["mongodb"]

        # Check if from_config method exists, otherwise use parameters directly
        if hasattr(mongodb_class, "from_config"):
            database = mongodb_class.from_config(config)
        else:
            # Fall back to traditional instantiation
            database = mongodb_class(
                host=config.host,
                port=config.port,
                database=config.database,
                collection=config.collection,
                **config.extra_params,
            )

        self.set_database(database)

        # Load data if in engine mode
        if self.engine:
            self._load_data_from_source(config.symbol)

        return self.engine if self.engine else self

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange | str,
        interval: Interval,
        start,
        end,
        **kwargs,
    ) -> list[BarData]:
        """
        统一的K线数据加载方法

        参数:
            symbol: 交易对名称
            exchange: 交易所, 可以是Exchange枚举或字符串
            interval: K线周期
            start: 开始时间
            end: 结束时间
            **kwargs: 其他参数, 如gateway_name等

        返回:
            BarData对象列表
        """
        # 提取基本符号和交易所
        if "." in symbol and not isinstance(exchange, Exchange):
            base_symbol, exchange_str = extract_symbol(symbol)
            if isinstance(exchange, str):
                exchange = Exchange(exchange)
        else:
            base_symbol = symbol
            if isinstance(exchange, str):
                exchange = Exchange(exchange)

        # 使用配置的数据库加载数据
        bars = self.database.load_bar_data(
            symbol=base_symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end,
        )

        # 应用其他配置 (例如gateway_name)
        if kwargs.get("gateway_name") and bars:
            for bar in bars:
                bar.gateway_name = kwargs["gateway_name"]

        return bars

    def load_tick_data(
        self, symbol: str, exchange: Exchange | str, start, end, **kwargs
    ) -> list[TickData]:
        """
        统一的Tick数据加载方法

        参数:
            symbol: 交易对代码
            exchange: 交易所
            start: 开始时间
            end: 结束时间
        """
        # 提取基本符号和交易所
        if "." in symbol and not isinstance(exchange, Exchange):
            base_symbol, exchange_str = extract_symbol(symbol)
            if isinstance(exchange, str):
                exchange = Exchange(exchange)
        else:
            base_symbol = symbol
            if isinstance(exchange, str):
                exchange = Exchange(exchange)

        # 使用配置的数据库加载数据
        ticks = self.database.load_tick_data(
            symbol=base_symbol, exchange=exchange, start=start, end=end
        )

        # 应用其他配置 (例如gateway_name)
        if kwargs.get("gateway_name") and ticks:
            for tick in ticks:
                tick.gateway_name = kwargs["gateway_name"]

        return ticks

    def _load_data_from_source(self, symbol_name=None):
        """
        从数据源加载数据到引擎

        参数:
            symbol_name: 可选, 指定要加载的交易对名称
        """
        if not hasattr(self.engine, "history_data"):
            self.engine.history_data = {}
            self.engine.dts = []

        # 确定要处理的交易对
        symbols = self.engine.symbols
        if symbol_name:
            # 精确匹配完整的交易对名称或仅交易对部分
            filtered_symbols = []
            for s in symbols:
                symbol, _ = s.split(".")
                if symbol_name == symbol or symbol_name == s:
                    filtered_symbols.append(s)

            symbols = filtered_symbols
            if not symbols:
                logger.warning(f"未找到匹配的交易对: {symbol_name}")
                return

        # 加载每个交易对的数据
        for symbol in symbols:
            base_symbol, exchange = extract_symbol(symbol)

            # 加载数据
            bars = self.database.load_bar_data(
                symbol=base_symbol,
                exchange=exchange,
                interval=self.engine.interval,
                start=self.engine.start,
                end=self.engine.end,
            )

            # 处理数据
            for bar in bars:
                bar.symbol = symbol
                self.engine.dts.append(bar.datetime)
                self.engine.history_data.setdefault(bar.datetime, {})[symbol] = bar

            logger.info(f"加载了 {len(bars)} 条 {symbol} 的历史数据")

        # 对时间点从小到大排序
        self.engine.dts = sorted(set(self.engine.dts))
        logger.info(f"历史数据加载完成, 数据量: {len(self.engine.dts)}")
