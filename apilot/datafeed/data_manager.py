"""
数据管理模块

负责加载和管理回测数据, 将数据加载逻辑从回测引擎中分离.
提供统一的数据访问接口, 支持各种数据库后端.
"""

from apilot.core import BarData, Exchange, Interval, TickData
from apilot.core.database import BaseDatabase, get_database
from apilot.core.utility import extract_symbol
from apilot.datafeed.providers.csv_provider import CsvDatabase
from apilot.utils.logger import get_logger

# 获取日志记录器
logger = get_logger("DataManager")


class DataManager:
    """
    数据管理类, 负责加载和管理回测数据

    作为所有数据访问的统一入口点, 可以配置不同的数据库后端.
    支持两种使用模式:
    1. 与回测引擎集成: 设置engine参数, 会自动加载数据到引擎
    2. 独立使用: 不设置engine参数, 直接使用load_bar_data等方法加载数据
    """

    def __init__(self, engine=None):
        """
        初始化数据管理器

        参数:
            engine: 回测引擎实例.如果不提供, 则处于独立模式.
        """
        self.engine = engine
        self.data_sources = {}
        self._database: BaseDatabase | None = None

    @property
    def database(self) -> BaseDatabase:
        """获取当前数据库实例, 如果未指定则使用默认配置"""
        if self._database is None:
            self._database = get_database()
        return self._database

    def set_database(self, database: BaseDatabase) -> None:
        """
        设置要使用的数据库实例

        参数:
            database: BaseDatabase的实现实例
        """
        self._database = database

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
            symbol: 交易对名称
            exchange: 交易所, 可以是Exchange枚举或字符串
            start: 开始时间
            end: 结束时间
            **kwargs: 其他参数, 如gateway_name等

        返回:
            TickData对象列表
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
            symbol: 交易对名称
            exchange: 交易所, 可以是Exchange枚举或字符串
            start: 开始时间
            end: 结束时间
            **kwargs: 其他参数, 如gateway_name等

        返回:
            TickData对象列表
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
