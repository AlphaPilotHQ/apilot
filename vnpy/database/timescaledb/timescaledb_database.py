from datetime import datetime
from typing import Dict, List, Any, Optional

import psycopg2
from psycopg2.extras import execute_batch, DictCursor

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import (
    BaseDatabase,
    BarOverview,
    TickOverview,
    DB_TZ
)
from vnpy.trader.setting import SETTINGS

from .timescaledb_scripts import (
    CREATE_BAR_TABLE_SCRIPT,
    CREATE_BAR_HYPERTABLE_SCRIPT,
    CREATE_BAR_OVERVIEW_TABLE_SCRIPT,
    CREATE_TICK_OVERVIEW_TABLE_SCRIPT,
    LOAD_BAR_OVERVIEW_QUERY,
    LOAD_TICK_OVERVIEW_QUERY,
    COUNT_BAR_QUERY,
    SAVE_BAR_OVERVIEW_QUERY,
    SAVE_TICK_OVERVIEW_QUERY,
    DELETE_BAR_QUERY,
    DELETE_BAR_OVERVIEW_QUERY,
    DELETE_TICK_OVERVIEW_QUERY,
    LOAD_ALL_BAR_OVERVIEW_QUERY,
    LOAD_ALL_TICK_OVERVIEW_QUERY,
    LOAD_BAR_QUERY,
    CREATE_TICK_TABLE_SCRIPT,
    CREATE_TICK_HYPERTABLE_SCRIPT,
    COUNT_TICK_QUERY,
    DELETE_TICK_QUERY,
    LOAD_TICK_QUERY,
    SAVE_BAR_QUERY,
    SAVE_TICK_QUERY
)


class TimescaleDBDatabase(BaseDatabase):
    """TimescaleDB数据库接口"""

    def __init__(self) -> None:
        """初始化数据库"""
        # 读取配置
        self.database: str = SETTINGS["database.database"]
        self.host: str = SETTINGS["database.host"]
        self.port: int = SETTINGS["database.port"]
        self.username: str = SETTINGS["database.user"]
        self.password: str = SETTINGS["database.password"]

        # 创建连接
        self.connection_string = f"postgres://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        self.connection = psycopg2.connect(self.connection_string)
        self.cursor = self.connection.cursor(cursor_factory=DictCursor)

        # 初始化数据表
        self._initialize_tables()

    def _initialize_tables(self) -> None:
        """初始化数据库表结构"""
        # 创建K线数据表
        self.cursor.execute(CREATE_BAR_TABLE_SCRIPT)
        self.cursor.execute(CREATE_BAR_HYPERTABLE_SCRIPT)
        
        # 创建Tick数据表
        self.cursor.execute(CREATE_TICK_TABLE_SCRIPT)
        self.cursor.execute(CREATE_TICK_HYPERTABLE_SCRIPT)
        
        # 创建概览表
        self.cursor.execute(CREATE_BAR_OVERVIEW_TABLE_SCRIPT)
        self.cursor.execute(CREATE_TICK_OVERVIEW_TABLE_SCRIPT)
        
        self.connection.commit()

    def execute(self, query: str, params: Any = None) -> None:
        """执行SQL查询"""
        if params is None:
            self.cursor.execute(query)
        else:
            if isinstance(params, list):
                execute_batch(self.cursor, query, params)
            else:
                self.cursor.execute(query, params)
        self.connection.commit()

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """保存K线数据"""
        if not bars:
            return False
            
        # 缓存字段参数
        bar: BarData = bars[0]
        symbol: str = bar.symbol
        exchange: Exchange = bar.exchange
        interval: Interval = bar.interval

        # 准备数据
        bar_data: List[dict] = []
        for bar in bars:
            # 转换为数据库格式
            bar_dict: Dict[str, Any] = bar.__dict__.copy()
            bar_dict["exchange"] = bar_dict["exchange"].value
            bar_dict["interval"] = bar_dict["interval"].value
            
            # 添加到批量操作列表
            bar_data.append(bar_dict)

        # 批量插入K线数据
        self.execute(SAVE_BAR_QUERY, bar_data)

        # 更新汇总信息
        self._update_bar_overview(symbol, exchange, interval, bars, stream)

        return True

    def _update_bar_overview(
        self, 
        symbol: str, 
        exchange: Exchange, 
        interval: Interval, 
        bars: List[BarData], 
        stream: bool
    ) -> None:
        """更新K线汇总信息"""
        # 查询汇总信息
        params: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value
        }
        self.execute(LOAD_BAR_OVERVIEW_QUERY, params)
        row = self.cursor.fetchone()

        data: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value
        }

        # 没有该合约信息
        if not row:
            data["starttime"] = bars[0].datetime
            data["endtime"] = bars[-1].datetime
            data["count"] = len(bars)
        # 增量更新
        elif stream:
            data["starttime"] = row["starttime"]
            data["endtime"] = bars[-1].datetime
            data["count"] = row["count"] + len(bars)
        # 全量更新
        else:
            self.execute(COUNT_BAR_QUERY, params)
            count = self.cursor.fetchone()[0]

            data["starttime"] = min(bars[0].datetime, row["starttime"])
            data["endtime"] = max(bars[-1].datetime, row["endtime"])
            data["count"] = count

        self.execute(SAVE_BAR_OVERVIEW_QUERY, data)

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """保存TICK数据"""
        if not ticks:
            return False
            
        # 缓存字段参数
        tick: TickData = ticks[0]
        symbol: str = tick.symbol
        exchange: Exchange = tick.exchange

        # 准备数据
        tick_data: List[dict] = []
        for tick in ticks:
            # 转换为数据库格式
            tick_dict: Dict[str, Any] = tick.__dict__.copy()
            tick_dict["exchange"] = tick_dict["exchange"].value
            tick_dict["localt"] = tick_dict.pop("localtime")
            if not tick_dict["localt"]:
                tick_dict["localt"] = datetime.now()
                
            # 添加到批量操作列表
            tick_data.append(tick_dict)

        # 批量插入Tick数据
        self.execute(SAVE_TICK_QUERY, tick_data)

        # 更新汇总信息
        self._update_tick_overview(symbol, exchange, ticks, stream)

        return True

    def _update_tick_overview(
        self, 
        symbol: str, 
        exchange: Exchange, 
        ticks: List[TickData], 
        stream: bool
    ) -> None:
        """更新Tick汇总信息"""
        # 查询Tick汇总信息
        params: dict = {
            "symbol": symbol,
            "exchange": exchange.value
        }
        self.execute(LOAD_TICK_OVERVIEW_QUERY, params)
        row = self.cursor.fetchone()

        data: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
        }

        # 没有该合约信息
        if not row:
            data["starttime"] = ticks[0].datetime
            data["endtime"] = ticks[-1].datetime
            data["count"] = len(ticks)
        # 增量更新
        elif stream:
            data["starttime"] = row["starttime"]
            data["endtime"] = ticks[-1].datetime
            data["count"] = row["count"] + len(ticks)
        # 全量更新
        else:
            self.execute(COUNT_TICK_QUERY, params)
            count = self.cursor.fetchone()[0]

            data["starttime"] = min(ticks[0].datetime, row["starttime"])
            data["endtime"] = max(ticks[-1].datetime, row["endtime"])
            data["count"] = count

        self.execute(SAVE_TICK_OVERVIEW_QUERY, data)

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """读取K线数据"""
        # 转换时区
        start = start.astimezone(DB_TZ)
        end = end.astimezone(DB_TZ)
        
        # 构建查询参数
        params: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value,
            "start": str(start),
            "end": str(end)
        }
        
        # 执行查询
        self.execute(LOAD_BAR_QUERY, params)
        rows = self.cursor.fetchall()

        # 转换为BarData对象
        bars: List[BarData] = []
        for row in rows:
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                datetime=row["datetime"].replace(tzinfo=DB_TZ),
                volume=row["volume"],
                turnover=row["turnover"],
                open_interest=row["open_interest"],
                open_price=row["open_price"],
                high_price=row["high_price"],
                low_price=row["low_price"],
                close_price=row["close_price"],
                gateway_name="DB"
            )
            bars.append(bar)

        return bars

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """读取TICK数据"""
        # 转换时区
        start = start.astimezone(DB_TZ)
        end = end.astimezone(DB_TZ)
        
        # 构建查询参数
        params: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "start": str(start),
            "end": str(end)
        }
        
        # 执行查询
        self.execute(LOAD_TICK_QUERY, params)
        rows = self.cursor.fetchall()

        # 转换为TickData对象
        ticks: List[TickData] = []
        for row in rows:
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"].replace(tzinfo=DB_TZ),
                name=row["name"],
                volume=row["volume"],
                turnover=row["turnover"],
                open_interest=row["open_interest"],
                last_price=row["last_price"],
                last_volume=row["last_volume"],
                limit_up=row["limit_up"],
                limit_down=row["limit_down"],
                open_price=row["open_price"],
                high_price=row["high_price"],
                low_price=row["low_price"],
                pre_close=row["pre_close"],
                bid_price_1=row["bid_price_1"],
                bid_price_2=row["bid_price_2"],
                bid_price_3=row["bid_price_3"],
                bid_price_4=row["bid_price_4"],
                bid_price_5=row["bid_price_5"],
                ask_price_1=row["ask_price_1"],
                ask_price_2=row["ask_price_2"],
                ask_price_3=row["ask_price_3"],
                ask_price_4=row["ask_price_4"],
                ask_price_5=row["ask_price_5"],
                bid_volume_1=row["bid_volume_1"],
                bid_volume_2=row["bid_volume_2"],
                bid_volume_3=row["bid_volume_3"],
                bid_volume_4=row["bid_volume_4"],
                bid_volume_5=row["bid_volume_5"],
                ask_volume_1=row["ask_volume_1"],
                ask_volume_2=row["ask_volume_2"],
                ask_volume_3=row["ask_volume_3"],
                ask_volume_4=row["ask_volume_4"],
                ask_volume_5=row["ask_volume_5"],
                localtime=row["localt"],
                gateway_name="DB"
            )
            ticks.append(tick)

        return ticks

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        # 构建删除参数
        params: dict = {
            "symbol": symbol,
            "exchange": exchange.value,
            "interval": interval.value
        }
        
        # 统计删除数量
        self.execute(COUNT_BAR_QUERY, params)
        count = self.cursor.fetchone()[0]
        
        # 执行删除
        self.execute(DELETE_BAR_QUERY, params)
        self.execute(DELETE_BAR_OVERVIEW_QUERY, params)
        
        return count

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除TICK数据"""
        # 构建删除参数
        params: dict = {
            "symbol": symbol,
            "exchange": exchange.value
        }
        
        # 统计删除数量
        self.execute(COUNT_TICK_QUERY, params)
        count = self.cursor.fetchone()[0]
        
        # 执行删除
        self.execute(DELETE_TICK_QUERY, params)
        self.execute(DELETE_TICK_OVERVIEW_QUERY, params)
        
        return count

    def get_bar_overview(self) -> List[BarOverview]:
        """查询数据库中的K线汇总信息"""
        # 执行查询
        self.execute(LOAD_ALL_BAR_OVERVIEW_QUERY)
        rows = self.cursor.fetchall()
        
        # 转换为BarOverview对象
        overviews: List[BarOverview] = []
        for row in rows:
            overview = BarOverview(
                symbol=row["symbol"],
                exchange=Exchange(row["exchange"]),
                interval=Interval(row["interval"]),
                count=row["count"],
                start=row["starttime"].replace(tzinfo=DB_TZ),
                end=row["endtime"].replace(tzinfo=DB_TZ)
            )
            overviews.append(overview)
            
        return overviews

    def get_tick_overview(self) -> List[TickOverview]:
        """查询数据库中的Tick汇总信息"""
        # 执行查询
        self.execute(LOAD_ALL_TICK_OVERVIEW_QUERY)
        rows = self.cursor.fetchall()
        
        # 转换为TickOverview对象
        overviews: List[TickOverview] = []
        for row in rows:
            overview = TickOverview(
                symbol=row["symbol"],
                exchange=Exchange(row["exchange"]),
                count=row["count"],
                start=row["starttime"].replace(tzinfo=DB_TZ),
                end=row["endtime"].replace(tzinfo=DB_TZ)
            )
            overviews.append(overview)
            
        return overviews
