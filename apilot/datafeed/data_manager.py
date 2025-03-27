"""
数据管理模块

负责加载和管理回测数据，将数据加载逻辑从回测引擎中分离。
"""

from typing import Dict, List, Optional
from datetime import datetime
import os
import logging

from apilot.core import BarData, Interval, Exchange
from apilot.core.utility import extract_vt_symbol
from apilot.utils.logger import get_logger
from apilot.datafeed.csv_database import CsvDatabase

# 获取日志记录器
logger = get_logger("DataManager")

class DataManager:
    """数据管理类，负责加载和管理回测数据"""

    def __init__(self, engine):
        """
        初始化数据管理器

        参数:
            engine: 回测引擎实例，用于获取回测参数
        """
        self.engine = engine
        self.data_sources = {}

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
        dtformat='%Y-%m-%d %H:%M:%S',
        **kwargs
    ):
        """
        使用列索引从CSV文件加载数据

        参数:
            data_path (str): CSV文件路径
            symbol_name (str): 要加载数据的交易对名称，如果不指定则尝试匹配所有符号
            datetime (int): 日期时间列的索引位置
            open (int): 开盘价列的索引位置
            high (int): 最高价列的索引位置
            low (int): 最低价列的索引位置
            close (int): 收盘价列的索引位置
            volume (int): 成交量列的索引位置
            openinterest (int): 持仓量列的索引位置，-1表示不使用
            dtformat (str): 日期时间格式，例如'%Y-%m-%d %H:%M:%S'
            **kwargs: 其他关键字参数
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
            dtformat=dtformat
        )

        # 加载数据到引擎
        self._load_data_from_source(database, symbol_name)

        return self.engine

    def _load_data_from_source(self, database, symbol_name=None):
        """从数据源加载数据到引擎"""
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
        for vt_symbol in symbols:
            symbol, exchange = extract_vt_symbol(vt_symbol)

            # 加载数据
            bars = database.load_bar_data(
                symbol=symbol,
                exchange=exchange,
                interval=self.engine.interval,
                start=self.engine.start,
                end=self.engine.end
            )

            # 处理数据
            for bar in bars:
                bar.vt_symbol = vt_symbol
                self.engine.dts.append(bar.datetime)
                self.engine.history_data.setdefault(bar.datetime, {})[vt_symbol] = bar

            logger.info(f"加载了 {len(bars)} 条 {vt_symbol} 的历史数据")

        # 对时间点从小到大排序
        self.engine.dts = sorted(list(set(self.engine.dts)))
        logger.info(f"历史数据加载完成，数据量：{len(self.engine.dts)}")
