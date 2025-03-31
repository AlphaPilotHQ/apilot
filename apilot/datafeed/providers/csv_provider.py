"""
CSV file data provider

Implements data storage based on CSV files, supports bar data loading.
"""

import os
from datetime import datetime

import pandas as pd

from apilot.core import BarData, Exchange, Interval, TickData
from apilot.core.database import BaseDatabase
from apilot.utils.logger import get_logger, set_level

# Get CSV database logger
logger = get_logger("csv_provider")
set_level("debug", "csv_provider")


class CsvDatabase(BaseDatabase):
    def __init__(self, filepath=None, **kwargs):
        """初始化CSV数据库

        Args:
            filepath: CSV文件路径
            **kwargs: 其他参数, 支持以下选项:
                - datetime_index: 日期时间列索引
                - open_index: 开盘价列索引
                - high_index: 最高价列索引
                - low_index: 最低价列索引
                - close_index: 收盘价列索引
                - volume_index: 成交量列索引
                - dtformat: 日期时间格式
        """
        self.csv_path = filepath

        # 设置默认索引
        self.datetime_index = kwargs.get("datetime_index", 0)
        self.open_index = kwargs.get("open_index", 1)
        self.high_index = kwargs.get("high_index", 2)
        self.low_index = kwargs.get("low_index", 3)
        self.close_index = kwargs.get("close_index", 4)
        self.volume_index = kwargs.get("volume_index", 5)
        self.openinterest_index = kwargs.get("openinterest_index", -1)

        # 日期格式
        self.dtformat = kwargs.get("dtformat", "%Y-%m-%d %H:%M:%S")

    def _get_file_path(
        self, symbol: str, exchange: Exchange | str, interval: Interval
    ) -> str:
        """获取CSV文件路径"""
        # 如果直接设置了CSV文件路径,则使用它
        if self.csv_path:
            logger.info(f"使用CSV文件路径: {self.csv_path}")
            if not os.path.exists(self.csv_path):
                raise FileNotFoundError(f"CSV文件不存在: {self.csv_path}")
            return self.csv_path

        # Get exchange value
        exchange_value = exchange.value if hasattr(exchange, "value") else exchange

        # Use global data directory or absolute path
        from apilot.utils.utility import get_data_dir

        data_dir = get_data_dir()  # Get configured data directory
        path = os.path.join(data_dir, f"{symbol}_{exchange_value}_{interval.value}.csv")

        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        return path

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange | str,
        interval: Interval,
        start: datetime,
        end: datetime,
    ) -> list[BarData]:
        """加载K线数据"""
        # 获取交易所值
        exchange_value = exchange.value if hasattr(exchange, "value") else exchange
        logger.info(
            f"CSV数据库: 请求加载 {symbol}.{exchange_value} 数据,区间 {interval},时间 {start} 至 {end}"
        )

        # 获取文件路径
        file_path = self._get_file_path(symbol, exchange, interval)

        # 加载CSV文件
        try:
            # 读取CSV文件到DataFrame
            df = pd.read_csv(file_path)
            logger.info(f"CSV文件已加载,行数: {len(df)}")

            # 处理数据
            bars = []

            # 转换日期时间列
            dt_col = df.iloc[:, self.datetime_index]
            logger.info(
                f"转换日期时间字段: 索引{self.datetime_index}, 格式{self.dtformat}"
            )

            # 转换日期时间列为datetime格式
            df.iloc[:, self.datetime_index] = pd.to_datetime(
                dt_col, format=self.dtformat
            )

            # 筛选时间范围内的数据
            df = df[
                (df.iloc[:, self.datetime_index] >= pd.Timestamp(start))
                & (df.iloc[:, self.datetime_index] <= pd.Timestamp(end))
            ]

            logger.info(f"筛选后数据行数: {len(df)}")

            # 生成Bar对象
            for _, row in df.iterrows():
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=row.iloc[self.datetime_index],
                    interval=interval,
                    volume=float(row.iloc[self.volume_index]),
                    open_price=float(row.iloc[self.open_index]),
                    high_price=float(row.iloc[self.high_index]),
                    low_price=float(row.iloc[self.low_index]),
                    close_price=float(row.iloc[self.close_index]),
                    gateway_name="CSV",
                )

                # 添加持仓量(如果有)
                if self.openinterest_index >= 0:
                    bar.open_interest = float(row.iloc[self.openinterest_index])

                bars.append(bar)

            logger.info(f"成功创建 {len(bars)} 个Bar对象")
            if bars:
                logger.info(f"第一个Bar: {bars[0]}")
                logger.info(f"最后一个Bar: {bars[-1]}")

            return bars

        except Exception as e:
            logger.error(f"加载CSV数据时出错: {e!s}")
            import traceback

            logger.error(traceback.format_exc())
            return []

    def load_tick_data(
        self, symbol: str, exchange: Exchange, start: datetime, end: datetime
    ) -> list[TickData]:
        """Load tick data (not implemented)"""
        return []

    def delete_bar_data(
        self, symbol: str, exchange: Exchange, interval: Interval
    ) -> int:
        """Delete bar data (not implemented)"""
        return 0

    def delete_tick_data(self, symbol: str, exchange: Exchange) -> int:
        """Delete tick data (not implemented)"""
        return 0

    def get_bar_overview(self) -> list[dict]:
        """Get bar data overview (not implemented)"""
        return []

    def get_tick_overview(self) -> list[dict]:
        """Get tick data overview (not implemented)"""
        return []
