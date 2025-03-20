from datetime import datetime
from typing import List, Dict, Optional
import os
import csv
import pandas as pd
from pathlib import Path

from .database import BaseDatabase, register_database
from .constant import Exchange, Interval
from .object import BarData, TickData


class CsvDatabase(BaseDatabase):
    """CSV文件数据库"""

    def __init__(self, data_path: str = "csv_database") -> None:
        """构造函数"""
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """保存K线数据"""
        if not bars:
            return False

        bar = bars[0]
        symbol = bar.symbol
        exchange = bar.exchange
        interval = bar.interval

        path = self._get_bar_path(symbol, exchange, interval)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "datetime", "open", "high", "low", "close", "volume"
            ])

            if os.stat(path).st_size == 0:
                writer.writeheader()

            for bar in bars:
                writer.writerow({
                    "datetime": bar.datetime.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                    "volume": bar.volume
                })

        return True

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """保存Tick数据"""
        if not ticks:
            return False

        tick = ticks[0]
        symbol = tick.symbol
        exchange = tick.exchange

        path = self._get_tick_path(symbol, exchange)
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "datetime", "last_price", "volume", "bid_price_1", "ask_price_1",
                "bid_volume_1", "ask_volume_1"
            ])

            if os.stat(path).st_size == 0:
                writer.writeheader()

            for tick in ticks:
                writer.writerow({
                    "datetime": tick.datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "last_price": tick.last_price,
                    "volume": tick.volume,
                    "bid_price_1": tick.bid_price_1,
                    "ask_price_1": tick.ask_price_1,
                    "bid_volume_1": tick.bid_volume_1,
                    "ask_volume_1": tick.ask_volume_1
                })

        return True

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """加载K线数据"""
        path = self._get_bar_path(symbol, exchange, interval)
        if not os.path.exists(path):
            return []

        data = pd.read_csv(path)
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data[(data["datetime"] >= start) & (data["datetime"] <= end)]

        bars = []
        for _, row in data.iterrows():
            dt = row["datetime"].to_pydatetime()
            
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                interval=interval,
                datetime=dt,
                open_price=row["open"],
                high_price=row["high"],
                low_price=row["low"],
                close_price=row["close"],
                volume=row["volume"],
                gateway_name="CSV"
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
        """加载Tick数据"""
        path = self._get_tick_path(symbol, exchange)
        if not os.path.exists(path):
            return []

        data = pd.read_csv(path)
        data["datetime"] = pd.to_datetime(data["datetime"])
        data = data[(data["datetime"] >= start) & (data["datetime"] <= end)]

        ticks = []
        for _, row in data.iterrows():
            dt = row["datetime"].to_pydatetime()
            
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                last_price=row["last_price"],
                volume=row["volume"],
                bid_price_1=row["bid_price_1"],
                ask_price_1=row["ask_price_1"],
                bid_volume_1=row["bid_volume_1"],
                ask_volume_1=row["ask_volume_1"],
                gateway_name="CSV"
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
        path = self._get_bar_path(symbol, exchange, interval)
        if os.path.exists(path):
            os.remove(path)
            return 1
        return 0

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除Tick数据"""
        path = self._get_tick_path(symbol, exchange)
        if os.path.exists(path):
            os.remove(path)
            return 1
        return 0

    def get_bar_overview(self) -> List[Dict]:
        """获取K线数据概览"""
        result = []
        
        for root, dirs, files in os.walk(Path(self.data_path) / "bar"):
            for file_name in files:
                if file_name.endswith(".csv"):
                    path = os.path.join(root, file_name)
                    parts = path.split(os.sep)
                    
                    # 从路径中提取信息
                    symbol = parts[-3]
                    exchange_name = parts[-2]
                    interval_name = parts[-1].split(".")[0]
                    
                    try:
                        exchange = Exchange(exchange_name)
                        interval = Interval(interval_name)
                    except ValueError:
                        continue
                        
                    # 读取CSV获取统计信息
                    data = pd.read_csv(path)
                    
                    # 添加到概览
                    overview = {
                        "symbol": symbol,
                        "exchange": exchange.value,
                        "interval": interval.value,
                        "count": len(data),
                        "start": data["datetime"].iloc[0] if not data.empty else "",
                        "end": data["datetime"].iloc[-1] if not data.empty else ""
                    }
                    result.append(overview)
        
        return result

    def get_tick_overview(self) -> List[Dict]:
        """获取Tick数据概览"""
        result = []
        
        for root, dirs, files in os.walk(Path(self.data_path) / "tick"):
            for file_name in files:
                if file_name.endswith(".csv"):
                    path = os.path.join(root, file_name)
                    parts = path.split(os.sep)
                    
                    # 从路径中提取信息
                    symbol = parts[-2]
                    exchange_name = parts[-1].split(".")[0]
                    
                    try:
                        exchange = Exchange(exchange_name)
                    except ValueError:
                        continue
                        
                    # 读取CSV获取统计信息
                    data = pd.read_csv(path)
                    
                    # 添加到概览
                    overview = {
                        "symbol": symbol,
                        "exchange": exchange.value,
                        "count": len(data),
                        "start": data["datetime"].iloc[0] if not data.empty else "",
                        "end": data["datetime"].iloc[-1] if not data.empty else ""
                    }
                    result.append(overview)
        
        return result

    def _get_bar_path(self, symbol: str, exchange: Exchange, interval: Interval) -> str:
        """获取K线数据文件路径"""
        path = Path(self.data_path) / "bar" / symbol / exchange.value / f"{interval.value}.csv"
        return str(path)

    def _get_tick_path(self, symbol: str, exchange: Exchange) -> str:
        """获取Tick数据文件路径"""
        path = Path(self.data_path) / "tick" / symbol / f"{exchange.value}.csv"
        return str(path)


# 注册CSV数据库
register_database("csv", CsvDatabase)
