"""
CSV文件数据库

基于CSV文件的数据存储实现，支持K线和Tick数据的读写
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from apilot.core import BarData, Exchange, Interval, TickData, SETTINGS

from .database import BaseDatabase, register_database


class CsvDatabase(BaseDatabase):
    """CSV文件数据库"""

    def __init__(self, data_path: str = "csv_database") -> None:
        """构造函数"""
        self.data_path = data_path

        # 检查data_path是否是直接指向CSV文件
        self.is_direct_file = data_path.endswith(".csv") if isinstance(data_path, str) else False

        # 如果是目录，确保存在
        if not self.is_direct_file:
            os.makedirs(self.data_path, exist_ok=True)

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """加载K线数据"""
        print(f"CSV数据库: 请求加载 {symbol}.{exchange} 数据，区间 {interval}，时间 {start} 至 {end}")

        # 处理直接指定的CSV文件
        if self.is_direct_file:
            print(f"使用直接CSV文件路径: {self.data_path}")
            if not os.path.exists(self.data_path):
                print(f"CSV文件不存在: {self.data_path}")
                return []

            try:
                # 读取CSV文件
                data = pd.read_csv(self.data_path)
                print(f"CSV文件已加载，行数: {len(data)}")

                # 尝试使用配置中的字段映射
                from apilot.trader.setting import SETTINGS
                datetime_field = SETTINGS.get("csv_datetime_field", "datetime")
                open_field = SETTINGS.get("csv_open_field", "open")
                high_field = SETTINGS.get("csv_high_field", "high")
                low_field = SETTINGS.get("csv_low_field", "low")
                close_field = SETTINGS.get("csv_close_field", "close")
                volume_field = SETTINGS.get("csv_volume_field", "volume")

                print(f"字段映射: datetime={datetime_field}, open={open_field}, high={high_field}, low={low_field}, close={close_field}, volume={volume_field}")

                # 确保所需字段存在
                missing_fields = []
                for field_name, default in [
                    (datetime_field, "datetime"),
                    (open_field, "open"),
                    (high_field, "high"),
                    (low_field, "low"),
                    (close_field, "close"),
                    (volume_field, "volume")
                ]:
                    if field_name not in data.columns:
                        # 尝试使用默认字段名
                        if default != field_name and default in data.columns:
                            if field_name == datetime_field:
                                datetime_field = default
                            elif field_name == open_field:
                                open_field = default
                            elif field_name == high_field:
                                high_field = default
                            elif field_name == low_field:
                                low_field = default
                            elif field_name == close_field:
                                close_field = default
                            elif field_name == volume_field:
                                volume_field = default
                        else:
                            missing_fields.append(field_name)

                if missing_fields:
                    print(f"CSV文件缺少必要字段: {', '.join(missing_fields)}")
                    print(f"可用字段: {', '.join(data.columns)}")
                    return []

                # 转换日期时间
                print(f"转换日期时间字段: {datetime_field}")
                data[datetime_field] = pd.to_datetime(data[datetime_field])

                # 根据起止时间筛选数据
                data = data[(data[datetime_field] >= start) & (data[datetime_field] <= end)]
                print(f"筛选后数据行数: {len(data)}")

                # 构建bar数据
                bars = []
                for _, row in data.iterrows():
                    dt = row[datetime_field].to_pydatetime()

                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        datetime=dt,
                        open_price=row[open_field],
                        high_price=row[high_field],
                        low_price=row[low_field],
                        close_price=row[close_field],
                        volume=row[volume_field],
                        gateway_name="CSV"
                    )
                    bars.append(bar)

                print(f"成功创建 {len(bars)} 个Bar对象")
                if bars:
                    print(f"第一个Bar: {bars[0]}")
                    print(f"最后一个Bar: {bars[-1]}")

                return bars

            except Exception as e:
                print(f"加载CSV文件失败: {e}")
                import traceback
                traceback.print_exc()
                return []

        # 原有的目录结构处理逻辑
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
