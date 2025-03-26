"""
CSV文件数据库

基于CSV文件的数据存储实现，支持K线和Tick数据的读写
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from apilot.core import BarData, Exchange, Interval, TickData
from apilot.core.database import BaseDatabase, register_database
from apilot.utils.logger import get_logger

# 获取CSV数据库专用日志记录器
logger = get_logger("csv_database")


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
            
        # 从JSON配置加载CSV字段映射（只加载一次）
        from apilot.core.utility import load_json
        
        # 配置文件名
        self.setting_filename = "apilot_setting.json"
        
        # 加载配置
        config = load_json(self.setting_filename)
        
        # CSV字段映射默认值
        self.csv_field_mapping = {
            "datetime_field": config.get("csv_datetime_field", "datetime"),
            "open_field": config.get("csv_open_field", "open"),
            "high_field": config.get("csv_high_field", "high"),
            "low_field": config.get("csv_low_field", "low"),
            "close_field": config.get("csv_close_field", "close"),
            "volume_field": config.get("csv_volume_field", "volume")
        }

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """加载K线数据"""
        logger.info(f"CSV数据库: 请求加载 {symbol}.{exchange} 数据，区间 {interval}，时间 {start} 至 {end}")

        # 处理直接指定的CSV文件
        if self.is_direct_file:
            logger.info(f"使用直接CSV文件路径: {self.data_path}")
            if not os.path.exists(self.data_path):
                logger.error(f"CSV文件不存在: {self.data_path}")
                return []

            try:
                # 读取CSV文件
                data = pd.read_csv(self.data_path)
                logger.info(f"CSV文件已加载，行数: {len(data)}")

                # 获取字段映射（不在循环中重复访问）
                datetime_field = self.csv_field_mapping["datetime_field"]
                open_field = self.csv_field_mapping["open_field"]
                high_field = self.csv_field_mapping["high_field"]
                low_field = self.csv_field_mapping["low_field"]
                close_field = self.csv_field_mapping["close_field"]
                volume_field = self.csv_field_mapping["volume_field"]

                logger.info(f"字段映射: datetime={datetime_field}, open={open_field}, high={high_field}, low={low_field}, close={close_field}, volume={volume_field}")

                # 检查所有必要字段是否同时存在（单次检查）
                required_fields = [datetime_field, open_field, high_field, low_field, close_field, volume_field]
                missing_fields = [field for field in required_fields if field not in data.columns]

                if missing_fields:
                    logger.error(f"CSV文件缺少必要字段: {', '.join(missing_fields)}")
                    logger.error(f"可用字段: {', '.join(data.columns)}")
                    return []

                # 转换日期时间
                logger.info(f"转换日期时间字段: {datetime_field}")
                data[datetime_field] = pd.to_datetime(data[datetime_field])

                # 根据起止时间筛选数据（单次操作）
                mask = (data[datetime_field] >= start)
                if end:
                    mask &= (data[datetime_field] <= end)
                data = data[mask]
                logger.info(f"筛选后数据行数: {len(data)}")

                # 如果CSV中有symbol列，可以进一步筛选
                if 'symbol' in data.columns:
                    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                    data = data[data['symbol'] == base_symbol]
                    logger.info(f"按symbol={base_symbol}筛选后数据行数: {len(data)}")

                # 优化创建bar对象的过程，减少循环内的字典访问
                bars = []
                for _, row in data.iterrows():
                    dt = row[datetime_field]
                    
                    # 提前获取所有字段值，减少循环内访问
                    open_price = row[open_field]
                    high_price = row[high_field]
                    low_price = row[low_field]
                    close_price = row[close_field]
                    volume = row[volume_field]
                    
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=dt,
                        interval=interval,
                        volume=volume,
                        open_price=open_price,
                        high_price=high_price,
                        low_price=low_price,
                        close_price=close_price,
                        gateway_name="CSV",
                    )
                    bars.append(bar)

                logger.info(f"成功创建 {len(bars)} 个Bar对象")
                if bars:
                    logger.info(f"第一个Bar: {bars[0]}")
                    logger.info(f"最后一个Bar: {bars[-1]}")

                return bars

            except Exception as e:
                logger.error(f"加载CSV文件失败: {e}")
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
