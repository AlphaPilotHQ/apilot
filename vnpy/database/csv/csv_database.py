"""
CSV Database Adapter for APilot
"""
import os
import glob
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.object import BarData, TickData
from vnpy.trader.database import BaseDatabase, BarOverview, TickOverview
from vnpy.trader.setting import SETTINGS


class CsvDatabase(BaseDatabase):
    """CSV文件数据库接口"""

    def __init__(self) -> None:
        """初始化CSV数据库"""
        # 读取配置
        self.data_path: str = SETTINGS.get("database.data_path", "csv_database")
        
        # 创建数据目录（如果不存在）
        self.bar_path: str = os.path.join(self.data_path, "bar_data")
        self.tick_path: str = os.path.join(self.data_path, "tick_data")
        
        os.makedirs(self.bar_path, exist_ok=True)
        os.makedirs(self.tick_path, exist_ok=True)
        
        # 用于缓存已加载的数据
        self.bar_cache: Dict[str, pd.DataFrame] = {}
        self.tick_cache: Dict[str, pd.DataFrame] = {}

    def save_bar_data(self, bars: List[BarData], stream: bool = False) -> bool:
        """保存K线数据到CSV文件"""
        if not bars:
            return False
            
        # 按照合约和周期分组
        bar_data_map = {}
        for bar in bars:
            key = f"{bar.symbol}_{bar.exchange.value}_{bar.interval.value}"
            if key not in bar_data_map:
                bar_data_map[key] = []
            bar_data_map[key].append(bar)
            
        # 保存每个分组的数据到单独的CSV文件
        for key, bar_list in bar_data_map.items():
            symbol, exchange, interval = key.split("_")
            
            # 创建DataFrame
            data = []
            for bar in bar_list:
                data.append({
                    "datetime": bar.datetime,
                    "open": bar.open_price,
                    "high": bar.high_price,
                    "low": bar.low_price,
                    "close": bar.close_price,
                    "volume": bar.volume,
                    "open_interest": bar.open_interest if hasattr(bar, "open_interest") else 0
                })
            
            df = pd.DataFrame(data)
            
            # 生成文件路径
            filename = f"{symbol}_{exchange}_{interval}.csv"
            filepath = os.path.join(self.bar_path, filename)
            
            # 如果文件已存在，则合并数据
            if os.path.exists(filepath):
                # 清除相应的缓存
                cache_key = f"{symbol}_{exchange}_{interval}"
                if cache_key in self.bar_cache:
                    del self.bar_cache[cache_key]
                    
                # 读取现有数据
                existing_df = pd.read_csv(filepath)
                if "datetime" in existing_df.columns:
                    existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])
                    
                    # 合并并去重
                    df = pd.concat([existing_df, df])
                    df = df.drop_duplicates(subset=["datetime"])
                    df = df.sort_values("datetime")
            
            # 保存CSV文件
            df.to_csv(filepath, index=False)
        
        return True

    def save_tick_data(self, ticks: List[TickData], stream: bool = False) -> bool:
        """保存Tick数据到CSV文件"""
        if not ticks:
            return False
            
        # 按照合约分组
        tick_data_map = {}
        for tick in ticks:
            key = f"{tick.symbol}_{tick.exchange.value}"
            if key not in tick_data_map:
                tick_data_map[key] = []
            tick_data_map[key].append(tick)
            
        # 保存每个分组的数据到单独的CSV文件
        for key, tick_list in tick_data_map.items():
            symbol, exchange = key.split("_")
            
            # 创建DataFrame
            data = []
            for tick in tick_list:
                # 提取Tick对象的所有相关字段
                tick_dict = {
                    "datetime": tick.datetime,
                    "last_price": tick.last_price,
                    "volume": tick.volume,
                    "bid_price_1": tick.bid_price_1,
                    "bid_volume_1": tick.bid_volume_1,
                    "ask_price_1": tick.ask_price_1,
                    "ask_volume_1": tick.ask_volume_1,
                }
                
                # 添加可能存在的额外字段
                for i in range(2, 6):
                    bid_price_key = f"bid_price_{i}"
                    bid_volume_key = f"bid_volume_{i}"
                    ask_price_key = f"ask_price_{i}"
                    ask_volume_key = f"ask_volume_{i}"
                    
                    if hasattr(tick, bid_price_key):
                        tick_dict[bid_price_key] = getattr(tick, bid_price_key)
                        tick_dict[bid_volume_key] = getattr(tick, bid_volume_key)
                        tick_dict[ask_price_key] = getattr(tick, ask_price_key)
                        tick_dict[ask_volume_key] = getattr(tick, ask_volume_key)
                
                data.append(tick_dict)
            
            df = pd.DataFrame(data)
            
            # 生成文件路径
            filename = f"{symbol}_{exchange}_tick.csv"
            filepath = os.path.join(self.tick_path, filename)
            
            # 如果文件已存在，则合并数据
            if os.path.exists(filepath):
                # 清除相应的缓存
                cache_key = f"{symbol}_{exchange}"
                if cache_key in self.tick_cache:
                    del self.tick_cache[cache_key]
                    
                # 读取现有数据
                existing_df = pd.read_csv(filepath)
                if "datetime" in existing_df.columns:
                    existing_df["datetime"] = pd.to_datetime(existing_df["datetime"])
                    
                    # 合并并去重
                    df = pd.concat([existing_df, df])
                    df = df.drop_duplicates(subset=["datetime"])
                    df = df.sort_values("datetime")
            
            # 保存CSV文件
            df.to_csv(filepath, index=False)
        
        return True

    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """从CSV文件加载K线数据"""
        # 生成标准文件名
        filename = f"{symbol}_{exchange.value}_{interval.value}.csv"
        
        # 先检查在项目根目录是否存在同名文件
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        root_filepath = os.path.join(project_root, filename)
        
        # 如果根目录有对应文件，直接使用
        if os.path.exists(root_filepath):
            filepath = root_filepath
            print(f"使用根目录文件: {filepath}")
        else:
            # 回退到标准路径
            filepath = os.path.join(self.bar_path, filename)
            if not os.path.exists(filepath):
                print(f"错误: 未找到数据文件: {filename}，在根目录或数据目录均不存在")
                return []
        
        # 尝试从缓存加载数据
        cache_key = f"{symbol}_{exchange.value}_{interval.value}"
        df = self.bar_cache.get(cache_key, None)
        
        # 如果缓存中没有，则从文件加载
        if df is None:
            try:
                df = pd.read_csv(filepath)
                
                # 转换时间列
                datetime_col = None
                for col in ["datetime", "candle_begin_time", "date", "time", "Date", "Time"]:
                    if col in df.columns:
                        datetime_col = col
                        break
                
                if datetime_col:
                    # 确保datetime列已转换为datetime类型
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    
                    # 如果列名不是'datetime'，则重命名
                    if datetime_col != "datetime":
                        df = df.rename(columns={datetime_col: "datetime"})
                else:
                    print(f"CSV文件中未找到时间列: {filepath}")
                    return []
                
                # 标准化列名
                column_mapping = {
                    "open_price": "open",
                    "high_price": "high",
                    "low_price": "low",
                    "close_price": "close"
                }
                
                for target, source in column_mapping.items():
                    if source in df.columns and target not in df.columns:
                        df[target] = df[source]
                
                # 检查并映射标准列名
                if "open" in df.columns and "open_price" not in df.columns:
                    df["open_price"] = df["open"]
                if "high" in df.columns and "high_price" not in df.columns:
                    df["high_price"] = df["high"]
                if "low" in df.columns and "low_price" not in df.columns:
                    df["low_price"] = df["low"]
                if "close" in df.columns and "close_price" not in df.columns:
                    df["close_price"] = df["close"]
                
                # 缓存数据
                self.bar_cache[cache_key] = df
            except Exception as e:
                print(f"加载CSV文件失败: {e}")
                return []
        
        # 确保datetime列是datetime类型
        if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
        
        # 筛选时间范围
        df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        
        # 转换为BarData对象列表
        bars = []
        for _, row in df.iterrows():
            # 检查必要的价格列是否存在
            if not all(x in row.index for x in ["open_price", "high_price", "low_price", "close_price"]):
                continue
                
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=row["datetime"],
                interval=interval,
                open_price=row["open_price"],
                high_price=row["high_price"],
                low_price=row["low_price"],
                close_price=row["close_price"],
                volume=row["volume"] if "volume" in row else 0,
                open_interest=row["open_interest"] if "open_interest" in row else 0,
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
        """从CSV文件加载Tick数据"""
        # 生成文件路径
        filename = f"{symbol}_{exchange.value}_tick.csv"
        filepath = os.path.join(self.tick_path, filename)
        
        # 文件不存在，则返回空列表
        if not os.path.exists(filepath):
            return []
        
        # 尝试从缓存加载数据
        cache_key = f"{symbol}_{exchange.value}"
        df = self.tick_cache.get(cache_key, None)
        
        # 如果缓存中没有，则从文件加载
        if df is None:
            try:
                df = pd.read_csv(filepath)
                
                # 转换时间列
                datetime_col = None
                for col in ["datetime", "date", "time", "Date", "Time"]:
                    if col in df.columns:
                        datetime_col = col
                        break
                
                if datetime_col:
                    # 确保datetime列已转换为datetime类型
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    
                    # 如果列名不是'datetime'，则重命名
                    if datetime_col != "datetime":
                        df = df.rename(columns={datetime_col: "datetime"})
                else:
                    print(f"CSV文件中未找到时间列: {filepath}")
                    return []
                
                # 缓存数据
                self.tick_cache[cache_key] = df
            except Exception as e:
                print(f"加载CSV文件失败: {e}")
                return []
        
        # 确保datetime列是datetime类型
        if "datetime" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = pd.to_datetime(df["datetime"])
        
        # 筛选时间范围
        df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]
        
        # 转换为TickData对象列表
        ticks = []
        for _, row in df.iterrows():
            tick_dict = {
                "symbol": symbol,
                "exchange": exchange,
                "datetime": row["datetime"],
                "gateway_name": "CSV"
            }
            
            # 添加其他可能存在的字段
            for field in [
                "name", "volume", "last_price", "last_volume",
                "limit_up", "limit_down",
                "open_price", "high_price", "low_price", "close_price",
                "pre_close", "bid_price_1", "bid_price_2", "bid_price_3", "bid_price_4", "bid_price_5",
                "ask_price_1", "ask_price_2", "ask_price_3", "ask_price_4", "ask_price_5",
                "bid_volume_1", "bid_volume_2", "bid_volume_3", "bid_volume_4", "bid_volume_5",
                "ask_volume_1", "ask_volume_2", "ask_volume_3", "ask_volume_4", "ask_volume_5",
            ]:
                if field in row:
                    field_name = field
                    # 将列名从snake_case转换为CamelCase (如果需要)
                    tick_dict[field_name] = row[field]
            
            tick = TickData(**tick_dict)
            ticks.append(tick)
        
        return ticks

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除指定合约的K线数据"""
        # 生成文件路径
        filename = f"{symbol}_{exchange.value}_{interval.value}.csv"
        filepath = os.path.join(self.bar_path, filename)
        
        # 文件不存在，则返回0
        if not os.path.exists(filepath):
            return 0
        
        # 统计行数
        try:
            df = pd.read_csv(filepath)
            count = len(df)
            
            # 删除文件
            os.remove(filepath)
            
            # 删除缓存
            cache_key = f"{symbol}_{exchange.value}_{interval.value}"
            if cache_key in self.bar_cache:
                del self.bar_cache[cache_key]
                
            return count
        except Exception as e:
            print(f"删除CSV文件失败: {e}")
            return 0

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除指定合约的Tick数据"""
        # 生成文件路径
        filename = f"{symbol}_{exchange.value}_tick.csv"
        filepath = os.path.join(self.tick_path, filename)
        
        # 文件不存在，则返回0
        if not os.path.exists(filepath):
            return 0
        
        # 统计行数
        try:
            df = pd.read_csv(filepath)
            count = len(df)
            
            # 删除文件
            os.remove(filepath)
            
            # 删除缓存
            cache_key = f"{symbol}_{exchange.value}"
            if cache_key in self.tick_cache:
                del self.tick_cache[cache_key]
                
            return count
        except Exception as e:
            print(f"删除CSV文件失败: {e}")
            return 0

    def get_bar_overview(self) -> List[BarOverview]:
        """获取K线数据概览"""
        overviews = []
        
        # 列出所有CSV文件
        for filename in os.listdir(self.bar_path):
            if filename.endswith(".csv"):
                try:
                    # 解析文件名以获取合约信息
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        symbol = parts[0]
                        exchange_value = parts[1]
                        interval_value = parts[2].replace(".csv", "")
                        
                        exchange = Exchange(exchange_value)
                        interval = Interval(interval_value)
                        
                        # 读取CSV文件以获取数据概览
                        filepath = os.path.join(self.bar_path, filename)
                        df = pd.read_csv(filepath)
                        
                        # 转换时间列
                        datetime_col = None
                        for col in ["datetime", "candle_begin_time", "date", "time", "Date", "Time"]:
                            if col in df.columns:
                                datetime_col = col
                                break
                        
                        if datetime_col:
                            # 确保datetime列已转换为datetime类型
                            df[datetime_col] = pd.to_datetime(df[datetime_col])
                            
                            # 如果列名不是'datetime'，则重命名
                            if datetime_col != "datetime":
                                df = df.rename(columns={datetime_col: "datetime"})
                        else:
                            print(f"CSV文件中未找到时间列: {filepath}")
                            continue
                        
                        # 创建概览对象
                        overview = BarOverview(
                            symbol=symbol,
                            exchange=exchange,
                            interval=interval,
                            count=len(df),
                            start=df["datetime"].min() if not df.empty else None,
                            end=df["datetime"].max() if not df.empty else None
                        )
                        
                        overviews.append(overview)
                except Exception as e:
                    print(f"处理CSV文件失败: {filename}, 错误: {e}")
        
        return overviews

    def get_tick_overview(self) -> List[TickOverview]:
        """获取Tick数据概览"""
        overviews = []
        
        # 列出所有CSV文件
        for filename in os.listdir(self.tick_path):
            if filename.endswith(".csv"):
                try:
                    # 解析文件名以获取合约信息
                    parts = filename.split("_")
                    if len(parts) >= 3 and parts[2] == "tick.csv":
                        symbol = parts[0]
                        exchange_value = parts[1]
                        
                        exchange = Exchange(exchange_value)
                        
                        # 读取CSV文件以获取数据概览
                        filepath = os.path.join(self.tick_path, filename)
                        df = pd.read_csv(filepath)
                        
                        # 转换时间列
                        datetime_col = None
                        for col in ["datetime", "date", "time", "Date", "Time"]:
                            if col in df.columns:
                                datetime_col = col
                                break
                        
                        if datetime_col:
                            # 确保datetime列已转换为datetime类型
                            df[datetime_col] = pd.to_datetime(df[datetime_col])
                            
                            # 如果列名不是'datetime'，则重命名
                            if datetime_col != "datetime":
                                df = df.rename(columns={datetime_col: "datetime"})
                        else:
                            print(f"CSV文件中未找到时间列: {filepath}")
                            continue
                        
                        # 创建概览对象
                        overview = TickOverview(
                            symbol=symbol,
                            exchange=exchange,
                            count=len(df),
                            start=df["datetime"].min() if not df.empty else None,
                            end=df["datetime"].max() if not df.empty else None
                        )
                        
                        overviews.append(overview)
                except Exception as e:
                    print(f"处理CSV文件失败: {filename}, 错误: {e}")
        
        return overviews


# 数据库实例
Database = CsvDatabase
