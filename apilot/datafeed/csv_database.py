"""
CSV文件数据库

基于CSV文件的数据存储实现，支持K线和Tick数据的读写
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import glob

import pandas as pd

from apilot.core import BarData, Exchange, Interval, TickData
from apilot.core.database import BaseDatabase, register_database
from apilot.utils.logger import get_logger

# 获取CSV数据库专用日志记录器
logger = get_logger("csv_database")


class CsvDatabase(BaseDatabase):
    """CSV文件数据库"""

    def __init__(self, csv_path: str = None):
        """
        初始化CSV数据库
        
        Args:
            csv_path: CSV文件的路径，如果不提供则需要在运行时指定
        """
        self.csv_path = csv_path
        
        # 默认使用列名方式
        self.use_column_index = False
        
        # 列名映射
        self.datetime_field = "datetime"  
        self.open_field = "open"
        self.high_field = "high"
        self.low_field = "low"
        self.close_field = "close"
        self.volume_field = "volume"
        
        # 列索引映射
        self.datetime_index = 0
        self.open_index = 1
        self.high_index = 2
        self.low_index = 3
        self.close_index = 4
        self.volume_index = 5
        self.openinterest_index = -1  # 默认不使用
        
        # 日期格式
        self.dtformat = '%Y-%m-%d %H:%M:%S'
    
    def set_column_mode(self, use_index: bool = False):
        """设置使用列索引还是列名模式"""
        self.use_column_index = use_index
    
    def set_index_mapping(self, **kwargs):
        """
        设置列索引映射
        
        Args:
            **kwargs: 包含datetime, open, high, low, close, volume, 
                     openinterest, dtformat等字段的列索引
        """
        self.use_column_index = True
        
        if 'datetime' in kwargs:
            self.datetime_index = kwargs['datetime']
        if 'open' in kwargs:
            self.open_index = kwargs['open']
        if 'high' in kwargs:
            self.high_index = kwargs['high']
        if 'low' in kwargs:
            self.low_index = kwargs['low']
        if 'close' in kwargs:
            self.close_index = kwargs['close']
        if 'volume' in kwargs:
            self.volume_index = kwargs['volume']
        if 'openinterest' in kwargs:
            self.openinterest_index = kwargs['openinterest']
        if 'dtformat' in kwargs:
            self.dtformat = kwargs['dtformat']

    def _get_file_path(self, symbol: str, exchange: Exchange, interval: Interval) -> str:
        """获取文件路径"""
        if self.csv_path:
            # 如果设置了直接文件路径，则直接返回
            logger.info(f"使用直接CSV文件路径: {self.csv_path}")
            return self.csv_path
        
        # 否则尝试从数据目录查找文件
        file_pattern = f"{symbol}_{exchange.value}_{interval.value}*.csv"
        data_dir = "./data"  # 默认数据目录
        pattern_path = os.path.join(data_dir, file_pattern)
        
        matching_files = glob.glob(pattern_path)
        if matching_files:
            path = matching_files[0]
            logger.info(f"找到匹配的CSV文件: {path}")
            return path
        
        # 如果找不到匹配的文件，创建一个标准文件名
        path = os.path.join(data_dir, f"{symbol}_{exchange.value}_{interval.value}.csv")
        logger.warning(f"未找到匹配的CSV文件，将使用: {path}")
        return path
    
    def load_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval,
        start: datetime,
        end: datetime
    ) -> List[BarData]:
        """
        加载K线数据
        
        参数:
            symbol: 交易对名称
            exchange: 交易所
            interval: K线周期
            start: 开始时间
            end: 结束时间
            
        返回:
            List[BarData]: K线数据列表
        """
        logger.info(f"CSV数据库: 请求加载 {symbol}.{exchange} 数据，区间 {interval}，时间 {start} 至 {end}")
        
        # 获取文件路径
        file_path = self._get_file_path(symbol, exchange, interval)
        
        # 如果文件不存在，返回空列表
        if not os.path.exists(file_path):
            logger.warning(f"CSV文件不存在: {file_path}")
            return []
        
        # 加载CSV文件
        try:
            # 读取CSV文件到DataFrame
            df = pd.read_csv(file_path)
            logger.info(f"CSV文件已加载，行数: {len(df)}")
            
            # 处理数据
            bars = []
            
            if self.use_column_index:
                # 使用列索引模式
                if len(df.columns) <= max(
                    self.datetime_index, 
                    self.open_index, 
                    self.high_index, 
                    self.low_index, 
                    self.close_index, 
                    self.volume_index
                ):
                    logger.error(f"CSV文件列数不足，需要至少 {max(self.datetime_index, self.open_index, self.high_index, self.low_index, self.close_index, self.volume_index) + 1} 列")
                    return []
                
                # 转换日期时间列
                dt_col = df.iloc[:, self.datetime_index]
                logger.info(f"转换日期时间字段: 索引{self.datetime_index}, 格式{self.dtformat}")
                
                # 转换日期时间列为datetime格式
                df.iloc[:, self.datetime_index] = pd.to_datetime(dt_col, format=self.dtformat)
                
                # 筛选时间范围内的数据
                df = df[(df.iloc[:, self.datetime_index] >= pd.Timestamp(start)) & 
                         (df.iloc[:, self.datetime_index] <= pd.Timestamp(end))]
                
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
                        gateway_name="CSV"
                    )
                    
                    # 添加持仓量（如果有）
                    if self.openinterest_index >= 0:
                        bar.open_interest = float(row.iloc[self.openinterest_index])
                        
                    bars.append(bar)
            else:
                # 使用列名模式（向后兼容）
                # 检查必要的列是否存在
                required_fields = [
                    self.datetime_field,
                    self.open_field,
                    self.high_field,
                    self.low_field,
                    self.close_field,
                    self.volume_field
                ]
                
                for field in required_fields:
                    if field not in df.columns:
                        logger.error(f"CSV文件缺少必要的列: {field}")
                        return []
                
                logger.info(f"字段映射: datetime={self.datetime_field}, open={self.open_field}, high={self.high_field}, low={self.low_field}, close={self.close_field}, volume={self.volume_field}")
                
                # 转换日期时间列为datetime格式
                if df[self.datetime_field].dtype == 'object':
                    logger.info(f"转换日期时间字段: {self.datetime_field}")
                    df[self.datetime_field] = pd.to_datetime(df[self.datetime_field])
                
                # 筛选时间范围内的数据
                df = df[(df[self.datetime_field] >= pd.Timestamp(start)) & 
                         (df[self.datetime_field] <= pd.Timestamp(end))]
                
                logger.info(f"筛选后数据行数: {len(df)}")
                
                # 生成Bar对象
                for _, row in df.iterrows():
                    bar = BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=row[self.datetime_field],
                        interval=interval,
                        volume=float(row[self.volume_field]),
                        open_price=float(row[self.open_field]),
                        high_price=float(row[self.high_field]),
                        low_price=float(row[self.low_field]),
                        close_price=float(row[self.close_field]),
                        gateway_name="CSV"
                    )
                    bars.append(bar)
            
            logger.info(f"成功创建 {len(bars)} 个Bar对象")
            if bars:
                logger.info(f"第一个Bar: {bars[0]}")
                logger.info(f"最后一个Bar: {bars[-1]}")
            
            return bars
        
        except Exception as e:
            logger.error(f"加载CSV数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def load_tick_data(
        self,
        symbol: str,
        exchange: Exchange,
        start: datetime,
        end: datetime
    ) -> List[TickData]:
        """加载Tick数据"""
        # 暂不实现
        return []

    def delete_bar_data(
        self,
        symbol: str,
        exchange: Exchange,
        interval: Interval
    ) -> int:
        """删除K线数据"""
        # 当前实现不支持删除操作
        return 0

    def delete_tick_data(
        self,
        symbol: str,
        exchange: Exchange
    ) -> int:
        """删除Tick数据"""
        # 当前实现不支持删除操作
        return 0

    def get_bar_overview(self) -> List[Dict]:
        """获取K线数据概览"""
        # 简单实现，仅返回空列表
        return []

    def get_tick_overview(self) -> List[Dict]:
        """获取Tick数据概览"""
        # 简单实现，仅返回空列表
        return []


# 注册CSV数据库
register_database("csv", CsvDatabase)
