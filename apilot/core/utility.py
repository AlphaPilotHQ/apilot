"""
General utility functions.
"""

import json
import logging
import sys
import random
from datetime import datetime, time
from pathlib import Path
from typing import Callable, Dict, Tuple, Union, Optional, Any, TypeVar
from decimal import Decimal
from math import floor, ceil
from functools import wraps
from time import sleep

import numpy as np
from .object import BarData, TickData
from .constant import Exchange, Interval

# 定义泛型返回类型
T = TypeVar('T')


# 日志格式定义
log_formatter: logging.Formatter = logging.Formatter("[%(asctime)s] %(message)s")


def extract_vt_symbol(vt_symbol: str) -> Tuple[str, Exchange]:
    """从交易所符号中提取标的和交易所"""
    symbol, exchange_str = vt_symbol.rsplit(".", 1)
    return symbol, Exchange(exchange_str)


def generate_vt_symbol(symbol: str, exchange: Exchange) -> str:
    """生成交易所符号"""
    return f"{symbol}.{exchange.value}"


def _get_trader_dir(temp_name: str) -> Tuple[Path, Path]:
    """获取交易程序运行路径

    首先检查当前工作目录是否存在指定的临时目录，
    如果存在则使用当前目录作为交易路径，
    否则使用系统主目录。
    """
    cwd: Path = Path.cwd()
    temp_path: Path = cwd.joinpath(temp_name)

    # 如果临时目录存在于当前工作目录中，则使用当前目录作为交易路径
    if temp_path.exists():
        return cwd, temp_path

    # 否则使用系统主目录
    home_path: Path = Path.home()
    temp_path: Path = home_path.joinpath(temp_name)

    # 如果主目录下不存在临时目录，则创建
    if not temp_path.exists():
        temp_path.mkdir()

    return home_path, temp_path


# 使用.apilot作为临时目录名，替代原先的.vntrader
TRADER_DIR, TEMP_DIR = _get_trader_dir(".apilot")
sys.path.append(str(TRADER_DIR))


def get_file_path(filename: str) -> Path:
    """获取临时文件完整路径"""
    return TEMP_DIR.joinpath(filename)


def get_folder_path(folder_name: str) -> Path:
    """获取文件夹路径，如不存在则创建"""
    folder_path: Path = TEMP_DIR.joinpath(folder_name)
    if not folder_path.exists():
        folder_path.mkdir()
    return folder_path


def load_json(filename: str) -> dict:
    """
    Load data from json file in temp path.
    """
    filepath: Path = get_file_path(filename)
    if not filepath.exists():
        return {}

    with open(filepath, mode="r", encoding="UTF-8") as f:
        data: dict = json.load(f)
    return data


def save_json(filename: str, data: dict) -> None:
    """
    Save data into json file in temp path.
    """
    filepath: Path = get_file_path(filename)
    with open(filepath, mode="w+", encoding="UTF-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )


def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    rounded: float = float(int(round(value / target)) * target)
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(int(floor(value / target)) * target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(int(ceil(value / target)) * target)
    return result


def get_digits(value: float) -> int:
    """
    Get number of digits after decimal point.
    """
    value_str: str = str(value)

    if "e-" in value_str:
        _, buf = value_str.split("e-")
        return int(buf)
    elif "." in value_str:
        _, buf = value_str.split(".")
        return len(buf)
    else:
        return 0


class BarGenerator:
    """
    For:
    1. generating 1 minute bar data from tick data
    2. generating x minute bar/x hour bar data from 1 minute data
    Notice:
    1. for x minute bar, x must be able to divide 60: 2, 3, 5, 6, 10, 15, 20, 30
    2. for x hour bar, x can be any number
    """

    def __init__(
        self,
        on_bars: Callable,
        window: int = 0,
        on_window_bars: Callable = None,
        interval: Interval = Interval.MINUTE,
        daily_end: time = None
    ) -> None:
        """构造函数"""
        self.on_bars: Callable = on_bars

        self.interval: Interval = interval
        self.interval_count: int = 0

        self.bars: dict[str, BarData] = {}
        self.last_ticks: dict[str, TickData] = {}

        self.hour_bars: dict[str, BarData] = {}
        self.finished_hour_bars: dict[str, BarData] = {}

        self.window: int = window
        self.window_bars: dict[str, BarData] = {}
        self.on_window_bars: Callable = on_window_bars

        self.last_dt: datetime = None

        self.daily_end: time = daily_end
        if self.interval == Interval.DAILY and not self.daily_end:
            raise RuntimeError(_("合成日K线必须传入每日收盘时间"))

    def update_tick(self, tick: TickData) -> None:
        """更新行情切片数据"""
        if not tick.last_price:
            return

        if self.last_dt and self.last_dt.minute != tick.datetime.minute:
            for bar in self.bars.values():
                bar.datetime = bar.datetime.replace(second=0, microsecond=0)

            self.on_bars(self.bars)
            self.bars = {}

        bar: Optional[BarData] = self.bars.get(tick.vt_symbol, None)
        if not bar:
            bar = BarData(
                symbol=tick.symbol,
                exchange=tick.exchange,
                interval=Interval.MINUTE,
                datetime=tick.datetime,
                gateway_name=tick.gateway_name,
                open_price=tick.last_price,
                high_price=tick.last_price,
                low_price=tick.last_price,
                close_price=tick.last_price,
                open_interest=tick.open_interest
            )
            self.bars[bar.vt_symbol] = bar
        else:
            bar.high_price = max(bar.high_price, tick.last_price)
            bar.low_price = min(bar.low_price, tick.last_price)
            bar.close_price = tick.last_price
            bar.open_interest = tick.open_interest
            bar.datetime = tick.datetime

        last_tick: Optional[TickData] = self.last_ticks.get(tick.vt_symbol, None)
        if last_tick:
            bar.volume += max(tick.volume - last_tick.volume, 0)
            bar.turnover += max(tick.turnover - last_tick.turnover, 0)

        self.last_ticks[tick.vt_symbol] = tick
        self.last_dt = tick.datetime

    def update_bars(self, bars: dict[str, BarData]) -> None:
        """更新一分钟K线"""
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bars)
        else:
            self.update_bar_hour_window(bars)

    def update_bar_minute_window(self, bars: dict[str, BarData]) -> None:
        """更新N分钟K线"""
        for vt_symbol, bar in bars.items():
            window_bar: Optional[BarData] = self.window_bars.get(vt_symbol, None)

            # 如果没有N分钟K线则创建
            if not window_bar:
                dt: datetime = bar.datetime.replace(second=0, microsecond=0)
                window_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=dt,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price
                )
                self.window_bars[vt_symbol] = window_bar

            # 更新K线内最高价及最低价
            else:
                window_bar.high_price = max(
                    window_bar.high_price,
                    bar.high_price
                )
                window_bar.low_price = min(
                    window_bar.low_price,
                    bar.low_price
                )

            # 更新K线内收盘价、数量、成交额、持仓量
            window_bar.close_price = bar.close_price
            window_bar.volume += bar.volume
            window_bar.turnover += bar.turnover
            window_bar.open_interest = bar.open_interest

        # 检查K线是否合成完毕
        if not (bar.datetime.minute + 1) % self.window:
            self.on_window_bars(self.window_bars)
            self.window_bars = {}

    def update_bar_hour_window(self, bars: dict[str, BarData]) -> None:
        """更新小时K线"""
        for vt_symbol, bar in bars.items():
            hour_bar: Optional[BarData] = self.hour_bars.get(vt_symbol, None)

            # 如果没有小时K线则创建
            if not hour_bar:
                dt: datetime = bar.datetime.replace(minute=0, second=0, microsecond=0)
                hour_bar = BarData(
                    symbol=bar.symbol,
                    exchange=bar.exchange,
                    datetime=dt,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price,
                    close_price=bar.close_price,
                    volume=bar.volume,
                    turnover=bar.turnover,
                    open_interest=bar.open_interest
                )
                self.hour_bars[vt_symbol] = hour_bar

            else:
                # 如果收到59分的分钟K线，更新小时K线并推送
                if bar.datetime.minute == 59:
                    hour_bar.high_price = max(
                        hour_bar.high_price,
                        bar.high_price
                    )
                    hour_bar.low_price = min(
                        hour_bar.low_price,
                        bar.low_price
                    )

                    hour_bar.close_price = bar.close_price
                    hour_bar.volume += bar.volume
                    hour_bar.turnover += bar.turnover
                    hour_bar.open_interest = bar.open_interest

                    self.finished_hour_bars[vt_symbol] = hour_bar
                    self.hour_bars[vt_symbol] = None

                # 如果收到新的小时的分钟K线，直接推送当前的小时K线
                elif bar.datetime.hour != hour_bar.datetime.hour:
                    self.finished_hour_bars[vt_symbol] = hour_bar

                    dt: datetime = bar.datetime.replace(minute=0, second=0, microsecond=0)
                    hour_bar = BarData(
                        symbol=bar.symbol,
                        exchange=bar.exchange,
                        datetime=dt,
                        gateway_name=bar.gateway_name,
                        open_price=bar.open_price,
                        high_price=bar.high_price,
                        low_price=bar.low_price,
                        close_price=bar.close_price,
                        volume=bar.volume,
                        turnover=bar.turnover,
                        open_interest=bar.open_interest
                    )
                    self.hour_bars[vt_symbol] = hour_bar

                # 否则直接更新小时K线
                else:
                    hour_bar.high_price = max(
                        hour_bar.high_price,
                        bar.high_price
                    )
                    hour_bar.low_price = min(
                        hour_bar.low_price,
                        bar.low_price
                    )

                    hour_bar.close_price = bar.close_price
                    hour_bar.volume += bar.volume
                    hour_bar.turnover += bar.turnover
                    hour_bar.open_interest = bar.open_interest

        # 推送合成完毕的小时K线
        if self.finished_hour_bars:
            self.on_hour_bars(self.finished_hour_bars)
            self.finished_hour_bars = {}

    def on_hour_bars(self, bars: dict[str, BarData]) -> None:
        """推送小时K线"""
        if self.window == 1:
            self.on_window_bars(bars)
        else:
            for vt_symbol, bar in bars.items():
                window_bar: Optional[BarData] = self.window_bars.get(vt_symbol, None)
                if not window_bar:
                    window_bar = BarData(
                        symbol=bar.symbol,
                        exchange=bar.exchange,
                        datetime=bar.datetime,
                        gateway_name=bar.gateway_name,
                        open_price=bar.open_price,
                        high_price=bar.high_price,
                        low_price=bar.low_price
                    )
                    self.window_bars[vt_symbol] = window_bar
                else:
                    window_bar.high_price = max(
                        window_bar.high_price,
                        bar.high_price
                    )
                    window_bar.low_price = min(
                        window_bar.low_price,
                        bar.low_price
                    )

                window_bar.close_price = bar.close_price
                window_bar.volume += bar.volume
                window_bar.turnover += bar.turnover
                window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bars(self.window_bars)
                self.window_bars = {}


    def update_bar_daily_window(self, bar: BarData) -> None:
        """"""
        # If not inited, create daily bar object
        if not self.daily_bar:
            self.daily_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=bar.datetime,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price
            )
        # Otherwise, update high/low price into daily bar
        else:
            self.daily_bar.high_price = max(
                self.daily_bar.high_price,
                bar.high_price
            )
            self.daily_bar.low_price = min(
                self.daily_bar.low_price,
                bar.low_price
            )

        # Update close price/volume/turnover into daily bar
        self.daily_bar.close_price = bar.close_price
        self.daily_bar.volume += bar.volume
        self.daily_bar.turnover += bar.turnover
        self.daily_bar.open_interest = bar.open_interest

        # Check if daily bar completed
        if bar.datetime.time() == self.daily_end:
            self.daily_bar.datetime = bar.datetime.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            )
            self.on_window_bar(self.daily_bar)

            self.daily_bar = None

    def generate(self) -> Optional[BarData]:
        """
        Generate the bar data and call callback immediately.
        """
        bar: BarData = self.bar

        if self.bar:
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
            self.on_bar(bar)

        self.bar = None
        return bar


class ArrayManager(object):
    """
    For:
    1. time series container of bar data
    2. calculating technical indicator value
    """

    def __init__(self, size: int = 100) -> None:
        """Constructor"""
        self.count: int = 0
        self.size: int = size
        self.inited: bool = False

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)
        self.turnover_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

    def update_bar(self, bar: BarData) -> None:
        """
        Update new bar data into array manager.
        """
        self.count += 1
        if not self.inited and self.count >= self.size:
            self.inited = True

        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]
        self.turnover_array[:-1] = self.turnover_array[1:]
        self.open_interest_array[:-1] = self.open_interest_array[1:]

        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume
        self.turnover_array[-1] = bar.turnover
        self.open_interest_array[-1] = bar.open_interest

    @property
    def open(self) -> np.ndarray:
        """
        Get open price time series.
        """
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        """
        Get high price time series.
        """
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        """
        Get low price time series.
        """
        return self.low_array

    @property
    def close(self) -> np.ndarray:
        """
        Get close price time series.
        """
        return self.close_array

    @property
    def volume(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.volume_array

    @property
    def turnover(self) -> np.ndarray:
        """
        Get trading turnover time series.
        """
        return self.turnover_array

    @property
    def open_interest(self) -> np.ndarray:
        """
        Get trading volume time series.
        """
        return self.open_interest_array

    # TODO：改成numpy而非talib计算
    def sma(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Simple moving average.
        """
        result: np.ndarray = talib.SMA(self.close, n)
        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Exponential moving average.
        """
        result: np.ndarray = talib.EMA(self.close, n)
        if array:
            return result
        return result[-1]

    def std(self, n: int, nbdev: int = 1, array: bool = False) -> Union[float, np.ndarray]:
        """
        计算标准差 - 高效NumPy实现
        """
        # 确保数据足够计算
        if not self.inited:
            return 0 if not array else np.zeros(len(self.close))

        # 创建结果数组
        result = np.full_like(self.close, 0.0)

        # 计算有效位置的标准差 (使用纯NumPy操作)
        for i in range(n-1, len(self.close)):
            # 使用NumPy的std函数直接计算窗口的标准差
            result[i] = np.std(self.close[i-n+1:i+1]) * nbdev

        if array:
            return result
        return result[-1]

    def atr(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Average True Range (ATR).
        """
        result: np.ndarray = talib.ATR(self.high, self.low, self.close, n)
        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Relative Strenght Index (RSI).
        """
        result: np.ndarray = talib.RSI(self.close, n)
        if array:
            return result
        return result[-1]

    def macd(
        self,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[float, float, float]
    ]:
        """
        MACD.
        """
        macd, signal, hist = talib.MACD(
            self.close, fast_period, slow_period, signal_period
        )
        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]


    def boll(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Bollinger Channel.
        """
        mid: Union[float, np.ndarray] = self.sma(n, array)
        std: Union[float, np.ndarray] = self.std(n, 1, array)

        up: Union[float, np.ndarray] = mid + std * dev
        down: Union[float, np.ndarray] = mid - std * dev

        return up, down

    def keltner(
        self,
        n: int,
        dev: float,
        array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Keltner Channel.
        """
        mid: Union[float, np.ndarray] = self.sma(n, array)
        atr: Union[float, np.ndarray] = self.atr(n, array)

        up: Union[float, np.ndarray] = mid + atr * dev
        down: Union[float, np.ndarray] = mid - atr * dev

        return up, down

    def donchian(
        self, n: int, array: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[float, float]
    ]:
        """
        Donchian Channel.
        """
        up: np.ndarray = talib.MAX(self.high, n)
        down: np.ndarray = talib.MIN(self.low, n)

        if array:
            return up, down
        return up[-1], down[-1]


    def mfi(self, n: int, array: bool = False) -> Union[float, np.ndarray]:
        """
        Money Flow Index.
        """
        result: np.ndarray = talib.MFI(self.high, self.low, self.close, self.volume, n)
        if array:
            return result
        return result[-1]


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func


import threading

# 全局变量
_logger_lock = threading.Lock()  # 添加线程锁
_file_loggers = {}               # 缓存日志器对象（不只是处理器）

def get_file_logger(filename: str) -> logging.Logger:
    """
    获取写入指定文件的日志记录器（线程安全单例模式）
    """
    # 使用线程锁保护共享资源访问
    with _logger_lock:
        # 检查是否已有该文件的logger
        if filename in _file_loggers:
            return _file_loggers[filename]

        # 创建新logger
        logger = logging.getLogger(filename)

        # 检查是否已经有处理器添加到这个logger
        has_file_handler = False
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and \
               getattr(handler, 'baseFilename', None) == filename:
                has_file_handler = True
                break

        # 只有在没有相应处理器时才添加新的
        if not has_file_handler:
            handler = logging.FileHandler(filename)
            handler.setFormatter(log_formatter)
            logger.addHandler(handler)

        # 缓存并返回logger对象
        _file_loggers[filename] = logger
        return logger
