"""
General utility functions.
"""

from collections.abc import Callable
from datetime import datetime, time
from decimal import Decimal
from math import ceil, floor

import numpy as np

from .constant import Exchange, Interval
from .object import BarData, TickData


def extract_symbol(symbol: str) -> tuple[str, Exchange]:
    """Extract base symbol and exchange from full trading symbol"""
    # 使用新的工具函数实现
    from apilot.utils import split_symbol

    base_symbol, exchange_str = split_symbol(symbol)
    return base_symbol, Exchange(exchange_str)


def generate_symbol(base_symbol: str, exchange: Exchange) -> str:
    """Generate full trading symbol from base symbol and exchange"""
    return f"{base_symbol}.{exchange.value}"


def round_to(value: float, target: float) -> float:
    """
    Round price to price tick value.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    rounded: float = float(round(value / target) * target)
    return rounded


def floor_to(value: float, target: float) -> float:
    """
    Similar to math.floor function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(floor(value / target) * target)
    return result


def ceil_to(value: float, target: float) -> float:
    """
    Similar to math.ceil function, but to target float number.
    """
    value: Decimal = Decimal(str(value))
    target: Decimal = Decimal(str(target))
    result: float = float(ceil(value / target) * target)
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
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable | None = None,
        interval: Interval = Interval.MINUTE,
        daily_end: time | None = None,
    ) -> None:
        """Constructor"""
        self.on_bar: Callable = on_bar

        self.interval: Interval = interval
        self.interval_count: int = 0

        self.bars: dict[str, BarData] = {}
        self.last_ticks: dict[str, TickData] = {}

        self.hour_bars: dict[str, BarData] = {}
        self.finished_hour_bars: dict[str, BarData] = {}

        self.window: int = window
        self.window_bars: dict[str, BarData] = {}
        self.on_window_bar: Callable | None = on_window_bar

        self.last_dt: datetime = None

        self.daily_end: time | None = daily_end
        if self.interval == Interval.DAILY and not self.daily_end:
            raise RuntimeError("Daily bar generation requires daily end time")

    def update_tick(self, tick: TickData) -> None:
        """Update tick data"""
        if not tick.last_price:
            return

        if self.last_dt and self.last_dt.minute != tick.datetime.minute:
            for bar in self.bars.values():
                bar.datetime = bar.datetime.replace(second=0, microsecond=0)

            self.on_bar(self.bars)
            self.bars = {}

        bar: BarData | None = self.bars.get(tick.symbol, None)
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
                open_interest=tick.open_interest,
            )
            self.bars[bar.symbol] = bar
        else:
            bar.high_price = max(bar.high_price, tick.last_price)
            bar.low_price = min(bar.low_price, tick.last_price)
            bar.close_price = tick.last_price
            bar.open_interest = tick.open_interest
            bar.datetime = tick.datetime

        last_tick: TickData | None = self.last_ticks.get(tick.symbol, None)
        if last_tick:
            bar.volume += max(tick.volume - last_tick.volume, 0)
            bar.turnover += max(tick.turnover - last_tick.turnover, 0)

        self.last_ticks[tick.symbol] = tick
        self.last_dt = tick.datetime

    def update_bars(self, bars: dict[str, BarData]) -> None:
        """Update bars"""
        if self.interval == Interval.MINUTE:
            self.update_bar_minute_window(bars)
        else:
            self.update_bar_hour_window(bars)

    def update_bar_minute_window(self, bars: dict[str, BarData]) -> None:
        """Update minute bar"""
        for symbol, bar in bars.items():
            window_bar: BarData | None = self.window_bars.get(symbol, None)

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
                    low_price=bar.low_price,
                )
                self.window_bars[symbol] = window_bar

            # 更新K线内最高价及最低价
            else:
                window_bar.high_price = max(window_bar.high_price, bar.high_price)
                window_bar.low_price = min(window_bar.low_price, bar.low_price)

            # 更新K线内收盘价、数量、成交额、持仓量
            window_bar.close_price = bar.close_price
            window_bar.volume += bar.volume
            window_bar.turnover += bar.turnover
            window_bar.open_interest = bar.open_interest

        # 检查K线是否合成完毕
        if not (bar.datetime.minute + 1) % self.window:
            self.on_window_bar(self.window_bars)
            self.window_bars = {}

    def update_bar_hour_window(self, bars: dict[str, BarData]) -> None:
        """Update hour bar"""
        for symbol, bar in bars.items():
            hour_bar: BarData | None = self.hour_bars.get(symbol, None)

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
                    open_interest=bar.open_interest,
                )
                self.hour_bars[symbol] = hour_bar

            else:
                # 如果收到59分的分钟K线,更新小时K线并推送
                if bar.datetime.minute == 59:
                    hour_bar.high_price = max(hour_bar.high_price, bar.high_price)
                    hour_bar.low_price = min(hour_bar.low_price, bar.low_price)

                    hour_bar.close_price = bar.close_price
                    hour_bar.volume += bar.volume
                    hour_bar.turnover += bar.turnover
                    hour_bar.open_interest = bar.open_interest

                    self.finished_hour_bars[symbol] = hour_bar
                    self.hour_bars[symbol] = None

                # 如果收到新的小时的分钟K线,直接推送当前的小时K线
                elif bar.datetime.hour != hour_bar.datetime.hour:
                    self.finished_hour_bars[symbol] = hour_bar

                    dt: datetime = bar.datetime.replace(
                        minute=0, second=0, microsecond=0
                    )
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
                        open_interest=bar.open_interest,
                    )
                    self.hour_bars[symbol] = hour_bar

                # 否则直接更新小时K线
                else:
                    hour_bar.high_price = max(hour_bar.high_price, bar.high_price)
                    hour_bar.low_price = min(hour_bar.low_price, bar.low_price)

                    hour_bar.close_price = bar.close_price
                    hour_bar.volume += bar.volume
                    hour_bar.turnover += bar.turnover
                    hour_bar.open_interest = bar.open_interest

        # 推送合成完毕的小时K线
        if self.finished_hour_bars:
            self.on_hour_bar(self.finished_hour_bars)
            self.finished_hour_bars = {}

    def update_bar_daily_window(self, bar: BarData) -> None:
        """Update daily bar"""
        symbol = bar.symbol

        window_bar = self.window_bars.get(symbol, None)
        if not window_bar:
            # Create if not exists
            window_bar = BarData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                datetime=bar.datetime,
                gateway_name=bar.gateway_name,
                open_price=bar.open_price,
                high_price=bar.high_price,
                low_price=bar.low_price,
                close_price=bar.close_price,
                volume=bar.volume,
                turnover=bar.turnover,
                open_interest=bar.open_interest,
            )
            self.window_bars[symbol] = window_bar

    def on_hour_bar(self, bars: dict[str, BarData]) -> None:
        """Push hour bar"""
        if self.window == 1:
            self.on_window_bar(bars)
        else:
            for symbol, bar in bars.items():
                window_bar: BarData | None = self.window_bars.get(symbol, None)
                if not window_bar:
                    window_bar = BarData(
                        symbol=bar.symbol,
                        exchange=bar.exchange,
                        datetime=bar.datetime,
                        gateway_name=bar.gateway_name,
                        open_price=bar.open_price,
                        high_price=bar.high_price,
                        low_price=bar.low_price,
                    )
                    self.window_bars[symbol] = window_bar
                else:
                    window_bar.high_price = max(window_bar.high_price, bar.high_price)
                    window_bar.low_price = min(window_bar.low_price, bar.low_price)

                window_bar.close_price = bar.close_price
                window_bar.volume += bar.volume
                window_bar.turnover += bar.turnover
                window_bar.open_interest = bar.open_interest

            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bar(self.window_bars)
                self.window_bars = {}

    def generate(self) -> BarData | None:
        """
        Generate the bar data and call callback immediately.
        """
        bar: BarData = self.bar

        if self.bar:
            bar.datetime = bar.datetime.replace(second=0, microsecond=0)
            self.on_bar(bar)

        self.bar = None
        return bar


class ArrayManager:
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

    # 使用numpy实现的技术指标
    def sma(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Simple moving average.
        """
        if not self.close:
            return 0.0

        # 使用numpy实现
        weights = np.ones(n) / n
        result = np.convolve(self.close, weights, mode="valid")
        # 补齐长度与self.close一致
        padding = np.full(n - 1, np.nan)
        result = np.concatenate((padding, result))

        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Exponential moving average.
        """
        if not self.close:
            return 0.0

        # 使用numpy实现EMA
        alpha = 2 / (n + 1)
        result = np.zeros_like(self.close)
        result[0] = self.close[0]
        for i in range(1, len(self.close)):
            result[i] = alpha * self.close[i] + (1 - alpha) * result[i - 1]

        if array:
            return result
        return result[-1]

    def std(self, n: int, nbdev: int = 1, array: bool = False) -> float | np.ndarray:
        """
        Calculate standard deviation.
        """
        if not self.inited:
            return 0.0

        # Efficiently calculate standard deviation with NumPy
        result = np.std(self.close[-n:], ddof=1) * nbdev

        # np.std 对一维数组返回标量值,不需要索引
        if array:
            return result
        return result  # 直接返回结果,不使用索引

    def atr(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Average True Range (ATR).
        """
        if not self.close:
            return 0.0

        # 使用numpy实现ATR
        tr = np.zeros_like(self.close)
        for i in range(1, len(self.close)):
            high_low = self.high[i] - self.low[i]
            high_close = abs(self.high[i] - self.close[i - 1])
            low_close = abs(self.low[i] - self.close[i - 1])
            tr[i] = max(high_low, high_close, low_close)
        tr[0] = tr[1]  # 第一个值取第二个值

        # 计算ATR
        result = np.zeros_like(self.close)
        result[0] = tr[0]
        for i in range(1, len(tr)):
            result[i] = (result[i - 1] * (n - 1) + tr[i]) / n

        if array:
            return result
        return result[-1]

    def rsi(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Relative Strenght Index (RSI).
        """
        if not self.close:
            return 0.0

        # 使用numpy实现RSI
        diff = np.diff(self.close)
        diff = np.append([0], diff)  # 补齐长度

        # 计算上涨和下跌
        up = np.where(diff > 0, diff, 0)
        down = np.where(diff < 0, -diff, 0)

        # 计算平均上涨和平均下跌
        up_avg = np.zeros_like(self.close)
        down_avg = np.zeros_like(self.close)

        # 初始值
        up_avg[n] = np.mean(up[1 : n + 1])
        down_avg[n] = np.mean(down[1 : n + 1])

        # 计算平均值
        for i in range(n + 1, len(self.close)):
            up_avg[i] = (up_avg[i - 1] * (n - 1) + up[i]) / n
            down_avg[i] = (down_avg[i - 1] * (n - 1) + down[i]) / n

        # 计算相对强度和RSI
        rs = up_avg / np.where(down_avg == 0, 0.001, down_avg)  # 避免除零
        result = 100 - (100 / (1 + rs))
        result[:n] = np.nan  # 前n个值置为NaN

        if array:
            return result
        return result[-1]

    def macd(
        self,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        array: bool = False,
    ) -> tuple[float, float, float] | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MACD.
        """
        if not self.close:
            return 0.0, 0.0, 0.0

        # 使用numpy实现MACD
        # 计算快速和慢速EMA
        ema_fast = np.zeros_like(self.close)
        ema_slow = np.zeros_like(self.close)

        # 计算初始EMA
        ema_fast[0] = self.close[0]
        ema_slow[0] = self.close[0]

        # EMA权重
        alpha_fast = 2 / (fast_period + 1)
        alpha_slow = 2 / (slow_period + 1)

        # 计算EMA序列
        for i in range(1, len(self.close)):
            ema_fast[i] = (
                alpha_fast * self.close[i] + (1 - alpha_fast) * ema_fast[i - 1]
            )
            ema_slow[i] = (
                alpha_slow * self.close[i] + (1 - alpha_slow) * ema_slow[i - 1]
            )

        # 计算MACD线
        macd = ema_fast - ema_slow

        # 计算信号线 (MACD的EMA)
        signal = np.zeros_like(self.close)
        signal[0] = macd[0]
        alpha_signal = 2 / (signal_period + 1)

        for i in range(1, len(macd)):
            signal[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal[i - 1]

        # 计算直方图
        hist = macd - signal

        if array:
            return macd, signal, hist
        return macd[-1], signal[-1], hist[-1]

    def donchian(
        self, n: int, array: bool = False
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Donchian Channel.
        """
        if not self.close:
            return 0.0, 0.0

        # 使用numpy实现唐奇安通道
        up = np.zeros_like(self.high)
        down = np.zeros_like(self.low)

        for i in range(len(self.high)):
            if i >= n - 1:
                up[i] = np.max(self.high[i - n + 1 : i + 1])
                down[i] = np.min(self.low[i - n + 1 : i + 1])
            else:
                up[i] = np.nan
                down[i] = np.nan

        if array:
            return up, down
        return up[-1], down[-1]

    def mfi(self, n: int, array: bool = False) -> float | np.ndarray:
        """
        Money Flow Index.
        """
        if not self.close or not self.volume:
            return 0.0

        # 使用numpy实现MFI
        # 计算典型价格
        tp = (self.high + self.low + self.close) / 3

        # 计算资金流
        mf = tp * self.volume

        # 计算正负资金流
        diff = np.diff(tp)
        diff = np.append([0], diff)  # 补齐长度

        positive_flow = np.where(diff > 0, mf, 0)
        negative_flow = np.where(diff < 0, mf, 0)

        # 计算n周期的正负资金流
        positive_mf = np.zeros_like(self.close)
        negative_mf = np.zeros_like(self.close)

        for i in range(n, len(self.close)):
            positive_mf[i] = np.sum(positive_flow[i - n + 1 : i + 1])
            negative_mf[i] = np.sum(negative_flow[i - n + 1 : i + 1])

        # 计算资金流比率和MFI
        mfr = np.divide(
            positive_mf,
            negative_mf,
            out=np.ones_like(positive_mf),
            where=negative_mf != 0,
        )
        result = 100 - (100 / (1 + mfr))
        result[:n] = np.nan  # 前n个值置为NaN

        if array:
            return result
        return result[-1]

    def boll(
        self, n: int, dev: float, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Bollinger Channel.
        """
        mid: float | np.ndarray = self.sma(n, array)
        std: float | np.ndarray = self.std(n, 1, array)

        up: float | np.ndarray = mid + std * dev
        down: float | np.ndarray = mid - std * dev

        return up, down

    def keltner(
        self, n: int, dev: float, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """
        Keltner Channel.
        """
        mid: float | np.ndarray = self.sma(n, array)
        atr: float | np.ndarray = self.atr(n, array)

        up: float | np.ndarray = mid + atr * dev
        down: float | np.ndarray = mid - atr * dev

        return up, down


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func
