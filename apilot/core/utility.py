"""
General utility functions.
"""

from collections.abc import Callable
from datetime import datetime
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
    Enhanced bar generator for time-based bar aggregation.

    Key features:
    1. Generate 1-minute bars from tick data
    2. Generate x-minute bars from 1-minute bars
    3. Generate hourly bars from 1-minute bars

    Time intervals:
    - Minute: x must divide 60 evenly (2, 3, 5, 6, 10, 15, 20, 30)
    - Hour: any positive integer is valid
    """

    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable | None = None,
        interval: Interval = Interval.MINUTE,
    ) -> None:
        """
        Initialize bar generator with callbacks and configuration.

        Args:
            on_bar: Callback when bar is generated
            window: Number of source bars to aggregate (0 means no aggregation)
            on_window_bar: Callback when window bar is complete
            interval: Bar interval type (minute or hour)
        """
        # Callbacks
        self.on_bar: Callable = on_bar
        self.on_window_bar: Callable | None = on_window_bar

        # Configuration
        self.window: int = window
        self.interval: Interval = interval
        self.interval_count: int = 0

        # State tracking
        self.last_dt: datetime = None

        # For tick to bar conversion
        self.bars: dict[str, BarData] = {}
        self.last_ticks: dict[str, TickData] = {}

        # For bar aggregation
        self.window_bars: dict[str, BarData] = {}

        # For hourly bar handling
        self.hour_bars: dict[str, BarData] = {}
        self.finished_hour_bars: dict[str, BarData] = {}

    def update_tick(self, tick: TickData) -> None:
        """
        Update tick data and generate 1-minute bars.

        Args:
            tick: The tick data to process
        """
        if not tick.last_price:
            return

        # Check if we need to finish the current bar
        if self.last_dt and self.last_dt.minute != tick.datetime.minute:
            for bar in self.bars.values():
                bar.datetime = bar.datetime.replace(second=0, microsecond=0)

            # Call the callback with the current bars
            self.on_bar(self.bars)
            self.bars = {}

        # Get or create bar for this symbol
        bar = self._get_or_create_tick_bar(tick)

        # Update the bar with this tick
        self._update_tick_bar(bar, tick)

        # Update tracking state
        self.last_dt = tick.datetime

    def _get_or_create_tick_bar(self, tick: TickData) -> BarData:
        """Get existing bar or create a new one for this tick."""
        bar: BarData | None = self.bars.get(tick.symbol)
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
        return bar

    def _update_tick_bar(self, bar: BarData, tick: TickData) -> None:
        """Update a bar with new tick data."""
        # Update OHLC
        bar.high_price = max(bar.high_price, tick.last_price)
        bar.low_price = min(bar.low_price, tick.last_price)
        bar.close_price = tick.last_price
        bar.open_interest = tick.open_interest
        bar.datetime = tick.datetime

        # Update volume and turnover based on the difference from last tick
        last_tick: TickData | None = self.last_ticks.get(tick.symbol)
        if last_tick:
            bar.volume += max(tick.volume - last_tick.volume, 0)
            bar.turnover += max(tick.turnover - last_tick.turnover, 0)

        # Store the tick for the next update
        self.last_ticks[tick.symbol] = tick

    def update_bar(self, bar: BarData) -> None:
        """
        Update with a single bar data and generate aggregated bars.

        Args:
            bar: Single bar data to process
        """
        bars_dict = {bar.symbol: bar}
        if self.interval == Interval.MINUTE:
            self._update_minute_window(bars_dict)
        else:
            self._update_hour_window(bars_dict)

    def _update_minute_window(self, bars: dict[str, BarData]) -> None:
        """Process bars for minute-based aggregation."""
        self._process_window_bars(bars)

        # Check if window is complete (based on minute count)
        if bars and any(bars.values()):
            # Get any bar to check the time
            sample_bar = next(iter(bars.values()))
            # Check if we completed a window
            if not (sample_bar.datetime.minute + 1) % self.window:
                self._finalize_window_bars()

    def _update_hour_window(self, bars: dict[str, BarData]) -> None:
        """Process bars for hourly aggregation."""
        # Process each bar
        for symbol, bar in bars.items():
            hour_bar = self._get_or_create_hour_bar(symbol, bar)

            # Check for hour boundary conditions
            if bar.datetime.minute == 59:
                # End of hour - update and finalize
                self._update_bar_data(hour_bar, bar)
                self.finished_hour_bars[symbol] = hour_bar
                self.hour_bars[symbol] = None
            elif hour_bar and bar.datetime.hour != hour_bar.datetime.hour:
                # New hour - finalize old bar and create new one
                self.finished_hour_bars[symbol] = hour_bar

                # Create new hour bar
                dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
                new_hour_bar = self._create_new_bar(bar, dt)
                self.hour_bars[symbol] = new_hour_bar
            else:
                # Within same hour - just update
                self._update_bar_data(hour_bar, bar)

        # Send completed hour bars
        if self.finished_hour_bars:
            self.on_hour_bar(self.finished_hour_bars)
            self.finished_hour_bars = {}

    def _get_or_create_hour_bar(self, symbol: str, bar: BarData) -> BarData:
        """Get existing hour bar or create a new one."""
        hour_bar = self.hour_bars.get(symbol)
        if not hour_bar:
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            hour_bar = self._create_new_bar(bar, dt)
            self.hour_bars[symbol] = hour_bar
        return hour_bar

    def _process_window_bars(self, bars: dict[str, BarData]) -> None:
        """Process bars for window aggregation."""
        for symbol, bar in bars.items():
            window_bar = self.window_bars.get(symbol)

            # Create window bar if it doesn't exist
            if not window_bar:
                # Align time to window start
                dt = self._align_bar_datetime(bar)
                window_bar = self._create_window_bar(bar, dt)
                self.window_bars[symbol] = window_bar
            else:
                # Update existing window bar
                self._update_window_bar(window_bar, bar)

    def _align_bar_datetime(self, bar: BarData) -> datetime:
        """Align datetime to appropriate window boundary."""
        dt = bar.datetime.replace(second=0, microsecond=0)
        if self.interval == Interval.HOUR:
            dt = dt.replace(minute=0)
        elif self.window > 1:
            # For X-minute bars, align to the window start
            minute = (dt.minute // self.window) * self.window
            dt = dt.replace(minute=minute)
        return dt

    def _create_window_bar(self, source: BarData, dt: datetime) -> BarData:
        """Create a new window bar from source bar."""
        return BarData(
            symbol=source.symbol,
            exchange=source.exchange,
            datetime=dt,
            gateway_name=source.gateway_name,
            open_price=source.open_price,
            high_price=source.high_price,
            low_price=source.low_price,
        )

    def _create_new_bar(self, source: BarData, dt: datetime) -> BarData:
        """Create a complete new bar from source bar."""
        return BarData(
            symbol=source.symbol,
            exchange=source.exchange,
            datetime=dt,
            gateway_name=source.gateway_name,
            open_price=source.open_price,
            high_price=source.high_price,
            low_price=source.low_price,
            close_price=source.close_price,
            volume=source.volume,
            turnover=source.turnover,
            open_interest=source.open_interest,
        )

    def _update_window_bar(self, target: BarData, source: BarData) -> None:
        """Update window bar with new source bar data."""
        # Update OHLC
        target.high_price = max(target.high_price, source.high_price)
        target.low_price = min(target.low_price, source.low_price)
        target.close_price = source.close_price

        # Accumulate volume, turnover, etc.
        target.volume = getattr(target, "volume", 0) + source.volume
        target.turnover = getattr(target, "turnover", 0) + source.turnover
        target.open_interest = source.open_interest

    def _update_bar_data(self, target: BarData, source: BarData) -> None:
        """Update bar data with new source values."""
        if target:
            target.high_price = max(target.high_price, source.high_price)
            target.low_price = min(target.low_price, source.low_price)
            target.close_price = source.close_price
            target.volume += source.volume
            target.turnover += source.turnover
            target.open_interest = source.open_interest

    def _finalize_window_bars(self) -> None:
        """Finalize and send window bars, then reset."""
        if self.window_bars and self.on_window_bar:
            self.on_window_bar(self.window_bars)
            self.window_bars = {}

    def on_hour_bar(self, bars: dict[str, BarData]) -> None:
        """
        Process completed hour bars.

        Args:
            bars: Dictionary of hour bars
        """
        if self.window == 1:
            # Direct pass-through for 1-hour window
            self.on_window_bar(bars)
        else:
            # Process for X-hour window
            for symbol, bar in bars.items():
                window_bar = self.window_bars.get(symbol)
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

            # Check if window is complete
            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bar(self.window_bars)
                self.window_bars = {}


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
