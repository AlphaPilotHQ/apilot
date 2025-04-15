"""
General utility functions.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from math import ceil, floor

from .constant import Interval
from .object import BarData

logger = logging.getLogger(__name__)


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


class BarGenerator:
    """
    Enhanced bar generator supporting multi-symbol synchronization.

    Main features:
    1. Generate 1-minute bars from tick data
    2. Generate X-minute bars from 1-minute bars
    3. Generate hourly bars from 1-minute bars
    4. Support multiple trading symbols simultaneously

    Time intervals:
    - Minutes: x must be divisible by 60 (2, 3, 5, 6, 10, 15, 20, 30)
    - Hours: any positive integer is valid
    """

    def __init__(
        self,
        on_bar: Callable,
        window: int = 0,
        on_window_bar: Callable | None = None,
        interval: Interval = Interval.MINUTE,
        symbols: list[str]
        | None = None,  # New parameter: list of symbols to synchronize
    ) -> None:
        """
        Initialize bar generator with callbacks and configuration.

        Args:
            on_bar: Callback when bar is generated
            window: Number of source bars to aggregate (0 means no aggregation)
            on_window_bar: Callback when window bar is complete
            interval: Bar interval type (minute or hour)
            symbols: List of symbols to synchronize (None means single-symbol mode)
        """
        # Callbacks
        self.on_bar: Callable = on_bar
        self.on_window_bar: Callable | None = on_window_bar

        # Configuration
        self.window: int = window
        self.interval: Interval = interval
        self.interval_count: int = 0

        # Multi-symbol support
        self.symbols: set[str] | None = set(symbols) if symbols else None
        self.multi_symbol_mode: bool = self.symbols is not None
        self.window_time: datetime = None  # Current window time

        # State tracking
        self.last_dt: datetime = None

        # For bar storage
        self.bars: dict[str, BarData] = {}

        # For bar aggregation - dictionary to store all symbols
        self.window_bars: dict[str, BarData] = {}

        # For hourly bar handling
        self.hour_bars: dict[str, BarData] = {}
        self.finished_hour_bars: dict[str, BarData] = {}

    def update_bar(self, bar: BarData) -> None:
        """
        Update with a single bar data and generate aggregated bars.

        Args:
            bar: Single bar data to process
        """
        # Convert single bar to dictionary format
        bars_dict = {bar.symbol: bar}
        self.update_bars(bars_dict)

    def update_bars(self, bars: dict[str, BarData]) -> None:
        """
        Update bar data for multiple symbols

        Args:
            bars: Dictionary of bar data for multiple symbols, format {symbol: bar}
        """
        if self.interval == Interval.MINUTE:
            self._update_minute_window(bars)
        else:
            self._update_hour_window(bars)

    def _update_minute_window(self, bars: dict[str, BarData]) -> None:
        """Process minute-level bars with multi-symbol support"""
        if not bars:
            return

        # Get current window time
        sample_bar = next(iter(bars.values()))
        current_window_time = self._align_bar_datetime(sample_bar)

        # If this is a new window time
        if self.window_time is not None and current_window_time != self.window_time:
            # Process data from previous window
            if self.window_bars:
                # In multi-symbol mode, check if all symbols are collected
                if not self.multi_symbol_mode or self._is_window_complete():
                    self._finalize_window_bars()

        # Update window time
        self.window_time = current_window_time

        # Process bar data for each symbol
        for symbol, bar in bars.items():
            # In multi-symbol mode, filter symbols not in target set
            if self.multi_symbol_mode and self.symbols and symbol not in self.symbols:
                continue

            # Get or create window_bar for current symbol
            if symbol not in self.window_bars:
                # Create new window_bar with aligned time
                dt = current_window_time  # Use already aligned time
                self.window_bars[symbol] = BarData(
                    symbol=bar.symbol,
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
            else:
                # Update existing window_bar
                window_bar = self.window_bars[symbol]
                window_bar.high_price = max(window_bar.high_price, bar.high_price)
                window_bar.low_price = min(window_bar.low_price, bar.low_price)
                window_bar.close_price = bar.close_price
                window_bar.volume += getattr(bar, "volume", 0)
                window_bar.turnover += getattr(bar, "turnover", 0)
                window_bar.open_interest = bar.open_interest

        # Check if we've reached window boundary
        if self.window > 0 and self.on_window_bar:
            # Check if we've reached end of window (e.g., every 5 minutes)
            if not (sample_bar.datetime.minute + 1) % self.window:
                # In multi-symbol mode, only trigger callback when all symbols are ready
                if not self.multi_symbol_mode or self._is_window_complete():
                    self._finalize_window_bars()

    def _is_window_complete(self) -> bool:
        """Check if current window contains data for all required symbols"""
        if not self.multi_symbol_mode or not self.symbols:
            return True
        return set(self.window_bars.keys()) >= self.symbols

    def _finalize_window_bars(self) -> None:
        """Process and send window bar data"""
        if self.window_bars and self.on_window_bar:
            # Log what we're about to send
            bar_info = []
            for symbol, bar in self.window_bars.items():
                bar_info.append(f"{symbol}@{bar.datetime}")
            logger.debug(
                f"BarGenerator: sending window bar data [{', '.join(bar_info)}] to callback {self.on_window_bar.__name__}"
            )

            # Send a copy of window data
            self.on_window_bar(self.window_bars.copy())
            self.window_bars = {}

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
        if self.finished_hour_bars and self.on_window_bar:
            self.on_window_bar(self.finished_hour_bars)
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

    def _create_new_bar(self, source: BarData, dt: datetime) -> BarData:
        """Create a complete new bar from source bar."""
        return BarData(
            symbol=source.symbol,
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


def virtual(func: Callable) -> Callable:
    """
    mark a function as "virtual", which means that this function can be override.
    any base class should use this or @abstractmethod to decorate all functions
    that can be (re)implemented by subclasses.
    """
    return func
