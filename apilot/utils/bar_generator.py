"""
Bar/K-line aggregation and synchronization utilities for multi-symbol trading.
"""

import logging
from collections.abc import Callable
from datetime import datetime

from ..core.constant import Interval
from ..core.models import BarData

logger = logging.getLogger(__name__)


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
        symbols: list[str] | None = None,
    ) -> None:
        self.on_bar: Callable = on_bar
        self.on_window_bar: Callable | None = on_window_bar
        self.window: int = window
        self.interval: Interval = interval
        self.interval_count: int = 0
        self.symbols: set[str] | None = set(symbols) if symbols else None
        self.multi_symbol_mode: bool = self.symbols is not None
        self.window_time: datetime = None
        self.last_dt: datetime = None
        self.bars: dict[str, BarData] = {}
        self.window_bars: dict[str, BarData] = {}
        self.hour_bars: dict[str, BarData] = {}
        self.finished_hour_bars: dict[str, BarData] = {}

    def update_bar(self, bar: BarData) -> None:
        bars_dict = {bar.symbol: bar}
        self.update_bars(bars_dict)

    def update_bars(self, bars: dict[str, BarData]) -> None:
        if self.interval == Interval.MINUTE:
            self._update_minute_window(bars)
        else:
            self._update_hour_window(bars)

    def _update_minute_window(self, bars: dict[str, BarData]) -> None:
        if not bars:
            return
        sample_bar = next(iter(bars.values()))
        current_window_time = self._align_bar_datetime(sample_bar)
        if self.window_time is not None and current_window_time != self.window_time:
            if self.window_bars:
                if not self.multi_symbol_mode or self._is_window_complete():
                    self._finalize_window_bars()
        self.window_time = current_window_time
        for symbol, bar in bars.items():
            if self.multi_symbol_mode and self.symbols and symbol not in self.symbols:
                continue
            if symbol not in self.window_bars:
                dt = current_window_time
                self.window_bars[symbol] = BarData(
                    symbol=bar.symbol,
                    datetime=dt,
                    gateway_name=bar.gateway_name,
                    open_price=bar.open_price,
                    high_price=bar.high_price,
                    low_price=bar.low_price,
                    close_price=bar.close_price,
                    volume=bar.volume,
                )
            else:
                window_bar = self.window_bars[symbol]
                window_bar.high_price = max(window_bar.high_price, bar.high_price)
                window_bar.low_price = min(window_bar.low_price, bar.low_price)
                window_bar.close_price = bar.close_price
                window_bar.volume += getattr(bar, "volume", 0)
        if self.window > 0 and self.on_window_bar:
            if not (sample_bar.datetime.minute + 1) % self.window:
                if not self.multi_symbol_mode or self._is_window_complete():
                    self._finalize_window_bars()

    def _is_window_complete(self) -> bool:
        if not self.multi_symbol_mode or not self.symbols:
            return True
        return set(self.window_bars.keys()) >= self.symbols

    def _finalize_window_bars(self) -> None:
        if self.window_bars and self.on_window_bar:
            bar_info = []
            for symbol, bar in self.window_bars.items():
                bar_info.append(f"{symbol}@{bar.datetime}")
            # logger.debug(
            #     f"BarGenerator: sending window bar data [{', '.join(bar_info)}] to callback {self.on_window_bar.__name__}"
            # )
            self.on_window_bar(self.window_bars.copy())
            self.window_bars = {}

    def _update_hour_window(self, bars: dict[str, BarData]) -> None:
        for symbol, bar in bars.items():
            hour_bar = self._get_or_create_hour_bar(symbol, bar)
            if bar.datetime.minute == 59:
                self._update_bar_data(hour_bar, bar)
                self.finished_hour_bars[symbol] = hour_bar
                self.hour_bars[symbol] = None
            elif hour_bar and bar.datetime.hour != hour_bar.datetime.hour:
                self.finished_hour_bars[symbol] = hour_bar
                dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
                new_hour_bar = self._create_new_bar(bar, dt)
                self.hour_bars[symbol] = new_hour_bar
            else:
                self._update_bar_data(hour_bar, bar)
        if self.finished_hour_bars and self.on_window_bar:
            self.on_window_bar(self.finished_hour_bars)
            self.finished_hour_bars = {}

    def _get_or_create_hour_bar(self, symbol: str, bar: BarData) -> BarData:
        hour_bar = self.hour_bars.get(symbol)
        if not hour_bar:
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            hour_bar = self._create_new_bar(bar, dt)
            self.hour_bars[symbol] = hour_bar
        return hour_bar

    def _process_window_bars(self, bars: dict[str, BarData]) -> None:
        for symbol, bar in bars.items():
            window_bar = self.window_bars.get(symbol)
            if not window_bar:
                dt = self._align_bar_datetime(bar)
                window_bar = self._create_window_bar(bar, dt)
                self.window_bars[symbol] = window_bar
            else:
                self._update_window_bar(window_bar, bar)

    def _align_bar_datetime(self, bar: BarData) -> datetime:
        dt = bar.datetime.replace(second=0, microsecond=0)
        if self.interval == Interval.HOUR:
            dt = dt.replace(minute=0)
        elif self.window > 1:
            minute = (dt.minute // self.window) * self.window
            dt = dt.replace(minute=minute)
        return dt

    def _create_window_bar(self, source: BarData, dt: datetime) -> BarData:
        return BarData(
            symbol=source.symbol,
            datetime=dt,
            gateway_name=source.gateway_name,
            open_price=source.open_price,
            high_price=source.high_price,
            low_price=source.low_price,
            close_price=source.close_price,
            volume=source.volume,
        )

    def _create_new_bar(self, source: BarData, dt: datetime) -> BarData:
        return BarData(
            symbol=source.symbol,
            datetime=dt,
            gateway_name=source.gateway_name,
            open_price=source.open_price,
            high_price=source.high_price,
            low_price=source.low_price,
            close_price=source.close_price,
            volume=source.volume,
        )

    def _update_window_bar(self, target: BarData, source: BarData) -> None:
        target.high_price = max(target.high_price, source.high_price)
        target.low_price = min(target.low_price, source.low_price)
        target.close_price = source.close_price
        target.volume = getattr(target, "volume", 0) + source.volume

    def _update_bar_data(self, target: BarData, source: BarData) -> None:
        if target:
            target.high_price = max(target.high_price, source.high_price)
            target.low_price = min(target.low_price, source.low_price)
            target.close_price = source.close_price
            target.volume += source.volume

    def on_hour_bar(self, bars: dict[str, BarData]) -> None:
        if self.window == 1:
            self.on_window_bar(bars)
        else:
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
                    )
                    self.window_bars[symbol] = window_bar
                else:
                    window_bar.high_price = max(window_bar.high_price, bar.high_price)
                    window_bar.low_price = min(window_bar.low_price, bar.low_price)
                    window_bar.close_price = bar.close_price
                    window_bar.volume += bar.volume
            self.interval_count += 1
            if not self.interval_count % self.window:
                self.interval_count = 0
                self.on_window_bar(self.window_bars)
                self.window_bars = {}
