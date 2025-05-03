"""
Bar/K-line aggregation utilities for single-symbol trading.
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..core.constant import Interval
from ..core.models import BarData

logger = logging.getLogger(__name__)


class BarGenerator:

    def __init__(
        self,
        on_bar: Callable[[BarData], Any],
        window: int = 1,
        on_window_bar: Callable[[BarData], Any] | None = None,
        interval: Interval = Interval.MINUTE,
    ) -> None:

        self.window: int = max(1, window)

        self.on_bar: Callable[[BarData], Any] = on_bar
        self.on_window_bar: Callable[[BarData], Any] | None = on_window_bar

        self.interval: Interval = interval
        self.hour_count: int = 0

        self.window_bar: BarData | None = None
        self.hour_bar: BarData | None = None

        self.window_time: datetime | None = None

    def update_bar(self, bar: BarData) -> None:
        # Trigger the raw on_bar callback immediately
        self.on_bar(bar)

        # If no aggregation callback is set, we are done
        if not self.on_window_bar:
            return

        # Route to appropriate updater based on interval for aggregation
        if self.interval == Interval.MINUTE:
            self._update_minute_window(bar)
        else:
            self._update_hour_window(bar)

    def _update_minute_window(self, bar: BarData) -> None:
        """Process minute-based window aggregation for a single symbol."""
        current_window_time = self._align_bar_datetime(bar)

        # Finalize previous window if time has changed
        if self.window_time is not None and current_window_time != self.window_time:
            self._finalize_window_bar() # Send previous window bar if exists

        self.window_time = current_window_time

        # Create or update bar for the current window
        if self.window_bar is None:
            self.window_bar = self._create_bar(bar, current_window_time)
        else:
            self._update_bar(self.window_bar, bar)

        # Finalize when (minute + 1) is a multiple of window size
        if (bar.datetime.minute + 1) % self.window == 0:
            self._finalize_window_bar()

    def _finalize_window_bar(self) -> None:
        """Send the completed window bar data to callback and clear buffer."""
        if self.window_bar and self.on_window_bar:
            self.on_window_bar(self.window_bar)
        self.window_bar = None  # Reset for the next window
        self.window_time = None

    def _update_hour_window(self, bar: BarData) -> None:
        """Process hour-based window aggregation for a single symbol."""
        # Get or create the bar for the current hour
        hour_bar = self._get_or_create_hour_bar(bar)

        # Detect hour switch
        if bar.datetime.hour != hour_bar.datetime.hour:
            # Previous hour finished
            self.hour_count += 1
            if self.hour_count >= self.window and self.on_window_bar:
                self.on_window_bar(hour_bar)
                self.hour_count = 0

            # Start new hour bar aligned to hour/window
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            if self.window > 1:
                aligned_hour = (dt.hour // self.window) * self.window
                dt = dt.replace(hour=aligned_hour)

            self.hour_bar = self._create_bar(bar, dt)
            self._update_bar(self.hour_bar, bar)
        else:
            # Still within same hour; keep updating
            self._update_bar(hour_bar, bar)

    def _get_or_create_hour_bar(self, bar: BarData) -> BarData:
        """Get existing hour bar or create a new one if needed."""
        if self.hour_bar is None:
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            if self.window > 1:
                aligned_hour = (dt.hour // self.window) * self.window
                dt = dt.replace(hour=aligned_hour)
            self.hour_bar = self._create_bar(bar, dt)
        return self.hour_bar

    def _align_bar_datetime(self, bar: BarData) -> datetime:
        """Align bar datetime to the start of its window boundary."""
        dt = bar.datetime.replace(second=0, microsecond=0)
        minute = (dt.minute // self.window) * self.window
        return dt.replace(minute=minute)

    def _create_bar(self, source: BarData, dt: datetime) -> BarData:
        """Create a new bar with aligned datetime based on source bar."""
        new_bar = BarData(
            symbol=source.symbol,
            datetime=dt,
            gateway_name=source.gateway_name,
            open_price=source.open_price,
            high_price=source.high_price,
            low_price=source.low_price,
            close_price=source.close_price,
            volume=source.volume,
            interval=self.interval,
        )
        return new_bar

    def _update_bar(self, target: BarData, source: BarData) -> None:
        """Update target bar with new data from source bar."""
        target.high_price = max(target.high_price, source.high_price)
        target.low_price = min(target.low_price, source.low_price)
        target.close_price = source.close_price

        # Accumulate volume, ensuring both target and source have the attribute
        target_volume = getattr(target, "volume", 0)
        source_volume = getattr(source, "volume", 0)
        setattr(target, "volume", target_volume + source_volume)
