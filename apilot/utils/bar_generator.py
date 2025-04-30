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
    """
    Simplified bar generator for single-symbol synchronization.

    Main features:
    1. Generate 1-minute bars from tick data (Implicitly handled by input bars)
    2. Generate X-minute bars from 1-minute bars
    3. Generate hourly bars from 1-minute bars

    Time intervals:
    - Minutes: x must be divisible by 60 (2, 3, 5, 6, 10, 15, 20, 30)
    - Hours: any positive integer is valid
    """

    def __init__(
        self,
        on_bar: Callable[[BarData], Any],  # Callback for each incoming bar
        window: int = 1,
        on_window_bar: Callable[[BarData], Any] | None = None, # Callback for aggregated bar
        interval: Interval = Interval.MINUTE,
    ) -> None:
        """
        Initialize bar generator with window size and callback function.

        Args:
            on_bar: Callback function triggered for each incoming bar.
                    Note: If using window > 1 or interval=HOUR, this might
                          not be the primary callback you need.
            window: Size of the aggregation window. Default is 1.
            on_window_bar: Callback function triggered when a window/hour is complete.
                           This is typically the main callback for aggregated bars.
            interval: Time interval for aggregation, default is MINUTE.
        """
        if window <= 0:
            window = 1
        if interval == Interval.HOUR:
             # window * hourly bar, window is only available for HOUR interval
            self.window: int = window
        else:
            # window * minute bar, window only means minute level window.
            self.window: int = window # Minute window size

        self.on_bar: Callable[[BarData], Any] = on_bar # Original bar callback
        self.on_window_bar: Callable[[BarData], Any] | None = on_window_bar # Aggregated bar callback

        self.interval: Interval = interval
        self.interval_count: int = 0 # Used for hourly aggregation count

        # Store the single bar being aggregated for the current window/hour
        self.window_bar: BarData | None = None
        self.hour_bar: BarData | None = None
        self.finished_hour_bar: BarData | None = None # Store completed hourly bar

        # Track the datetime of the current aggregation window start
        self.window_time: datetime | None = None

    def update_bar(self, bar: BarData) -> None:
        """
        Update bar data with a single BarData object.

        Args:
            bar: The incoming BarData object.
        """
        logger = logging.getLogger("BarGenerator")
        logger.info(f"BarGenerator.update_bar: get {bar.symbol} {bar.datetime} close: {bar.close_price}")

        # Trigger the raw on_bar callback immediately
        logger.info(f"BarGenerator.update_bar: 准备调用 on_bar 回调: {self.on_bar.__qualname__}")
        self.on_bar(bar)
        logger.info(f"BarGenerator.update_bar: on_bar 回调完成: {self.on_bar.__qualname__}")

        # If no aggregation callback is set, we are done
        if not self.on_window_bar:
            logger.info(f"BarGenerator.update_bar: 无聚合回调，处理完成")
            return

        # Route to appropriate updater based on interval for aggregation
        logger.info(f"BarGenerator.update_bar: 根据间隔类型 {self.interval} 路由到对应的更新器")
        if self.interval == Interval.MINUTE:
            self._update_minute_window(bar)
        else: # Interval.HOUR or other future intervals potentially
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

        # Check if the window is complete based on minute boundary
        # window=1 means trigger every minute bar
        # window>1 means trigger every `window` minutes
        should_trigger = False
        if self.window == 1:
            should_trigger = True
        elif self.window > 1:
            # Trigger when the bar's minute + 1 is a multiple of the window size
            if (bar.datetime.minute + 1) % self.window == 0:
                 should_trigger = True

        # Finalize if trigger condition met
        if should_trigger:
            self._finalize_window_bar()

    def _finalize_window_bar(self) -> None:
        """Send the completed window bar data to callback and clear buffer."""
        if self.window_bar and self.on_window_bar:
            self.on_window_bar(self.window_bar)
        self.window_bar = None # Reset for the next window

    def _update_hour_window(self, bar: BarData) -> None:
        """Process hour-based window aggregation for a single symbol."""
        # Get or create the bar for the current hour
        hour_bar = self._get_or_create_hour_bar(bar)

        # If the incoming bar is the last minute of the hour (e.g., 59)
        if bar.datetime.minute == 59:
            self._update_bar(hour_bar, bar)
            self.interval_count += 1 # Increment hour count
            # Check if the number of hours collected reaches the window size
            if self.interval_count >= self.window:
                self.finished_hour_bar = hour_bar # Mark as finished
                self.hour_bar = None             # Reset current hour bar
                self.interval_count = 0          # Reset hour count
            # else: hour bar continues to accumulate

        # If the incoming bar's hour is different from the current hour bar's hour
        elif hour_bar and bar.datetime.hour != hour_bar.datetime.hour:
            self.interval_count += 1 # Increment hour count for the completed hour
            # Check if the window is complete
            if self.interval_count >= self.window:
                 self.finished_hour_bar = hour_bar # Mark previous hour bar as finished
                 self.interval_count = 0          # Reset hour count
            # Start a new hour bar regardless of window completion
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            new_hour_bar = self._create_bar(bar, dt)
            self.hour_bar = new_hour_bar

        # Otherwise, it's a regular update within the same hour
        else:
            self._update_bar(hour_bar, bar)

        # If an hourly bar (or multi-hour bar) is finished, send it
        if self.finished_hour_bar and self.on_window_bar:
            self.on_window_bar(self.finished_hour_bar)
            self.finished_hour_bar = None # Clear the finished bar

    def _get_or_create_hour_bar(self, bar: BarData) -> BarData:
        """Get existing hour bar or create a new one if needed."""
        if self.hour_bar is None:
            # Create based on the start of the current bar's hour or window start
            dt = bar.datetime.replace(minute=0, second=0, microsecond=0)
            # If window > 1, align to the start hour of the multi-hour window
            if self.interval == Interval.HOUR and self.window > 1:
                 aligned_hour = (dt.hour // self.window) * self.window
                 dt = dt.replace(hour=aligned_hour)

            self.hour_bar = self._create_bar(bar, dt)
        return self.hour_bar

    def _align_bar_datetime(self, bar: BarData) -> datetime:
        """Align bar datetime to the start of its window boundary."""
        dt = bar.datetime.replace(second=0, microsecond=0)
        if self.interval == Interval.HOUR:
            # Align to the start hour of the potentially multi-hour window
            aligned_hour = (dt.hour // self.window) * self.window
            dt = dt.replace(hour=aligned_hour, minute=0)
        elif self.window > 1: # Minute interval with window > 1
            minute = (dt.minute // self.window) * self.window
            dt = dt.replace(minute=minute)
        # If window is 1 (minute or hour), the original dt (aligned to minute/hour start) is fine
        return dt

    def _create_bar(self, source: BarData, dt: datetime) -> BarData:
        """Create a new bar with aligned datetime based on source bar."""
        # Ensure volume attribute exists, default to 0 if not
        volume = getattr(source, "volume", 0)

        new_bar = BarData(
            symbol=source.symbol,
            datetime=dt,
            gateway_name=source.gateway_name,
            open_price=source.open_price,
            high_price=source.high_price,
            low_price=source.low_price,
            close_price=source.close_price,
            interval=self.interval, # Set interval on the new bar
            # volume=volume, # Initialize volume if needed, depends on BarData model
        )
        # Explicitly set volume if BarData expects it
        setattr(new_bar, "volume", volume)
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

        # Update datetime only if source is later (though typically target.datetime is fixed)
        # target.datetime = max(target.datetime, source.datetime) # Usually not needed for aggregated bars
