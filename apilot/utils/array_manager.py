"""
Technical indicators and array-based calculations.
"""

import numpy as np

from apilot.core.models import BarData


class ArrayManager:
    """
    Manages time series bar data and calculates technical indicators.
    """

    def __init__(self, size: int = 100) -> None:
        self.count: int = 0
        self.size: int = size
        self.inited: bool = False

        self.open_array: np.ndarray = np.zeros(size)
        self.high_array: np.ndarray = np.zeros(size)
        self.low_array: np.ndarray = np.zeros(size)
        self.close_array: np.ndarray = np.zeros(size)
        self.volume_array: np.ndarray = np.zeros(size)

    def update_bar(self, bar: BarData) -> None:
        """
        Updates the arrays with the latest bar data.
        """
        self.count += 1

        if not self.inited and self.count >= self.size:
            self.inited = True

        self.open_array[:-1] = self.open_array[1:]
        self.high_array[:-1] = self.high_array[1:]
        self.low_array[:-1] = self.low_array[1:]
        self.close_array[:-1] = self.close_array[1:]
        self.volume_array[:-1] = self.volume_array[1:]

        self.open_array[-1] = bar.open_price
        self.high_array[-1] = bar.high_price
        self.low_array[-1] = bar.low_price
        self.close_array[-1] = bar.close_price
        self.volume_array[-1] = bar.volume

    @property
    def open(self) -> np.ndarray:
        return self.open_array

    @property
    def high(self) -> np.ndarray:
        return self.high_array

    @property
    def low(self) -> np.ndarray:
        return self.low_array

    @property
    def close(self) -> np.ndarray:
        return self.close_array

    @property
    def volume(self) -> np.ndarray:
        return self.volume_array

    def sma(self, n: int, array: bool = False) -> float | np.ndarray:
        """Calculates Simple Moving Average (SMA)."""
        if len(self.close_array) == 0 or n <= 0 or n > len(self.close_array):
            return np.nan if not array else np.full(self.size, np.nan)

        weights = np.ones(n) / n
        result = np.convolve(self.close_array, weights, mode="valid")
        padding = np.full(n - 1, np.nan)
        result = np.concatenate((padding, result))

        if array:
            return result
        return result[-1]

    def ema(self, n: int, array: bool = False) -> float | np.ndarray:
        """Calculates Exponential Moving Average (EMA)."""
        if len(self.close_array) == 0 or n <= 0 or n > len(self.close_array):
            return np.nan if not array else np.full(self.size, np.nan)

        alpha = 2.0 / (n + 1)
        result = np.zeros_like(self.close_array)
        if len(result) > 0:
            result[0] = self.close_array[0]
            for i in range(1, len(self.close_array)):
                result[i] = alpha * self.close_array[i] + (1 - alpha) * result[i - 1]

        if array:
            return result
        return result[-1] if len(result) > 0 else np.nan

    def donchian(
        self, n: int, array: bool = False
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Calculates Donchian Channel upper and lower bands."""
        if len(self.high_array) < n or n <= 0:
            nan_array = np.full(self.size, np.nan)
            return (
                (np.nan, np.nan) if not array else (nan_array.copy(), nan_array.copy())
            )

        up = np.zeros_like(self.high_array)
        down = np.zeros_like(self.low_array)

        for i in range(len(self.high_array)):
            if i >= n - 1:
                up[i] = np.max(self.high_array[i - n + 1 : i + 1])
                down[i] = np.min(self.low_array[i - n + 1 : i + 1])
            else:
                up[i] = np.nan
                down[i] = np.nan

        if array:
            return up, down
        return up[-1] if len(up) > 0 else np.nan, down[-1] if len(down) > 0 else np.nan

    def std(self, n: int, array: bool = False) -> float | np.ndarray:
        """Calculates Standard Deviation (STD)."""
        if not self.inited or n <= 0 or n > len(self.close_array):
            return np.nan if not array else np.full(self.size, np.nan)

        std_dev = np.zeros_like(self.close_array)
        for i in range(n - 1, len(self.close_array)):
            std_dev[i] = np.std(self.close_array[i - n + 1 : i + 1], ddof=1)
        std_dev[: n - 1] = np.nan

        if array:
            return std_dev
        return std_dev[-1]

    def boll(
        self, n: int, dev: float, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """Calculates Bollinger Bands (BOLL)."""
        mid: float | np.ndarray = self.sma(n, array)
        std_dev: float | np.ndarray = self.std(n, array=array)

        if isinstance(mid, np.ndarray) and isinstance(std_dev, np.ndarray):
            up: np.ndarray = mid + std_dev * dev
            down: np.ndarray = mid - std_dev * dev
        elif isinstance(mid, float) and isinstance(std_dev, float):
            up: float = mid + std_dev * dev
            down: float = mid - std_dev * dev
        else:
            nan_array = np.full(self.size, np.nan)
            return (
                (np.nan, np.nan) if not array else (nan_array.copy(), nan_array.copy())
            )

        return up, down

    def atr(self, n: int, array: bool = False) -> float | np.ndarray:
        """Calculates Average True Range (ATR)."""
        if len(self.close_array) < 1 or n <= 0 or n > len(self.close_array):
            return np.nan if not array else np.full(self.size, np.nan)

        tr = np.zeros_like(self.close_array)
        for i in range(1, len(self.close_array)):
            high_low = self.high_array[i] - self.low_array[i]
            high_close = abs(self.high_array[i] - self.close_array[i - 1])
            low_close = abs(self.low_array[i] - self.close_array[i - 1])
            tr[i] = max(high_low, high_close, low_close)
        tr[0] = self.high_array[0] - self.low_array[0]

        atr_result = np.zeros_like(self.close_array)
        if len(tr) >= n:
            atr_result[n - 1] = np.mean(tr[0:n])
            for i in range(n, len(self.close_array)):
                atr_result[i] = (atr_result[i - 1] * (n - 1) + tr[i]) / n
        atr_result[: n - 1] = np.nan

        if array:
            return atr_result
        return atr_result[-1] if len(atr_result) > 0 else np.nan

    def keltner(
        self, n: int, dev: float, array: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[float, float]:
        """Calculates Keltner Channel."""
        mid: float | np.ndarray = self.sma(n, array)
        atr_val: float | np.ndarray = self.atr(n, array)

        if isinstance(mid, np.ndarray) and isinstance(atr_val, np.ndarray):
            up: np.ndarray = mid + atr_val * dev
            down: np.ndarray = mid - atr_val * dev
        elif isinstance(mid, float) and isinstance(atr_val, float):
            up: float = mid + atr_val * dev
            down: float = mid - atr_val * dev
        else:
            nan_array = np.full(self.size, np.nan)
            return (
                (np.nan, np.nan) if not array else (nan_array.copy(), nan_array.copy())
            )

        return up, down
