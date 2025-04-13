"""
Technical indicators and array-based calculations.
"""

import numpy as np

from apilot.core.object import BarData


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
        self.turnover_array: np.ndarray = np.zeros(size)
        self.open_interest_array: np.ndarray = np.zeros(size)

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

    @property
    def turnover(self) -> np.ndarray:
        return self.turnover_array

    @property
    def open_interest(self) -> np.ndarray:
        return self.open_interest_array

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

    def rsi(self, n: int, array: bool = False) -> float | np.ndarray:
        """Calculates Relative Strength Index (RSI)."""
        if len(self.close_array) < n + 1 or n <= 0:
            return np.nan if not array else np.full(self.size, np.nan)

        delta = np.diff(self.close_array)
        delta = np.insert(delta, 0, 0)

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = np.zeros_like(self.close_array)
        avg_loss = np.zeros_like(self.close_array)

        if len(gain) >= n and len(loss) >= n:
            avg_gain[n] = np.mean(gain[1 : n + 1])
            avg_loss[n] = np.mean(loss[1 : n + 1])

        for i in range(n + 1, len(self.close_array)):
            avg_gain[i] = (avg_gain[i - 1] * (n - 1) + gain[i]) / n
            avg_loss[i] = (avg_loss[i - 1] * (n - 1) + loss[i]) / n

        rs = np.divide(
            avg_gain,
            avg_loss,
            out=np.full_like(avg_gain, np.inf),
            where=avg_loss != 0,
        )

        rsi = 100 - (100 / (1 + rs))
        rsi[:n] = np.nan

        if array:
            return rsi
        return rsi[-1] if len(rsi) > 0 else np.nan

    def macd(
        self,
        fast_period: int,
        slow_period: int,
        signal_period: int,
        array: bool = False,
    ) -> (
        tuple[float, float, float]
        | tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
    ):
        """Calculates Moving Average Convergence Divergence (MACD)."""
        if (
            len(self.close_array) < slow_period
            or fast_period <= 0
            or slow_period <= 0
            or signal_period <= 0
            or fast_period >= slow_period
        ):
            nan_array = np.full(self.size, np.nan)
            return (
                (np.nan, np.nan, np.nan)
                if not array
                else (nan_array.copy(), nan_array.copy(), nan_array.copy())
            )

        ema_fast = self.ema(fast_period, array=True)
        ema_slow = self.ema(slow_period, array=True)

        if not isinstance(ema_fast, np.ndarray) or not isinstance(ema_slow, np.ndarray):
            nan_array = np.full(self.size, np.nan)
            return (
                (np.nan, np.nan, np.nan)
                if not array
                else (nan_array.copy(), nan_array.copy(), nan_array.copy())
            )

        macd = ema_fast - ema_slow

        signal = np.zeros_like(macd)
        alpha_signal = 2.0 / (signal_period + 1)
        start_idx = np.argmax(~np.isnan(macd))
        if start_idx < len(signal):
            signal[start_idx] = macd[start_idx]
            for i in range(start_idx + 1, len(macd)):
                signal[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal[i - 1]
        signal[:start_idx] = np.nan

        hist = macd - signal

        if array:
            return macd, signal, hist
        last_macd = macd[-1] if len(macd) > 0 else np.nan
        last_signal = signal[-1] if len(signal) > 0 else np.nan
        last_hist = hist[-1] if len(hist) > 0 else np.nan
        return last_macd, last_signal, last_hist

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

    def mfi(self, n: int, array: bool = False) -> float | np.ndarray:
        """Calculates Money Flow Index (MFI)."""
        required_len = n + 1
        if (
            len(self.close_array) < required_len
            or len(self.volume_array) < required_len
            or n <= 0
        ):
            return np.nan if not array else np.full(self.size, np.nan)

        tp = (self.high_array + self.low_array + self.close_array) / 3

        mf = tp * self.volume_array

        diff = np.diff(tp)
        diff = np.insert(diff, 0, 0)

        positive_flow = np.where(diff > 0, mf, 0)
        negative_flow = np.where(diff < 0, mf, 0)

        positive_mf_sum = np.zeros_like(self.close_array)
        negative_mf_sum = np.zeros_like(self.close_array)

        for i in range(n, len(self.close_array)):
            positive_mf_sum[i] = np.sum(positive_flow[i - n + 1 : i + 1])
            negative_mf_sum[i] = np.sum(negative_flow[i - n + 1 : i + 1])

        mfr = np.full_like(positive_mf_sum, np.nan)
        non_zero_mask = negative_mf_sum != 0
        mfr[non_zero_mask] = (
            positive_mf_sum[non_zero_mask] / negative_mf_sum[non_zero_mask]
        )

        result = 100 - (100 / (1 + mfr))
        result[:n] = np.nan

        zero_mask = (positive_mf_sum == 0) & (negative_mf_sum == 0)
        result[zero_mask] = 50.0

        if array:
            return result
        return result[-1] if len(result) > 0 else np.nan

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
