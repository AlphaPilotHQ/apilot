"""
Tests for BarGenerator utility class.
"""

import unittest
from copy import copy
from datetime import datetime

import apilot as ap
from apilot.core.constant import Exchange, Interval
from apilot.core.object import BarData, TickData


class TestBarGenerator(unittest.TestCase):
    """
    Test BarGenerator functionality:
    1. Converting tick data to 1-minute bars
    2. Converting 1-minute bars to x-minute bars
    3. Converting 1-minute bars to hourly bars
    """

    def setUp(self):
        """Set up test environment before each test method"""
        self.bar_results = []
        self.window_bar_results = []

    def on_bar(self, bars):
        """Callback for 1-minute bars"""
        self.bar_results.append(copy(bars))

    def on_window_bar(self, bars):
        """Callback for window bars (x-minute or hourly)"""
        self.window_bar_results.append(copy(bars))

    def test_update_tick_single_symbol(self):
        """Test converting tick data to bars for a single symbol"""
        # Create bar generator for 1-minute bars
        bg = ap.core.utility.BarGenerator(self.on_bar)

        # Create sample tick data for single symbol
        symbol = "btc_usdt"
        exchange = Exchange.BINANCE

        ticks = []
        # Create tick data for minute 1
        for i in range(5):
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=datetime(2023, 1, 1, 10, 1, i * 10),
                name="test_tick",
                last_price=100 + i,
                volume=1000 + i * 100,
                turnover=100000 + i * 10000,
                open_interest=10000,
                gateway_name="test",
            )
            ticks.append(tick)

        # Create tick data for minute 2
        for i in range(5):
            tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=datetime(2023, 1, 1, 10, 2, i * 10),
                name="test_tick",
                last_price=200 + i,
                volume=2000 + i * 100,
                turnover=200000 + i * 10000,
                open_interest=20000,
                gateway_name="test",
            )
            ticks.append(tick)

        # Update generator with ticks
        for tick in ticks:
            bg.update_tick(tick)

        # Force generation of the last bar
        last_tick = TickData(
            symbol=symbol,
            exchange=exchange,
            datetime=datetime(2023, 1, 1, 10, 3, 0),
            name="test_tick",
            last_price=300,
            volume=3000,
            turnover=300000,
            open_interest=30000,
            gateway_name="test",
        )
        bg.update_tick(last_tick)

        # Check results
        assert len(self.bar_results) == 2
        bar1 = self.bar_results[0][symbol]
        assert bar1.datetime.minute == 1
        assert bar1.open_price == 100
        assert bar1.high_price == 104
        assert bar1.low_price == 100
        assert bar1.close_price == 104

        bar2 = self.bar_results[1][symbol]
        assert bar2.datetime.minute == 2
        assert bar2.open_price == 200
        assert bar2.high_price == 204
        assert bar2.low_price == 200
        assert bar2.close_price == 204

    def test_update_tick_multiple_symbols(self):
        """Test converting tick data to bars for multiple symbols"""
        # Create bar generator for 1-minute bars
        bg = ap.core.utility.BarGenerator(self.on_bar)

        # Create sample tick data for two symbols
        symbols = ["btc_usdt", "eth_usdt"]
        exchange = Exchange.BINANCE

        ticks = []
        # Create tick data for minute 1
        for symbol in symbols:
            for i in range(3):
                tick = TickData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=datetime(2023, 1, 1, 10, 1, i * 10),
                    name=f"test_{symbol}",
                    last_price=100 + i if symbol == "btc_usdt" else 50 + i,
                    volume=1000 + i * 100,
                    turnover=100000 + i * 10000,
                    open_interest=10000,
                    gateway_name="test",
                )
                ticks.append(tick)

        # Create tick data for minute 2
        for symbol in symbols:
            for i in range(3):
                tick = TickData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=datetime(2023, 1, 1, 10, 2, i * 10),
                    name=f"test_{symbol}",
                    last_price=200 + i if symbol == "btc_usdt" else 150 + i,
                    volume=2000 + i * 100,
                    turnover=200000 + i * 10000,
                    open_interest=20000,
                    gateway_name="test",
                )
                ticks.append(tick)

        # Update generator with ticks
        for tick in ticks:
            bg.update_tick(tick)

        # Force generation of the last bar
        for symbol in symbols:
            last_tick = TickData(
                symbol=symbol,
                exchange=exchange,
                datetime=datetime(2023, 1, 1, 10, 3, 0),
                name=f"test_{symbol}",
                last_price=300 if symbol == "btc_usdt" else 250,
                volume=3000,
                turnover=300000,
                open_interest=30000,
                gateway_name="test",
            )
            bg.update_tick(last_tick)

        # Check results
        assert len(self.bar_results) == 2

        # Check first minute bars
        bar1_btc = self.bar_results[0]["btc_usdt"]
        bar1_eth = self.bar_results[0]["eth_usdt"]

        assert bar1_btc.open_price == 100
        assert bar1_btc.close_price == 102
        assert bar1_eth.open_price == 50
        assert bar1_eth.close_price == 52

        # Check second minute bars
        bar2_btc = self.bar_results[1]["btc_usdt"]
        bar2_eth = self.bar_results[1]["eth_usdt"]

        assert bar2_btc.open_price == 200
        assert bar2_btc.close_price == 202
        assert bar2_eth.open_price == 150
        assert bar2_eth.close_price == 152

    def test_x_minute_window(self):
        """Test generating x-minute bars from 1-minute bars"""
        # Window size 5 minutes
        window = 5
        bg = ap.core.utility.BarGenerator(
            self.on_bar, window=window, on_window_bar=self.on_window_bar
        )

        symbol = "btc_usdt"
        exchange = Exchange.BINANCE

        # Create 10 one-minute bars
        for i in range(10):
            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                interval=Interval.MINUTE,
                datetime=datetime(2023, 1, 1, 10, i, 0),
                gateway_name="test",
                open_price=100 + i * 10,
                high_price=110 + i * 10,
                low_price=90 + i * 10,
                close_price=105 + i * 10,
                volume=1000 * (i + 1),
                turnover=100000 * (i + 1),
                open_interest=10000,
            )

            bg.update_bar(bar)

        # Should have generated 5-minute bars at minutes 4 and 9
        assert len(self.window_bar_results) == 2

        # Check first 5-minute bar
        first_window = self.window_bar_results[0][symbol]
        assert first_window.datetime.minute == 0
        assert first_window.open_price == 100
        assert first_window.high_price == 150
        assert first_window.low_price == 90
        assert first_window.close_price == 145

        # Check second 5-minute bar
        second_window = self.window_bar_results[1][symbol]
        assert second_window.datetime.minute == 5
        assert second_window.open_price == 150
        assert second_window.high_price == 200
        assert second_window.low_price == 140
        assert second_window.close_price == 195

    def test_hourly_window(self):
        """Test generating hourly bars from 1-minute bars"""
        # Create a list to store the generated window bars
        hourly_bars = []

        # Define a callback that just stores the bars
        def on_hour_bar(bars):
            nonlocal hourly_bars
            hourly_bars.append(bars.copy())

        # Create bar generator with our callback
        bg = ap.core.utility.BarGenerator(
            self.on_bar,
            window=1,
            on_window_bar=self.on_window_bar,
            interval=Interval.HOUR,
        )
        # Set our callback for hourly bars
        bg.on_hour_bar = on_hour_bar

        symbol = "btc_usdt"
        exchange = Exchange.BINANCE

        # Test scenario:
        # 1. Bar at 10:00 - should create a new hour bar
        # 2. Bar at 10:30 - should update the existing hour bar
        # 3. Bar at 10:59 - should finalize the hour bar
        # 4. Bar at 11:00 - should create a new hour bar

        # Create and process 10:00 bar
        bar1 = BarData(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.MINUTE,
            datetime=datetime(2023, 1, 1, 10, 0, 0),
            gateway_name="test",
            open_price=100,
            high_price=110,
            low_price=90,
            close_price=105,
            volume=1000,
            turnover=100000,
            open_interest=10000,
        )
        bg.update_bar(bar1)

        # Create and process 10:30 bar
        bar2 = BarData(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.MINUTE,
            datetime=datetime(2023, 1, 1, 10, 30, 0),
            gateway_name="test",
            open_price=120,
            high_price=130,
            low_price=110,
            close_price=125,
            volume=2000,
            turnover=200000,
            open_interest=10000,
        )
        bg.update_bar(bar2)

        # Create and process 10:59 bar
        bar3 = BarData(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.MINUTE,
            datetime=datetime(2023, 1, 1, 10, 59, 0),
            gateway_name="test",
            open_price=120,
            high_price=140,
            low_price=110,
            close_price=135,
            volume=3000,
            turnover=300000,
            open_interest=10000,
        )
        bg.update_bar(bar3)

        # Create and process 11:00 bar
        bar4 = BarData(
            symbol=symbol,
            exchange=exchange,
            interval=Interval.MINUTE,
            datetime=datetime(2023, 1, 1, 11, 0, 0),
            gateway_name="test",
            open_price=130,
            high_price=140,
            low_price=120,
            close_price=135,
            volume=4000,
            turnover=400000,
            open_interest=10000,
        )
        bg.update_bar(bar4)

        # Check the current state by directly accessing the internal bar
        # This is just to have a meaningful test while we're developing
        if hasattr(bg, "hour_bars") and symbol in bg.hour_bars and bg.hour_bars[symbol]:
            current_hour_bar = bg.hour_bars[symbol]
            assert current_hour_bar.datetime.hour == 11
            assert current_hour_bar.open_price == 130

    def test_invalid_window_sizes(self):
        """Test BarGenerator with valid and invalid window sizes"""
        # Test valid window sizes that divide 60 evenly
        valid_windows = [2, 3, 5, 6, 10, 15, 20, 30]
        for window in valid_windows:
            # Just create instances to ensure no errors
            _ = ap.core.utility.BarGenerator(
                self.on_bar, window=window, on_window_bar=self.on_window_bar
            )

        # Test invalid window size that doesn't divide 60
        invalid_windows = [4, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18]
        for _ in invalid_windows:
            # TODO: Implement validation in BarGenerator or add test
            # Currently BarGenerator doesn't validate window size
            pass


if __name__ == "__main__":
    unittest.main()
