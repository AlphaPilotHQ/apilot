"""
Momentum Strategy Backtesting Example

This module implements a trend-following strategy based on momentum indicators with dynamic stop-loss.
Includes single backtest and parameter optimization functionality.
"""

import logging
from datetime import datetime
from typing import ClassVar

import apilot as ap
from apilot.utils.logger import setup_logging

setup_logging("bt_stdmom_lo")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StdMomentumStrategy(ap.PATemplate):
    """
    Strategy Logic:
    1. Core Idea: Long-term trend-following strategy using momentum signals and dynamic stop-loss
    2. Entry Signals:
       - Based on momentum indicator (current price/N-period price - 1)
       - Enter long when momentum > threshold (5%)
       - Use full account capital for position sizing
    3. Risk Management:
       - Dynamic trailing stop based on standard deviation
       - Long positions: stop-loss set at highest price - 4x standard deviation
    """

    # Strategy parameters
    std_period = 48
    mom_threshold = 0.05
    trailing_std_scale = 4

    parameters: ClassVar[list[str]] = [
        "std_period",
        "mom_threshold",
        "trailing_std_scale",
    ]
    variables: ClassVar[list[str]] = [
        "momentum",
        "intra_trade_high",
        "pos",
    ]

    def __init__(self, pa_engine, strategy_name, symbols, setting):
        super().__init__(pa_engine, strategy_name, symbols, setting)
        # Initialize BarGenerator for all trading symbols
        self.bg = ap.BarGenerator(
            self.on_bar,
            5,
            self.on_5min_bar,
            symbols=self.symbols,
        )

        # Create ArrayManager for each symbol
        self.ams = {}
        for symbol in self.symbols:
            self.ams[symbol] = ap.ArrayManager(size=200)

        # Initialize tracking variables for each symbol
        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.pos = {}

        # Initialize each symbol's state
        for symbol in self.symbols:
            self.momentum[symbol] = 0.0
            self.std_value[symbol] = 0.0
            self.intra_trade_high[symbol] = 0
            self.pos[symbol] = 0

    def on_init(self):
        self.load_bar(self.std_period)

    def on_bar(self, bar: ap.BarData):
        """Process 1-minute bar data"""
        self.bg.update_bar(bar)

    def on_5min_bar(self, bars: dict):
        self.cancel_all()

        # Process each symbol's 5-minute bar data
        for symbol, bar in bars.items():
            if symbol not in self.ams:
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            if not am.inited:
                continue

            # Calculate technical indicators
            self.std_value[symbol] = am.std(self.std_period)

            # Calculate momentum
            if len(am.close_array) > self.std_period + 1:
                old_price = am.close_array[-self.std_period - 1]
                current_price = am.close_array[-1]
                self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1

            # Get current position
            current_pos = self.pos.get(symbol, 0)

            # Update trailing stop for long positions
            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )

            # Trading logic
            if current_pos == 0:
                # Initialize trailing price
                self.intra_trade_high[symbol] = bar.high_price

                # Calculate position size
                risk_percent = 1 / len(self.symbols)
                current_capital = self.pa_engine.get_current_capital()
                capital_to_use = current_capital * risk_percent
                size = max(1, int(capital_to_use / bar.close_price))

                # Enter long if momentum > threshold
                if self.momentum[symbol] > self.mom_threshold:
                    self.buy(symbol=symbol, price=bar.close_price, volume=size)

            elif current_pos > 0:  # Long position → trailing stop
                # Calculate trailing stop price
                long_stop = (
                    self.intra_trade_high[symbol]
                    - self.trailing_std_scale * self.std_value[symbol]
                )

                # Exit if price falls below stop
                if bar.close_price < long_stop:
                    self.sell(
                        symbol=symbol, price=bar.close_price, volume=abs(current_pos)
                    )

    def on_order(self, order: ap.OrderData):
        """Order callback"""
        pass

    def on_trade(self, trade: ap.TradeData):
        """Trade callback"""
        symbol = trade.symbol

        # Update position
        position_change = (
            trade.volume if trade.direction == ap.Direction.LONG else -trade.volume
        )
        self.pos[symbol] = self.pos.get(symbol, 0) + position_change

        # Update high price tracking
        current_pos = self.pos[symbol]

        if current_pos > 0:
            self.intra_trade_high[symbol] = max(
                self.intra_trade_high.get(symbol, trade.price), trade.price
            )

        logger.debug(
            f"Trade: {symbol} {trade.orderid} {trade.direction} "
            f"{trade.volume}@{trade.price}, current_pos: {current_pos}"
        )


def run_backtesting(
    strategy_class=StdMomentumStrategy,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 6, 29),
    std_period=48,
    mom_threshold=0.02,
    trailing_std_scale=2.0,
):
    # 1 Create backtesting engine
    engine = ap.BacktestingEngine()
    logger.info("1 Create backtesting engine completed")

    # 2 Set engine parameters
    symbols = ["SOL-USDT", "BTC-USDT"]

    engine.set_parameters(
        symbols=symbols,
        interval=ap.Interval.MINUTE,
        start=start,
        end=end,
    )
    logger.info("2 Set engine parameters completed")

    # 3 Add strategy
    engine.add_strategy(
        strategy_class,
        {
            "std_period": std_period,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale,
        },
    )
    logger.info("3 Add strategy completed")

    # 4 Add data
    engine.add_csv_data(
        symbol="SOL-USDT",
        filepath="data/SOL-USDT_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )
    engine.add_csv_data(
        symbol="BTC-USDT",
        filepath="data/BTC-USDT_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )
    logger.info("4 Add data completed")

    # 5 Run backtest
    engine.run_backtesting()
    logger.info("5 Run backtest completed")

    # 6 Generate report
    engine.report()
    logger.info("6 Generate report completed")

    return engine


if __name__ == "__main__":
    # Single backtest mode
    run_backtesting()
