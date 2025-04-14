from datetime import datetime
from typing import ClassVar

import apilot as ap
from apilot.utils.logger import get_logger, set_level

# Get logger
logger = get_logger("momentum_strategy")
set_level("info", "momentum_strategy")


class StdMomentumStrategy(ap.PATemplate):
    """
    Strategy logic:
    1. Core idea: Combine momentum signal with standard deviation dynamic stop-loss for medium-term trend tracking
    2. Entry signal:
       - Generate trading signal based on momentum indicator (current price/N-period ago price-1)
       - Momentum > threshold (5%) to go long
       - Momentum < -threshold (-5%) to go short
       - Use all account funds for position management
    3. Risk management:
       - Use standard deviation-based dynamic trailing stop-loss
       - Long position: Set stop-loss at highest price - 4 times standard deviation
       - Short position: Set stop-loss at lowest price + 4 times standard deviation
    """

    # Strategy parameters
    std_period = 30
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
        "intra_trade_low",
        "pos",
    ]

    def __init__(self, pa_engine, strategy_name, symbols, setting):
        super().__init__(pa_engine, strategy_name, symbols, setting)
        # Use enhanced BarGenerator instance to process all trading symbols, add symbols support for multi-symbol synchronization
        self.bg = ap.BarGenerator(
            self.on_bar,
            5,
            self.on_5min_bar,
            symbols=self.symbols,  # Pass in trading symbol list to enable multi-symbol synchronization
        )

        # Create ArrayManager for each trading pair
        self.ams = {}
        for symbol in self.symbols:
            self.ams[symbol] = ap.ArrayManager(
                size=200
            )  # Keep up to 200 bars TODO: Change to dynamic

        # Create state tracking dictionary for each trading pair
        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.intra_trade_low = {}
        self.pos = {}

        # Initialize state for each trading pair
        for symbol in self.symbols:
            self.momentum[symbol] = 0.0
            self.std_value[symbol] = 0.0
            self.intra_trade_high[symbol] = 0
            self.intra_trade_low[symbol] = 0
            self.pos[symbol] = 0

    def on_init(self):
        self.load_bar(self.std_period)

    def on_bar(self, bar: ap.BarData):
        """
        Process bar data
        """
        # Pass bar data to BarGenerator for aggregation
        self.bg.update_bar(bar)

    def on_5min_bar(self, bars: dict):
        self.cancel_all()

        # Record received multi-symbol bar data
        logger.debug(
            f"on_5min_bar received complete multi-symbol data: {list(bars.keys())}"
        )

        # Execute data update and trading logic for each trading symbol
        for symbol, bar in bars.items():
            if symbol not in self.ams:
                logger.debug(f"Ignore symbol {symbol} because it is not in ams")
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            # Skip trading logic if data is insufficient
            if not am.inited:
                continue

            # Calculate technical indicators
            self.std_value[symbol] = am.std(self.std_period)

            # Calculate momentum factor
            if len(am.close_array) > self.std_period + 1:
                old_price = am.close_array[-self.std_period - 1]
                current_price = am.close_array[-1]
                self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1

            # Get current position
            current_pos = self.pos.get(symbol, 0)

            # Update trailing stop price in position
            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )
            elif current_pos < 0:
                self.intra_trade_low[symbol] = min(
                    self.intra_trade_low[symbol], bar.low_price
                )

            # Trading logic
            if current_pos == 0:
                # Initialize trailing price
                self.intra_trade_high[symbol] = bar.high_price
                self.intra_trade_low[symbol] = bar.low_price

                # Allocate funds evenly to all symbols (full position)
                risk_percent = 1 / len(self.symbols)
                # Use current account value instead of initial capital
                current_capital = self.pa_engine.get_current_capital()
                capital_to_use = current_capital * risk_percent
                size = max(1, int(capital_to_use / bar.close_price))

                # Open position based on momentum signal
                logger.debug(
                    f"{bar.datetime}: {symbol} momentum value {self.momentum[symbol]:.4f}, threshold {self.mom_threshold:.4f}"
                )

                if self.momentum[symbol] > self.mom_threshold:
                    logger.debug(
                        f"{bar.datetime}: {symbol} issued long signal: momentum {self.momentum[symbol]:.4f} > threshold {self.mom_threshold}"
                    )
                    self.buy(symbol=symbol, price=bar.close_price, volume=size)
                elif self.momentum[symbol] < -self.mom_threshold:
                    logger.debug(
                        f"{bar.datetime}: {symbol} issued short signal: momentum {self.momentum[symbol]:.4f} < threshold {-self.mom_threshold}"
                    )
                    self.sell(symbol=symbol, price=bar.close_price, volume=size)

            elif (
                current_pos > 0
            ):  # Long position -> standard deviation trailing stop-loss
                # Calculate trailing stop price
                long_stop = (
                    self.intra_trade_high[symbol]
                    - self.trailing_std_scale * self.std_value[symbol]
                )

                # Close position if price falls below stop level
                if bar.close_price < long_stop:
                    self.sell(
                        symbol=symbol, price=bar.close_price, volume=abs(current_pos)
                    )

            elif (
                current_pos < 0
            ):  # Short position -> standard deviation trailing stop-loss
                # Calculate trailing stop price
                short_stop = (
                    self.intra_trade_low[symbol]
                    + self.trailing_std_scale * self.std_value[symbol]
                )

                # Close position if price exceeds stop level
                if bar.close_price > short_stop:
                    self.buy(
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

        # Update high/low price tracking
        current_pos = self.pos[symbol]

        # Update tracking prices only when position exists
        if current_pos > 0:
            # Long position, update highest price
            self.intra_trade_high[symbol] = max(
                self.intra_trade_high.get(symbol, trade.price), trade.price
            )
        elif current_pos < 0:
            # Short position, update lowest price
            self.intra_trade_low[symbol] = min(
                self.intra_trade_low.get(symbol, trade.price), trade.price
            )

        logger.debug(
            f"Trade: {symbol} {trade.orderid} {trade.direction} "
            f"{trade.volume}@{trade.price}, current_pos: {current_pos}"
        )


@ap.log_exceptions()
def run_backtesting(
    strategy_class=StdMomentumStrategy,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 6, 29),
    std_period=20,
    mom_threshold=0.02,
    trailing_std_scale=2.0,
    run_optimization=False,
):
    """
    Run backtesting or parameter optimization

    Args:
        strategy_class: Strategy class
        start: Start date
        end: End date
        std_period: Standard deviation period
        mom_threshold: Momentum threshold
        trailing_std_scale: Trailing stop-loss coefficient
        run_optimization: Whether to run parameter optimization
        (Removed) optimization_method parameter
    """
    # 1 Create backtesting engine
    engine = ap.BacktestingEngine()
    logger.info("1 Created backtesting engine")

    # 2 Set engine parameters
    symbols = ["SOL-USDT.LOCAL", "BTC-USDT.LOCAL"]

    engine.set_parameters(
        symbols=symbols,
        interval=ap.Interval.MINUTE,
        start=start,
        end=end,
    )
    logger.info("2 Set engine parameters")

    # 3 Add strategy
    engine.add_strategy(
        strategy_class,
        {
            "std_period": std_period,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale,
        },
    )
    logger.info("3 Added strategy")

    # 4 Add data
    engine.add_csv_data(
        symbol="SOL-USDT.LOCAL",
        filepath="data/SOL-USDT_LOCAL_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )
    engine.add_csv_data(
        symbol="BTC-USDT.LOCAL",
        filepath="data/BTC-USDT_LOCAL_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )
    logger.info("4 Added data")

    # 5. If optimization mode is enabled, run parameter optimization first
    if run_optimization:
        # Create optimization parameter configuration
        from apilot.optimizer import OptimizationSetting

        setting = OptimizationSetting()
        setting.set_target("total_return")

        # Set parameter ranges
        setting.add_parameter("std_period", 15, 50, 5)  # Standard deviation period
        setting.add_parameter("mom_threshold", 0.02, 0.06, 0.01)  # Momentum threshold
        setting.add_parameter(
            "trailing_std_scale", 2.0, 7.0, 1.0
        )  # Stop-loss coefficient

        # Run optimization
        results = engine.optimize(strategy_setting=setting)

        # Apply optimal parameters (if found)
        if results:
            best_setting = results[0].copy()
            fitness = best_setting.pop("fitness", 0)

            logger.info(
                f"Optimal parameter combination: {best_setting}, fitness: {fitness:.4f}"
            )

            # Reconfigure strategy with optimal parameters
            engine.strategy = None  # Clear original strategy
            engine.add_strategy(strategy_class, best_setting)

    # 6 Run backtesting
    engine.run_backtesting()
    logger.info("5 Ran backtesting")

    # 7 Calculate results and generate report
    engine.report()
    logger.info("6 Calculated results and generated report")

    return engine


if __name__ == "__main__":
    # Single backtesting mode
    # run_backtesting()

    # Optimization mode - using grid search
    run_backtesting(run_optimization=True)
