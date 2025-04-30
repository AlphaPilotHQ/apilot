# """
# Improvement plan:
# Monitor all symbols, buy 1% on signal, add positions up to 4%.
# Continuously select coins.
# Position management is complex and needs further design.
# """

# import logging
# from datetime import datetime
# from typing import ClassVar

# import numpy as np

# import apilot as ap
# from apilot.utils.logger import setup_logging

# # Setup logging system
# setup_logging("bt_turtle")

# # Get module logger
# logger = logging.getLogger(__name__)


# class TurtleSignalStrategy(ap.PATemplate):
#     """
#     Turtle Trading Signal Strategy

#     A trend-following system based on Donchian Channel, implementing classic Turtle Trading rules.
#     Uses two channel periods (long and short) with ATR for position sizing and stop loss.
#     """

#     # Strategy parameters
#     entry_window: ClassVar[int] = 20  # Entry channel period, 20 days
#     exit_window: ClassVar[int] = 10  # Exit channel period, 10 days
#     atr_window: ClassVar[int] = 20  # ATR calculation period, 20 days
#     fixed_size: ClassVar[int] = 1  # Base trading unit

#     # Strategy variables
#     entry_up: ClassVar[float] = 0  # Entry channel upper band (highest price)
#     entry_down: ClassVar[float] = 0  # Entry channel lower band (lowest price)
#     exit_up: ClassVar[float] = 0  # Exit channel upper band
#     exit_down: ClassVar[float] = 0  # Exit channel lower band
#     atr_value: ClassVar[float] = 0  # ATR value for position sizing and stop loss

#     long_entry: ClassVar[float] = 0  # Long entry price
#     short_entry: ClassVar[float] = 0  # Short entry price
#     long_stop: ClassVar[float] = 0  # Long stop loss price
#     short_stop: ClassVar[float] = 0  # Short stop loss price

#     # Parameter and variable lists for UI display and optimization
#     parameters: ClassVar[list[str]] = [
#         "entry_window",
#         "exit_window",
#         "atr_window",
#         "fixed_size",
#     ]
#     variables: ClassVar[list[str]] = [
#         "entry_up",
#         "entry_down",
#         "exit_up",
#         "exit_down",
#         "atr_value",
#     ]

#     def __init__(self, pa_engine, strategy_name, symbol, setting):
#         """Initialize"""
#         super().__init__(pa_engine, strategy_name, symbol, setting)

#         # Create bar generator and array manager
#         self.bg = ap.BarGenerator(self.on_bar)
#         self.am = ap.ArrayManager()

#     def on_init(self):
#         """
#         Strategy initialization callback
#         """
#         # Use engine's logging method to record initialization info

#     def on_start(self):
#         """
#         Strategy start callback
#         """

#     def on_stop(self):
#         """
#         Strategy stop callback
#         """

#     def on_bar(self, bar: ap.BarData):
#         """
#         Bar data update callback - core trading logic
#         """
#         # Update bar data to array manager
#         self.am.update_bar(bar)
#         if not self.am.inited:
#             return

#         # Cancel all previous orders
#         self.cancel_all()

#         # Get current position for the trading symbol
#         current_pos = self.get_pos(self.symbols[0])

#         # Only calculate new entry channel when no position
#         if self.is_pos_equal(current_pos, 0):  # Using helper method
#             # Calculate Donchian channel (N-day high and low)
#             self.entry_up, self.entry_down = self.am.donchian(self.entry_window)

#         # Always calculate short-period exit channel
#         self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

#         # Calculate ATR for risk management - update on each bar for accurate risk calculation
#         self.atr_value = self.am.atr(self.atr_window)

#         # Trading logic: handle based on position
#         symbol = self.symbols[0]  # Get current trading symbol
#         current_pos = self.get_pos(symbol)

#         # No position state
#         if self.is_pos_equal(current_pos, 0):
#             # Reset entry and stop loss prices
#             self.long_entry = 0
#             self.short_entry = 0
#             self.long_stop = 0
#             self.short_stop = 0

#             # Send long and short breakout orders
#             self.send_buy_orders(self.entry_up)  # Upper band breakout - go long
#             self.send_short_orders(self.entry_down)  # Lower band breakout - go short

#         # Long position
#         elif self.is_pos_greater(current_pos, 0):
#             # Check if exit needed - price below exit channel lower band or stop loss
#             if bar.close_price < self.exit_down or bar.close_price < self.long_stop:
#                 self.sell(symbol, bar.close_price, abs(current_pos))
#                 return
#             # Continue adding position logic - break new high to add long
#             self.send_buy_orders(self.entry_up)

#             # Long stop loss logic: take max of ATR stop and 10-day low
#             sell_price = max(self.long_stop, self.exit_down)
#             self.sell(symbol, sell_price, abs(current_pos), True)  # Close long position

#         # Short position
#         elif self.is_pos_less(current_pos, 0):
#             # Check if exit needed - price above exit channel upper band or stop loss
#             if bar.close_price > self.exit_up or bar.close_price > self.short_stop:
#                 self.buy(symbol, bar.close_price, abs(current_pos))
#                 return
#             # Continue adding position logic - break new low to add short
#             self.send_short_orders(self.entry_down)

#             # Short stop loss logic: take min of ATR stop and 10-day high
#             cover_price = min(self.short_stop, self.exit_up)
#             self.buy(
#                 symbol, cover_price, abs(current_pos), True
#             )  # Close short position

#     def on_trade(self, trade: ap.TradeData):
#         """
#         Trade callback: record trade price and set stop loss
#         """
#         if trade.direction == ap.Direction.LONG:
#             # Long trade, set long stop loss
#             self.long_entry = trade.price  # Record long entry price
#             self.long_stop = (
#                 self.long_entry - 2 * self.atr_value
#             )  # Stop loss at 2x ATR below entry price
#             logger.info(
#                 f"Long trade: {trade.symbol} {trade.orderid} {trade.volume}@{trade.price}, stop: {self.long_stop}"
#             )
#         else:
#             # Short trade, set short stop loss
#             self.short_entry = trade.price  # Record short entry price
#             self.short_stop = (
#                 self.short_entry + 2 * self.atr_value
#             )  # Stop loss at 2x ATR above entry price
#             logger.info(
#                 f"Short trade: {trade.symbol} {trade.orderid} {trade.volume}@{trade.price}, stop: {self.short_stop}"
#             )

#     def on_order(self, order: ap.OrderData):
#         """
#         Order callback
#         """
#         # Order callback doesn't need implementation now, may add logic later

#     def send_buy_orders(self, price):
#         """
#         Send long orders, including initial entry and pyramid scaling

#         A key feature of the Turtle system is pyramid position scaling, up to 4 units
#         """
#         symbol = self.symbols[0]  # Get current trading symbol

#         # Calculate current position units
#         pos = self.get_pos(symbol)
#         # Handle possible numpy array
#         if isinstance(pos, np.ndarray):
#             pos = float(pos)
#         t = pos / self.fixed_size

#         # First unit: enter at channel breakout
#         if isinstance(t, np.ndarray):
#             t = float(t)  # Ensure t is a scalar value

#         if t < 1:
#             self.buy(symbol, price, self.fixed_size, True)

#         # Second unit: add at first unit price + 0.5 ATR
#         if t < 2:
#             self.buy(symbol, price + self.atr_value * 0.5, self.fixed_size, True)

#         # Third unit: add at first unit price + 1 ATR
#         if t < 3:
#             self.buy(symbol, price + self.atr_value, self.fixed_size, True)

#         # Fourth unit: add at first unit price + 1.5 ATR
#         if t < 4:
#             self.buy(symbol, price + self.atr_value * 1.5, self.fixed_size, True)

#     def send_short_orders(self, price):
#         """
#         Send short orders, including initial entry and pyramid scaling

#         Opposite to long logic, prices decrease gradually
#         """
#         symbol = self.symbols[0]  # Get current trading symbol

#         # Calculate current position units
#         pos = self.get_pos(symbol)
#         # Handle possible numpy array
#         if isinstance(pos, np.ndarray):
#             pos = float(pos)
#         t = pos / self.fixed_size

#         # First unit: enter at channel breakout
#         if isinstance(t, np.ndarray):
#             t = float(t)  # Ensure t is a scalar value

#         if t > -1:
#             self.sell(symbol, price, self.fixed_size, True)

#         # Second unit: add at first unit price - 0.5 ATR
#         if t > -2:
#             self.sell(symbol, price - self.atr_value * 0.5, self.fixed_size, True)

#         # Third unit: add at first unit price - 1 ATR
#         if t > -3:
#             self.sell(symbol, price - self.atr_value, self.fixed_size, True)

#         # Fourth unit: add at first unit price - 1.5 ATR
#         if t > -4:
#             self.sell(symbol, price - self.atr_value * 1.5, self.fixed_size, True)


# def run_backtesting(show_chart=True):
#     """
#     Run Turtle Signal Strategy backtest
#     """
#     # Note: logging is already configured at module level
#     # Initialize backtest engine
#     bt_engine = ap.BacktestingEngine()
#     logger.info("1 Backtest engine created")

#     # Set backtest parameters
#     bt_engine.set_parameters(
#         symbols=["SOL-USDT.LOCAL"],  # Must use list format
#         interval=ap.Interval.MINUTE,
#         start=datetime(2023, 1, 1),
#         end=datetime(2023, 6, 30),  # Using 6 months of data
#     )
#     logger.info("2 Engine parameters set")

#     # Add strategy
#     bt_engine.add_strategy(
#         TurtleSignalStrategy,
#         {"entry_window": 20, "exit_window": 10, "atr_window": 20, "fixed_size": 1},
#     )

#     # Add data - using index-based data loading method
#     bt_engine.add_data(
#         symbol="SOL-USDT.LOCAL",
#         filepath="data/SOL-USDT_1m.csv",
#         datetime_index=0,
#         open_index=1,
#         high_index=2,
#         low_index=3,
#         close_index=4,
#         volume_index=5,
#     )

#     # Run backtest
#     bt_engine.run_backtesting()
#     logger.info("5 Backtest completed")

#     # Calculate results and generate report
#     bt_engine.report()
#     logger.info("6 Results and report generated")

#     # Log if chart display is skipped
#     if not show_chart:
#         logger.info("Chart display skipped")
#     logger.info("7 Chart display completed")

#     # Calculate and get data for return
#     df = bt_engine.daily_df
#     stats = bt_engine.calculate_statistics(output=False)
#     return df, stats, bt_engine


# if __name__ == "__main__":
#     run_backtesting(show_chart=True)
