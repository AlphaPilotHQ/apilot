"""
Backtesting Engine Module

For strategy backtesting and optimization.
"""

import logging
import time
from collections.abc import Callable
from datetime import date, datetime

from pandas import DataFrame

from apilot.core.constant import (
    Direction,
    EngineType,
    Interval,
    Status,
)
from apilot.core.models import (
    BarData,
    OrderData,
    TradeData,
)
from apilot.datafeed.csv_provider import CsvDatabase
from apilot.performance.calculator import calculate_statistics
from apilot.performance.report import PerformanceReport
from apilot.strategy.pa_template import PATemplate
from apilot.utils.utility import round_to

logger = logging.getLogger("BacktestEngine")


# Class to store daily performance results
class DailyResult:
    def __init__(self, date):
        self.date = date
        self.close_prices = {}  # Symbol: close_price
        self.net_pnl = 0.0  # Daily net profit/loss
        self.turnover = 0.0  # Daily trading volume value
        self.trade_count = 0  # Number of trades on this day

    def add_close_price(self, symbol, price):
        """Adds the closing price for a symbol on this date."""
        self.close_prices[symbol] = price

    def add_trade(self, trade, profit=0.0):
        """Adds trade details and associated profit for the day."""
        self.turnover += trade.price * trade.volume
        self.trade_count += 1
        self.net_pnl += profit


class BacktestingEngine:
    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self, main_engine=None) -> None:
        self.main_engine = main_engine
        self.symbols: list[str] = []
        self.start: datetime | None = None
        self.end: datetime | None = None
        self.sizes: dict[str, float] | None = None
        self.priceticks: dict[str, float] | None = None
        self.capital: int = 100_000
        self.annual_days: int = 365

        self.strategy_class: type[PATemplate] | None = None
        self.strategy: PATemplate | None = None
        self.bars: dict[str, BarData] = {}  # Current bars for active symbols
        self.datetime: datetime | None = None  # Current backtesting time

        self.interval: Interval | None = None
        self.callback: Callable | None = None  # Callback function for progress update
        self.history_data: dict[
            datetime, dict[str, BarData]
        ] = {}  # Stores all historical bar data
        self.dts: list[datetime] = []  # Sorted unique timestamps from history_data

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}  # All orders sent
        self.active_limit_orders: dict[
            str, OrderData
        ] = {}  # Orders not yet filled or cancelled

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}  # All trades executed

        self.logs: list = []  # Stores log messages (if needed)

        self.daily_results: dict[date, DailyResult] = {}
        self.daily_df: DataFrame | None = None  # DataFrame for daily results storage
        self.accounts = {"balance": self.capital}  # Account balance tracking
        self.positions: dict[
            str, dict
        ] = {}  # Position tracking: {symbol: {"volume": float, "avg_price": float}}

    def clear_data(self) -> None:
        """Reset engine state for a new backtest run."""
        self.__init__(self.main_engine)

    def set_parameters(
        self,
        symbols: list[str],
        interval: Interval,
        start: datetime,
        sizes: dict[str, float] | None = None,
        priceticks: dict[str, float] | None = None,
        capital: int = 100_000,
        end: datetime | None = None,
        annual_days: int = 240,
    ) -> None:
        # Parameters from user settings
        # self.mode removed - framework now uses bar-based data only
        self.symbols = symbols  # List of symbols to trade
        self.interval = Interval(interval)
        self.sizes = sizes if sizes is not None else {}
        self.priceticks = priceticks if priceticks is not None else {}
        self.start = start

        # Store symbols (no need to cache exchange objects anymore)

        self.capital = capital
        self.annual_days = annual_days

        if not end:
            end = datetime.now()
        self.end = end.replace(hour=23, minute=59, second=59)

        if self.start >= self.end:
            logger.warning(
                f"Error: Start date ({self.start}) must be before end date ({self.end})"
            )

    def add_strategy(
        self, strategy_class: type[PATemplate], setting: dict | None = None
    ) -> None:
        """Adds and initializes the trading strategy."""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.symbols, setting
        )

    def add_data(self, symbol, **kwargs):
        """Loads historical data for a symbol using the default provider (csv)."""
        start_time = time.time()
        provider = CsvDatabase(**kwargs)
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        data_params = {
            k: kwargs[k] for k in ["downsample_minutes", "limit_count"] if k in kwargs
        }
        bars = provider.load_bar_data(
            symbol=symbol,
            interval=self.interval,
            start=self.start,
            end=self.end,
            **data_params,
        )
        for bar in bars:
            bar.symbol = symbol
            self.dts.append(bar.datetime)
            self.history_data.setdefault(bar.datetime, {})[symbol] = bar
        self.dts = sorted(set(self.dts))
        logger.info(
            f"Loaded {len(bars)} bars for {symbol} in {time.time() - start_time:.2f}s"
        )
        return self

    def run_backtesting(self) -> None:
        self.strategy.on_init()
        logger.debug("Strategy on_init() called")

        # Pre-warming phase - use first N bars to initialize strategy
        if not self.dts:
            logger.error("No valid data points found, check data loading")
            return

        warmup_bars = min(100, len(self.dts))
        logger.debug(f"Using {warmup_bars} time points for strategy pre-warming")

        for i in range(warmup_bars):
            dt = self.dts[i]
            for _, bar in self.history_data[dt].items():
                try:
                    self.new_bar(bar)
                except Exception as e:
                    logger.error(f"Pre-warming phase error: {e}")

        # Set to trading mode after pre-warming
        self.strategy.inited = True
        self.strategy.trading = True
        self.strategy.on_start()

        # Use remaining historical data for strategy backtesting
        logger.debug(
            f"Starting backtesting, start index: {warmup_bars}, end index: {len(self.dts)}"
        )
        for dt in self.dts[warmup_bars:]:
            for _, bar in self.history_data[dt].items():
                try:
                    self.new_bar(bar)
                except Exception as e:
                    logger.error(f"Backtesting phase error: {e}")
            logger.debug(
                f"Backtest finished: "
                f"trade_count: {self.trade_count}, "
                f"active_limit_orders: {len(self.active_limit_orders)}, "
                f"limit_orders: {len(self.limit_orders)}"
            )

    def update_daily_close(self, price: float, symbol: str) -> None:
        d: date = self.datetime.date()

        daily_result: DailyResult = self.daily_results.get(d, None)
        if daily_result:
            daily_result.add_close_price(symbol, price)
        else:
            self.daily_results[d] = DailyResult(d)
            self.daily_results[d].add_close_price(symbol, price)

    def new_bar(self, bar: BarData) -> None:
        """
        Process new bar data for a single symbol. Robust and simple.
        """
        self.bars[bar.symbol] = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        try:
            self.strategy.on_bar(bar)
        except Exception as e:
            logger.error(f"on_bar error for {bar.symbol} @ {bar.datetime}: {e}")

        price = getattr(bar, "close_price", None)
        if price is not None:
            self.update_daily_close(price, bar.symbol)
        else:
            logger.warning(f"Bar missing close_price: {bar.symbol} @ {bar.datetime}")

    def cross_limit_order(self) -> None:
        """
        Match limit orders
        """
        for order in list(self.active_limit_orders.values()):
            # Update order status
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)
                logger.debug(f"Order {order.orderid} status: {order.status}")

            # Get price for order's symbol
            symbol = order.symbol

            bar = self.bars.get(symbol)
            if not bar:
                logger.info(
                    f"No bar data found for order's symbol: {symbol}, current time: {self.datetime}, order ID: {order.orderid}"
                )
                continue
            buy_price = bar.low_price
            sell_price = bar.high_price

            # Check if order is filled
            buy_cross = (
                order.direction == Direction.LONG
                and order.price >= buy_price
                and buy_price > 0
            )
            sell_cross = (
                order.direction == Direction.SHORT
                and order.price <= sell_price
                and sell_price > 0
            )

            if not buy_cross and not sell_cross:
                continue

            # Set order as filled
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)
            logger.debug(f"Order {order.orderid} status: {order.status}")

            if order.orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.orderid)

            # Create trade record
            self.trade_count += 1

            # Determine fill price and position change
            trade_price = buy_price if buy_cross else sell_price
            # pos_change = order.volume if buy_cross else -order.volume

            # Create trade object
            trade = TradeData(
                symbol=order.symbol,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            trade.orderid = order.orderid
            trade.tradeid = f"{self.gateway_name}.{trade.tradeid}"

            self.trades[trade.tradeid] = trade
            logger.debug(
                f"Trade record created: {trade.tradeid}, direction: {trade.direction}, price: {trade.price}, volume: {trade.volume}"
            )

            # Update current position and account balance
            self.update_account_balance(trade)

            self.strategy.on_trade(trade)

    def update_account_balance(self, trade: TradeData) -> None:
        """
        Update account balance after each trade
        Using net position model: no distinction between open and close, only direction and position change
        """
        # If position dictionary doesn't exist, initialize it
        if not hasattr(self, "positions"):
            self.positions = {}  # Format: {symbol: {"volume": 0, "avg_price": 0.0}}

        symbol = trade.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {"volume": 0, "avg_price": 0.0}

        # Get current position
        position = self.positions[symbol]
        old_volume = position["volume"]

        # Calculate position change
        volume_change = (
            trade.volume if trade.direction == Direction.LONG else -trade.volume
        )
        new_volume = old_volume + volume_change

        # Calculate profit/loss
        profit = 0.0

        # If reducing position
        if (old_volume > 0 and volume_change < 0) or (
            old_volume < 0 and volume_change > 0
        ):
            # Determine if closing long or short position
            if old_volume > 0:  # Closing long position
                # Calculate profit/loss for closing part
                profit = (trade.price - position["avg_price"]) * min(
                    abs(volume_change), abs(old_volume)
                )
            else:  # Closing short position
                # Calculate profit/loss for closing part
                profit = (position["avg_price"] - trade.price) * min(
                    abs(volume_change), abs(old_volume)
                )

            # If fully closing or reversing position
            if old_volume * new_volume <= 0:
                # If reversing direction, reset average price for remaining part
                if abs(new_volume) > 0:
                    # Reset average price to current price
                    position["avg_price"] = trade.price
                else:
                    # Fully closing, reset average price
                    position["avg_price"] = 0.0
            else:
                # Partially closing, average price remains the same
                pass
        else:
            # If increasing position
            if new_volume != 0:
                # Calculate new average price
                if old_volume == 0:
                    position["avg_price"] = trade.price
                else:
                    # Same direction, update average price
                    position["avg_price"] = (
                        position["avg_price"] * abs(old_volume)
                        + trade.price * abs(volume_change)
                    ) / abs(new_volume)

        # Update position volume
        position["volume"] = new_volume

        # Update account balance
        self.accounts["balance"] += profit
        # Improved log message
        profit_type = "Profit" if profit > 0 else "Loss" if profit < 0 else "Break-even"
        logger.info(
            f"Trade {trade.tradeid}: {profit_type} {profit:.2f}, account balance: {self.accounts['balance']:.2f}, position: {new_volume}, average price: {position['avg_price']:.4f}"
        )

        # Add profit/loss value to trade object's profit attribute
        trade.profit = profit

        # Update daily result
        trade_date = trade.datetime.date()
        if trade_date in self.daily_results:
            self.daily_results[trade_date].add_trade(trade, profit)
        else:
            # If no record for this date, create a new daily result
            self.daily_results[trade_date] = DailyResult(trade_date)
            self.daily_results[trade_date].add_trade(trade, profit)

    def send_order(
        self,
        strategy: PATemplate,
        symbol: str,
        direction: Direction,
        price: float,
        volume: float,
    ) -> list:
        """
        Send order
        """
        price_tick = self.priceticks.get(symbol, 0.001)
        price: float = round_to(price, price_tick)
        orderid: str = self.send_limit_order(symbol, direction, price, volume)
        return [orderid]

    def send_limit_order(
        self,
        symbol: str,
        direction: Direction,
        price: float,
        volume: float,
    ) -> str:
        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=symbol,
            orderid=str(self.limit_order_count),
            direction=direction,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime,
        )

        self.active_limit_orders[order.orderid] = order
        self.limit_orders[order.orderid] = order

        logger.debug(
            f"Created order: {order.orderid}, symbol: {symbol}, direction: {direction}, price: {price}, volume: {volume}"
        )
        return order.orderid

    def cancel_order(self, strategy: PATemplate, orderid: str) -> None:
        """
        Cancel order
        """
        if orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: PATemplate) -> None:
        """
        Cancel all orders
        """
        orderids: list = list(self.active_limit_orders.keys())
        for orderid in orderids:
            self.cancel_order(strategy, orderid)

    def sync_strategy_data(self, strategy: PATemplate) -> None:
        """
        Sync strategy data
        """
        pass

    def get_engine_type(self) -> EngineType:
        return self.engine_type

    def get_pricetick(self, strategy: PATemplate, symbol: str) -> float:
        return self.priceticks.get(symbol, 0.0001)

    def get_size(self, strategy: PATemplate, symbol: str) -> int:
        # If symbol not in sizes dictionary, return default value 1
        return self.sizes.get(symbol, 1)

    def get_all_trades(self) -> list:
        return list(self.trades.values())

    def get_all_orders(self) -> list:
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list:
        return list(self.daily_results.values())

    def calculate_result(self) -> DataFrame:
        import pandas as pd

        # Check if there are trades
        if not self.trades:
            return pd.DataFrame()

        # Collect daily data
        daily_results = []

        # Ensure all trade dates have records
        all_dates = sorted(self.daily_results.keys())

        # Initialize first day's balance to initial capital
        current_balance = self.capital

        for d in all_dates:
            daily_result = self.daily_results[d]

            # Get daily result data
            result = {
                "date": d,
                "trade_count": daily_result.trade_count,
                "turnover": daily_result.turnover,
                "net_pnl": daily_result.net_pnl,
            }

            # Update current balance
            current_balance += daily_result.net_pnl
            result["balance"] = current_balance

            daily_results.append(result)

        # Create DataFrame
        self.daily_df = pd.DataFrame(daily_results)

        if not self.daily_df.empty:
            self.daily_df.set_index("date", inplace=True)

            # Calculate drawdown
            self.daily_df["highlevel"] = self.daily_df["balance"].cummax()
            self.daily_df["ddpercent"] = (
                (self.daily_df["balance"] - self.daily_df["highlevel"])
                / self.daily_df["highlevel"]
                * 100
            )

            # Calculate return
            pre_balance = self.daily_df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital

            # Safely calculate return
            self.daily_df["return"] = (
                self.daily_df["balance"].pct_change().fillna(0) * 100
            )
            self.daily_df.loc[self.daily_df.index[0], "return"] = (
                (self.daily_df["balance"].iloc[0] / self.capital) - 1
            ) * 100

        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=False) -> dict:
        """Calculate performance statistics (only for valid daily results)."""
        if df is None:
            df = self.daily_df
        if df is None or df.empty:
            return {}
        stats = calculate_statistics(
            df=df,
            trades=list(self.trades.values()),
            capital=self.capital,
            annual_days=self.annual_days,
        )
        if output:
            self._print_statistics(stats)
        return stats

    def _print_statistics(self, stats):
        """Print statistics"""
        logger.info(
            f"Trade day:\t{stats.get('start_date', '')} - {stats.get('end_date', '')}"
        )
        logger.info(f"Profit days:\t{stats.get('profit_days', 0)}")
        logger.info(f"Loss days:\t{stats.get('loss_days', 0)}")
        logger.info(f"Initial capital:\t{self.capital:.2f}")
        logger.info(f"Final capital:\t{stats.get('final_capital', 0):.2f}")
        logger.info(f"Total return:\t{stats.get('total_return', 0):.2f}%")
        logger.info(f"Annual return:\t{stats.get('annual_return', 0):.2f}%")
        logger.info(f"Max drawdown:\t{stats.get('max_drawdown', 0):.2f}%")
        logger.info(f"Total turnover:\t{stats.get('total_turnover', 0):.2f}")
        logger.info(f"Total trades:\t{stats.get('total_trade_count', 0)}")
        logger.info(f"Sharpe ratio:\t{stats.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Return/Drawdown:\t{stats.get('return_drawdown_ratio', 0):.2f}")

    def load_bar(
        self,
        symbol: str,
        count: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable | None = None,
    ) -> list[BarData]:
        """Placeholder method for backtesting environment, actual data loaded via add_csv_data etc."""
        logger.debug(f"Backtest engine load_bar called: {symbol}, {count} count bar")
        return []

    def get_current_capital(self) -> float:
        """
        Get current account value (initial capital + realized profit/loss)

        Simplified implementation: directly use account balance
        """
        return self.accounts.get("balance", self.capital)

    def report(self) -> None:
        """
        Generate and display performance report
        """
        # Calculate results if not already done
        if self.daily_df is None:
            self.calculate_result()

        # Create and display performance report
        report = PerformanceReport(
            df=self.daily_df,
            trades=list(self.trades.values()),
            capital=self.capital,
            annual_days=self.annual_days,
        )
        report.show()

    def optimize(self, strategy_setting=None, max_workers=4) -> list[dict]:
        """Run parameter optimization (grid search)."""
        from apilot.optimizer import OptimizationSetting, run_grid_search

        if not self.strategy_class:
            logger.error("Cannot optimize: strategy class not set")
            return []
        if strategy_setting is None:
            strategy_setting = OptimizationSetting()
            strategy_setting.set_target("total_return")
            if hasattr(self.strategy_class, "parameters"):
                for param in self.strategy_class.parameters:
                    if hasattr(self.strategy, param):
                        current_value = getattr(self.strategy, param)
                        if isinstance(current_value, int | float):
                            strategy_setting.add_parameter(
                                param,
                                max(1, current_value // 2)
                                if isinstance(current_value, int)
                                else current_value * 0.5,
                                current_value * 2,
                                max(1, current_value // 10)
                                if isinstance(current_value, int)
                                else current_value * 0.1,
                            )

        def evaluate_setting(setting):
            test_engine = BacktestingEngine()
            test_engine.set_parameters(
                symbols=self.symbols.copy(),
                interval=self.interval,
                start=self.start,
                end=self.end,
                capital=self.capital,
            )
            for dt in self.dts:
                if dt in self.history_data:
                    test_engine.history_data[dt] = self.history_data[dt].copy()
            test_engine.dts = self.dts.copy()
            test_engine.add_strategy(self.strategy_class, setting)
            try:
                test_engine.run_backtesting()
                test_engine.calculate_result()
                stats = test_engine.calculate_statistics()
                target_name = strategy_setting.target_name or "total_return"
                fitness = stats.get(target_name, 0)
                return fitness
            except Exception as e:
                logger.error(f"Parameter evaluation failed: {e!s}")
                return -999999

        return run_grid_search(
            strategy_class=self.strategy_class,
            optimization_setting=strategy_setting,
            key_func=evaluate_setting,
            max_workers=max_workers,
        )
