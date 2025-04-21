import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from typing import Any, ClassVar

from apilot.core.constant import Direction, EngineType, Interval
from apilot.core.models import BarData, OrderData, TradeData

logger = logging.getLogger(__name__)


class PATemplate(ABC):
    parameters: ClassVar[list] = []
    variables: ClassVar[list] = []

    def __init__(
        self,
        pa_engine: Any,
        strategy_name: str,
        symbols: str | list[str],
        setting: dict,
    ) -> None:
        self.pa_engine = pa_engine
        self.strategy_name = strategy_name

        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols if symbols else []

        self.inited: bool = False
        self.trading: bool = False

        # Dictionary to manage positions and targets uniformly
        self.pos_dict: dict[str, int] = defaultdict(int)  # Actual position
        self.target_dict: dict[str, int] = defaultdict(int)  # Target position

        # Order cache container
        self.orders: dict[str, OrderData] = {}
        self.active_orderids: set[str] = set()

        # Copy variable list and insert default variables
        self.variables = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos_dict")
        self.variables.insert(3, "target_dict")

        # Set strategy parameters
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls) -> dict:
        """Gets the default parameter dictionary for the strategy class."""
        return {name: getattr(cls, name) for name in cls.parameters}

    def get_parameters(self) -> dict:
        """Gets the parameter dictionary for the strategy instance."""
        return {name: getattr(self, name) for name in self.parameters}

    def get_variables(self) -> dict:
        """Gets the strategy variable dictionary."""
        return {name: getattr(self, name) for name in self.variables}

    def get_data(self) -> dict:
        """Gets the strategy data."""
        data = {
            "strategy_name": self.strategy_name,
            "class_name": self.__class__.__name__,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
            "symbols": self.symbols,
        }
        return data

    @abstractmethod
    def on_init(self) -> None:
        pass

    @abstractmethod
    def on_start(self) -> None:
        pass

    @abstractmethod
    def on_stop(self) -> None:
        pass

    @abstractmethod
    def on_bar(self, bar: BarData) -> None:
        pass

    def on_trade(self, trade: TradeData) -> None:
        # Update position data
        self.pos_dict[trade.symbol] += (
            trade.volume if trade.direction == Direction.LONG else -trade.volume
        )

    def on_order(self, order: OrderData) -> None:
        # Update order cache
        self.orders[order.orderid] = order

        # If the order is no longer active, remove it from the active order set
        if not order.is_active() and order.orderid in self.active_orderids:
            self.active_orderids.remove(order.orderid)

    def buy(self, symbol: str, price: float, volume: float) -> list[str]:
        """
        Sends a buy order.
        """
        return self.send_order(symbol, Direction.LONG, price, volume)

    def sell(self, symbol: str, price: float, volume: float) -> list[str]:
        """
        Sends a sell order.
        """
        return self.send_order(symbol, Direction.SHORT, price, volume)

    def send_order(
        self,
        symbol: str,
        direction: Direction,
        price: float,
        volume: float,
    ) -> list[str]:
        """Sends an order to the trading engine."""
        try:
            if self.trading:
                # Send the order
                orderids: list[str] = self.pa_engine.send_order(
                    self,
                    symbol,
                    direction,
                    price,
                    volume,
                )

                # Add to active order set
                for orderid in orderids:
                    self.active_orderids.add(orderid)

                return orderids
            else:
                logger.warning(
                    f"[{self.strategy_name}] Strategy is not trading, order not sent"
                )
                return []
        except Exception as e:
            logger.error(f"[{self.strategy_name}] Sending order failed: {e!s}")
            return []

    def cancel_order(self, orderid: str) -> None:
        """Cancels a specific order."""
        if self.trading:
            self.pa_engine.cancel_order(self, orderid)

    def cancel_all(self) -> None:
        """Cancels all active orders for the strategy."""
        if self.trading:
            for orderid in list(self.active_orderids):
                self.cancel_order(orderid)

    def get_pos(self, symbol: str) -> float:
        """
        Gets the current position for a specific symbol.

        Safely returns the position, handling potential numpy array types
        and ensuring a float is returned.
        """
        import numpy as np

        pos = self.pos_dict.get(symbol, 0)
        if isinstance(pos, np.ndarray):
            pos = float(pos)
        return pos

    def is_pos_equal(self, pos, value) -> bool:
        """
        Safely compares if the position is equal to a value.
        Handles numpy arrays.
        """
        import numpy as np

        if isinstance(pos, np.ndarray):
            return np.array_equal(pos, value)
        return pos == value

    def is_pos_greater(self, pos, value) -> bool:
        """
        Safely compares if the position is greater than a value.
        Handles numpy arrays.
        """
        import numpy as np

        if isinstance(pos, np.ndarray):
            return np.all(pos > value)
        return pos > value

    def is_pos_less(self, pos, value) -> bool:
        """
        Safely compares if the position is less than a value.
        Handles numpy arrays.
        """
        import numpy as np

        if isinstance(pos, np.ndarray):
            return np.all(pos < value)
        return pos < value

    def get_target(self, symbol: str) -> int:
        """Gets the target position for a specific symbol."""
        return self.target_dict.get(symbol, 0)

    def set_target(self, symbol: str, target: int) -> None:
        """Sets the target position for a specific symbol."""
        self.target_dict[symbol] = target

    def get_engine_type(self) -> EngineType:
        """Gets the type of the trading engine (backtesting or live)."""
        return self.pa_engine.get_engine_type()

    def get_pricetick(self, symbol: str) -> float:
        return self.pa_engine.get_pricetick(self, symbol)

    def get_size(self, symbol: str) -> int:
        return self.pa_engine.get_size(self, symbol)

    def load_bar(
        self,
        count: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable | None = None,
    ) -> None:
        """Loads historical bar data, suitable for backtesting and live trading."""

        if not self.symbols:
            logger.warning("symbols is none, can not load bar")
            return

        callback = callback or self.on_bar

        for symbol in self.symbols:
            bars = self.pa_engine.load_bar(symbol, count, interval, callback)

            if bars and callback:
                for bar in bars:
                    callback(bar)

    def sync_data(self) -> None:
        """Synchronizes strategy variable values to disk storage."""
        if self.trading:
            self.pa_engine.sync_strategy_data(self)

    def calculate_price(
        self, symbol: str, direction: Direction, reference: float
    ) -> float:
        """Calculates the order price for portfolio rebalancing (can be overridden)."""
        return reference

    def rebalance_portfolio(self, bars: dict[str, BarData]) -> None:
        """Executes portfolio rebalancing trades based on target positions."""
        self.cancel_all()

        # Only send orders for contracts that have data in the current bar slice
        for symbol, bar in bars.items():
            # Calculate position difference
            target: int = self.get_target(symbol)
            pos: int = self.get_pos(symbol)
            diff: int = target - pos

            # Long position adjustment
            if diff > 0:
                # Calculate long order price
                order_price: float = self.calculate_price(
                    symbol, Direction.LONG, bar.close_price
                )
                # Send buy order
                self.buy(symbol, order_price, diff)
            # Short position adjustment
            elif diff < 0:
                # Calculate short order price
                order_price: float = self.calculate_price(
                    symbol, Direction.SHORT, bar.close_price
                )
                # Send sell order
                self.sell(symbol, order_price, abs(diff))

    def get_order(self, orderid: str) -> OrderData | None:
        """Queries order data by order ID."""
        return self.orders.get(orderid, None)

    def get_all_active_orderids(self) -> list[str]:
        """Gets all active order IDs for the strategy."""
        return list(self.active_orderids)
