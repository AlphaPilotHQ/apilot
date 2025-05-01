import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from typing import Any, ClassVar

from apilot.core.constant import Direction, Interval
from apilot.core.models import BarData, OrderData, TradeData

logger = logging.getLogger(__name__)


class PATemplate(ABC):
    parameters: ClassVar[list] = []
    variables: ClassVar[list] = []

    def __init__(
        self,
        pa_engine: Any,
        strategy_name: str,
        symbol: str,
        setting: dict,
    ) -> None:
        self.pa_engine = pa_engine
        self.strategy_name = strategy_name

        self.symbol: str = symbol

        self.inited: bool = False
        self.trading: bool = False
        self.pos: int = 0
        self.orders: dict[str, OrderData] = {}
        self.active_orderids: set[str] = set()

        self.variables = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos")

        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

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
        if trade.symbol == self.symbol:
            if trade.direction == Direction.LONG:
                self.pos += trade.volume
            else:
                self.pos -= trade.volume
        else:
            logger.warning(
                f"[{self.strategy_name}] Received trade for unexpected symbol: {trade.symbol}"
            )

    def on_order(self, order: OrderData) -> None:
        self.orders[order.orderid] = order

        if not order.is_active() and order.orderid in self.active_orderids:
            self.active_orderids.remove(order.orderid)

    def buy(self, price: float, volume: float) -> list[str]:
        return self.send_order(self.symbol, Direction.LONG, price, volume)

    def sell(self, price: float, volume: float) -> list[str]:
        return self.send_order(self.symbol, Direction.SHORT, price, volume)

    def send_order(
        self,
        symbol: str,
        direction: Direction,
        price: float,
        volume: float,
    ) -> list[str]:
        try:
            if self.trading:
                orderids = self.pa_engine.send_order(
                    self,
                    symbol,
                    direction,
                    price,
                    volume,
                )

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
        if self.trading:
            self.pa_engine.cancel_order(self, orderid)

    def cancel_all(self) -> None:
        if self.trading:
            for orderid in list(self.active_orderids):
                self.cancel_order(orderid)

    def get_pos(self) -> int:
        return self.pos

    def get_pricetick(self, symbol: str) -> float:
        return self.pa_engine.get_pricetick(self, symbol)

    def get_size(self, symbol: str) -> int:
        return self.pa_engine.get_size(self, symbol)

    def get_order(self, orderid: str) -> OrderData | None:
        return self.orders.get(orderid, None)

    def get_all_active_orderids(self) -> list[str]:
        return list(self.active_orderids)

    def load_bar(
        self,
        count: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable[[BarData], Any] | None = None,
    ) -> list[BarData] | None:

        try:
            bars = self.pa_engine.load_bar(self.symbol, count, interval)
        except Exception as e:
            logger.error(f"Failed to load bars for {self.symbol}: {e}")
            return None

        if callback is None:
            if hasattr(self, "bg") and callable(getattr(self, "bg").update_bar):
                callback = self.bg.update_bar
            else:
                callback = self.on_bar

        for bar in bars or []:
            callback(bar)

        return bars
