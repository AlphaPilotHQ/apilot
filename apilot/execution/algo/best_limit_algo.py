from random import uniform
from typing import ClassVar

from apilot.core import BaseEngine, Direction, OrderData, TickData, TradeData

from .algo_template import AlgoTemplate


class BestLimitAlgo(AlgoTemplate):
    """Best Limit Price Algorithm"""

    default_setting: ClassVar[dict] = {
        "min_volume": 0,
        "max_volume": 0,
    }

    variables: ClassVar[list] = ["orderid", "order_price"]

    def __init__(
        self,
        algo_engine: BaseEngine,
        algo_name: str,
        symbol: str,
        direction: str,
        price: float,
        volume: float,
        setting: dict,
    ) -> None:
        """Constructor"""
        super().__init__(
            algo_engine, algo_name, symbol, direction, price, volume, setting
        )

        # Parameters
        self.min_volume: float = setting["min_volume"]
        self.max_volume: float = setting["max_volume"]

        # Variables
        self.orderid: str = ""
        self.order_price: float = 0

        self.put_event()

        # Check max/min order volume
        if self.min_volume <= 0:
            self.write_log(
                "Minimum order volume must be greater than 0, algorithm failed to start"
            )
            self.finish()
            return

        if self.max_volume < self.min_volume:
            self.write_log(
                "Maximum order volume must not be less than minimum order volume, algorithm failed to start"
            )
            self.finish()
            return

    def on_tick(self, tick: TickData) -> None:
        """Tick data callback"""
        if self.direction == Direction.LONG:
            if not self.orderid:
                self.buy_best_limit(tick.bid_price_1)
            elif self.order_price != tick.bid_price_1:
                self.cancel_all()
        else:
            if not self.orderid:
                self.sell_best_limit(tick.ask_price_1)
            elif self.order_price != tick.ask_price_1:
                self.cancel_all()

        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """Trade callback"""
        if self.traded >= self.volume:
            self.write_log(f"Traded volume: {self.traded}, total volume: {self.volume}")
            self.finish()
        else:
            self.put_event()

    def on_order(self, order: OrderData) -> None:
        """Order callback"""
        if not order.is_active():
            self.orderid = ""
            self.order_price = 0
            self.put_event()

    def buy_best_limit(self, bid_price_1: float) -> None:
        """Buy with best limit price"""
        volume_left: float = self.volume - self.traded

        rand_volume: int = self.generate_rand_volume()
        order_volume: float = min(rand_volume, volume_left)

        self.order_price = bid_price_1
        self.orderid = self.buy(self.order_price, order_volume)

    def sell_best_limit(self, ask_price_1: float) -> None:
        """Sell with best limit price"""
        volume_left: float = self.volume - self.traded

        rand_volume: int = self.generate_rand_volume()
        order_volume: float = min(rand_volume, volume_left)

        self.order_price = ask_price_1
        self.orderid = self.sell(self.order_price, order_volume)

    def generate_rand_volume(self) -> int:
        """Generate random order volume"""
        rand_volume: float = uniform(self.min_volume, self.max_volume)
        return int(rand_volume)
