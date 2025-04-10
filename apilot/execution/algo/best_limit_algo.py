from random import uniform
from typing import ClassVar

from apilot.core import BaseEngine, Direction, OrderData, TickData, TradeData

from .algo_template import AlgoTemplate


class BestLimitAlgo(AlgoTemplate):
    """最优限价算法类"""

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
        """构造函数"""
        super().__init__(
            algo_engine, algo_name, symbol, direction, price, volume, setting
        )

        # 参数
        self.min_volume: float = setting["min_volume"]
        self.max_volume: float = setting["max_volume"]

        # 变量
        self.orderid: str = ""
        self.order_price: float = 0

        self.put_event()

        # 检查最大/最小挂单量
        if self.min_volume <= 0:
            self.write_log("最小挂单量必须大于0, 算法启动失败")
            self.finish()
            return

        if self.max_volume < self.min_volume:
            self.write_log("最大挂单量必须不小于最小委托量, 算法启动失败")
            self.finish()
            return

    def on_tick(self, tick: TickData) -> None:
        """Tick行情回调"""
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
        """成交回调"""
        if self.traded >= self.volume:
            self.write_log(f"已交易数量: {self.traded}, 总数量: {self.volume}")
            self.finish()
        else:
            self.put_event()

    def on_order(self, order: OrderData) -> None:
        """委托回调"""
        if not order.is_active():
            self.orderid = ""
            self.order_price = 0
            self.put_event()

    def buy_best_limit(self, bid_price_1: float) -> None:
        """最优限价买入"""
        volume_left: float = self.volume - self.traded

        rand_volume: int = self.generate_rand_volume()
        order_volume: float = min(rand_volume, volume_left)

        self.order_price = bid_price_1
        self.orderid = self.buy(self.order_price, order_volume)

    def sell_best_limit(self, ask_price_1: float) -> None:
        """最优限价卖出"""
        volume_left: float = self.volume - self.traded

        rand_volume: int = self.generate_rand_volume()
        order_volume: float = min(rand_volume, volume_left)

        self.order_price = ask_price_1
        self.orderid = self.sell(self.order_price, order_volume)

    def generate_rand_volume(self) -> int:
        """随机生成委托数量"""
        rand_volume: float = uniform(self.min_volume, self.max_volume)
        return int(rand_volume)
