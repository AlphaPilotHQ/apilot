from apilot.core import BaseEngine, Direction, OrderData, TickData, TradeData

from .algo_template import AlgoTemplate


class StopAlgo(AlgoTemplate):
    """条件委托算法类"""

    display_name: str = "Stop 条件委托"

    default_setting: dict = {
        "price_add": 0.0
    }

    variables: list = [
        "orderid",
        "order_status",
    ]

    def __init__(
        self,
        algo_engine: BaseEngine,
        algo_name: str,
        symbol: str,
        direction: str,
        offset: str,
        price: float,
        volume: float,
        setting: dict
    ) -> None:
        """构造函数"""
        super().__init__(algo_engine, algo_name, symbol, direction, offset, price, volume, setting)

        # 参数
        self.price_add: float = setting["price_add"]

        # 变量
        self.orderid: str = ""
        self.order_status: str = ""

        self.put_event()

    def on_tick(self, tick: TickData) -> None:
        """Tick行情回调"""
        if self.orderid:
            return

        if self.direction == Direction.LONG:
            if tick.last_price >= self.price:
                price: float = self.price + self.price_add

                if tick.limit_up:
                    price = min(price, tick.limit_up)

                self.orderid = self.buy(
                    price,
                    self.volume,
                    offset=self.offset
                )
                self.write_log(
                    f"停止单已触发，代码：{self.symbol}，方向：{self.direction}, 价格：{self.price}，数量：{self.volume}，开平：{self.offset}")

        else:
            if tick.last_price <= self.price:
                price: float = self.price - self.price_add

                if tick.limit_down:
                    price = max(price, tick.limit_down)

                self.orderid = self.sell(
                    price,
                    self.volume,
                    offset=self.offset
                )
                self.write_log(
                    f"停止单已触发，代码：{self.symbol}，方向：{self.direction}, 价格：{self.price}，数量：{self.volume}，开平：{self.offset}")

        self.put_event()

    def on_order(self, order: OrderData) -> None:
        """委托回调"""
        self.order_status = order.status
        self.put_event()

    def on_trade(self, trade: TradeData) -> None:
        """成交回调"""
        if self.traded == self.volume:
            self.finish()
        self.put_event()
