import logging
from typing import TYPE_CHECKING, ClassVar

from apilot.core import (
    BaseEngine,
    ContractData,
    Direction,
    OrderData,
    OrderType,
    TradeData,
)
from apilot.core.constant import AlgoStatus
from apilot.core.utility import virtual

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .engine import AlgoEngine


class AlgoTemplate:
    """Algorithm Template"""

    _count: int = 0  # Instance counter

    display_name: str = ""  # Display name
    default_setting: ClassVar[dict] = {}  # Default parameters
    variables: ClassVar[list] = []  # Variable names

    def __init__(
        self,
        algo_engine: "AlgoEngine",
        algo_name: str,
        symbol: str,
        direction: Direction,
        price: float,
        volume: int,
        setting: dict,
    ) -> None:
        """Constructor"""
        self.algo_engine: BaseEngine = algo_engine
        self.algo_name: str = algo_name

        self.symbol: str = symbol
        self.direction: Direction = direction
        self.price: float = price
        self.volume: int = volume

        self.status: AlgoStatus = AlgoStatus.PAUSED
        self.traded: float = 0
        self.traded_price: float = 0

        self.active_orders: dict[str, OrderData] = {}

    def update_order(self, order: OrderData) -> None:
        """Update order data"""
        if order.is_active():
            self.active_orders[order.orderid] = order
        elif order.orderid in self.active_orders:
            self.active_orders.pop(order.orderid)

        self.on_order(order)

    def update_trade(self, trade: TradeData) -> None:
        """Update trade data"""
        cost: float = self.traded_price * self.traded + trade.price * trade.volume
        self.traded += trade.volume
        self.traded_price = cost / self.traded

        self.on_trade(trade)

    def update_timer(self) -> None:
        """Update timer every second"""
        if self.status == AlgoStatus.RUNNING:
            self.on_timer()

    @virtual
    def on_order(self, order: OrderData) -> None:
        """Order callback"""
        pass

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        """Trade callback"""
        pass

    @virtual
    def on_timer(self) -> None:
        """Timer callback"""
        pass

    def start(self) -> None:
        """Start algorithm"""
        self.status = AlgoStatus.RUNNING
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] Algorithm started")

    def stop(self) -> None:
        """Stop algorithm"""
        self.status = AlgoStatus.STOPPED
        self.cancel_all()
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] Algorithm stopped")

    def finish(self) -> None:
        """Finish algorithm"""
        self.status = AlgoStatus.FINISHED
        self.cancel_all()
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] Algorithm finished")

    def pause(self) -> None:
        """Pause algorithm"""
        self.status = AlgoStatus.PAUSED
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] Algorithm paused")

    def resume(self) -> None:
        """Resume algorithm"""
        self.status = AlgoStatus.RUNNING
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] Algorithm resumed")

    def buy(
        self,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> None:
        """Buy order"""
        if self.status != AlgoStatus.RUNNING:
            return

        msg: str = f"{self.symbol}, buy order {order_type.value}, {volume}@{price}"
        logger.info(f"[Algo:{self.algo_name}] {msg}")

        return self.algo_engine.send_order(
            self, Direction.LONG, price, volume, order_type
        )

    def sell(
        self,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> None:
        """Sell order"""
        if self.status != AlgoStatus.RUNNING:
            return

        msg: str = f"{self.symbol}, sell order {order_type.value}, {volume}@{price}"
        logger.info(f"[Algo:{self.algo_name}] {msg}")

        return self.algo_engine.send_order(
            self, Direction.SHORT, price, volume, order_type
        )

    def cancel_order(self, orderid: str) -> None:
        """Cancel order"""
        self.algo_engine.cancel_order(self, orderid)

    def cancel_all(self) -> None:
        """Cancel all orders"""
        if not self.active_orders:
            return

        for orderid in self.active_orders.keys():
            self.cancel_order(orderid)

    def get_contract(self) -> ContractData | None:
        """Get contract info"""
        return self.algo_engine.get_contract(self)

    def get_parameters(self) -> dict:
        """Get algorithm parameters"""
        strategy_parameters: dict = {}
        for name in self.default_setting.keys():
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self) -> dict:
        """Get algorithm variables"""
        strategy_variables: dict = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self) -> dict:
        """Get algorithm information"""
        algo_data: dict = {
            "algo_name": self.algo_name,
            "symbol": self.symbol,
            "direction": self.direction,
            "price": self.price,
            "volume": self.volume,
            "status": self.status,
            "traded": self.traded,
            "traded_price": self.traded_price,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }
        return algo_data

    def put_event(self) -> None:
        """Push update"""
        data: dict = self.get_data()
        self.algo_engine.put_algo_event(self, data)
