from typing import TYPE_CHECKING, ClassVar

from apilot.core import (
    BaseEngine,
    ContractData,
    Direction,
    OrderData,
    OrderType,
    TickData,
    TradeData,
)
from apilot.core.constant import AlgoStatus
from apilot.core.utility import virtual
from apilot.utils.logger import get_logger

# 模块级初始化日志器
logger = get_logger("AlgoTrading")

if TYPE_CHECKING:
    from .engine import AlgoEngine


class AlgoTemplate:
    """算法模板"""

    _count: int = 0  # 实例计数

    display_name: str = ""  # 显示名称
    default_setting: ClassVar[dict] = {}  # 默认参数
    variables: ClassVar[list] = []  # 变量名称

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
        """构造函数"""
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

    def update_tick(self, tick: TickData) -> None:
        """行情数据更新"""
        if self.status == AlgoStatus.RUNNING:
            self.on_tick(tick)

    def update_order(self, order: OrderData) -> None:
        """委托数据更新"""
        if order.is_active():
            self.active_orders[order.orderid] = order
        elif order.orderid in self.active_orders:
            self.active_orders.pop(order.orderid)

        self.on_order(order)

    def update_trade(self, trade: TradeData) -> None:
        """成交数据更新"""
        cost: float = self.traded_price * self.traded + trade.price * trade.volume
        self.traded += trade.volume
        self.traded_price = cost / self.traded

        self.on_trade(trade)

    def update_timer(self) -> None:
        """每秒定时更新"""
        if self.status == AlgoStatus.RUNNING:
            self.on_timer()

    @virtual
    def on_tick(self, tick: TickData) -> None:
        """行情回调"""
        pass

    @virtual
    def on_order(self, order: OrderData) -> None:
        """委托回调"""
        pass

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        """成交回调"""
        pass

    @virtual
    def on_timer(self) -> None:
        """定时回调"""
        pass

    def start(self) -> None:
        """启动"""
        self.status = AlgoStatus.RUNNING
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] 算法启动")

    def stop(self) -> None:
        """停止"""
        self.status = AlgoStatus.STOPPED
        self.cancel_all()
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] 算法停止")

    def finish(self) -> None:
        """结束"""
        self.status = AlgoStatus.FINISHED
        self.cancel_all()
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] 算法结束")

    def pause(self) -> None:
        """暂停"""
        self.status = AlgoStatus.PAUSED
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] 算法暂停")

    def resume(self) -> None:
        """恢复"""
        self.status = AlgoStatus.RUNNING
        self.put_event()

        logger.info(f"[Algo:{self.algo_name}] 算法恢复")

    def buy(
        self,
        price: float,
        volume: float,
        order_type: OrderType = OrderType.LIMIT,
    ) -> None:
        """买入"""
        if self.status != AlgoStatus.RUNNING:
            return

        msg: str = f"{self.symbol}, 委托买入{order_type.value}, {volume}@{price}"
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
        """卖出"""
        if self.status != AlgoStatus.RUNNING:
            return

        msg: str = f"{self.symbol}委托卖出{order_type.value}, {volume}@{price}"
        logger.info(f"[Algo:{self.algo_name}] {msg}")

        return self.algo_engine.send_order(
            self, Direction.SHORT, price, volume, order_type
        )

    def cancel_order(self, orderid: str) -> None:
        """撤销委托"""
        self.algo_engine.cancel_order(self, orderid)

    def cancel_all(self) -> None:
        """全撤委托"""
        if not self.active_orders:
            return

        for orderid in self.active_orders.keys():
            self.cancel_order(orderid)

    def get_tick(self) -> TickData | None:
        """查询行情"""
        return self.algo_engine.get_tick(self)

    def get_contract(self) -> ContractData | None:
        """查询合约"""
        return self.algo_engine.get_contract(self)

    def get_parameters(self) -> dict:
        """获取算法参数"""
        strategy_parameters: dict = {}
        for name in self.default_setting.keys():
            strategy_parameters[name] = getattr(self, name)
        return strategy_parameters

    def get_variables(self) -> dict:
        """获取算法变量"""
        strategy_variables: dict = {}
        for name in self.variables:
            strategy_variables[name] = getattr(self, name)
        return strategy_variables

    def get_data(self) -> dict:
        """获取算法信息"""
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
        """推送更新"""
        data: dict = self.get_data()
        self.algo_engine.put_algo_event(self, data)
