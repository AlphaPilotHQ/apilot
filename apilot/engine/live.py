"""
实时交易引擎模块

实现交易策略的实时运行与管理,包括信号处理、订单执行与风控
"""

import copy
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

from apilot.core import (
    # 常量和工具函数
    EVENT_ORDER,
    EVENT_TICK,
    EVENT_TRADE,
    # 数据类和常量
    BarData,
    BaseEngine,
    CancelRequest,
    ContractData,
    Direction,
    EngineType,
    Event,
    EventEngine,
    Exchange,
    Interval,
    MainEngine,
    OrderData,
    OrderRequest,
    OrderType,
    SubscribeRequest,
    TickData,
    TradeData,
    # 工具函数
    round_to,
)
from apilot.core.database import DATABASE_CONFIG, BaseDatabase, use_database
from apilot.strategy import PATemplate
from apilot.utils.logger import get_logger
from apilot.utils.symbol import split_symbol

# 模块级初始化日志器
logger = get_logger("LiveTrading")


class PAEngine(BaseEngine):
    engine_type: EngineType = EngineType.LIVE
    setting_filename: str = "pa_strategy_setting.json"
    data_filename: str = "pa_strategy_data.json"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        super().__init__(main_engine, event_engine, "APILOT")

        self.strategy_setting = {}
        self.strategy_data = {}
        self.strategies = {}
        self.symbol_strategy_map = defaultdict(list)
        self.orderid_strategy_map = {}
        self.strategy_orderid_map = defaultdict(set)
        self.init_executor = ThreadPoolExecutor(max_workers=1)
        self.tradeids = set()
        self.database: BaseDatabase = use_database(
            DATABASE_CONFIG.get("name", ""), **DATABASE_CONFIG.get("params", {})
        )

    def init_engine(self) -> None:
        """
        初始化引擎
        """
        self.load_strategy_setting()
        self.load_strategy_data()
        self.register_event()
        logger.info("PA策略引擎初始化成功")

    def close(self) -> None:
        self.stop_all_strategies()

    def register_event(self) -> None:
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)

    def process_tick_event(self, event: Event) -> None:
        tick = event.data

        strategies = self.symbol_strategy_map[tick.symbol]
        if not strategies:
            return

        for strategy in strategies:
            if strategy.inited:
                self.call_strategy_func(strategy, strategy.on_tick, tick)

    def process_order_event(self, event: Event) -> None:
        order = event.data

        strategy: type | None = self.orderid_strategy_map.get(order.orderid, None)
        if not strategy:
            return

        # Remove orderid if order is no longer active.
        orderids: set = self.strategy_orderid_map[strategy.strategy_name]
        if order.orderid in orderids and not order.is_active():
            orderids.remove(order.orderid)

        # Call strategy on_order function
        self.call_strategy_func(strategy, strategy.on_order, order)

    def process_trade_event(self, event: Event) -> None:
        """
        处理成交事件
        """
        trade: TradeData = event.data

        # Avoid processing duplicate trade
        if trade.tradeid in self.tradeids:
            return
        self.tradeids.add(trade.tradeid)

        strategy: PATemplate | None = self.orderid_strategy_map.get(trade.orderid, None)
        if not strategy:
            return

        # Update strategy pos before calling on_trade method
        if trade.direction == Direction.LONG:
            strategy.pos += trade.volume
        else:
            strategy.pos -= trade.volume

        # Call strategy on_trade function
        self.call_strategy_func(strategy, strategy.on_trade, trade)

        # Sync strategy variables to data file
        self.sync_strategy_data(strategy)

    def send_server_order(
        self,
        strategy: PATemplate,
        contract: ContractData,
        direction: Direction,
        price: float,
        volume: float,
        type: OrderType,
    ) -> list:
        # Create request and send order.
        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            type=type,
            price=price,
            volume=volume,
            reference=f"APILOT_{strategy.strategy_name}",
        )

        # Convert with offset converter
        req_list: list = self.main_engine.convert_order_request(
            original_req, contract.gateway_name
        )

        # Send Orders
        orderids: list = []

        for req in req_list:
            orderid: str = self.main_engine.send_order(req, contract.gateway_name)

            # Check if sending order successful
            if not orderid:
                continue

            orderids.append(orderid)

            self.main_engine.update_order_request(req, orderid, contract.gateway_name)

            # Save relationship between orderid and strategy.
            self.orderid_strategy_map[orderid] = strategy
            self.strategy_orderid_map[strategy.strategy_name].add(orderid)

        return orderids

    def send_limit_order(
        self,
        strategy: PATemplate,
        contract: ContractData,
        direction: Direction,
        price: float,
        volume: float,
    ) -> list:
        return self.send_server_order(
            strategy, contract, direction, price, volume, OrderType.LIMIT
        )

    def send_order(
        self,
        strategy: PATemplate,
        direction: Direction,
        price: float,
        volume: float,
        stop: bool = False,
    ) -> list:
        contract: ContractData | None = self.main_engine.get_contract(strategy.symbol)
        if not contract:
            error_msg = (
                f"[{strategy.strategy_name}] 委托失败,找不到合约:{strategy.symbol}"
            )
            logger.error(f"{error_msg}")
            return ""

        # Round order price and volume to nearest incremental value
        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_volume)

        return self.send_limit_order(
            strategy, contract, direction, price, volume
        )

    def cancel_server_order(self, orderid: str, strategy=None) -> None:
        """
        Cancel existing order by orderid.
        """
        order: OrderData | None = self.main_engine.get_order(orderid)
        if not order:
            if strategy:
                error_msg = f"[{strategy.strategy_name}] 撤单失败,找不到委托{orderid}"
                logger.error(f"{error_msg}")
            else:
                error_msg = f"撤单失败,找不到委托{orderid}"
                logger.error(f"{error_msg}")
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_order(self, strategy: PATemplate, orderid: str) -> None:
        """
        取消策略委托
        """
        self.cancel_server_order(orderid, strategy)

    def cancel_all(self, strategy: PATemplate) -> None:
        orderids: set = self.strategy_orderid_map[strategy.strategy_name]
        if not orderids:
            return

        for orderid in copy(orderids):
            self.cancel_order(strategy, orderid)

    def get_engine_type(self) -> EngineType:
        return self.engine_type

    def get_pricetick(self, strategy: PATemplate) -> float:
        contract: ContractData | None = self.main_engine.get_contract(strategy.symbol)

        if contract:
            return contract.pricetick
        else:
            return None

    def get_size(self, strategy: PATemplate) -> int:
        contract: ContractData | None = self.main_engine.get_contract(strategy.symbol)

        if contract:
            return contract.size
        else:
            return None

    def load_bar(
        self,
        symbol: str,
        count: int,
        interval: Interval,
        callback: Callable[[BarData], None],
        use_database: bool,
    ) -> list:
        symbol_str, exchange_str = split_symbol(symbol)
        end: datetime = datetime.now()
        start: datetime = end - timedelta(days=count)
        bars: list = []

        # Try to query bars from database, if not found, load from database.
        bars: list = self.database.load_bar_data(
            symbol_str, exchange_str, interval, start, end
        )

        return bars

    def load_tick(
        self, symbol: str, count: int, callback: Callable[[TickData], None]
    ) -> list:
        symbol_str, exchange_str = split_symbol(symbol)
        end: datetime = datetime.now()
        start: datetime = end - timedelta(days=count)
        ticks: list = self.database.load_tick_data(symbol_str, exchange_str, start, end)

        return ticks

    def call_strategy_func(
        self, strategy: PATemplate, func: Callable, params: Any = None
    ) -> None:
        try:
            func(params) if params is not None else func()
        except Exception:
            strategy.trading = strategy.inited = False
            error_msg = (
                f"[{strategy.strategy_name}] 触发异常已停止\n{traceback.format_exc()}"
            )
            logger.critical(f"{error_msg}")

    def add_strategy(
        self, strategy_class: type, strategy_name: str, symbol: str, setting: dict
    ) -> None:
        if strategy_name in self.strategies:
            error_msg = f"创建策略失败,存在重名{strategy_name}"
            logger.error(f"{error_msg}")
            return

        if "." not in symbol:
            error_msg = "创建策略失败,本地代码缺失交易所后缀"
            logger.error(f"{error_msg}")
            return

        symbol_str, exchange_str = split_symbol(symbol)
        if exchange_str not in Exchange.__members__:
            error_msg = "创建策略失败,本地代码的交易所后缀不正确"
            logger.error(f"{error_msg}")
            return

        strategy: PATemplate = strategy_class(self, strategy_name, symbol, setting)
        self.strategies[strategy_name] = strategy

        # Add symbol to strategy map.
        strategies: list = self.symbol_strategy_map[symbol]
        strategies.append(strategy)

        # Update to setting file.
        self.update_strategy_setting(strategy_name, setting)

    def init_strategy(self, strategy_name: str) -> Future:
        return self.init_executor.submit(self._init_strategy, strategy_name)

    def _init_strategy(self, strategy_name: str) -> None:
        """
        Init strategies in queue.
        """
        strategy: PATemplate = self.strategies[strategy_name]

        if strategy.inited:
            error_msg = f"{strategy_name}已经完成初始化,禁止重复操作"
            logger.error(f"{error_msg}")
            return

        logger.info(f"{strategy_name}开始执行初始化")

        # Call on_init function of strategy
        self.call_strategy_func(strategy, strategy.on_init)

        # Restore strategy data(variables)
        data: dict | None = self.strategy_data.get(strategy_name, None)
        if data:
            for name in strategy.variables:
                value = data.get(name, None)
                if value is not None:
                    setattr(strategy, name, value)

        # Subscribe market data
        contract: ContractData | None = self.main_engine.get_contract(strategy.symbol)
        if contract:
            req: SubscribeRequest = SubscribeRequest(
                symbol=contract.symbol, exchange=contract.exchange
            )
            self.main_engine.subscribe(req, contract.gateway_name)
        else:
            error_msg = f"行情订阅失败,找不到合约{strategy.symbol}"
            logger.error(f"{error_msg}")

        # Put event to update init completed status.
        strategy.inited = True
        logger.info(f"{strategy_name}初始化完成")

    def start_strategy(self, strategy_name: str) -> None:
        strategy: PATemplate = self.strategies[strategy_name]
        if not strategy.inited:
            error_msg = f"策略{strategy_name}启动失败,请先初始化"
            logger.error(f"{error_msg}")
            return

        if strategy.trading:
            error_msg = f"{strategy_name}已经启动,请勿重复操作"
            logger.error(f"{error_msg}")
            return
        self.call_strategy_func(strategy, strategy.on_start)
        strategy.trading = True

    def stop_strategy(self, strategy_name: str) -> None:
        strategy: PATemplate = self.strategies[strategy_name]
        if not strategy.trading:
            return

        # Call on_stop function of the strategy
        self.call_strategy_func(strategy, strategy.on_stop)

        # Change trading status of strategy to False
        strategy.trading = False

        # Cancel all orders of the strategy
        self.cancel_all(strategy)

        # Sync strategy variables to data file
        self.sync_strategy_data(strategy)

    def edit_strategy(self, strategy_name: str, setting: dict) -> None:
        strategy: PATemplate = self.strategies[strategy_name]
        strategy.update_setting(setting)

        self.update_strategy_setting(strategy_name, setting)

    def remove_strategy(self, strategy_name: str) -> bool:
        strategy: PATemplate = self.strategies[strategy_name]
        if strategy.trading:
            error_msg = f"策略{strategy_name}移除失败,请先停止"
            logger.error(f"{error_msg}")
            return

        # Remove setting
        self.remove_strategy_setting(strategy_name)

        # Remove from symbol strategy map
        strategies: list = self.symbol_strategy_map[strategy.symbol]
        strategies.remove(strategy)

        # Remove from active orderid map
        if strategy_name in self.strategy_orderid_map:
            orderids: set = self.strategy_orderid_map.pop(strategy_name)

            # Remove orderid strategy map
            for orderid in orderids:
                if orderid in self.orderid_strategy_map:
                    self.orderid_strategy_map.pop(orderid)

        # Remove from strategies
        self.strategies.pop(strategy_name)

        logger.info(f"策略{strategy_name}移除成功")
        return True

    def sync_strategy_data(self, strategy: PATemplate) -> None:
        """
        Sync strategy data into json file.
        """
        data: dict = strategy.get_variables()
        data.pop("inited")  # Strategy status (inited, trading) should not be synced.
        data.pop("trading")

        self.strategy_data[strategy.strategy_name] = data

    def stop_all_strategies(self) -> None:
        """停止所有策略"""
        for strategy_name in self.strategies.keys():
            self.stop_strategy(strategy_name)
