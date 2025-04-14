"""
Live Trading Engine Module

Implements real-time operation and management of trading strategies, including signal processing, order execution, and risk control
"""

import copy
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from apilot.core import (
    # Constants and utility functions
    EVENT_ORDER,
    EVENT_TRADE,
    # Data classes and constants
    BarData,
    BaseEngine,
    CancelRequest,
    ContractData,
    Direction,
    EngineType,
    Event,
    EventEngine,
    Interval,
    MainEngine,
    OrderData,
    OrderRequest,
    OrderType,
    SubscribeRequest,
    TradeData,
    # Utility functions
    round_to,
)
from apilot.strategy import PATemplate
from apilot.utils.logger import get_logger

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

    def init_engine(self) -> None:
        self.register_event()
        logger.info("PA strategy engine initialized successfully")

    def close(self) -> None:
        self.stop_all_strategies()

    def register_event(self) -> None:
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)

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
            error_msg = f"[{strategy.strategy_name}] Order failed, contract not found: {strategy.symbol}"
            logger.error(f"{error_msg}")
            return ""

        # Round order price and volume to nearest incremental value
        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_volume)

        return self.send_limit_order(strategy, contract, direction, price, volume)

    def cancel_server_order(self, orderid: str, strategy=None) -> None:
        """
        Cancel existing order by orderid.
        """
        order: OrderData | None = self.main_engine.get_order(orderid)
        if not order:
            if strategy:
                error_msg = f"[{strategy.strategy_name}] Cancel order failed, order not found: {orderid}"
                logger.error(f"{error_msg}")
            else:
                error_msg = f"Cancel order failed, order not found: {orderid}"
                logger.error(f"{error_msg}")
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_order(self, strategy: PATemplate, orderid: str) -> None:
        """
        Cancel strategy order
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
        logger.info(f"From exchange get {symbol} history data")
        return []

    def call_strategy_func(
        self, strategy: PATemplate, func: Callable, params: Any = None
    ) -> None:
        try:
            func(params) if params is not None else func()
        except Exception:
            strategy.trading = strategy.inited = False
            error_msg = f"[{strategy.strategy_name}] Exception triggered, stopped\n{traceback.format_exc()}"
            logger.critical(f"{error_msg}")

    def add_strategy(
        self,
        strategy_class: type,
        strategy_name: str,
        symbols: str | list[str],
        setting: dict,
    ) -> None:
        if strategy_name in self.strategies:
            error_msg = f"Strategy {strategy_name} already exists, duplicate creation not allowed"
            logger.error(f"{error_msg}")
            return

        # Check symbols parameter - can be string or list of strings
        if isinstance(symbols, str):
            if not symbols:
                error_msg = "Symbol cannot be empty"
                logger.error(f"{error_msg}")
                return
        elif isinstance(symbols, list) and symbols:
            # All symbols should be non-empty
            for symbol in symbols:
                if not symbol:
                    error_msg = "Symbol cannot be empty"
                    logger.error(f"{error_msg}")
                    return
        else:
            error_msg = "Symbols must be a non-empty string or list"
            logger.error(f"{error_msg}")
            return

        # Create strategy instance
        strategy: PATemplate = strategy_class(self, strategy_name, symbols, setting)
        self.strategies[strategy_name] = strategy

        # Add strategy to each symbol's mapping
        for symbol in strategy.symbols:
            strategies: list = self.symbol_strategy_map[symbol]
            strategies.append(strategy)

    def init_strategy(self, strategy_name: str) -> Future:
        return self.init_executor.submit(self._init_strategy, strategy_name)

    def _init_strategy(self, strategy_name: str) -> None:
        """
        Init strategies in queue.
        """
        strategy: PATemplate = self.strategies[strategy_name]

        if strategy.inited:
            error_msg = (
                f"{strategy_name} already initialized, duplicate operation prohibited"
            )
            logger.error(f"{error_msg}")
            return

        logger.info(f"{strategy_name} starting initialization")

        # Call on_init function of strategy
        self.call_strategy_func(strategy, strategy.on_init)

        # Restore strategy data(variables)
        data: dict | None = self.strategy_data.get(strategy_name, None)
        if data:
            for name in strategy.variables:
                value = data.get(name, None)
                if value is not None:
                    setattr(strategy, name, value)

        # 订阅策略中所有交易对的行情数据
        subscription_failed = False
        for symbol in strategy.symbols:
            logger.info(f"From exchange get {symbol} history data")
            contract: ContractData | None = self.main_engine.get_contract(symbol)
            if contract:
                req: SubscribeRequest = SubscribeRequest(symbol=contract.symbol)
                self.main_engine.subscribe(req, contract.gateway_name)
            else:
                error_msg = (
                    f"Market data subscription failed, contract {symbol} not found"
                )
                logger.error(f"{error_msg}")
                subscription_failed = True

        # 即使某些交易对订阅失败，也标记策略初始化完成
        # 但日志中会记录失败的交易对
        strategy.inited = True

        if subscription_failed:
            logger.warning(
                f"{strategy_name} initialization completed with subscription failures"
            )
        else:
            logger.info(f"{strategy_name} initialization completed")

    def start_strategy(self, strategy_name: str) -> None:
        strategy: PATemplate = self.strategies[strategy_name]
        if not strategy.inited:
            error_msg = (
                f"Strategy {strategy_name} start failed, please initialize first"
            )
            logger.error(f"{error_msg}")
            return

        if strategy.trading:
            error_msg = (
                f"{strategy_name} already started, please do not repeat operation"
            )
            logger.error(f"{error_msg}")
            return
        self.call_strategy_func(strategy, strategy.on_start)
        strategy.trading = True

    def sync_strategy_data(self, strategy: PATemplate) -> None:
        """
        Sync strategy data into json file.
        """
        data: dict = strategy.get_variables()
        data.pop("inited")  # Strategy status (inited, trading) should not be synced.
        data.pop("trading")

        self.strategy_data[strategy.strategy_name] = data

    def stop_all_strategies(self) -> None:
        """Stop all strategies"""
        for strategy_name in list(self.strategies.keys()):
            strategy = self.strategies[strategy_name]
            if not strategy.trading:
                continue

            # Call on_stop function of the strategy
            self.call_strategy_func(strategy, strategy.on_stop)

            # Change trading status of strategy to False
            strategy.trading = False

            # Cancel all orders of the strategy
            self.cancel_all(strategy)

            # Sync strategy variables to data file
            self.sync_strategy_data(strategy)
