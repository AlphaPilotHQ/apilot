"""
Live Trading Engine Module

Implements real-time operation and management of trading strategies, including signal processing, order execution, and risk control
"""

import copy
import logging
import traceback
from collections import defaultdict, OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from apilot.core import (
    EVENT_ORDER,
    EVENT_QUOTE,
    EVENT_TRADE,
    BarData,
    CancelRequest,
    ContractData,
    Direction,
    Event,
    EventEngine,
    Interval,
    OrderData,
    OrderRequest,
    OrderType,
    SubscribeRequest,
    TradeData,
)
from apilot.core.models import HistoryRequest
from apilot.strategy import PATemplate
from apilot.utils.utility import round_to

from .base_engine import BaseEngine
from .main_engine import MainEngine

logger = logging.getLogger("LiveEngine")


class LiveEngine(BaseEngine):

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        super().__init__(main_engine, event_engine, "LiveEngine")

        self.strategy_setting = {}
        self.strategies = {}
        self.symbol_strategy_map = defaultdict(set)
        self.orderid_strategy_map = {}
        self.strategy_orderid_map = defaultdict(set)

        # LRU cache for trade IDs to prevent processing duplicates
        # Limited to 10000 most recent trade IDs to control memory usage
        self.MAX_TRADE_CACHE_SIZE = 10000
        self.tradeids = OrderedDict()

        self.init_engine()

    def init_engine(self) -> None:
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_QUOTE, self.process_quote_event)
        logger.info("Engine initialized successfully: ORDER, TRADE, QUOTE")

    def close(self) -> None:
        self.stop_all_strategies()

        self.event_engine.unregister(EVENT_ORDER, self.process_order_event)
        self.event_engine.unregister(EVENT_TRADE, self.process_trade_event)
        self.event_engine.unregister(EVENT_QUOTE, self.process_quote_event)

        # Clear trade ID cache on shutdown
        self.tradeids.clear()
        logger.info("Live engine shutdown completed")

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

    def process_quote_event(self, event: Event) -> None:
        data = event.data

        bar: BarData = data
        symbol = bar.symbol
        logger.info(f"process_quote_event: get bar data {symbol} {bar.datetime} close: {bar.close_price}")

        strategies = self.symbol_strategy_map.get(symbol, set())
        if not strategies:
            logger.warning(f"process_quote_event: no strategy subscription {symbol}")
            return

        for strategy in strategies:
            strategy_name = getattr(strategy, "strategy_name")
            if strategy.inited and strategy.trading:

                if hasattr(strategy, 'bg'):
                    strategy.bg.update_bar(bar)
                else:
                    self.call_strategy_func(strategy, strategy.on_bar, bar)
            else:
                logger.warning(f"process_quote_event: 策略 {strategy_name} 未初始化或未启动，跳过")

    def process_trade_event(self, event: Event) -> None:
        trade: TradeData = event.data

        # Avoid processing duplicate trade
        if trade.tradeid in self.tradeids:
            # Move this trade ID to the end (most recently used)
            self.tradeids.move_to_end(trade.tradeid)
            return

        # Add new trade ID to OrderedDict
        self.tradeids[trade.tradeid] = None

        # If cache exceeds max size, remove oldest item (first item)
        if len(self.tradeids) > self.MAX_TRADE_CACHE_SIZE:
            self.tradeids.popitem(last=False)

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

    def send_limit_order(
        self,
        strategy: PATemplate,
        contract: ContractData,
        direction: Direction,
        price: float,
        volume: float,
    ) -> list[str]:
        # Create request and send order.
        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            direction=direction,
            type=OrderType.LIMIT,
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

    def send_order(
        self,
        strategy: PATemplate,
        direction: Direction,
        price: float,
        volume: float,
        stop: bool = False,
    ) -> list[str]:
        contract: ContractData | None = self.main_engine.get_contract(strategy.symbol)
        if not contract:
            error_msg = f"[{strategy.strategy_name}] Order failed, contract not found: {strategy.symbol}"
            logger.error(f"{error_msg}")
            return []

        # Round order price and volume to nearest incremental value
        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_amount)

        return self.send_limit_order(strategy, contract, direction, price, volume)

    def cancel_order(self, strategy: PATemplate, orderid: str) -> None:
        """
        Cancel strategy order by orderid.
        """
        # Fetch existing order
        order: OrderData | None = self.main_engine.get_order(orderid)
        if not order:
            prefix = f"[{strategy.strategy_name}] "
            logger.error(f"{prefix}Cancel order failed, order not found: {orderid}")
            return

        # Create and send cancel request
        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_all(self, strategy: PATemplate) -> None:
        orderids: set = self.strategy_orderid_map[strategy.strategy_name]
        if not orderids:
            return

        for orderid in orderids.copy():
            self.cancel_order(strategy, orderid)

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

    def load_bar(self, symbol: str, count: int, interval: Interval,) -> list[BarData]:
        """
        Load historical bars from the exchange for the given symbol.
        """
        end = datetime.now(timezone.utc)
        minutes_per_bar = self._get_interval_minutes(interval)
        start = end - timedelta(minutes=minutes_per_bar * (count + 1))
        req = HistoryRequest(symbol=symbol, interval=interval, start=start, end=end)

        try:
            bars = self.main_engine.query_history(req, gateway_name="Binance", count=count)
            return bars
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            return []

    def _get_interval_minutes(self, interval: Interval) -> int:
        """将K线周期转换为对应的分钟数"""
        # 获取周期值
        val = getattr(interval, "value", None)

        # 如果是整数，直接返回
        if isinstance(val, int):
            return val

        # 如果是字符串，通过映射转换
        if isinstance(val, str):
            interval_map = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "30m": 30,
                "1h": 60,
                "4h": 240,
                "1d": 1440,
                "d": 1440,
                "w": 10080,
            }
            return interval_map.get(val, 1)

        return 1

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
        symbol: str,
        setting: dict,
    ) -> None:
        if strategy_name in self.strategies:
            error_msg = f"Strategy {strategy_name} already exists, duplicate creation not allowed"
            logger.error(f"{error_msg}")
            return

        # Create strategy instance
        strategy: PATemplate = strategy_class(self, strategy_name, symbol, setting)
        self.strategies[strategy_name] = strategy

        # Add strategy to the symbol's mapping
        self.symbol_strategy_map[strategy.symbol].add(strategy)
        logger.info(f"Strategy {strategy_name} added")

    def init_strategy(self, strategy_name: str) -> None:
        # Check if strategy exists
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")

        strategy = self.strategies[strategy_name]
        if strategy.inited:
            raise RuntimeError(f"{strategy_name} already initialized, duplicate operation prohibited")

        # Call on_init function of strategy
        self.call_strategy_func(strategy, strategy.on_init)

        # sub market data for the single symbol
        contract = self.main_engine.get_contract(strategy.symbol)
        if not contract:
            raise RuntimeError(f"Market data subscription failed, contract {strategy.symbol} not found")

        req = SubscribeRequest(symbol=contract.symbol)
        self.main_engine.subscribe(req, contract.gateway_name)

        strategy.inited = True
        logger.info(f"{strategy_name} initialization completed")

    def start_strategy(self, strategy_name: str) -> None:

        strategy: PATemplate = self.strategies[strategy_name]
        if not strategy.inited:
            logger.error(f"Strategy {name} start failed, please initialize first")
            return

        if strategy.trading:
            logger.error(f"{name} already started, please do not repeat operation")
            return

        self.call_strategy_func(strategy, strategy.on_start)
        strategy.trading = True
        logger.info(f"{strategy_name} started")

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
