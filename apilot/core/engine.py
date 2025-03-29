"""
核心引擎模块

包含交易平台主引擎和基础引擎类,负责管理事件、网关和应用程序.
"""

import os
from abc import ABC, abstractmethod
from typing import Any

from apilot.utils.logger import get_logger

from .event import (
    EventEngine,
)
from .gateway import BaseGateway
from .object import (
    BarData,
    CancelRequest,
    Exchange,
    HistoryRequest,
    OrderRequest,
    QuoteRequest,
    SubscribeRequest,
)
from .utility import TRADER_DIR


class MainEngine:
    """
    Acts as the core of the trading platform.
    """

    def __init__(self, event_engine: EventEngine = None) -> None:
        """"""
        if event_engine:
            self.event_engine: EventEngine = event_engine
        else:
            self.event_engine = EventEngine()
        self.event_engine.start()

        self.gateways: dict[str, BaseGateway] = {}
        self.engines: dict[str, BaseEngine] = {}
        self.exchanges: list[Exchange] = []

        os.chdir(TRADER_DIR)  # Change working directory
        self.init_engines()  # Initialize function engines

    def add_engine(self, engine_class: Any) -> "BaseEngine":
        """
        Add function engine.
        """
        engine: BaseEngine = engine_class(self, self.event_engine)
        self.engines[engine.engine_name] = engine
        return engine

    def add_gateway(
        self, gateway_class: type[BaseGateway], gateway_name: str = ""
    ) -> BaseGateway:
        """
        Add gateway.
        """
        # Use default name if gateway_name not passed
        if not gateway_name:
            gateway_name: str = gateway_class.default_name

        gateway: BaseGateway = gateway_class(self.event_engine, gateway_name)
        self.gateways[gateway_name] = gateway

        # Add gateway supported exchanges into engine
        for exchange in gateway.exchanges:
            if exchange not in self.exchanges:
                self.exchanges.append(exchange)

        return gateway

    def init_engines(self) -> None:
        """
        Init all engines.
        """
        # Import from engine package instead of individual modules
        from apilot.engine import OmsEngine

        self.add_engine(OmsEngine)

    def get_gateway(self, gateway_name: str) -> BaseGateway:
        """
        Return gateway object by name.
        """
        gateway: BaseGateway = self.gateways.get(gateway_name, None)
        if not gateway:
            get_logger().error(f"找不到底层接口:{gateway_name}")
        return gateway

    def get_engine(self, engine_name: str) -> "BaseEngine":
        """
        Return engine object by name.
        """
        engine: BaseEngine = self.engines.get(engine_name, None)
        if not engine:
            get_logger().error(f"找不到引擎:{engine_name}")
        return engine

    def get_default_setting(self, gateway_name: str) -> dict[str, Any] | None:
        """
        Get default setting dict of a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.get_default_setting()
        return None

    def get_all_gateway_names(self) -> list[str]:
        """
        Get all names of gateway added in main engine.
        """
        return list(self.gateways.keys())

    def get_all_exchanges(self) -> list[Exchange]:
        """
        Get all exchanges.
        """
        return self.exchanges

    def connect(self, setting: dict, gateway_name: str) -> None:
        """
        Start connection of a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.connect(setting)

    def subscribe(self, req: SubscribeRequest, gateway_name: str) -> None:
        """
        Subscribe tick data update of a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.subscribe(req)

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        """
        Send new order request to a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.send_order(req)
        else:
            return ""

    def cancel_order(self, req: CancelRequest, gateway_name: str) -> None:
        """
        Send cancel order request to a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.cancel_order(req)

    def send_quote(self, req: QuoteRequest, gateway_name: str) -> str:
        """
        Send new quote request to a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.send_quote(req)
        else:
            return ""

    def cancel_quote(self, req: CancelRequest, gateway_name: str) -> None:
        """
        Send cancel quote request to a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.cancel_quote(req)

    def query_history(
        self, req: HistoryRequest, gateway_name: str
    ) -> list[BarData] | None:
        """
        Query bar history data from a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.query_history(req)
        else:
            return None

    def close(self) -> None:
        """
        Make sure every gateway and engine is closed properly before
        programme exit.
        """
        # Stop event engine first to prevent new timer event.
        self.event_engine.stop()

        for engine in self.engines.values():
            engine.close()

        for gateway in self.gateways.values():
            gateway.close()


class BaseEngine(ABC):
    """
    Abstract class for implementing a function engine.
    """

    def __init__(
        self,
        main_engine: MainEngine,
        event_engine: EventEngine,
        engine_name: str,
    ) -> None:
        """"""
        self.main_engine: MainEngine = main_engine
        self.event_engine: EventEngine = event_engine
        self.engine_name: str = engine_name

    @abstractmethod
    def close(self):
        """
        Close the engine and release resources.
        """
        pass
