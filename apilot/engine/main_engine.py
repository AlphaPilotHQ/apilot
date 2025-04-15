"""
Main Engine Module

Implements the MainEngine class for managing events, gateways, and applications.
"""

import logging
from typing import Any

from apilot.core.event import EventEngine
from apilot.core.object import (
    BarData,
    CancelRequest,
    HistoryRequest,
    OrderRequest,
    SubscribeRequest,
)
from apilot.engine.base_engine import BaseEngine
from apilot.gateway.gateway import BaseGateway

logger = logging.getLogger(__name__)


class MainEngine:
    """
    Acts as the core of the trading platform.
    """

    def __init__(self, event_engine: EventEngine = None) -> None:
        if event_engine:
            self.event_engine: EventEngine = event_engine
        else:
            self.event_engine = EventEngine()
        self.event_engine.start()

        self.gateways: dict[str, BaseGateway] = {}
        self.engines: dict[str, BaseEngine] = {}

        self.init_engines()  # Initialize function engines

    def add_engine(self, engine_class: Any) -> BaseEngine:
        engine: BaseEngine = engine_class(self, self.event_engine)
        self.engines[engine.engine_name] = engine
        return engine

    def add_gateway(
        self, gateway_class: type[BaseGateway], gateway_name: str = ""
    ) -> BaseGateway:
        if not gateway_name:
            gateway_name: str = gateway_class.default_name

        gateway: BaseGateway = gateway_class(self.event_engine, gateway_name)
        self.gateways[gateway_name] = gateway
        return gateway

    def init_engines(self) -> None:
        from apilot.engine.oms_engine import OmsEngine

        self.add_engine(OmsEngine)

    def get_gateway(self, gateway_name: str) -> BaseGateway:
        gateway: BaseGateway = self.gateways.get(gateway_name, None)
        if not gateway:
            logger.error(f"Gateway not found: {gateway_name}")
        return gateway

    def get_engine(self, engine_name: str) -> BaseEngine:
        engine: BaseEngine = self.engines.get(engine_name, None)
        if not engine:
            logger.error(f"Engine not found: {engine_name}")
        return engine

    def get_default_setting(self, gateway_name: str) -> dict[str, Any] | None:
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.get_default_setting()
        return None

    def get_all_gateway_names(self) -> list[str]:
        return list(self.gateways.keys())

    def get_all_exchanges(self) -> list[str]:
        return []

    def connect(self, setting: dict, gateway_name: str) -> None:
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.connect(setting)

    def subscribe(self, req: SubscribeRequest, gateway_name: str) -> None:
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.subscribe(req)

    def send_order(self, req: OrderRequest, gateway_name: str) -> str:
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.send_order(req)
        else:
            return ""

    def cancel_order(self, req: CancelRequest, gateway_name: str) -> None:
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            gateway.cancel_order(req)

    def query_history(
        self, req: HistoryRequest, gateway_name: str
    ) -> list[BarData] | None:
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.query_history(req)
        else:
            return None

    def close(self) -> None:
        self.event_engine.stop()
        for engine in self.engines.values():
            engine.close()
        for gateway in self.gateways.values():
            gateway.close()
