"""
Main Engine Module

Implements the MainEngine class for managing events, gateways, and applications.
"""

import logging
from typing import Any

from apilot.core.event import EventEngine
from apilot.core.models import (
    BarData,
    CancelRequest,
    HistoryRequest,
    OrderRequest,
    SubscribeRequest,
)
from apilot.engine.base_engine import BaseEngine
from apilot.gateway.gateway import BaseGateway

logger = logging.getLogger("MainEngine")

class MainEngine:
    """
    Acts as the core of the trading platform.
    """
    def __init__(self) -> None:
        self.event_engine = EventEngine()
        self.event_engine.start()

        self.gateways: dict[str, BaseGateway] = {}
        self.engines: dict[str, BaseEngine] = {}
        logger.info("MainEngine inited")


    def add_engine(self, engine_class: type[BaseEngine]) -> BaseEngine:
        """Register a new function engine. Raise if name exists."""
        engine = engine_class(self, self.event_engine)
        name = engine.engine_name

        if name in self.engines:
            logger.warning(f"Engine '{name}' already exists. Registration skipped.")
            return self.engines[name]

        self.engines[name] = engine

        logger.info(f"Engine '{name}' added.")
        
        return engine

    def add_gateway(
        self, gateway_class: type[BaseGateway], name: str | None = None
    ) -> BaseGateway:
        """Register a new gateway. Raise if name exists."""
        gateway_name = name or gateway_class.default_name

        if gateway_name in self.gateways:
            logger.warning(f"Gateway '{gateway_name}' already exists.")
            return self.gateways[gateway_name]

        gateway = gateway_class(self.event_engine, gateway_name)
        self.gateways[gateway_name] = gateway

        logger.info(f"Gateway '{gateway_name}' added.")

        return gateway

    def get_gateway(self, gateway_name: str) -> BaseGateway | None:
        gateway: BaseGateway | None = self.gateways.get(gateway_name, None)
        if not gateway:
            logger.error(f"Gateway not found: {gateway_name}")
        return gateway

    def get_default_setting(self, gateway_name: str) -> dict[str, Any] | None:
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            return gateway.get_default_setting()
        return None

    def connect(self, setting: dict, gateway_name: str) -> None:
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            gateway.connect(setting)
        else:
            logger.error(f"Connect failed: Gateway '{gateway_name}' not found. Setting: {setting}")

    def subscribe(self, req: SubscribeRequest, gateway_name: str) -> None:
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            gateway.subscribe(req)
        else:
            logger.error(f"Subscribe failed: Gateway '{gateway_name}' not found. Request: {req}")

    def send_order(self, req: OrderRequest, gateway_name: str) -> str | None:
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            return gateway.send_order(req)
        else:
            logger.error(f"Send order failed: Gateway '{gateway_name}' not found. Request: {req}")
            return None

    def cancel_order(self, req: CancelRequest, gateway_name: str) -> None:
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if gateway:
            gateway.cancel_order(req)
        else:
            logger.error(f"Cancel order failed: Gateway '{gateway_name}' not found. Request: {req}")

    def query_history(self, req: HistoryRequest, gateway_name: str, count: int) -> list[BarData]:
        gateway: BaseGateway | None = self.get_gateway(gateway_name)
        if not gateway:
            logger.error(f"Query history failed: Gateway '{gateway_name}' not found.")
            return []
        return gateway.query_history(req, count)

    def close(self) -> None:
        self.event_engine.close()
        for engine in self.engines.values():
            engine.close()
        for gateway in self.gateways.values():
            gateway.close()
