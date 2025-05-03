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
from apilot.engine.base_engine import ENGINE_REGISTRY, BaseEngine
from apilot.gateway.gateway import BaseGateway

logger = logging.getLogger("MainEngine")


class MainEngine:
    """
    Acts as the core of the trading platform.
    """

    def __init__(self) -> None:
        print(">>> [MainEngine] __init__ entered.") # 确认 __init__ 开始
        try:
            print(">>> [MainEngine] Attempting to create EventEngine...") # 确认尝试创建
            self.event_engine = EventEngine()
            print(">>> [MainEngine] EventEngine created successfully.") # 确认创建成功

            print(">>> [MainEngine] Attempting to call event_engine.start()...") # 确认尝试启动
            self.event_engine.start() # 这里会调用我们之前加了 print 的 start 方法
            print(">>> [MainEngine] event_engine.start() called successfully.") # 确认启动调用完成

            print(">>> [MainEngine] Initializing gateways and engines dictionaries...") # 确认后续步骤开始
            self.gateways: dict[str, BaseGateway] = {}
            self.engines: dict[str, BaseEngine] = {}
            print(">>> [MainEngine] Dictionaries initialized.") # 确认后续步骤完成

            print(">>> [MainEngine] Calling self.init_engines()...") # 确认再后续步骤开始
            self.init_engines()
            print(">>> [MainEngine] self.init_engines() finished.") # 确认再后续步骤完成

        except Exception as e:
            # 如果 __init__ 过程中任何地方出错，打印异常信息
            print(f">>> [MainEngine] !!! EXCEPTION during __init__: {e}")
            import traceback
            traceback.print_exc() # 打印完整的错误堆栈
            raise # 重新抛出异常，以便程序按预期失败

        print(">>> [MainEngine] __init__ finished successfully.") # 确认 __init__ 成功结束

    def add_engine(self, engine_class: type[BaseEngine]) -> BaseEngine:
        """Register a new function engine. Raise if name exists."""
        engine = engine_class(self, self.event_engine)
        name = engine.engine_name
        if name in self.engines:
            logger.warning(f"Engine '{name}' already exists. Registration skipped.")
            return self.engines[name]
        self.engines[name] = engine
        return engine

    def add_gateway(
        self, gateway_class: type[BaseGateway], gateway_name: str | None = None
    ) -> BaseGateway:
        """Register a new gateway. Raise if name exists."""
        name = gateway_name or gateway_class.default_name
        if name in self.gateways:
            logger.warning(f"Gateway '{name}' already exists. Registration skipped.")
            return self.gateways[name]
        gateway = gateway_class(self.event_engine, name)
        self.gateways[name] = gateway
        return gateway

    def init_engines(self) -> None:
        for engine_cls in ENGINE_REGISTRY:
            self.add_engine(engine_cls)

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
        return gateway.query_history(req, count)


    def close(self) -> None:
        self.event_engine.stop()
        for engine in self.engines.values():
            engine.close()
        for gateway in self.gateways.values():
            gateway.close()
