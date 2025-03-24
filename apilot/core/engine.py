"""
核心引擎模块

包含交易平台主引擎和基础引擎类，负责管理事件、网关和应用程序。
"""

import logging
import os
import smtplib
import traceback
from abc import ABC
from datetime import datetime
from email.message import EmailMessage
from logging import Logger
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Type

from .app import BaseApp
from .event import (
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_LOG,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_QUOTE,
    EVENT_TICK,
    EVENT_TRADE,
    Event,
    EventEngine,
)
from .gateway import BaseGateway
from .object import (
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    Exchange,
    HistoryRequest,
    LogData,
    OrderData,
    OrderRequest,
    PositionData,
    QuoteData,
    QuoteRequest,
    SubscribeRequest,
    TickData,
    TradeData,
)
from .setting import SETTINGS
from .utility import TRADER_DIR, get_folder_path


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

        self.gateways: Dict[str, BaseGateway] = {}
        self.engines: Dict[str, BaseEngine] = {}
        self.apps: Dict[str, BaseApp] = {}
        self.exchanges: List[Exchange] = []

        os.chdir(TRADER_DIR)    # Change working directory
        self.init_engines()     # Initialize function engines

    def add_engine(self, engine_class: Any) -> "BaseEngine":
        """
        Add function engine.
        """
        engine: BaseEngine = engine_class(self, self.event_engine)
        self.engines[engine.engine_name] = engine
        return engine

    def add_gateway(self, gateway_class: Type[BaseGateway], gateway_name: str = "") -> BaseGateway:
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

    def add_app(self, app_class: Type[BaseApp]) -> "BaseEngine":
        """
        Add app.
        """
        app: BaseApp = app_class()
        self.apps[app.app_name] = app

        engine: BaseEngine = self.add_engine(app.engine_class)
        return engine

    def init_engines(self) -> None:
        """
        Init all engines.
        """
        # Import from engine package instead of individual modules
        from apilot.engine import LogEngine, OmsEngine, EmailEngine

        self.add_engine(LogEngine)
        self.add_engine(OmsEngine)
        self.add_engine(EmailEngine)

    def _write_log(self, msg: str, source: str = "", gateway_name: str = "", level: int = logging.INFO) -> None:
        """
        内部方法：创建并发送日志事件
        """
        log = LogData(gateway_name=gateway_name, msg=msg, source=source, level=level)
        event = Event(EVENT_LOG, log)
        self.event_engine.put(event)

    def write_log(self, msg: str, source: str = "", level: int = logging.INFO) -> None:
        """记录日志消息（已弃用，请使用log_debug/log_info等方法）"""
        import warnings
        warnings.warn("write_log方法已弃用，请使用log_debug/log_info等专用方法", DeprecationWarning, stacklevel=2)
        self._write_log(msg, source, level=level)

    def log_debug(self, msg: str, source: str = "") -> None:
        """记录调试级别日志"""
        self._write_log(msg, source, level=logging.DEBUG)

    def log_info(self, msg: str, source: str = "") -> None:
        """记录信息级别日志"""
        self._write_log(msg, source, level=logging.INFO)

    def log_warning(self, msg: str, source: str = "") -> None:
        """记录警告级别日志"""
        self._write_log(msg, source, level=logging.WARNING)

    def log_error(self, msg: str, source: str = "") -> None:
        """记录错误级别日志"""
        self._write_log(msg, source, level=logging.ERROR)

    def log_critical(self, msg: str, source: str = "") -> None:
        """记录严重错误级别日志"""
        self._write_log(msg, source, level=logging.CRITICAL)

    def get_gateway(self, gateway_name: str) -> BaseGateway:
        """
        Return gateway object by name.
        """
        gateway: BaseGateway = self.gateways.get(gateway_name, None)
        if not gateway:
            self.log_error(f"找不到底层接口：{gateway_name}")
        return gateway

    def get_engine(self, engine_name: str) -> "BaseEngine":
        """
        Return engine object by name.
        """
        engine: BaseEngine = self.engines.get(engine_name, None)
        if not engine:
            self.log_error(f"找不到引擎：{engine_name}")
        return engine

    def get_default_setting(self, gateway_name: str) -> Optional[Dict[str, Any]]:
        """
        Get default setting dict of a specific gateway.
        """
        gateway: BaseGateway = self.get_gateway(gateway_name)
        if gateway:
            return gateway.get_default_setting()
        return None

    def get_all_gateway_names(self) -> List[str]:
        """
        Get all names of gateway added in main engine.
        """
        return list(self.gateways.keys())

    def get_all_apps(self) -> List[BaseApp]:
        """
        Get all app objects.
        """
        return list(self.apps.values())

    def get_all_exchanges(self) -> List[Exchange]:
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

    def query_history(self, req: HistoryRequest, gateway_name: str) -> Optional[List[BarData]]:
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
        Make sure every gateway and app is closed properly before
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

    def close(self):
        """"""
        pass
