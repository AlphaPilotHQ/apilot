import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from apilot.core import (
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_QUOTE,
    EVENT_TRADE,
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    Event,
    EventEngine,
    HistoryRequest,
    OrderData,
    OrderRequest,
    PositionData,
    SubscribeRequest,
    TradeData,
)

logger = logging.getLogger("BaseGateway")


class BaseGateway(ABC):
    default_name: str = ""

    def __init__(self, event_engine: EventEngine, gateway_name: str = "") -> None:
        self.event_engine = event_engine
        self.gateway_name = gateway_name

    def on_event(self, type: str, data: Any = None) -> None:
        event: Event = Event(type, data)
        self.event_engine.put(event)

    def on_trade(self, trade: TradeData) -> None:
        self.on_event(EVENT_TRADE, trade)

    def on_order(self, order: OrderData) -> None:
        self.on_event(EVENT_ORDER, order)

    def on_position(self, position: PositionData) -> None:
        self.on_event(EVENT_POSITION, position)

    def on_account(self, account: AccountData) -> None:
        self.on_event(EVENT_ACCOUNT, account)

    def on_quote(self, data: Any) -> None:
        self.on_event(EVENT_QUOTE, data)

    def on_contract(self, contract: ContractData) -> None:
        self.on_event(EVENT_CONTRACT, contract)

    @abstractmethod
    def connect(self, setting: dict) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        pass

    @abstractmethod
    def query_account(self) -> None:
        pass

    @abstractmethod
    def query_history(self, req: HistoryRequest) -> list[BarData]:
        pass

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        pass

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        pass




