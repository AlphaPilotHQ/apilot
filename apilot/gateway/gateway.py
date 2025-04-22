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
    default_setting: ClassVar[dict[str, Any]] = {}

    def __init__(self, event_engine: EventEngine, gateway_name: str = "") -> None:
        """
        Initialize a gateway instance.

        Args:
            event_engine: Event engine for pushing data updates
            gateway_name: Name of the gateway
        """
        self.event_engine = event_engine
        self.gateway_name = gateway_name

    def on_event(self, type: str, data: Any = None) -> None:
        """
        Push an event to event engine.

        Args:
            type: Event type string
            data: Event data object
        """
        event: Event = Event(type, data)
        self.event_engine.put(event)

    def on_trade(self, trade: TradeData) -> None:
        """
        Push trade event.

        Args:
            trade: Trade data object
        """
        self.on_event(EVENT_TRADE, trade)
        self.on_event(EVENT_TRADE + "_" + trade.symbol, trade)

    def on_order(self, order: OrderData) -> None:
        """
        Push order event.

        Args:
            order: Order data object
        """
        self.on_event(EVENT_ORDER, order)
        self.on_event(EVENT_ORDER + "_" + order.orderid_with_gateway, order)

    def on_position(self, position: PositionData) -> None:
        """
        Push position event.

        Args:
            position: Position data object
        """
        self.on_event(EVENT_POSITION, position)
        self.on_event(EVENT_POSITION + "_" + position.symbol, position)

    def on_account(self, account: AccountData) -> None:
        """
        Push account event.

        Args:
            account: Account data object
        """
        self.on_event(EVENT_ACCOUNT, account)
        self.on_event(EVENT_ACCOUNT + "_" + account.accountid, account)

    def on_quote(self, data: Any) -> None:
        """
        Push quote event. Can handle both QuoteData and BarData.

        Args:
            data: Quote data object or Bar data object
        """
        self.on_event(EVENT_QUOTE, data)
        self.on_event(EVENT_QUOTE + "_" + data.symbol, data)

    def on_contract(self, contract: ContractData) -> None:
        """
        Push contract event.

        Args:
            contract: Contract data object
        """
        self.on_event(EVENT_CONTRACT, contract)

    @abstractmethod
    def connect(self, setting: dict) -> None:
        """
        Connect to trading server.

        Implementation requirements:
        * Connect to server
        * Log connection status
        * Query account, position, orders, trades, contracts
        * Push data through on_* callbacks

        Args:
            setting: Connection settings
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close connection to trading server.
        """
        pass

    @abstractmethod
    def subscribe(self, req: SubscribeRequest) -> None:
        """
        Subscribe to market data.

        Args:
            req: Subscription request object
        """
        pass

    @abstractmethod
    def send_order(self, req: OrderRequest) -> str:
        """
        Send new order to server.

        Implementation requirements:
        * Create OrderData from request
        * Assign unique orderid
        * Send to server
        * Return orderid for reference

        Args:
            req: Order request object

        Returns:
            str: Unique order ID
        """
        pass

    @abstractmethod
    def cancel_order(self, req: CancelRequest) -> None:
        """
        Cancel existing order.
        Args:
            req: Cancel request object
        """
        pass

    @abstractmethod
    def query_account(self) -> None:
        """
        Query account balance from server.
        """
        pass

    @abstractmethod
    def query_history(self, req: HistoryRequest) -> list[BarData]:
        """
        Query bar history data from server.
        Args:
            req: History request object
        Returns:
            List[BarData]: List of bar data
        """
        pass

    def get_default_setting(self) -> dict[str, Any]:
        return self.default_setting
