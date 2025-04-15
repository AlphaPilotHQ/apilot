"""
This module defines the core data structures for a quantitative trading platform.

It includes:
- Standardized data classes for market data (BarData), orders (OrderData), trades (TradeData), positions (PositionData), accounts (AccountData), contracts (ContractData), quotes (QuoteData) and more.
- Request classes for actions such as order placement, cancellation, quote submission, and historical data queries.
- All structures are designed to ensure clear separation between request objects (for sending instructions) and data objects (for storing results and states).
- These classes provide a clean, extensible foundation for event-driven trading engines, supporting both crypto and stock markets.
"""

from dataclasses import dataclass, field
from datetime import datetime as dt  # Avoid module name conflict

from .constant import (
    Direction,
    Interval,
    OrderType,
    Product,
    Status,
)

ACTIVE_STATUSES = {Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED}


@dataclass
class BaseData:
    gateway_name: str = ""
    extra: dict | None = field(default=None, init=False)


@dataclass
class BarData(BaseData):
    symbol: str = ""
    datetime: dt = None
    interval: Interval | None = None
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0
    volume: float = 0

    @staticmethod
    def from_dict(data: dict) -> "BarData":
        bar = BarData(
            symbol=data["symbol"],
            datetime=data.get("datetime"),
            gateway_name=data.get("gateway_name", ""),
            interval=data.get("interval", None),
            open_price=data.get("open_price", 0),
            high_price=data.get("high_price", 0),
            low_price=data.get("low_price", 0),
            close_price=data.get("close_price", 0),
            volume=data.get("volume", 0),
        )
        return bar


@dataclass
class OrderData(BaseData):
    gateway_name: str = ""

    symbol: str = ""
    orderid: str = ""
    type: OrderType = OrderType.LIMIT
    direction: Direction | None = None
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: dt = None

    def __post_init__(self) -> None:
        """Initialize order ID with gateway prefix"""
        # Format orderid with gateway prefix
        if not self.orderid.startswith(f"{self.gateway_name}."):
            self.orderid: str = f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
        """
        Check if the order is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from order.
        """
        req: CancelRequest = CancelRequest(orderid=self.orderid, symbol=self.symbol)
        return req


@dataclass
class TradeData(BaseData):
    symbol: str = ""
    orderid: str = ""
    tradeid: str = ""
    direction: Direction | None = None

    price: float = 0
    volume: float = 0
    datetime: dt = None


@dataclass
class PositionData(BaseData):
    symbol: str = ""
    direction: Direction | None = None

    volume: float = 0
    frozen: float = 0
    price: float = 0
    pnl: float = 0

    def __post_init__(self) -> None:
        """Create position ID"""
        self.ap_positionid: str = (
            f"{self.gateway_name}.{self.symbol}.{self.direction.value}"
        )


@dataclass
class AccountData(BaseData):
    accountid: str = ""

    balance: float = 0
    frozen: float = 0

    def __post_init__(self) -> None:
        self.available: float = self.balance - self.frozen
        self.ap_accountid: str = f"{self.gateway_name}.{self.accountid}"


@dataclass
class ContractData(BaseData):
    symbol: str = ""
    product: Product | None = None
    pricetick: float = 0

    min_volume: float = 1  # minimum order volume
    max_volume: float | None = None  # maximum order volume


@dataclass
class QuoteData(BaseData):
    symbol: str = ""
    quoteid: str = ""
    bid_price: float = 0.0
    bid_volume: int = 0
    ask_price: float = 0.0
    ask_volume: int = 0
    status: Status = Status.SUBMITTING
    datetime: dt | None = None

    def __post_init__(self) -> None:
        """Format quote ID with gateway prefix"""
        if not self.quoteid.startswith(f"{self.gateway_name}."):
            self.ap_quoteid: str = f"{self.gateway_name}.{self.quoteid}"

    def is_active(self) -> bool:
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        req: CancelRequest = CancelRequest(orderid=self.quoteid, symbol=self.symbol)
        return req


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    symbol: str = ""


@dataclass
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    """

    symbol: str = ""
    direction: Direction | None = None
    type: OrderType | None = None
    volume: float = 0
    price: float = 0

    def create_order_data(self, orderid: str, gateway_name: str) -> "OrderData":
        """
        Create order data from request.
        """
        order: OrderData = OrderData(
            symbol=self.symbol,
            orderid=orderid,
            direction=self.direction,
            type=self.type,
            price=self.price,
            volume=self.volume,
            status=Status.SUBMITTING,
            gateway_name=gateway_name,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid: str = ""
    symbol: str = ""


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol: str = ""
    start: dt | None = None
    end: dt | None = None
    interval: Interval | None = None


@dataclass
class QuoteRequest:
    """
    Request sending to specific gateway for creating a new quote.
    """

    symbol: str = ""
    bid_price: float = 0
    bid_volume: int = 0
    ask_price: float = 0
    ask_volume: int = 0

    def create_quote_data(self, quoteid: str, gateway_name: str) -> "QuoteData":
        """
        Create quote data from request.
        """
        quote: QuoteData = QuoteData(
            symbol=self.symbol,
            quoteid=quoteid,
            bid_price=self.bid_price,
            bid_volume=self.bid_volume,
            ask_price=self.ask_price,
            ask_volume=self.ask_volume,
            gateway_name=gateway_name,
            status=Status.SUBMITTING,
        )
        return quote
