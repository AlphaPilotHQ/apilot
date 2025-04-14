"""
Basic data structure used for general trading function in the trading platform.
"""

import datetime
from dataclasses import dataclass, field
from datetime import datetime as dt  # Avoid module name conflict
from logging import INFO

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
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name: str = ""
    extra: dict | None = field(default=None, init=False)


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    datetime: dt = None
    interval: Interval | None = None
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0

    @staticmethod
    def from_dict(data: dict) -> "BarData":
        """
        Create bar data object from dictionary.
        """
        bar = BarData(
            symbol=data["symbol"],
            datetime=data.get("datetime"),
            gateway_name=data.get("gateway_name", ""),
            interval=data.get("interval", None),
            volume=data.get("volume", 0),
            turnover=data.get("turnover", 0),
            open_interest=data.get("open_interest", 0),
            open_price=data.get("open_price", 0),
            high_price=data.get("high_price", 0),
            low_price=data.get("low_price", 0),
            close_price=data.get("close_price", 0),
        )
        return bar


@dataclass
class OrderData(BaseData):
    """
    Order data contains information for tracking lastest status
    of a specific order.
    """

    gateway_name: str = ""

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    orderid: str = ""
    type: OrderType = OrderType.LIMIT
    direction: Direction | None = None
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: dt = None
    reference: str = ""

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
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    orderid: str = ""
    tradeid: str = ""
    direction: Direction | None = None

    price: float = 0
    volume: float = 0
    datetime: dt = None


@dataclass
class PositionData(BaseData):
    """
    Position data is used for tracking each individual position holding.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    direction: Direction | None = None

    volume: float = 0
    frozen: float = 0
    price: float = 0
    pnl: float = 0
    yd_volume: float = 0

    def __post_init__(self) -> None:
        """Create position ID"""
        self.vt_positionid: str = (
            f"{self.gateway_name}.{self.symbol}.{self.direction.value}"
        )


@dataclass
class AccountData(BaseData):
    """
    Account data contains information about balance, frozen and
    available.
    """

    accountid: str = ""

    balance: float = 0
    frozen: float = 0

    def __post_init__(self) -> None:
        """"""
        self.available: float = self.balance - self.frozen
        self.vt_accountid: str = f"{self.gateway_name}.{self.accountid}"


@dataclass
class LogData(BaseData):
    """
    Log data structure for event-driven logging system
    """

    msg: str = ""  # Log message
    level: int = INFO  # Log level
    source: str = ""  # Log source
    timestamp: dt = field(default_factory=datetime.datetime.now)  # Timestamp
    extra: dict = field(default_factory=dict)  # Extra information

    @property
    def level_name(self) -> str:
        """Get log level name"""
        import logging

        return logging.getLevelName(self.level)


@dataclass
class ContractData(BaseData):
    """
    Contract data contains basic information about each contract traded.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    name: str = ""
    product: Product | None = None
    size: float = 0
    pricetick: float = 0

    min_volume: float = 1  # minimum order volume
    max_volume: float | None = None  # maximum order volume
    stop_supported: bool = False  # whether server supports stop order
    net_position: bool = False  # whether gateway uses net position volume
    history_data: bool = False  # whether gateway provides bar history data

    option_strike: float = 0
    option_underlying: str = ""  # symbol of underlying contract
    option_expiry: dt | None = None
    option_portfolio: str = ""
    option_index: str = ""  # for identifying options with same strike price


@dataclass
class QuoteData(BaseData):
    """
    Quote data contains information for tracking lastest status
    of a specific quote.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    quoteid: str = ""

    bid_price: float = 0.0
    bid_volume: int = 0
    ask_price: float = 0.0
    ask_volume: int = 0
    status: Status = Status.SUBMITTING
    datetime: dt | None = None
    reference: str = ""

    def __post_init__(self) -> None:
        """Format quote ID with gateway prefix"""
        if not self.quoteid.startswith(f"{self.gateway_name}."):
            self.vt_quoteid: str = f"{self.gateway_name}.{self.quoteid}"

    def is_active(self) -> bool:
        """
        Check if the quote is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from quote.
        """
        req: CancelRequest = CancelRequest(orderid=self.quoteid, symbol=self.symbol)
        return req


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")


@dataclass
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    direction: Direction | None = None
    type: OrderType | None = None
    volume: float = 0
    price: float = 0
    reference: str = ""

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
            reference=self.reference,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid: str = ""
    symbol: str = ""  # Trading symbol (e.g. "BTC")


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    start: dt | None = None
    end: dt | None = None
    interval: Interval | None = None


@dataclass
class QuoteRequest:
    """
    Request sending to specific gateway for creating a new quote.
    """

    symbol: str = ""  # Trading symbol (e.g. "BTC")
    bid_price: float = 0
    bid_volume: int = 0
    ask_price: float = 0
    ask_volume: int = 0
    reference: str = ""

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
            reference=self.reference,
            gateway_name=gateway_name,
            status=Status.SUBMITTING,
        )
        return quote
