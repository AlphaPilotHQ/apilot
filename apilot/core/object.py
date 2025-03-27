"""
Basic data structure used for general trading function in the trading platform.
"""

from dataclasses import dataclass, field
from datetime import datetime
from logging import INFO
from typing import Optional

from .constant import (
    Direction,
    Exchange,
    Interval,
    Offset,
    OptionType,
    OrderType,
    Product,
    Status,
)

ACTIVE_STATUSES = set([Status.SUBMITTING, Status.NOTTRADED, Status.PARTTRADED])


@dataclass
class BaseData:
    """
    Any data object needs a gateway_name as source
    and should inherit base data.
    """

    gateway_name: str = ""
    extra: Optional[dict] = field(default=None, init=False)


@dataclass
class TickData(BaseData):
    """
    Tick data contains information about:
        * last trade in market
        * orderbook snapshot
        * intraday market statistics.
    """

    symbol: str = ""
    exchange: Optional[Exchange] = None
    datetime: Optional[datetime] = None

    name: str = ""
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    last_price: float = 0
    last_volume: float = 0
    limit_up: float = 0
    limit_down: float = 0

    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    pre_close: float = 0

    bid_price_1: float = 0
    bid_price_2: float = 0
    bid_price_3: float = 0
    bid_price_4: float = 0
    bid_price_5: float = 0

    ask_price_1: float = 0
    ask_price_2: float = 0
    ask_price_3: float = 0
    ask_price_4: float = 0
    ask_price_5: float = 0

    bid_volume_1: float = 0
    bid_volume_2: float = 0
    bid_volume_3: float = 0
    bid_volume_4: float = 0
    bid_volume_5: float = 0

    ask_volume_1: float = 0
    ask_volume_2: float = 0
    ask_volume_3: float = 0
    ask_volume_4: float = 0
    ask_volume_5: float = 0

    localtime: datetime = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}" if self.exchange else ""

    @staticmethod
    def from_dict(data: dict) -> "TickData":
        """
        Create tick data object from dictionary.
        """
        tick = TickData(
            symbol=data["symbol"],
            exchange=data.get("exchange"),
            datetime=data.get("datetime"),
            gateway_name=data.get("gateway_name", ""),
            name=data.get("name", ""),
            volume=data.get("volume", 0),
            turnover=data.get("turnover", 0),
            open_interest=data.get("open_interest", 0),
            last_price=data.get("last_price", 0),
            last_volume=data.get("last_volume", 0),
            limit_up=data.get("limit_up", 0),
            limit_down=data.get("limit_down", 0),
            open_price=data.get("open_price", 0),
            high_price=data.get("high_price", 0),
            low_price=data.get("low_price", 0),
            pre_close=data.get("pre_close", 0),
            bid_price_1=data.get("bid_price_1", 0),
            bid_price_2=data.get("bid_price_2", 0),
            bid_price_3=data.get("bid_price_3", 0),
            bid_price_4=data.get("bid_price_4", 0),
            bid_price_5=data.get("bid_price_5", 0),
            ask_price_1=data.get("ask_price_1", 0),
            ask_price_2=data.get("ask_price_2", 0),
            ask_price_3=data.get("ask_price_3", 0),
            ask_price_4=data.get("ask_price_4", 0),
            ask_price_5=data.get("ask_price_5", 0),
            bid_volume_1=data.get("bid_volume_1", 0),
            bid_volume_2=data.get("bid_volume_2", 0),
            bid_volume_3=data.get("bid_volume_3", 0),
            bid_volume_4=data.get("bid_volume_4", 0),
            bid_volume_5=data.get("bid_volume_5", 0),
            ask_volume_1=data.get("ask_volume_1", 0),
            ask_volume_2=data.get("ask_volume_2", 0),
            ask_volume_3=data.get("ask_volume_3", 0),
            ask_volume_4=data.get("ask_volume_4", 0),
            ask_volume_5=data.get("ask_volume_5", 0),
        )
        return tick


@dataclass
class BarData(BaseData):
    """
    Candlestick bar data of a certain trading period.
    """

    symbol: str = ""
    exchange: Exchange = None
    datetime: datetime = None

    interval: Interval = None
    volume: float = 0
    turnover: float = 0
    open_interest: float = 0
    open_price: float = 0
    high_price: float = 0
    low_price: float = 0
    close_price: float = 0

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    @staticmethod
    def from_dict(data: dict) -> "BarData":
        """
        Create bar data object from dictionary.
        """
        bar = BarData(
            symbol=data["symbol"],
            exchange=data["exchange"],
            datetime=data["datetime"],
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

    symbol: str = ""
    exchange: Exchange = None
    orderid: str = ""

    type: OrderType = OrderType.LIMIT
    direction: Direction = None
    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    traded: float = 0
    status: Status = Status.SUBMITTING
    datetime: datetime = None
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"

    def is_active(self) -> bool:
        """
        Check if the order is active.
        """
        return self.status in ACTIVE_STATUSES

    def create_cancel_request(self) -> "CancelRequest":
        """
        Create cancel request object from order.
        """
        req: CancelRequest = CancelRequest(
            orderid=self.orderid, symbol=self.symbol, exchange=self.exchange
        )
        return req


@dataclass
class TradeData(BaseData):
    """
    Trade data contains information of a fill of an order. One order
    can have several trade fills.
    """

    symbol: str = ""
    exchange: Exchange = None
    orderid: str = ""
    tradeid: str = ""
    direction: Direction = None

    offset: Offset = Offset.NONE
    price: float = 0
    volume: float = 0
    datetime: datetime = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_orderid: str = f"{self.gateway_name}.{self.orderid}"
        self.vt_tradeid: str = f"{self.gateway_name}.{self.tradeid}"


@dataclass
class PositionData(BaseData):
    """
    Position data is used for tracking each individual position holding.
    """

    symbol: str = ""
    exchange: Exchange = None
    direction: Direction = None

    volume: float = 0
    frozen: float = 0
    price: float = 0
    pnl: float = 0
    yd_volume: float = 0

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
        self.vt_positionid: str = f"{self.gateway_name}.{self.vt_symbol}.{self.direction.value}"


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
    """日志数据结构

    包含日志消息、级别、来源和时间戳等信息，用于事件驱动的日志系统
    """
    msg: str = ""  # 日志消息
    level: int = INFO  # 日志级别
    source: str = ""  # 日志来源
    timestamp: datetime = field(default_factory=datetime.now)  # 日志记录时间
    extra: dict = field(default_factory=dict)  # 额外信息字典，可用于扩展

    @property
    def level_name(self) -> str:
        """获取日志级别名称"""
        import logging
        return logging.getLevelName(self.level)


@dataclass
class ContractData(BaseData):
    """
    Contract data contains basic information about each contract traded.
    """

    symbol: str = ""
    exchange: Exchange = None
    name: str = ""
    product: Product = None
    size: float = 0
    pricetick: float = 0

    min_volume: float = 1           # minimum order volume
    max_volume: float = None        # maximum order volume
    stop_supported: bool = False    # whether server supports stop order
    net_position: bool = False      # whether gateway uses net position volume
    history_data: bool = False      # whether gateway provides bar history data

    option_strike: float = 0
    option_underlying: str = ""     # vt_symbol of underlying contract
    option_type: OptionType = None
    option_listed: datetime = None
    option_expiry: datetime = None
    option_portfolio: str = ""
    option_index: str = ""          # for identifying options with same strike price

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteData(BaseData):
    """
    Quote data contains information for tracking lastest status
    of a specific quote.
    """

    symbol: str = ""
    exchange: Exchange = None
    quoteid: str = ""

    bid_price: float = 0.0
    bid_volume: int = 0
    ask_price: float = 0.0
    ask_volume: int = 0
    bid_offset: Offset = Offset.NONE
    ask_offset: Offset = Offset.NONE
    status: Status = Status.SUBMITTING
    datetime: datetime = None
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"
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
        req: CancelRequest = CancelRequest(
            orderid=self.quoteid, symbol=self.symbol, exchange=self.exchange
        )
        return req


@dataclass
class SubscribeRequest:
    """
    Request sending to specific gateway for subscribing tick data update.
    """

    symbol: str = ""
    exchange: Exchange = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class OrderRequest:
    """
    Request sending to specific gateway for creating a new order.
    """

    symbol: str = ""
    exchange: Exchange = None
    direction: Direction = None
    type: OrderType = None
    volume: float = 0
    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_order_data(self, orderid: str, gateway_name: str) -> OrderData:
        """
        Create order data from request.
        """
        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=orderid,
            type=self.type,
            direction=self.direction,
            offset=self.offset,
            price=self.price,
            volume=self.volume,
            reference=self.reference,
        )
        return order


@dataclass
class CancelRequest:
    """
    Request sending to specific gateway for canceling an existing order.
    """

    orderid: str = ""
    symbol: str = ""
    exchange: Exchange = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class HistoryRequest:
    """
    Request sending to specific gateway for querying history data.
    """

    symbol: str = ""
    exchange: Exchange = None
    start: datetime = None
    end: datetime = None
    interval: Interval = None

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"


@dataclass
class QuoteRequest:
    """
    Request sending to specific gateway for creating a new quote.
    """

    symbol: str = ""
    exchange: Exchange = None
    bid_price: float = 0
    bid_volume: int = 0
    ask_price: float = 0
    ask_volume: int = 0
    bid_offset: Offset = Offset.NONE
    ask_offset: Offset = Offset.NONE
    reference: str = ""

    def __post_init__(self) -> None:
        """"""
        self.vt_symbol: str = f"{self.symbol}.{self.exchange.value}"

    def create_quote_data(self, quoteid: str, gateway_name: str) -> QuoteData:
        """
        Create quote data from request.
        """
        quote: QuoteData = QuoteData(
            symbol=self.symbol,
            exchange=self.exchange,
            quoteid=quoteid,
            bid_price=self.bid_price,
            bid_volume=self.bid_volume,
            ask_price=self.ask_price,
            ask_volume=self.ask_volume,
            bid_offset=self.bid_offset,
            ask_offset=self.ask_offset,
            reference=self.reference,
        )
        return quote
