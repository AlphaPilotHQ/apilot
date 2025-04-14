"""
apilot Binance Gateway using CCXT
"""

from datetime import datetime
from typing import Any, ClassVar

import ccxt

from apilot.core.constant import (
    Direction,
    Exchange,
    Interval,
    OrderType,
    Product,
    Status,
)
from apilot.core.event import EventEngine
from apilot.core.gateway import BaseGateway
from apilot.core.object import (
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    HistoryRequest,
    OrderData,
    OrderRequest,
    QuoteRequest,
    SubscribeRequest,
)
from apilot.utils.logger import get_logger

# Binance exchange symbols for apilot
EXCHANGE_BINANCE = Exchange.BINANCE

# Maps of CCXT orderType, orderStatus to apilot constants
ORDERTYPE_BINANCE2VT = {"limit": OrderType.LIMIT, "market": OrderType.MARKET}
ORDERTYPE_VT2BINANCE = {v: k for k, v in ORDERTYPE_BINANCE2VT.items()}

STATUS_BINANCE2VT = {
    "open": Status.NOTTRADED,
    "closed": Status.ALLTRADED,
    "canceled": Status.CANCELLED,
    "expired": Status.CANCELLED,
    "rejected": Status.REJECTED,
}

INTERVAL_VT2BINANCE = {
    Interval.MINUTE: "1m",
    Interval.HOUR: "1h",
    Interval.DAILY: "1d",
    Interval.WEEKLY: "1w",
}

# Error codes mapping
ORDER_NOT_EXISTS_ERROR = "order not found"
TRADING_RULE_NOT_EXISTS_ERROR = "Order would trigger immediately."

LOCAL_SYS_ORDER_ID_MAP = {}
SYS_LOCAL_ORDER_ID_MAP = {}


class BinanceGateway(BaseGateway):
    default_name = "BINANCE"
    default_setting: ClassVar[dict[str, Any]] = {
        "API Key": "",
        "Secret Key": "",
        "Session Number": 3,
        "Proxy Host": "",
        "Proxy Port": 0,
    }
    exchanges: ClassVar[list[Exchange]] = [Exchange.BINANCE]

    def __init__(self, event_engine: EventEngine, gateway_name: str = "BINANCE"):
        """Constructor"""
        super().__init__(event_engine, gateway_name)

        self.logger = get_logger(f"Binance_{gateway_name}")

        # Track connection time for order IDs
        self.connect_time = 0

        # REST API client
        self.rest_api = BinanceRestApi(self)

    def connect(self, setting: dict) -> None:
        api_key = setting["API Key"]
        secret_key = setting["Secret Key"]
        session_number = setting["Session Number"]
        proxy_host = setting["Proxy Host"]
        proxy_port = setting["Proxy Port"]

        self.rest_api.connect(
            api_key,
            secret_key,
            session_number,
            proxy_host,
            proxy_port,
        )
        self.connect_time = int(datetime.now().timestamp())

    def subscribe(self, req: SubscribeRequest) -> None:
        self.rest_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        return self.rest_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        self.rest_api.cancel_order(req)

    def query_account(self) -> None:
        self.rest_api.query_account()

    def query_position(self) -> None:
        self.rest_api.query_position()

    def query_history(self, req: HistoryRequest) -> list[BarData]:
        return self.rest_api.query_history(req)

    def close(self) -> None:
        self.rest_api.stop()

    def send_quote(self, req: QuoteRequest) -> str:
        return ""

    def cancel_quote(self, req: CancelRequest) -> None:
        pass


class BinanceRestApi:
    """Binance REST API implementation"""

    def __init__(self, gateway: BinanceGateway):
        """Constructor"""
        self.gateway = gateway
        self.gateway_name = gateway.gateway_name
        # Create a dedicated logger instead of relying on gateway's logger
        self.logger = get_logger(f"BinanceRestApi_{self.gateway_name}")

        self.api_key = ""
        self.secret_key = ""
        self.proxy_host = ""
        self.proxy_port = 0

        self.exchange = None
        self.order_count = 0
        self.connect_time = 0

        self.orders = {}

        # For restoring data from small network interruption
        self.contract_symbols = set()

    def connect(
        self,
        api_key: str,
        secret_key: str,
        session_number: int,
        proxy_host: str,
        proxy_port: int,
    ) -> None:
        """Connect to Binance"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

        # Create CCXT exchange instance
        exchange_class = ccxt.binance
        options = {}

        # Prepare exchange kwargs
        self.exchange_kwargs = {
            "apiKey": api_key,
            "secret": secret_key,
            "options": options,
        }

        # Set proxy if provided
        if proxy_host and proxy_port:
            proxy_url = f"http://{proxy_host}:{proxy_port}"
            self.logger.info(f"Using proxy: {proxy_url}")
            # Set proxy in options for CCXT
            options["proxies"] = {
                "http": proxy_url,
                "https": proxy_url.replace("http:", "https:"),
            }
            # Also set for requests library
            self.exchange_kwargs["proxies"] = {
                "http": proxy_url,
                "https": proxy_url,
            }

        # Create the exchange instance
        try:
            self.logger.info("Creating Binance exchange instance")
            self.exchange = exchange_class(self.exchange_kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create Binance exchange instance: {e}")

        # Start connection and initialization
        self.init()

    def init(self) -> None:
        """Initialize connection and fetch data"""
        if not self.exchange:
            return

        # Fetch exchange info and initialize
        try:
            self.logger.info("Initializing Binance interface")
            self.exchange.load_markets()
            self.query_account()
            self.init_contracts()
        except Exception as e:
            self.logger.error(f"Failed to initialize Binance interface: {e}")

    def init_contracts(self) -> None:
        """Initialize contract list"""
        for symbol_data in self.exchange.markets.values():
            if not symbol_data["active"]:
                continue

            symbol = symbol_data["symbol"]
            self.contract_symbols.add(symbol)

            contract = ContractData(
                symbol=symbol,
                exchange=EXCHANGE_BINANCE,
                name=symbol,
                product=Product.SPOT,
                size=1,
                pricetick=float(symbol_data["precision"]["price"]),
                min_volume=float(
                    symbol_data.get("limits", {}).get("amount", {}).get("min", 0)
                ),
                gateway_name=self.gateway_name,
            )

            self.gateway.on_contract(contract)

        self.logger.info(
            f"Contract info query successful: {len(self.contract_symbols)} contracts"
        )

    def query_account(self) -> None:
        """Query account balance"""
        try:
            self.logger.info("Getting account balance information...")

            # Test API connection with a simple call
            try:
                self.exchange.fetch_time()
            except Exception as conn_err:
                self.logger.error(f"Failed to connect to Binance API: {conn_err}")
                self.logger.error(
                    "Please check network connection and API key permissions"
                )
                return

            # Get account balance
            try:
                data = self.exchange.fetch_balance()
                self.logger.info("Successfully obtained account balance")
            except Exception as bal_err:
                self.logger.error(f"Failed to get account balance: {bal_err}")
                return

            # Process each currency
            for currency, balance_data in data["total"].items():
                # Skip zero balances
                if balance_data == 0:
                    continue

                account = AccountData(
                    accountid=currency,
                    balance=data["total"][currency],
                    frozen=data["total"][currency] - data["free"].get(currency, 0),
                    gateway_name=self.gateway_name,
                )

                self.gateway.on_account(account)
                self.logger.info(
                    f"Account {currency}: Total={data['total'][currency]}, Available={data['free'].get(currency, 0)}"
                )
        except Exception as e:
            import traceback

            self.logger.error(f"Account balance query failed: {e}")
            self.logger.error(f"Detailed error: {traceback.format_exc()}")

    def query_position(self) -> None:
        """Query position data - for spot market, positions are derived from account balances"""
        pass

    def send_order(self, req: OrderRequest) -> str:
        """Send an order to Binance"""
        try:
            # Generate local orderid
            self.order_count += 1
            local_orderid = f"{self.connect_time}{self.order_count}"

            # Convert request to Binance parameters
            side = "buy" if req.direction == Direction.LONG else "sell"
            ordertype = ORDERTYPE_VT2BINANCE.get(req.type, "")

            # Send order via CCXT
            result = self.exchange.create_order(
                req.symbol,
                ordertype,
                side,
                req.volume,
                req.price if req.type == OrderType.LIMIT else None,
            )

            # Store sys orderid and local orderid map
            sys_orderid = result["id"]
            self.orders[local_orderid] = OrderData(
                symbol=req.symbol,
                exchange=req.exchange,
                orderid=local_orderid,
                type=req.type,
                direction=req.direction,
                price=req.price,
                volume=req.volume,
                traded=0,
                status=Status.SUBMITTING,
                gateway_name=self.gateway_name,
                datetime=datetime.now(),
            )

            LOCAL_SYS_ORDER_ID_MAP[local_orderid] = sys_orderid
            SYS_LOCAL_ORDER_ID_MAP[sys_orderid] = local_orderid

            return local_orderid
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            return ""

    def cancel_order(self, req: CancelRequest) -> None:
        """Cancel an order"""
        try:
            # Get sys orderid from local orderid
            sys_orderid = LOCAL_SYS_ORDER_ID_MAP.get(req.orderid, "")
            if not sys_orderid:
                self.logger.error(
                    f"Cancel order failed: Cannot find order {req.orderid}"
                )
                return

            # Cancel order via CCXT
            self.exchange.cancel_order(sys_orderid, req.symbol)
            self.logger.info(f"Cancel order request sent successfully: {req.orderid}")
        except Exception as e:
            self.logger.error(f"Cancel order failed: {e}")
            if ORDER_NOT_EXISTS_ERROR in str(e):
                # Order already finished or canceled
                order = self.orders.get(req.orderid, None)
                if order:
                    order.status = Status.CANCELLED
                    self.gateway.on_order(order)

    def subscribe(self, req: SubscribeRequest) -> None:
        """Subscribe to market data for a specific symbol"""
        # Currently using REST API polling - subscription is handled internally

    def query_history(self, req: HistoryRequest) -> list[BarData]:
        """Query K-line history data"""
        try:
            # Convert apilot interval to Binance interval
            interval = INTERVAL_VT2BINANCE.get(req.interval, "")
            if not interval:
                self.logger.error(f"Unsupported K-line interval: {req.interval}")
                return []

            # Convert start time to millisecond timestamp for CCXT
            start_time = int(req.start.timestamp() * 1000) if req.start else None

            # Fetch OHLCV data
            data = self.exchange.fetch_ohlcv(
                req.symbol, interval, since=start_time, limit=1000
            )

            # Convert to apilot BarData
            bars = []
            for row in data:
                ts, open_price, high_price, low_price, close_price, volume = row
                dt = datetime.fromtimestamp(ts / 1000)
                bar = BarData(
                    symbol=req.symbol,
                    exchange=EXCHANGE_BINANCE,
                    interval=req.interval,
                    datetime=dt,
                    open_price=float(open_price),
                    high_price=float(high_price),
                    low_price=float(low_price),
                    close_price=float(close_price),
                    volume=float(volume),
                    gateway_name=self.gateway_name,
                )
                bars.append(bar)

            return bars
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return []

    def stop(self) -> None:
        """Stop the API"""
        self.logger.info("Binance interface disconnected")
