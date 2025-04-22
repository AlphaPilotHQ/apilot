import logging
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from typing import Any, ClassVar

import ccxt

from apilot.core.constant import Direction, Interval, OrderType, Product, Status
from apilot.core.event import EventEngine
from apilot.core.models import (
    AccountData,
    BarData,
    CancelRequest,
    ContractData,
    HistoryRequest,
    OrderData,
    OrderRequest,
    SubscribeRequest,
)

from .gateway import BaseGateway

logger = logging.getLogger(__name__)


class BinanceGateway(BaseGateway):
    default_name = "BINANCE"
    default_setting: ClassVar[dict[str, Any]] = {
        "API Key": "",
        "Secret Key": "",
        "Proxy Host": "",
        "Proxy Port": 0,
    }

    def __init__(self, event_engine: EventEngine, gateway_name: str = "BINANCE"):
        super().__init__(event_engine, gateway_name)
        self.api = BinanceApi(self)

    def connect(self, setting: dict):
        self.api.connect(
            api_key=setting["API Key"],
            secret_key=setting["Secret Key"],
            proxy_host=setting["Proxy Host"],
            proxy_port=setting["Proxy Port"],
        )

    def subscribe(self, req: SubscribeRequest):
        self.api.subscribe(req.symbol)

    def send_order(self, req: OrderRequest) -> str:
        return self.api.send_order(req)

    def cancel_order(self, req: CancelRequest):
        self.api.cancel_order(req)

    def query_account(self):
        self.api.query_account()

    def query_history(self, req: HistoryRequest) -> list[BarData]:
        return self.api.query_history(req)

    def close(self):
        self.api.close()


class BinanceApi:
    INTERVAL_MAP: ClassVar[dict[Interval, str]] = {
        Interval.MINUTE: "1m",
        Interval.HOUR: "1h",
        Interval.DAILY: "1d",
    }

    ORDER_TYPE_MAP: ClassVar[dict[OrderType, str]] = {
        OrderType.LIMIT: "limit",
        OrderType.MARKET: "market",
    }

    STATUS_MAP: ClassVar[dict[str, Status]] = {
        "open": Status.NOTTRADED,
        "closed": Status.ALLTRADED,
        "canceled": Status.CANCELLED,
    }

    def __init__(self, gateway: BinanceGateway):
        self.gateway = gateway
        self.exchange = None
        self.order_map = {}
        self.polling_symbols = set()
        self.stop_event = Event()
        self.last_timestamp = {}

    def connect(self, api_key, secret_key, proxy_host, proxy_port):
        params = {"apiKey": api_key, "secret": secret_key}
        if proxy_host and proxy_port:
            proxy = f"http://{proxy_host}:{proxy_port}"
            params["proxies"] = {"http": proxy, "https": proxy}

        self.exchange = ccxt.binance(params)
        try:
            self.exchange.load_markets()
            self._init_contracts()
            self.query_account()

            Thread(target=self._poll_market_data, daemon=True).start()
            logger.info("Connected and polling started.")
        except Exception as e:
            logger.error(f"Connect failed: {e}")

    def _init_contracts(self):
        for symbol, data in self.exchange.markets.items():
            if not data["active"]:
                continue
            contract = ContractData(
                symbol=symbol,
                product=Product.SPOT,
                pricetick=10 ** -data["precision"]["price"],
                min_volume=data.get("limits", {}).get("amount", {}).get("min", 1),
                max_volume=data.get("limits", {}).get("amount", {}).get("max"),
                gateway_name=self.gateway.gateway_name,
            )
            self.gateway.on_contract(contract)

    def subscribe(self, symbol):
        self.polling_symbols.add(symbol)
        # initialize last timestamp aligned to current minute
        aligned = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self.last_timestamp[symbol] = int(aligned.timestamp() * 1000)

    def query_account(self):
        try:
            balance = self.exchange.fetch_balance()
            for currency, total in balance["total"].items():
                if total > 0:
                    account = AccountData(
                        accountid=currency,
                        balance=total,
                        frozen=total - balance["free"][currency],
                        gateway_name=self.gateway.gateway_name,
                    )
                    self.gateway.on_account(account)
        except Exception as e:
            logger.error(f"Query account failed: {e}")

    def send_order(self, req: OrderRequest):
        try:
            params = {
                "symbol": req.symbol,
                "type": self.ORDER_TYPE_MAP[req.type],
                "side": "buy" if req.direction == Direction.LONG else "sell",
                "amount": req.volume,
                "price": req.price if req.type == OrderType.LIMIT else None,
            }
            result = self.exchange.create_order(**params)
            orderid = result["id"]
            order = OrderData(
                symbol=req.symbol,
                orderid=orderid,
                type=req.type,
                direction=req.direction,
                price=req.price,
                volume=req.volume,
                traded=0,
                status=Status.SUBMITTING,
                gateway_name=self.gateway.gateway_name,
                datetime=datetime.utcnow(),
            )
            self.order_map[orderid] = order
            return orderid
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return ""

    def cancel_order(self, req: CancelRequest):
        try:
            self.exchange.cancel_order(req.orderid, req.symbol)
        except Exception as e:
            logger.error(f"Cancel order failed: {e}")

    def query_history(self, req: HistoryRequest):
        try:
            timeframe = self.INTERVAL_MAP[req.interval]
            since = int(req.start.timestamp() * 1000)
            klines = self.exchange.fetch_ohlcv(req.symbol, timeframe, since)
            bars = [
                BarData(
                    symbol=req.symbol,
                    interval=req.interval,
                    datetime=datetime.fromtimestamp(t / 1000),
                    open_price=o,
                    high_price=h,
                    low_price=low_price,
                    close_price=c,
                    volume=v,
                    gateway_name=self.gateway.gateway_name,
                )
                for t, o, h, low_price, c, v in klines
            ]
            return bars
        except Exception as e:
            logger.error(f"Query history failed: {e}")
            return []

    def _poll_market_data(self):
        # aligned incremental polling for 1-minute bars
        while not self.stop_event.is_set():
            # sleep until next minute boundary plus small buffer
            now = datetime.now(timezone.utc)
            next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            wait_secs = (next_min - now).total_seconds()
            # interruptible sleep with small buffer
            if self.stop_event.wait(wait_secs + 1):
                break
            for symbol in list(self.polling_symbols):
                last_ts = self.last_timestamp.get(symbol, 0)
                try:
                    timeframe = self.INTERVAL_MAP[Interval.MINUTE]
                    klines = self.exchange.fetch_ohlcv(symbol, timeframe, last_ts, 1000)
                    for t, o, h, low_price, c, v in klines:
                        if t <= last_ts:
                            continue
                        bar = BarData(
                            symbol=symbol,
                            interval=Interval.MINUTE,
                            datetime=datetime.fromtimestamp(t / 1000, timezone.utc),
                            open_price=o,
                            high_price=h,
                            low_price=low_price,
                            close_price=c,
                            volume=v,
                            gateway_name=self.gateway.gateway_name,
                        )
                        logger.info(f"BinanceGateway get Bar: {bar}")
                        self.gateway.on_quote(bar)
                        last_ts = t
                    self.last_timestamp[symbol] = last_ts
                except Exception as e:
                    logger.error(f"Polling error: {e}")

    def close(self):
        self.stop_event.set()
        logger.info("Disconnected.")
