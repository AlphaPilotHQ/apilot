import logging
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from typing import Any, ClassVar
from time import sleep

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

    def __init__(self, event_engine: EventEngine, gateway_name: str = "BINANCE"):
        super().__init__(event_engine, gateway_name)
        self.api = BinanceRestApi(self)

    def connect(self, setting: dict):
        self.api.connect(
            api_key=setting["API Key"],
            secret_key=setting["Secret Key"],
            proxy_host=setting.get("Proxy Host", ""),
            proxy_port=setting.get("Proxy Port", 0),
            symbol=setting.get("Symbol", None),  # Pass symbol to only initialize specific contract
        )

        # Wait until API signals readiness or timeout
        timeout, interval = 10.0, 0.1  # seconds
        elapsed = 0.0
        while not self.api.ready and elapsed < timeout:
            sleep(interval)
            elapsed += interval

        if not self.api.ready:
            logger.error("Binance gateway init timeout after %.1f seconds", timeout)

    def subscribe(self, req: SubscribeRequest):
        self.api.subscribe(req.symbol)

    def send_order(self, req: OrderRequest) -> str:
        return self.api.send_order(req)

    def cancel_order(self, req: CancelRequest):
        self.api.cancel_order(req)

    def query_account(self):
        self.api.query_account()

    def query_history(self, req: HistoryRequest, count: int) -> list[BarData]:
        return self.api.query_history(req, count)

    def close(self):
        self.api.close()


class BinanceRestApi:
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
        self.ready = False

    def connect(self, api_key, secret_key, proxy_host, proxy_port, symbol=None):
        params = {"apiKey": api_key, "secret": secret_key}
        if proxy_host and proxy_port:
            proxy = f"http://{proxy_host}:{proxy_port}"
            params["proxies"] = {"http": proxy, "https": proxy}

        self.exchange = ccxt.binance(params)
        try:
            self.exchange.load_markets()
            self._init_contracts(symbol)
            # self.query_account()

            # Start polling thread then mark API ready
            Thread(target=self._poll_market_data, daemon=True).start()

            # Mark as ready so callers can proceed
            self.ready = True
        except Exception as e:
            logger.error(f"Connect failed: {e}")

    def _init_contracts(self, symbol=None):
        """
        Initialize contract data for a specific symbol.
        A valid symbol must be provided, otherwise an error will be logged.
        """
        if not symbol:
            logger.error("Symbol must be provided to initialize contracts")
            return

        if symbol in self.exchange.markets:
            data = self.exchange.markets[symbol]
            if data["active"]:
                contract = ContractData(
                    symbol=symbol,
                    product=Product.SPOT,
                    pricetick=10 ** -data["precision"]["price"],
                    min_amount=data.get("limits", {}).get("amount", {}).get("min", 1),
                    max_amount=data.get("limits", {}).get("amount", {}).get("max"),
                    gateway_name=self.gateway.gateway_name,
                )
                self.gateway.on_contract(contract)
                logger.info(f"Initialized contract for {symbol}")
            else:
                logger.warning(f"Symbol {symbol} is not active")
        else:
            logger.error(f"Symbol {symbol} not found in Binance markets")

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
            logger.info(f"Query account failed: {e}")

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

    def query_history(self, req: HistoryRequest, count:int):
        timeframe = self.INTERVAL_MAP[req.interval]
        since = int(req.start.timestamp() * 1000)
        klines = self.exchange.fetch_ohlcv(req.symbol, timeframe, since, limit=count+1)
        bars = []
        for t, o, h, l, c, v in klines:
            bar_time = datetime.fromtimestamp(t / 1000, timezone.utc)
            if bar_time + timedelta(seconds=60) <= datetime.now(timezone.utc):
                bars.append(BarData(
                    symbol=req.symbol,
                    interval=req.interval,
                    datetime=bar_time,
                    open_price=o,
                    high_price=h,
                    low_price=l,
                    close_price=c,
                    volume=v,
                    gateway_name=self.gateway.gateway_name,
                ))
        return bars

    def _poll_market_data(self):
        # aligned incremental polling for 1-minute bars
        while not self.stop_event.is_set():
            # sleep until next minute boundary plus small buffer
            now = datetime.now(timezone.utc)
            next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            wait_secs = (next_min - now).total_seconds() + 2  # 2-s safety buffer
            if self.stop_event.wait(wait_secs):
                break
            for symbol in list(self.polling_symbols):
                last_ts = self.last_timestamp.get(symbol, 0)
                try:
                    timeframe = self.INTERVAL_MAP[Interval.MINUTE]
                    klines = self.exchange.fetch_ohlcv(symbol, timeframe, last_ts, 1000)
                    current_min_ts = int(datetime.now(timezone.utc).replace(second=0,
                                                      microsecond=0).timestamp() * 1000)

                    for t, o, h, l, c, v in klines:
                        if t < last_ts:
                            continue

                        if t >= current_min_ts:
                            continue
                        bar = BarData(
                            symbol=symbol,
                            interval=Interval.MINUTE,
                            datetime=datetime.fromtimestamp(t / 1000, timezone.utc),
                            open_price=o,
                            high_price=h,
                            low_price=l,
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
