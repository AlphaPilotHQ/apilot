import logging
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from typing import Any, ClassVar
import time

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
    default_name = "Binance"

    def __init__(self, event_engine: EventEngine, gateway_name: str = "Binance"):
        super().__init__(event_engine, gateway_name)
        self.api = BinanceRestApi(self)

    def connect(self, setting: dict):
        self.api.connect(
            api_key=setting["API Key"],
            secret_key=setting["Secret Key"],
            proxy_host=setting.get("Proxy Host", ""),
            proxy_port=setting.get("Proxy Port", 0),
            symbol=setting.get("Symbol", None),
        )

        # Wait until API signals readiness or timeout
        timeout, interval = 10.0, 0.2  # seconds
        elapsed = 0.0
        while not self.api.ready and elapsed < timeout:
            sleep(interval)
            elapsed += interval

        if not self.api.ready:
            logger.error("Binance gateway init timeout after %.1f seconds", timeout)

    def close(self):
        self.api.close()

    def subscribe(self, req: SubscribeRequest):
        self.api.subscribe(req.symbol)

    def query_account(self):
        self.api.query_account()

    def query_history(self, req: HistoryRequest, count: int) -> list[BarData]:
        return self.api.query_history(req, count)

    def send_order(self, req: OrderRequest) -> str:
        return self.api.send_order(req)

    def cancel_order(self, req: CancelRequest):
        self.api.cancel_order(req)


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
            # Load all symbols and initialize symbols
            self.exchange.load_markets()
            self._init_symbols(symbol)

            Thread(target=self._poll_market_data, daemon=True).start()

            # Mark as ready so callers can proceed
            self.ready = True
            logger.info("Binance gateway connected.")
        except Exception as e:
            logger.error(f"Connect failed: {e}")

    def close(self):
        self.stop_event.set()
        logger.info("Disconnected.")

    def _init_symbols(self, symbol=None):
        """
        Initialize contract data for a specific symbol.
        """
        market = self.exchange.markets[symbol]

        if not market["active"]:
            logger.warning(f"Symbol {symbol} is not active")

        limits = market.get("limits", {}).get("amount", {})
        min_amount = limits.get("min", 1)
        max_amount = limits.get("max")
        price_precision = market["precision"]["price"]
        price_tick = 10 ** -price_precision

        contract = ContractData(
            symbol=symbol,
            product=Product.SPOT,
            pricetick=price_tick,
            min_amount=min_amount,
            max_amount=max_amount,
            gateway_name=self.gateway.gateway_name,
        )

        self.gateway.on_contract(contract)
        logger.info(f"Initialized contract for {symbol}")

    # def _poll_market_data(self):
    #     # aligned incremental polling for 1-minute bars
    #     while not self.stop_event.is_set():
    #         # sleep until next minute boundary plus small buffer
    #         now = datetime.now(timezone.utc)
    #         next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    #         wait_secs = (next_min - now).total_seconds() + 2  # 2-s safety buffer
    #         if self.stop_event.wait(wait_secs):
    #             break

    #         for symbol in list(self.polling_symbols):
    #             last_ts = self.last_timestamp.get(symbol, 0)
    #             try:
    #                 timeframe = self.INTERVAL_MAP[Interval.MINUTE]
    #                 klines = self.exchange.fetch_ohlcv(symbol, timeframe, last_ts, 1000)
    #                 current_ts = int(datetime.now(timezone.utc).replace(second=0,
    #                                                   microsecond=0).timestamp() * 1000)

    #                 for t, o, h, l, c, v in klines:
    #                     if t <= last_ts:
    #                         continue

    #                     if t >= current_ts:
    #                         continue
    #                     bar = BarData(
    #                         symbol=symbol,
    #                         interval=Interval.MINUTE,
    #                         datetime=datetime.fromtimestamp(t / 1000, timezone.utc),
    #                         open_price=o,
    #                         high_price=h,
    #                         low_price=l,
    #                         close_price=c,
    #                         volume=v,
    #                         gateway_name=self.gateway.gateway_name,
    #                     )
    #                     logger.info(f"BinanceGateway get Bar: {bar}")
    #                     self.gateway.on_quote(bar)
    #                     last_ts = t
    #                 self.last_timestamp[symbol] = last_ts
    #             except Exception as e:
    #                 logger.error(f"Polling error: {e}")

    def _poll_market_data(self) -> None:
        timeframe = self.INTERVAL_MAP[Interval.MINUTE]

        def safe_fetch(*args, retries: int = 5, **kwargs):
            """REST with exponential-backoff retry."""
            for i in range(retries):
                try:
                    return self.exchange.fetch_ohlcv(*args, **kwargs)
                except Exception as e:
                    if i == retries - 1:
                        raise
                    logger.warning("fetch_ohlcv retry %s/%s – %s", i + 1, retries, e)
                    time.sleep(2 ** i)  # 1s → 2s → 4s

        while not self.stop_event.is_set():
            now = datetime.now(timezone.utc)
            next_min = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
            wait_secs = (next_min - now).total_seconds() + 1.5
            if self.stop_event.wait(wait_secs):
                break

            for symbol in list(self.polling_symbols):
                last_ts = self.last_timestamp.get(symbol, 0)

                since = max(0, last_ts - 60_000)
                try:
                    klines = safe_fetch(symbol, timeframe, since, limit=3)
                except Exception as e:
                    logger.error("Polling error (%s): %s", symbol, e)
                    continue

                period_close_ts = int(datetime.now(timezone.utc)
                                    .replace(second=0, microsecond=0).timestamp() * 1000)

                for t, o, h, l, c, v in klines:
                    if t <= last_ts:          # ① 已处理或重叠 bar
                        continue
                    if t >= period_close_ts:  # ② 当期 bar 未收盘
                        continue

                    expected = last_ts + 60_000 if last_ts else t
                    if t > expected:
                        self._backfill_missing(symbol, expected, t - 60_000, timeframe)

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
                    logger.info("BinanceGateway get Bar: %s", bar)
                    self.gateway.on_quote(bar)
                    last_ts = t

                self.last_timestamp[symbol] = last_ts


    def _backfill_missing(self, symbol: str, start_ts: int, end_ts: int, timeframe: str) -> None:
        """Fetch and forward any missing bars between start_ts and end_ts (inclusive)."""
        bars_needed = (end_ts - start_ts) // 60_000 + 1
        klines = self.exchange.fetch_ohlcv(symbol, timeframe, start_ts, limit=bars_needed)

        for t, o, h, l, c, v in klines:
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
            logger.info("Back-filled bar: %s", bar)
            self.gateway.on_quote(bar)

    def subscribe(self, symbol):
        self.polling_symbols.add(symbol)
        aligned_close = (datetime.now(timezone.utc)
                        .replace(second=0, microsecond=0)
                        - timedelta(minutes=1))
        self.last_timestamp[symbol] = int(aligned_close.timestamp() * 1000)

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



