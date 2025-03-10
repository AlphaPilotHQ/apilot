"""
VeighNa Binance Gateway using CCXT
"""

import asyncio
import time
import hmac
import hashlib
import json
import base64
from copy import copy
from datetime import datetime, timedelta
from threading import Lock
from urllib.parse import urlencode
from typing import Any, Dict, List, Set, Optional, Callable

import ccxt
from ccxt.base.errors import NetworkError, ExchangeError, OrderNotFound, InsufficientFunds

from vnpy.event import Event, EventEngine
from vnpy.trader.constant import (
    Direction,
    Exchange,
    OrderType,
    Product,
    Status,
    Interval
)
from vnpy.trader.gateway import BaseGateway
from vnpy.trader.object import (
    TickData,
    OrderData,
    TradeData,
    PositionData,
    AccountData,
    ContractData,
    BarData,
    OrderRequest,
    CancelRequest,
    SubscribeRequest,
    HistoryRequest
)
from vnpy.trader.utility import round_to

# Binance exchange symbols for VeighNa
EXCHANGE_BINANCE = Exchange.BINANCE

# Maps of CCXT orderType, orderStatus to VeighNa constants
ORDERTYPE_BINANCE2VT = {
    "limit": OrderType.LIMIT,
    "market": OrderType.MARKET
}
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
    """
    VeighNa gateway for Binance connection using CCXT.
    """
    default_name = "BINANCE"
    default_setting = {
        "API Key": "",
        "Secret Key": "",
        "Session Number": 3,
        "Proxy Host": "",
        "Proxy Port": 0,
        "Testnet": False,
    }
    exchanges = [Exchange.BINANCE]

    def __init__(self, event_engine: EventEngine, gateway_name: str = "BINANCE"):
        """Constructor"""
        super().__init__(event_engine, gateway_name)

        self.trade_ws_api = None
        self.market_ws_api = None
        self.rest_api = BinanceRestApi(self)

        self.orders = {}
        self.order_count = 0
        self.connect_time = 0

    def connect(self, setting: dict) -> None:
        """Connect to exchange"""
        api_key = setting["API Key"]
        secret_key = setting["Secret Key"]
        session_number = setting["Session Number"]
        proxy_host = setting["Proxy Host"]
        proxy_port = setting["Proxy Port"]
        testnet = setting["Testnet"]

        self.rest_api.connect(
            api_key,
            secret_key,
            session_number,
            proxy_host,
            proxy_port,
            testnet
        )
        self.connect_time = int(datetime.now().timestamp())

    def subscribe(self, req: SubscribeRequest) -> None:
        """Subscribe to market data"""
        self.rest_api.subscribe(req)

    def send_order(self, req: OrderRequest) -> str:
        """Send order"""
        return self.rest_api.send_order(req)

    def cancel_order(self, req: CancelRequest) -> None:
        """Cancel order"""
        self.rest_api.cancel_order(req)

    def query_account(self) -> None:
        """Query account balance"""
        self.rest_api.query_account()

    def query_position(self) -> None:
        """Query positions"""
        self.rest_api.query_position()

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """Query history data"""
        return self.rest_api.query_history(req)

    def close(self) -> None:
        """Close connection"""
        self.rest_api.stop()


class BinanceRestApi:
    """Binance REST API implementation"""

    def __init__(self, gateway: BinanceGateway):
        """Constructor"""
        self.gateway = gateway
        self.gateway_name = gateway.gateway_name

        self.trade_ws_connected = False
        self.market_ws_connected = False

        self.api_key = ""
        self.secret_key = ""
        self.proxy_host = ""
        self.proxy_port = 0
        self.testnet = False

        self.exchange = None
        self.order_count = 0
        self.connect_time = 0

        self.positions = {}
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
        testnet: bool
    ) -> None:
        """Connect to Binance"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port

        # Create CCXT exchange instance
        exchange_class = getattr(ccxt, "binance")
        options = {}
        
        # Set proxy
        if proxy_host and proxy_port:
            self.gateway.write_log(f"使用代理: {proxy_host}:{proxy_port}")
            # CCXT的代理设置
            options["proxies"] = {
                "http": f"http://{proxy_host}:{proxy_port}",
                "https": f"https://{proxy_host}:{proxy_port}",
            }
            # 直接设置请求库的代理
            self.exchange_kwargs = {
                "apiKey": api_key,
                "secret": secret_key,
                "options": options,
                "proxies": {
                    "http": f"http://{proxy_host}:{proxy_port}",
                    "https": f"http://{proxy_host}:{proxy_port}"
                }
            }
        else:
            self.exchange_kwargs = {
                "apiKey": api_key,
                "secret": secret_key,
                "options": options
            }

        # Use testnet if specified
        urls = {}
        if testnet:
            urls["api"] = "https://testnet.binance.vision/api"
            
        # Create the exchange instance
        if urls:
            self.exchange_kwargs["urls"] = urls
            
        try:
            self.gateway.write_log(f"创建Binance交易所实例, 测试网: {testnet}")
            self.exchange = exchange_class(self.exchange_kwargs)
        except Exception as e:
            self.gateway.write_log(f"创建Binance交易所实例失败: {e}")

        # Start connection and initialization
        self.init()

    def init(self) -> None:
        """Initialize connection and fetch data"""
        if not self.exchange:
            return

        # Fetch exchange info, symbols, trading rules
        try:
            self.gateway.write_log("开始初始化Binance接口")
            self.exchange.load_markets()
            
            # Query account and positions
            self.query_account()
            self.query_position()
            
            # Initialize contract info
            self.init_contracts()
            
            self.gateway.write_log("Binance接口初始化成功")
        except Exception as e:
            self.gateway.write_log(f"Binance接口初始化失败: {e}")
        
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
                min_volume=float(symbol_data.get("limits", {}).get("amount", {}).get("min", 0)),
                gateway_name=self.gateway_name
            )
            
            self.gateway.on_contract(contract)
            
        self.gateway.write_log(f"合约信息查询成功: {len(self.contract_symbols)}个")

    def query_account(self) -> None:
        """Query account balance"""
        try:
            self.gateway.write_log("正在获取账户余额信息...")
            
            # 测试代理连接
            try:
                # 先用一个简单的API调用检查连接
                time_res = self.exchange.fetch_time()
                self.gateway.write_log(f"成功连接到Binance API, 服务器时间: {time_res}")
            except Exception as conn_err:
                self.gateway.write_log(f"连接到Binance API失败: {conn_err}")
                self.gateway.write_log("请检查代理设置是否正确, 以及代理是否已开启")
                return
                
            # 获取账户余额
            try:
                data = self.exchange.fetch_balance()
                self.gateway.write_log("成功获取账户余额")
            except Exception as bal_err:
                self.gateway.write_log(f"获取账户余额失败: {bal_err}")
                self.gateway.write_log("请检查API密钥权限是否正确设置")
                import traceback
                self.gateway.write_log(f"详细错误: {traceback.format_exc()}")
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
                    gateway_name=self.gateway_name
                )
                
                self.gateway.on_account(account)
                self.gateway.write_log(f"账户 {currency}: 总额={data['total'][currency]}, 可用={data['free'].get(currency, 0)}")
        except Exception as e:
            self.gateway.write_log(f"账户资金查询失败: {e}")
            import traceback
            self.gateway.write_log(f"详细错误堆栈: {traceback.format_exc()}")

    def query_position(self) -> None:
        """Query position data"""
        # For spot market, we use account balances
        self.query_account()

    def send_order(self, req: OrderRequest) -> str:
        """Send an order to Binance"""
        try:
            # Generate local orderid
            self.order_count += 1
            local_orderid = f"{self.connect_time}{self.order_count}"
            
            # Convert request to Binance params
            side = "buy" if req.direction == Direction.LONG else "sell"
            ordertype = ORDERTYPE_VT2BINANCE.get(req.type, "")
            
            params = {
                "symbol": req.symbol,
                "side": side,
                "type": ordertype,
            }
            
            # Add price and quantity according to order type
            if req.type == OrderType.LIMIT:
                params["price"] = str(req.price)
                params["quantity"] = str(req.volume)
                params["timeInForce"] = "GTC"
            else:
                params["quantity"] = str(req.volume)
            
            # Send order via CCXT
            result = self.exchange.create_order(
                req.symbol,
                ordertype,
                side,
                req.volume,
                req.price if req.type == OrderType.LIMIT else None
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
                datetime=datetime.now()
            )
            
            LOCAL_SYS_ORDER_ID_MAP[local_orderid] = sys_orderid
            SYS_LOCAL_ORDER_ID_MAP[sys_orderid] = local_orderid
            
            return local_orderid
        except Exception as e:
            self.gateway.write_log(f"委托下单失败: {e}")
            return ""

    def cancel_order(self, req: CancelRequest) -> None:
        """Cancel an order"""
        try:
            # Get sys orderid from local orderid
            sys_orderid = LOCAL_SYS_ORDER_ID_MAP.get(req.orderid, "")
            if not sys_orderid:
                self.gateway.write_log(f"撤单失败: 找不到对应的系统委托号")
                return
            
            # Send cancel request
            result = self.exchange.cancel_order(sys_orderid, req.symbol)
            
            # Process cancel result
            if result.get("status") == "canceled":
                order = self.orders.get(req.orderid)
                if order:
                    order.status = Status.CANCELLED
                    self.gateway.on_order(copy(order))
        except OrderNotFound:
            # If order already completed or not found, just finish
            order = self.orders.get(req.orderid)
            if order:
                order.status = Status.CANCELLED
                self.gateway.on_order(copy(order))
        except Exception as e:
            self.gateway.write_log(f"撤单失败: {e}")

    def subscribe(self, req: SubscribeRequest) -> None:
        """Subscribe to market data for a specific symbol"""
        try:
            # Query latest tick data
            result = self.exchange.fetch_ticker(req.symbol)
            
            # Convert data to TickData
            tick = TickData(
                symbol=req.symbol,
                exchange=req.exchange,
                datetime=datetime.fromtimestamp(result["timestamp"] / 1000),
                name=req.symbol,
                volume=float(result["quoteVolume"]),
                open_price=float(result["open"]),
                high_price=float(result["high"]),
                low_price=float(result["low"]),
                last_price=float(result["last"]),
                gateway_name=self.gateway_name
            )
            
            # If bid/ask prices are available
            if "bid" in result and "ask" in result:
                tick.bid_price_1 = float(result["bid"])
                tick.ask_price_1 = float(result["ask"])
            
            self.gateway.on_tick(tick)
            
            # Here you would typically set up a websocket subscription for real-time updates
            # but that would require more complex websocket handling which is beyond the 
            # scope of this simple example
            
        except Exception as e:
            self.gateway.write_log(f"订阅行情失败: {e}")

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """Query K-line history data"""
        history = []

        try:
            # Convert VeighNa interval to Binance interval
            interval = INTERVAL_VT2BINANCE.get(req.interval)
            if not interval:
                self.gateway.write_log("不支持的时间间隔")
                return []
                
            # Calculate start and end time
            end = int(req.end.timestamp() * 1000)
            start = int(req.start.timestamp() * 1000)
            
            # Limit parameter
            limit = 1000
            
            # Query CCXT for OHLCV data
            data = self.exchange.fetch_ohlcv(
                req.symbol,
                timeframe=interval,
                since=start,
                limit=limit
            )
            
            if not data:
                self.gateway.write_log("获取历史数据为空")
                return []
                
            # Convert data to BarData list
            for entry in data:
                ts, open_price, high_price, low_price, close_price, volume = entry
                
                bar = BarData(
                    symbol=req.symbol,
                    exchange=req.exchange,
                    datetime=datetime.fromtimestamp(ts / 1000),
                    interval=req.interval,
                    volume=volume,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    gateway_name=self.gateway_name
                )
                history.append(bar)
                
            return history
                
        except Exception as e:
            self.gateway.write_log(f"获取历史数据失败: {e}")
            return []

    def stop(self) -> None:
        """Stop the API"""
        # Clean up resources
        self.exchange = None
