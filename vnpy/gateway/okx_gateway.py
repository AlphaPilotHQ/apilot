"""
VeighNa OKX Gateway
"""

import time
import json
import hmac
import base64
import hashlib
import datetime
from datetime import timedelta
from copy import copy
from urllib.parse import urlencode
from typing import Any, Dict, List, Set, Tuple, Optional

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

# OKX常量定义
EXCHANGE_OKX = Exchange.OKX
TESTNET_REST_HOST = "https://www.okx.com"  # 模拟盘REST API地址
REST_HOST = "https://www.okx.com"  # 实盘REST API地址

# API路径
REST_PATH = "/api/v5"

# 请求头
CONTENT_TYPE = "application/json"

# 委托状态映射
STATUS_OKX2VT = {
    # 待填充
}

# 委托类型映射
ORDERTYPE_VT2OKX = {
    # 待填充
}
ORDERTYPE_OKX2VT = {v: k for k, v in ORDERTYPE_VT2OKX.items()}

# 买卖方向映射
DIRECTION_VT2OKX = {
    # 待填充
}
DIRECTION_OKX2VT = {v: k for k, v in DIRECTION_VT2OKX.items()}

# K线周期映射
INTERVAL_VT2OKX = {
    # 待填充
}


class OkxGateway(BaseGateway):
    """
    VeighNa用于对接OKX交易所的网关
    """
    default_name = "OKX"

    default_setting = {
        "API Key": "",
        "Secret Key": "",
        "Passphrase": "",
        "会话数": 3,
        "代理地址": "",
        "代理端口": 0,
        "测试模式": False
    }

    exchanges = [Exchange.OKX]

    def __init__(self, event_engine: EventEngine, gateway_name: str = "OKX"):
        """构造函数"""
        super().__init__(event_engine, gateway_name)

        # REST API对象
        self.rest_api = OkxRestApi(self)

        # 存储功能相关的变量
        self.orders = {}
        self.positions = {}

    def connect(self, setting: dict) -> None:
        """连接交易接口"""
        # 获取API配置
        key = setting["API Key"]
        secret = setting["Secret Key"]
        passphrase = setting["Passphrase"]
        session_number = setting["会话数"]
        proxy_host = setting["代理地址"]
        proxy_port = setting["代理端口"]
        testnet = setting["测试模式"]

        # 初始化REST API
        self.rest_api.connect(
            key,
            secret,
            passphrase,
            session_number,
            proxy_host,
            proxy_port,
            testnet
        )

    def subscribe(self, req: SubscribeRequest) -> None:
        """订阅行情"""
        # 待实现
        pass

    def send_order(self, req: OrderRequest) -> str:
        """委托下单"""
        # 待实现
        pass

    def cancel_order(self, req: CancelRequest) -> None:
        """委托撤单"""
        # 待实现
        pass

    def query_account(self) -> None:
        """查询资金"""
        # 待实现
        pass

    def query_position(self) -> None:
        """查询持仓"""
        # 待实现
        pass

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """查询历史数据"""
        # 待实现
        pass

    def close(self) -> None:
        """关闭连接"""
        self.rest_api.stop()


class OkxRestApi:
    """OKX REST API"""

    def __init__(self, gateway: OkxGateway):
        """构造函数"""
        self.gateway = gateway
        self.gateway_name = gateway.gateway_name

        # API相关
        self.key = ""
        self.secret = ""
        self.passphrase = ""
        self.proxy_host = ""
        self.proxy_port = 0
        self.testnet = False

        # HTTP会话相关
        self.session = None
        self.rest_host = ""
        self.rate_limit = False
        self.rate_limit_sleep = 0

        # 请求计数器
        self.request_id = 0

    def connect(
        self,
        key: str,
        secret: str,
        passphrase: str,
        session_number: int,
        proxy_host: str,
        proxy_port: int,
        testnet: bool
    ) -> None:
        """连接REST服务器"""
        # 待实现
        pass

    def sign(self, request: dict) -> dict:
        """生成OKX签名"""
        # 待实现
        pass

    def send_request(self, method, path, params=None, data=None) -> dict:
        """发送HTTP请求"""
        # 待实现
        pass

    def query_account(self) -> None:
        """查询资金"""
        # 待实现
        pass

    def query_position(self) -> None:
        """查询持仓"""
        # 待实现
        pass

    def query_order(self) -> None:
        """查询未成交委托"""
        # 待实现
        pass

    def send_order(self, req: OrderRequest) -> str:
        """委托下单"""
        # 待实现
        pass

    def cancel_order(self, req: CancelRequest) -> None:
        """委托撤单"""
        # 待实现
        pass

    def query_history(self, req: HistoryRequest) -> List[BarData]:
        """查询K线历史数据"""
        # 待实现
        pass

    def on_query_account(self, data, request) -> None:
        """资金查询回调"""
        # 待实现
        pass

    def on_query_position(self, data, request) -> None:
        """持仓查询回调"""
        # 待实现
        pass

    def on_query_order(self, data, request) -> None:
        """未成交委托查询回调"""
        # 待实现
        pass

    def on_send_order(self, data, request) -> None:
        """委托下单回调"""
        # 待实现
        pass

    def on_cancel_order(self, data, request) -> None:
        """委托撤单回调"""
        # 待实现
        pass

    def on_failed(self, status_code: int, request: dict) -> None:
        """请求失败回调"""
        # 待实现
        pass

    def on_error(self, exception_type: type, exception_value: Exception, tb) -> None:
        """异常回调"""
        # 待实现
        pass
        
    def stop(self) -> None:
        """停止REST API服务"""
        # 待实现
        pass
