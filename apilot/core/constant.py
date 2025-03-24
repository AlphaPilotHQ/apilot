"""
APilot量化交易平台使用的常量枚举定义。
"""

from enum import Enum
from datetime import datetime, timedelta
from typing import Dict


class Direction(Enum):
    """
    Direction of order/trade/position.
    """
    LONG = "多"
    SHORT = "空"
    NET = "净"


class Offset(Enum):
    """
    Offset of order/trade.
    """
    NONE = ""
    OPEN = "开"
    CLOSE = "平"
    CLOSETODAY = "平今"
    CLOSEYESTERDAY = "平昨"


class Status(Enum):
    """
    Order status.
    """
    SUBMITTING = "提交中"
    NOTTRADED = "未成交"
    PARTTRADED = "部分成交"
    ALLTRADED = "全部成交"
    CANCELLED = "已撤销"
    REJECTED = "拒单"


class Product(Enum):
    """
    产品类型
    """
    SPOT = "现货"        # 加密货币现货
    FUTURES = "合约"      # 加密货币合约
    MARGIN = "杠杆"       # 杠杆交易
    OPTION = "期权"       # 加密货币期权


class OrderType(Enum):
    """
    订单类型
    """
    LIMIT = "限价"          # 限价单
    MARKET = "市价"         # 市价单
    STOP = "止损"           # 止损单
    STOP_LIMIT = "止损限价"  # 止损限价单
    POST_ONLY = "只挂"      # 只挂单，不吃单
    FAK = "FAK"           # Fill and Kill
    FOK = "FOK"           # Fill or Kill


class OptionType(Enum):
    """
    Option type.
    """
    CALL = "看涨期权"
    PUT = "看跌期权"


class Exchange(Enum):
    """
    交易所枚举
    """
    # 加密货币交易所
    BINANCE = "BINANCE"     # 币安
    BINANCE_FUTURES = "BINANCEF"  # 币安合约
    OKEX = "OKEX"           # OKEx
    BYBIT = "BYBIT"         # Bybit
    COINBASE = "COINBASE"   # Coinbase
    DERIBIT = "DERIBIT"     # Deribit
    KRAKEN = "KRAKEN"       # Kraken
    BITFINEX = "BITFINEX"   # Bitfinex
    BITSTAMP = "BITSTAMP"   # Bitstamp

    LOCAL = "LOCAL"         # 本地生成数据（回测使用）


class Currency(Enum):
    """
    货币类型
    """
    USD = "USD"        # 美元
    USDT = "USDT"      # 泰达币
    BTC = "BTC"        # 比特币
    ETH = "ETH"        # 以太币
    BNB = "BNB"        # 币安币


class Interval(Enum):
    """
    K线数据周期
    """
    MINUTE = "1m"      # 1分钟
    MINUTE3 = "3m"     # 3分钟
    MINUTE5 = "5m"     # 5分钟
    MINUTE15 = "15m"   # 15分钟
    MINUTE30 = "30m"   # 30分钟
    HOUR = "1h"        # 1小时
    HOUR4 = "4h"       # 4小时
    DAILY = "d"        # 日线
    WEEKLY = "w"       # 周线
    TICK = "tick"      # Tick数据


# 引擎类型
class EngineType(Enum):
    LIVE = "实盘"
    BACKTESTING = "回测"


# 回测模式
class BacktestingMode(Enum):
    BAR = 1
    TICK = 2


# 时间间隔映射
INTERVAL_DELTA_MAP: Dict[Interval, timedelta] = {
    Interval.TICK: timedelta(milliseconds=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}


# 常量名称
APP_NAME = "CtaStrategy"
