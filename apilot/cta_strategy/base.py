from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from typing import Dict, List

from apilot.trader.constant import Direction, Offset, Interval


APP_NAME = "CtaStrategy"
STOPORDER_PREFIX = "STOP"

# TODO：改成EN
class StopOrderStatus(Enum):
    WAITING = "等待中"
    CANCELLED = "已撤销"
    TRIGGERED = "已触发"


class EngineType(Enum):
    LIVE = "实盘"
    BACKTESTING = "回测"


class BacktestingMode(Enum):
    BAR = 1
    TICK = 2

@dataclass
class StopOrder:
    """
    停止单数据结构
    
    用于记录和管理CTA策略中的停止单信息。停止单是在价格达到特定条件时触发的订单，
    主要用于止损、止盈等风险管理场景。
    
    属性:
        vt_symbol: 合约代码（包含交易所信息）
        direction: 买卖方向
        offset: 开平方向
        price: 触发价格
        volume: 交易数量
        stop_orderid: 停止单唯一标识
        strategy_name: 策略名称
        datetime: 创建时间
        net: 是否净仓交易
        vt_orderids: 关联的委托单ID列表
        status: 停止单状态
    """
    vt_symbol: str
    direction: Direction
    offset: Offset
    price: float
    volume: float
    stop_orderid: str
    strategy_name: str
    datetime: datetime
    net: bool = False
    vt_orderids: List[str] = field(default_factory=list)
    status: StopOrderStatus = StopOrderStatus.WAITING

    # TODO：数量必须是正数
    # def __post_init__(self) -> None:
    #     """
    #     数据初始化后的验证逻辑
    #     用于确保关键字段符合业务规则
    #     """
    #     # 验证价格必须为正数
    #     if self.price <= 0:
    #         raise ValueError(f"停止单价格必须为正数: {self.price}")
            
    #     # 验证交易量必须为正数
    #     if self.volume <= 0:
    #         raise ValueError(f"停止单数量必须为正数: {self.volume}")



EVENT_CTA_LOG: str = "eCtaLog"
EVENT_CTA_STRATEGY = "eCtaStrategy"
EVENT_CTA_STOPORDER = "eCtaStopOrder"

INTERVAL_DELTA_MAP: Dict[Interval, timedelta] = {
    Interval.TICK: timedelta(milliseconds=1),
    Interval.MINUTE: timedelta(minutes=1),
    Interval.HOUR: timedelta(hours=1),
    Interval.DAILY: timedelta(days=1),
}
