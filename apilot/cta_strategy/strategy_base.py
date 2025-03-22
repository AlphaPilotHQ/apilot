from abc import ABC
from copy import copy
from typing import Any, Callable, List, Dict
from collections import defaultdict
from datetime import datetime, timedelta

from apilot.trader.object import (
    TickData,
    BarData,
    TradeData,
    OrderData,
    ContractData
)
from apilot.trader.constant import Direction, Offset, Status, OrderType, Interval
from apilot.trader.utility import BarGenerator, ArrayManager, virtual

from .constants import StopOrder, EngineType


class CtaTemplate(ABC):
    parameters: list = []
    variables: list = []

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ) -> None:
        self.cta_engine = cta_engine
        self.strategy_name = strategy_name
        self.vt_symbol = vt_symbol

        self.inited: bool = False
        self.trading: bool = False
        self.pos: int = 0

        # Copy a new variables list here to avoid duplicate insert when multiple
        # strategy instances are created with the same strategy class.
        self.variables = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos")

        # 设置策略参数
        for name in self.parameters:
            if name in setting:
                setattr(self, name, setting[name])

    @classmethod
    def get_class_parameters(cls) -> dict:
        """获取策略类默认参数字典"""
        return {name: getattr(cls, name) for name in cls.parameters}

    def get_parameters(self) -> dict:
        """获取策略实例参数字典"""
        return {name: getattr(self, name) for name in self.parameters}

    def get_variables(self) -> dict:
        """获取策略变量字典"""
        return {name: getattr(self, name) for name in self.variables}

    def get_data(self) -> dict:
        """获取策略数据"""
        return {
            "strategy_name": self.strategy_name,
            "vt_symbol": self.vt_symbol,
            "class_name": self.__class__.__name__,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
        }

    @virtual
    def on_init(self) -> None:
        pass

    @virtual
    def on_start(self) -> None:
        pass

    @virtual
    def on_stop(self) -> None:
        pass

    @virtual
    def on_tick(self, tick: TickData) -> None:
        pass

    @virtual
    def on_bar(self, bar: BarData) -> None:
        pass

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        pass

    @virtual
    def on_order(self, order: OrderData) -> None:
        pass

    @virtual
    def on_stop_order(self, stop_order: StopOrder) -> None:
        pass

    # TODO：应该改成long short close三种状态比较好
    def buy(
        self,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False
    ) -> list:
        return self.send_order(Direction.LONG, Offset.OPEN, price, volume, stop, net)

    def sell(
        self,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False
    ) -> list:
        return self.send_order(Direction.SHORT, Offset.CLOSE, price, volume, stop, net)

    def short(
        self,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False
    ) -> list:
        """卖出开仓"""
        return self.send_order(Direction.SHORT, Offset.OPEN, price, volume, stop, net)

    def cover(
        self,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False
    ) -> list:
        """买入平仓"""
        return self.send_order(Direction.LONG, Offset.CLOSE, price, volume, stop, net)

    def send_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False
    ) -> list:

        try:
            if self.trading:
                vt_orderids: list = self.cta_engine.send_order(
                    self, direction, offset, price, volume, stop, net
                )
                return vt_orderids
            else:
                self.cta_engine.main_engine.log_warning("策略未启动交易，订单未发送", source="CTA_STRATEGY")
                return []
        except Exception as e:
            self.cta_engine.main_engine.log_error(f"发送订单异常: {str(e)}", source="CTA_STRATEGY")
            return []

    def cancel_order(self, vt_orderid: str) -> None:
        if self.trading:
            self.cta_engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        if self.trading:
            self.cta_engine.cancel_all(self)

    def write_log(self, msg: str) -> None:
        """
        记录日志消息（已弃用，请使用主引擎的日志方法）
        """
        import warnings
        warnings.warn(
            "策略的write_log方法已弃用，请使用cta_engine.main_engine.log_xxx方法代替",
            DeprecationWarning,
            stacklevel=2
        )

        # 检查引擎类型，区分实盘和回测环境
        if hasattr(self.cta_engine, "main_engine"):
            # 实盘环境
            self.cta_engine.main_engine.log_info(msg, source="CTA_STRATEGY")
        else:
            # 回测环境
            self.cta_engine.write_log(msg, self)

    def get_engine_type(self) -> EngineType:
        """
        Return whether the cta_engine is backtesting or live trading.
        """
        return self.cta_engine.get_engine_type()

    def get_pricetick(self) -> float:
        """
        Return pricetick data of trading contract.
        """
        return self.cta_engine.get_pricetick(self)

    def get_size(self) -> int:
        return self.cta_engine.get_size(self)

    def load_bar(
        self,
        days: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable = None,
        use_database: bool = False
    ) -> None:
        """Load historical bar data for initializing strategy."""
        if not callback:
            callback = self.on_bar

        bars: List[BarData] = self.cta_engine.load_bar(
            self.vt_symbol,
            days,
            interval,
            callback,
            use_database
        )

        for bar in bars:
            callback(bar)

    def load_tick(self, days: int) -> None:
        """
        Load historical tick data for initializing strategy.
        """
        ticks: List[TickData] = self.cta_engine.load_tick(self.vt_symbol, days, self.on_tick)

        for tick in ticks:
            self.on_tick(tick)

    def send_email(self, msg) -> None:
        if self.inited:
            self.cta_engine.send_email(msg, self)

    def sync_data(self) -> None:
        """
        Sync strategy variables value into disk storage.
        """
        if self.trading:
            self.cta_engine.sync_strategy_data(self)


class TargetPosTemplate(CtaTemplate):
    tick_add = 1

    last_tick: TickData = None
    last_bar: BarData = None
    target_pos = 0

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.active_orderids: list = []
        self.cancel_orderids: list = []

        self.variables.append("target_pos")

    @virtual
    def on_tick(self, tick: TickData) -> None:
        self.last_tick = tick

    @virtual
    def on_bar(self, bar: BarData) -> None:
        self.last_bar = bar

    @virtual
    def on_order(self, order: OrderData) -> None:
        vt_orderid: str = order.vt_orderid

        if not order.is_active():
            if vt_orderid in self.active_orderids:
                self.active_orderids.remove(vt_orderid)

            if vt_orderid in self.cancel_orderids:
                self.cancel_orderids.remove(vt_orderid)

    def check_order_finished(self) -> bool:
        if self.active_orderids:
            return False
        else:
            return True

    def set_target_pos(self, target_pos) -> None:
        self.target_pos = target_pos
        self.trade()

    def trade(self) -> None:
        if not self.check_order_finished():
            self.cancel_old_order()
        else:
            self.send_new_order()

    def cancel_old_order(self) -> None:
        for vt_orderid in self.active_orderids:
            if vt_orderid not in self.cancel_orderids:
                self.cancel_order(vt_orderid)
                self.cancel_orderids.append(vt_orderid)

    def send_new_order(self) -> None:
        # 计算仓位变化
        pos_change = self.target_pos - self.pos
        if not pos_change:
            return

        # 标记买卖方向
        is_long = pos_change > 0

        # 设置价格
        price = 0
        if self.last_tick:
            if is_long:
                price = self.last_tick.ask_price_1 + self.tick_add
                if self.last_tick.limit_up:
                    price = min(price, self.last_tick.limit_up)
            else:
                price = self.last_tick.bid_price_1 - self.tick_add
                if self.last_tick.limit_down:
                    price = max(price, self.last_tick.limit_down)
        elif self.last_bar:
            price = self.last_bar.close_price + (self.tick_add if is_long else -self.tick_add)
        else:
            return  # 无法确定价格时不交易

        # 回测模式直接发单
        if self.get_engine_type() == EngineType.BACKTESTING:
            func = self.buy if is_long else self.short
            vt_orderids = func(price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)
            return

        # 实盘模式，有活动订单时不交易
        if self.active_orderids:
            return

        # 实盘模式处理
        volume = abs(pos_change)

        if is_long:  # 做多
            if self.pos < 0:  # 持有空仓
                # 计算实际平仓量
                cover_volume = min(volume, abs(self.pos))
                vt_orderids = self.cover(price, cover_volume)
            else:  # 无仓位或持有多仓
                vt_orderids = self.buy(price, volume)
        else:  # 做空
            if self.pos > 0:  # 持有多仓
                # 计算实际平仓量
                sell_volume = min(volume, self.pos)
                vt_orderids = self.sell(price, sell_volume)
            else:  # 无仓位或持有空仓
                vt_orderids = self.short(price, volume)

        self.active_orderids.extend(vt_orderids)
