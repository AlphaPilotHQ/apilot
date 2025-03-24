from abc import ABC
from copy import copy
from typing import Any, Callable, List, Dict, Optional, Set, Union
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

from .constants import EngineType


class CtaTemplate(ABC):
    """
    PA策略模板
    统一采用多币种设计，单币种只是特殊情况
    """

    parameters: list = []
    variables: list = []

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbols: Union[str, List[str]],
        setting: dict
    ) -> None:
        """
        构造函数
        vt_symbols可以是单个字符串或字符串列表
        """
        self.cta_engine = cta_engine
        self.strategy_name = strategy_name

        # 统一处理为列表形式
        if isinstance(vt_symbols, str):
            self.vt_symbols = [vt_symbols]
        else:
            self.vt_symbols = vt_symbols if vt_symbols else []

        self.inited: bool = False
        self.trading: bool = False

        # 统一使用字典管理持仓和目标
        self.pos_dict: Dict[str, int] = defaultdict(int)  # 实际持仓
        self.target_dict: Dict[str, int] = defaultdict(int)  # 目标持仓

        # 委托缓存容器
        self.orders: Dict[str, OrderData] = {}
        self.active_orderids: Set[str] = set()

        # 复制变量列表并插入默认变量
        self.variables = copy(self.variables)
        self.variables.insert(0, "inited")
        self.variables.insert(1, "trading")
        self.variables.insert(2, "pos_dict")
        self.variables.insert(3, "target_dict")

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
        data = {
            "strategy_name": self.strategy_name,
            "class_name": self.__class__.__name__,
            "parameters": self.get_parameters(),
            "variables": self.get_variables(),
            "vt_symbols": self.vt_symbols
        }
        return data

    @virtual
    def on_init(self) -> None:
        """策略初始化回调"""
        pass

    @virtual
    def on_start(self) -> None:
        """策略启动回调"""
        pass

    @virtual
    def on_stop(self) -> None:
        """策略停止回调"""
        pass

    @virtual
    def on_tick(self, tick: TickData) -> None:
        """行情Tick推送回调"""
        pass

    @virtual
    def on_bar(self, bar: BarData) -> None:
        """K线推送回调"""
        pass

    @virtual
    def on_bars(self, bars: Dict[str, BarData]) -> None:
        """
        收到多个币种的K线数据时调用，由子类实现
        """
        # 遍历每个币种的K线并调用on_bar，保持向后兼容性
        for vt_symbol, bar in bars.items():
            self.on_bar(bar)

    @virtual
    def on_trade(self, trade: TradeData) -> None:
        """成交回调"""
        # 更新持仓数据
        self.pos_dict[trade.vt_symbol] += trade.volume if trade.direction == Direction.LONG else -trade.volume

    @virtual
    def on_order(self, order: OrderData) -> None:
        """委托回调"""
        # 更新委托缓存
        self.orders[order.vt_orderid] = order

        # 如果委托不再活跃，从活跃委托集合中移除
        if not order.is_active() and order.vt_orderid in self.active_orderids:
            self.active_orderids.remove(order.vt_orderid)

    # TODO：应该改成long short close三种状态比较好
    def buy(
        self,
        vt_symbol: str,
        price: float,
        volume: float,
        net: bool = False
    ) -> List[str]:
        """
        买入开仓
        """
        return self.send_order(vt_symbol, Direction.LONG, Offset.OPEN, price, volume, net)

    def sell(
        self,
        vt_symbol: str,
        price: float,
        volume: float,
        net: bool = False
    ) -> List[str]:
        """
        卖出平仓
        """
        return self.send_order(vt_symbol, Direction.SHORT, Offset.CLOSE, price, volume, net)

    def short(
        self,
        vt_symbol: str,
        price: float,
        volume: float,
        net: bool = False
    ) -> List[str]:
        """
        卖出开仓
        """
        return self.send_order(vt_symbol, Direction.SHORT, Offset.OPEN, price, volume, net)

    def cover(
        self,
        vt_symbol: str,
        price: float,
        volume: float,
        net: bool = False
    ) -> List[str]:
        """
        买入平仓
        """
        return self.send_order(vt_symbol, Direction.LONG, Offset.CLOSE, price, volume, net)

    def send_order(
        self,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        net: bool = False
    ) -> List[str]:
        """发送委托"""
        try:
            if self.trading:
                # 根据是否为多币种模式调用不同的发单接口
                vt_orderids: List[str] = self.cta_engine.send_order(
                    self, vt_symbol, direction, offset, price, volume, net
                )

                # 添加到活跃委托集合
                for vt_orderid in vt_orderids:
                    self.active_orderids.add(vt_orderid)

                return vt_orderids
            else:
                self.cta_engine.main_engine.log_warning("策略未启动交易，订单未发送", source="CTA_STRATEGY")
                return []
        except Exception as e:
            self.cta_engine.main_engine.log_error(f"发送订单异常: {str(e)}", source="CTA_STRATEGY")
            return []

    def cancel_order(self, vt_orderid: str) -> None:
        """撤销委托"""
        if self.trading:
            self.cta_engine.cancel_order(self, vt_orderid)

    def cancel_all(self) -> None:
        """全撤活动委托"""
        if self.trading:
            for vt_orderid in list(self.active_orderids):
                self.cancel_order(vt_orderid)

    def get_pos(self, vt_symbol: str) -> int:
        """
        查询持仓
        """
        return self.pos_dict.get(vt_symbol, 0)

    def get_target(self, vt_symbol: str) -> int:
        """查询目标仓位"""
        return self.target_dict.get(vt_symbol, 0)

    def set_target(self, vt_symbol: str, target: int) -> None:
        """设置目标仓位"""
        self.target_dict[vt_symbol] = target

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
        """查询引擎类型"""
        return self.cta_engine.get_engine_type()

    def get_pricetick(self, vt_symbol: str) -> float:
        """
        获取合约最小价格跳动
        """
        return self.cta_engine.get_pricetick(self, vt_symbol)

    def get_size(self, vt_symbol: str) -> int:
        """
        获取合约乘数
        """
        return self.cta_engine.get_size(self, vt_symbol)

    def load_bar(
        self,
        days: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable = None,
        use_database: bool = False
    ) -> None:
        """加载历史K线数据初始化策略"""
        if not callback:
            callback = self.on_bar

        bars: List[BarData] = self.cta_engine.load_bar(
            self.vt_symbols[0],
            days,
            interval,
            callback,
            use_database
        )

        for bar in bars:
            callback(bar)

    def load_bars(self, days: int, interval: Interval = Interval.MINUTE) -> None:
        """
        加载多币种历史K线数据
        适用于多币种策略
        """
        if self.vt_symbols:
            self.cta_engine.load_bars(self, days, interval)
        else:
            # 无币种模式下，使用传统load_bar
            self.load_bar(days, interval)

    def load_tick(self, days: int) -> None:
        """加载历史Tick数据初始化策略"""
        ticks: List[TickData] = self.cta_engine.load_tick(self.vt_symbols[0], days, self.on_tick)

        for tick in ticks:
            self.on_tick(tick)

    def send_email(self, msg: str) -> None:
        """发送电子邮件通知"""
        if self.inited:
            self.cta_engine.send_email(msg, self)

    def sync_data(self) -> None:
        """同步策略变量值到磁盘存储"""
        if self.trading:
            self.cta_engine.sync_strategy_data(self)

    def calculate_price(
        self,
        vt_symbol: str,
        direction: Direction,
        reference: float
    ) -> float:
        """计算调仓委托价格（支持按需重载实现）"""
        return reference

    def rebalance_portfolio(self, bars: Dict[str, BarData]) -> None:
        """基于目标执行调仓交易"""
        self.cancel_all()

        # 只发出当前K线切片有行情的合约的委托
        for vt_symbol, bar in bars.items():
            # 计算仓差
            target: int = self.get_target(vt_symbol)
            pos: int = self.get_pos(vt_symbol)
            diff: int = target - pos

            # 多头
            if diff > 0:
                # 计算多头委托价
                order_price: float = self.calculate_price(
                    vt_symbol,
                    Direction.LONG,
                    bar.close_price
                )

                # 计算买平和买开数量
                cover_volume: int = 0
                buy_volume: int = 0

                if pos < 0:
                    cover_volume = min(diff, abs(pos))
                    buy_volume = diff - cover_volume
                else:
                    buy_volume = diff

                # 发出对应委托
                if cover_volume:
                    self.cover(vt_symbol, order_price, cover_volume)

                if buy_volume:
                    self.buy(vt_symbol, order_price, buy_volume)
            # 空头
            elif diff < 0:
                # 计算空头委托价
                order_price: float = self.calculate_price(
                    vt_symbol,
                    Direction.SHORT,
                    bar.close_price
                )

                # 计算卖平和卖开数量
                sell_volume: int = 0
                short_volume: int = 0

                if pos > 0:
                    sell_volume = min(abs(diff), pos)
                    short_volume = abs(diff) - sell_volume
                else:
                    short_volume = abs(diff)

                # 发出对应委托
                if sell_volume:
                    self.sell(vt_symbol, order_price, sell_volume)

                if short_volume:
                    self.short(vt_symbol, order_price, short_volume)

    def get_order(self, vt_orderid: str) -> Optional[OrderData]:
        """查询委托数据"""
        return self.orders.get(vt_orderid, None)

    def get_all_active_orderids(self) -> List[str]:
        """获取全部活动状态的委托号"""
        return list(self.active_orderids)


class TargetPosTemplate(CtaTemplate):
    tick_add = 1

    last_tick: TickData = None
    last_bar: BarData = None
    target_pos = 0

    def __init__(self, cta_engine, strategy_name, vt_symbols, setting) -> None:
        super().__init__(cta_engine, strategy_name, vt_symbols, setting)

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
        """根据目标仓位和实际仓位计算并委托"""
        # 计算仓位变化
        pos_change = self.target_pos - self.get_pos(self.vt_symbols[0])
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
            vt_orderids = func(self.vt_symbols[0], price, abs(pos_change))
            self.active_orderids.extend(vt_orderids)
            return

        # 实盘模式，有活动订单时不交易
        if self.active_orderids:
            return

        # 实盘模式处理
        volume = abs(pos_change)

        if is_long:  # 做多
            if self.get_pos(self.vt_symbols[0]) < 0:  # 持有空仓
                # 计算实际平仓量
                cover_volume = min(volume, abs(self.get_pos(self.vt_symbols[0])))
                vt_orderids = self.cover(self.vt_symbols[0], price, cover_volume)
            else:  # 无仓位或持有多仓
                vt_orderids = self.buy(self.vt_symbols[0], price, volume)
        else:  # 做空
            if self.get_pos(self.vt_symbols[0]) > 0:  # 持有多仓
                # 计算实际平仓量
                sell_volume = min(volume, self.get_pos(self.vt_symbols[0]))
                vt_orderids = self.sell(self.vt_symbols[0], price, sell_volume)
            else:  # 无仓位或持有空仓
                vt_orderids = self.short(self.vt_symbols[0], price, volume)

        self.active_orderids.extend(vt_orderids)
