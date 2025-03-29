from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from typing import Any, ClassVar

from apilot.core.constant import Direction, EngineType, Interval, Offset
from apilot.core.object import BarData, OrderData, TickData, TradeData
from apilot.utils.logger import get_logger

# 模块级别初始化日志器
logger = get_logger("CtaStrategy")


class CtaTemplate(ABC):
    """
    PA策略模板
    统一采用多币种设计,单币种只是特殊情况
    """

    parameters: ClassVar[list] = []
    variables: ClassVar[list] = []

    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        symbols: str | list[str],
        setting: dict,
    ) -> None:
        self.cta_engine = cta_engine
        self.strategy_name = strategy_name

        # 统一处理为列表形式
        if isinstance(symbols, str):
            self.symbols = [symbols]
        else:
            self.symbols = symbols if symbols else []

        self.inited: bool = False
        self.trading: bool = False

        # 统一使用字典管理持仓和目标
        self.pos_dict: dict[str, int] = defaultdict(int)  # 实际持仓
        self.target_dict: dict[str, int] = defaultdict(int)  # 目标持仓

        # 委托缓存容器
        self.orders: dict[str, OrderData] = {}
        self.active_orderids: set[str] = set()

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
            "symbols": self.symbols,
        }
        return data

    @abstractmethod
    def on_init(self) -> None:
        """策略初始化回调"""
        pass

    @abstractmethod
    def on_start(self) -> None:
        """策略启动回调"""
        pass

    @abstractmethod
    def on_stop(self) -> None:
        """策略停止回调"""
        pass

    @abstractmethod
    def on_tick(self, tick: TickData) -> None:
        """行情Tick推送回调"""
        pass

    @abstractmethod
    def on_bar(self, bar: BarData) -> None:
        """K线推送回调"""
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线字典推送回调"""
        # 遍历每个币种的K线并调用on_bar,保持向后兼容性
        for _symbol, bar in bars.items():
            self.on_bar(bar)

    @abstractmethod
    def on_trade(self, trade: TradeData) -> None:
        """成交回调"""
        # 更新持仓数据
        self.pos_dict[trade.symbol] += (
            trade.volume if trade.direction == Direction.LONG else -trade.volume
        )

    @abstractmethod
    def on_order(self, order: OrderData) -> None:
        """委托回调"""
        # 更新委托缓存
        self.orders[order.orderid] = order

        # 如果委托不再活跃,从活跃委托集合中移除
        if not order.is_active() and order.orderid in self.active_orderids:
            self.active_orderids.remove(order.orderid)

    # TODO:应该改成long short close三种状态比较好
    def buy(
        self, symbol: str, price: float, volume: float, net: bool = False
    ) -> list[str]:
        """
        买入开仓
        """
        return self.send_order(symbol, Direction.LONG, Offset.OPEN, price, volume, net)

    def sell(
        self, symbol: str, price: float, volume: float, net: bool = False
    ) -> list[str]:
        """
        卖出平仓
        """
        return self.send_order(
            symbol, Direction.SHORT, Offset.CLOSE, price, volume, net
        )

    def short(
        self, symbol: str, price: float, volume: float, net: bool = False
    ) -> list[str]:
        """
        卖出开仓
        """
        return self.send_order(symbol, Direction.SHORT, Offset.OPEN, price, volume, net)

    def cover(
        self, symbol: str, price: float, volume: float, net: bool = False
    ) -> list[str]:
        """
        买入平仓
        """
        return self.send_order(symbol, Direction.LONG, Offset.CLOSE, price, volume, net)

    def send_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        net: bool = False,
    ) -> list[str]:
        """发送委托"""
        try:
            if self.trading:
                # 根据是否为多币种模式调用不同的发单接口
                orderids: list[str] = self.cta_engine.send_order(
                    self, symbol, direction, offset, price, volume, net
                )

                # 添加到活跃委托集合
                for orderid in orderids:
                    self.active_orderids.add(orderid)

                return orderids
            else:
                logger.warning(f"[{self.strategy_name}] 策略未启动交易,订单未发送")
                return []
        except Exception as e:
            logger.error(f"[{self.strategy_name}] 发送订单异常: {e!s}")
            return []

    def cancel_order(self, orderid: str) -> None:
        """撤销委托"""
        if self.trading:
            self.cta_engine.cancel_order(self, orderid)

    def cancel_all(self) -> None:
        """全撤活动委托"""
        if self.trading:
            for orderid in list(self.active_orderids):
                self.cancel_order(orderid)

    def get_pos(self, symbol: str) -> int:
        """
        查询持仓
        """
        return self.pos_dict.get(symbol, 0)

    def get_target(self, symbol: str) -> int:
        """查询目标仓位"""
        return self.target_dict.get(symbol, 0)

    def set_target(self, symbol: str, target: int) -> None:
        """设置目标仓位"""
        self.target_dict[symbol] = target

    def get_engine_type(self) -> EngineType:
        """查询引擎类型"""
        return self.cta_engine.get_engine_type()

    def get_pricetick(self, symbol: str) -> float:
        """
        获取合约最小价格跳动
        """
        return self.cta_engine.get_pricetick(self, symbol)

    def get_size(self, symbol: str) -> int:
        """
        获取合约乘数
        """
        return self.cta_engine.get_size(self, symbol)

    def load_bar(
        self,
        days: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable | None = None,
        use_database: bool = False,
    ) -> None:
        """加载历史K线数据初始化策略"""
        if not callback:
            callback = self.on_bar

        bars: list[BarData] = self.cta_engine.load_bar(
            self.symbols[0], days, interval, callback, use_database
        )

        for bar in bars:
            callback(bar)

    def load_bars(self, days: int, interval: Interval = Interval.MINUTE) -> None:
        """
        加载多币种历史K线数据
        适用于多币种策略
        """
        if self.symbols:
            self.cta_engine.load_bars(self, days, interval)
        else:
            # 无币种模式下,使用传统load_bar
            self.load_bar(days, interval)

    def load_tick(self, days: int) -> None:
        """加载历史Tick数据初始化策略"""
        ticks: list[TickData] = self.cta_engine.load_tick(
            self.symbols[0], days, self.on_tick
        )

        for tick in ticks:
            self.on_tick(tick)

    def sync_data(self) -> None:
        """同步策略变量值到磁盘存储"""
        if self.trading:
            self.cta_engine.sync_strategy_data(self)

    def calculate_price(
        self, symbol: str, direction: Direction, reference: float
    ) -> float:
        """计算调仓委托价格(支持按需重载实现)"""
        return reference

    def rebalance_portfolio(self, bars: dict[str, BarData]) -> None:
        """基于目标执行调仓交易"""
        self.cancel_all()

        # 只发出当前K线切片有行情的合约的委托
        for symbol, bar in bars.items():
            # 计算仓差
            target: int = self.get_target(symbol)
            pos: int = self.get_pos(symbol)
            diff: int = target - pos

            # 多头
            if diff > 0:
                # 计算多头委托价
                order_price: float = self.calculate_price(
                    symbol, Direction.LONG, bar.close_price
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
                    self.cover(symbol, order_price, cover_volume)

                if buy_volume:
                    self.buy(symbol, order_price, buy_volume)
            # 空头
            elif diff < 0:
                # 计算空头委托价
                order_price: float = self.calculate_price(
                    symbol, Direction.SHORT, bar.close_price
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
                    self.sell(symbol, order_price, sell_volume)

                if short_volume:
                    self.short(symbol, order_price, short_volume)

    def get_order(self, orderid: str) -> OrderData | None:
        """查询委托数据"""
        return self.orders.get(orderid, None)

    def get_all_active_orderids(self) -> list[str]:
        """获取全部活动状态的委托号"""
        return list(self.active_orderids)


class TargetPosTemplate(CtaTemplate):
    tick_add = 1

    last_tick: TickData = None
    last_bar: BarData = None
    target_pos = 0

    def __init__(self, cta_engine, strategy_name, symbols, setting) -> None:
        super().__init__(cta_engine, strategy_name, symbols, setting)

        self.active_orderids: list = []
        self.cancel_orderids: list = []

        self.variables.append("target_pos")

    @abstractmethod
    def on_tick(self, tick: TickData) -> None:
        self.last_tick = tick

    @abstractmethod
    def on_bar(self, bar: BarData) -> None:
        self.last_bar = bar

    @abstractmethod
    def on_order(self, order: OrderData) -> None:
        orderid: str = order.orderid

        if not order.is_active():
            if orderid in self.active_orderids:
                self.active_orderids.remove(orderid)

            if orderid in self.cancel_orderids:
                self.cancel_orderids.remove(orderid)

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
        for orderid in self.active_orderids:
            if orderid not in self.cancel_orderids:
                self.cancel_order(orderid)
                self.cancel_orderids.append(orderid)

    def send_new_order(self) -> None:
        """根据目标仓位和实际仓位计算并委托"""
        # 计算仓位变化
        pos_change = self.target_pos - self.get_pos(self.symbols[0])
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
            price = self.last_bar.close_price + (
                self.tick_add if is_long else -self.tick_add
            )
        else:
            return  # 无法确定价格时不交易

        # 回测模式直接发单
        if self.get_engine_type() == EngineType.BACKTESTING:
            func = self.buy if is_long else self.short
            orderids = func(self.symbols[0], price, abs(pos_change))
            self.active_orderids.extend(orderids)
            return

        # 实盘模式,有活动订单时不交易
        if self.active_orderids:
            return

        # 实盘模式处理
        volume = abs(pos_change)

        if is_long:  # 做多
            if self.get_pos(self.symbols[0]) < 0:  # 持有空仓
                # 计算实际平仓量
                cover_volume = min(volume, abs(self.get_pos(self.symbols[0])))
                orderids = self.cover(self.symbols[0], price, cover_volume)
            else:  # 无仓位或持有多仓
                orderids = self.buy(self.symbols[0], price, volume)
        else:  # 做空
            if self.get_pos(self.symbols[0]) > 0:  # 持有多仓
                # 计算实际平仓量
                sell_volume = min(volume, self.get_pos(self.symbols[0]))
                orderids = self.sell(self.symbols[0], price, sell_volume)
            else:  # 无仓位或持有空仓
                orderids = self.short(self.symbols[0], price, volume)

        self.active_orderids.extend(orderids)
