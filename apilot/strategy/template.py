from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from typing import Any, ClassVar

from apilot.core.constant import Direction, EngineType, Interval, Offset
from apilot.core.object import BarData, OrderData, TickData, TradeData
from apilot.core.utility import virtual
from apilot.utils.logger import get_logger

logger = get_logger("PAStrategy")


class PATemplate(ABC):
    parameters: ClassVar[list] = []
    variables: ClassVar[list] = []

    def __init__(
        self,
        pa_engine: Any,
        strategy_name: str,
        symbols: str | list[str],
        setting: dict,
    ) -> None:
        self.pa_engine = pa_engine
        self.strategy_name = strategy_name

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

    def on_bars(self, bars: dict[str, BarData]) -> None:
        for _symbol, bar in bars.items():
            self.on_bar(bar)

    def on_trade(self, trade: TradeData) -> None:
        # 更新持仓数据
        self.pos_dict[trade.symbol] += (
            trade.volume if trade.direction == Direction.LONG else -trade.volume
        )

    def on_order(self, order: OrderData) -> None:
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
                orderids: list[str] = self.pa_engine.send_order(
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
            self.pa_engine.cancel_order(self, orderid)

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
        return self.pa_engine.get_engine_type()

    def get_pricetick(self, symbol: str) -> float:
        """
        获取合约最小价格跳动
        """
        return self.pa_engine.get_pricetick(self, symbol)

    def get_size(self, symbol: str) -> int:
        """
        获取合约乘数
        """
        return self.pa_engine.get_size(self, symbol)

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

        bars: list[BarData] = self.pa_engine.load_bar(
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
            self.pa_engine.load_bars(self, days, interval)
        else:
            # 无币种模式下,使用传统load_bar
            self.load_bar(days, interval)

    def load_tick(self, days: int) -> None:
        """加载历史Tick数据初始化策略"""
        ticks: list[TickData] = self.pa_engine.load_tick(
            self.symbols[0], days, self.on_tick
        )

        for tick in ticks:
            self.on_tick(tick)

    def sync_data(self) -> None:
        """同步策略变量值到磁盘存储"""
        if self.trading:
            self.pa_engine.sync_strategy_data(self)

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
