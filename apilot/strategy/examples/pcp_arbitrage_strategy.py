from datetime import datetime
from typing import ClassVar

from apilot_portfoliostrategy import StrategyEngine, StrategyTemplate

from apilot.core.constant import Direction
from apilot.core.object import BarData, TickData
from apilot.core.utility import BarGenerator
from apilot.utils.symbol import split_symbol


class PcpArbitrageStrategy(StrategyTemplate):
    """期权平价套利策略"""

    author = "用Python的交易员"

    entry_level = 20
    price_add = 5
    fixed_size = 1

    strike_price = 0
    futures_price = 0
    synthetic_price = 0
    current_spread = 0
    futures_pos = 0
    call_pos = 0
    put_pos = 0
    futures_target = 0
    call_target = 0
    put_target = 0

    parameters: ClassVar[list[str]] = ["entry_level", "price_add", "fixed_size"]
    variables: ClassVar[list[str]] = [
        "strike_price",
        "futures_price",
        "synthetic_price",
        "current_spread",
        "futures_pos",
        "call_pos",
        "put_pos",
        "futures_target",
        "call_target",
        "put_target",
    ]

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        symbols: list[str],
        setting: dict,
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, symbols, setting)

        self.bgs: dict[str, BarGenerator] = {}
        self.last_tick_time: datetime = None

        # 绑定合约代码
        for symbol in self.symbols:
            base_symbol, _ = split_symbol(symbol)

            if "C" in base_symbol:
                self.call_symbol = symbol
                _, strike_str = base_symbol.split("-C-")  # CFFEX/DCE
                self.strike_price = int(strike_str)
            elif "P" in base_symbol:
                self.put_symbol = symbol
            else:
                self.futures_symbol = symbol

            def on_bar(bar: BarData):
                """"""
                pass

            self.bgs[symbol] = BarGenerator(on_bar)

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.load_bars(1)

    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """行情推送回调"""
        if self.last_tick_time and self.last_tick_time.minute != tick.datetime.minute:
            bars = {}
            for symbol, bg in self.bgs.items():
                bars[symbol] = bg.generate()
            self.on_bars(bars)

        bg: BarGenerator = self.bgs[tick.symbol]
        bg.update_tick(tick)

        self.last_tick_time = tick.datetime

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        self.cancel_all()

        # 计算PCP价差
        call_bar = bars[self.call_symbol]
        put_bar = bars[self.put_symbol]
        futures_bar = bars[self.futures_symbol]

        self.futures_price = futures_bar.close_price
        self.synthetic_price = (
            call_bar.close_price - put_bar.close_price + self.strike_price
        )
        self.current_spread = self.synthetic_price - self.futures_price

        # 计算目标仓位
        futures_target: int = self.get_target(self.futures_symbol)

        if not futures_target:
            if self.current_spread > self.entry_level:
                self.set_target(self.call_symbol, -self.fixed_size)
                self.set_target(self.put_symbol, self.fixed_size)
                self.set_target(self.futures_symbol, self.fixed_size)
            elif self.current_spread < -self.entry_level:
                self.set_target(self.call_symbol, self.fixed_size)
                self.set_target(self.put_symbol, -self.fixed_size)
                self.set_target(self.futures_symbol, -self.fixed_size)
        elif futures_target > 0:
            if self.current_spread <= 0:
                self.set_target(self.call_symbol, 0)
                self.set_target(self.put_symbol, 0)
                self.set_target(self.futures_symbol, 0)
        else:
            if self.current_spread >= 0:
                self.set_target(self.call_symbol, 0)
                self.set_target(self.put_symbol, 0)
                self.set_target(self.futures_symbol, 0)

        # 执行调仓交易
        self.rebalance_portfolio()

        # 更新策略状态
        self.call_pos = self.get_pos(self.call_symbol)
        self.put_pos = self.get_pos(self.put_symbol)
        self.futures_pos = self.get_pos(self.futures_symbol)

        self.call_target = self.get_target(self.call_symbol)
        self.put_target = self.get_target(self.put_symbol)
        self.futures_target = self.get_target(self.futures_symbol)

        self.put_event()

    def calculate_price(
        self, symbol: str, direction: Direction, reference: float
    ) -> float:
        """计算调仓委托价格(支持按需重载实现)"""
        if direction == Direction.LONG:
            price: float = reference + self.price_add
        else:
            price: float = reference - self.price_add

        return price
