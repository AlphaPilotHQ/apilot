from datetime import datetime
from typing import ClassVar

from apilot_portfoliostrategy import StrategyEngine, StrategyTemplate
from apilot_portfoliostrategy.utility import PortfolioBarGenerator

from apilot.core.object import BarData, TickData
from apilot.core.utility import ArrayManager, Interval


class PortfolioBollChannelStrategy(StrategyTemplate):
    """组合布林带通道策略"""

    author = "用Python的交易员"

    boll_window = 18
    boll_dev = 3.4
    cci_window = 10
    atr_window = 30
    sl_multiplier = 5.2
    fixed_size = 1
    price_add = 5

    parameters: ClassVar[list[str]] = [
        "boll_window",
        "boll_dev",
        "cci_window",
        "atr_window",
        "sl_multiplier",
        "fixed_size",
        "price_add",
    ]
    variables: ClassVar[list] = []

    def __init__(
        self,
        strategy_engine: StrategyEngine,
        strategy_name: str,
        symbols: list[str],
        setting: dict,
    ) -> None:
        """构造函数"""
        super().__init__(strategy_engine, strategy_name, symbols, setting)

        self.boll_up: dict[str, float] = {}
        self.boll_down: dict[str, float] = {}
        self.cci_value: dict[str, float] = {}
        self.atr_value: dict[str, float] = {}
        self.intra_trade_high: dict[str, float] = {}
        self.intra_trade_low: dict[str, float] = {}

        self.targets: dict[str, int] = {}
        self.last_tick_time: datetime = None

        # 获取合约信息
        self.ams: dict[str, ArrayManager] = {}
        for symbol in self.symbols:
            self.ams[symbol] = ArrayManager()
            self.targets[symbol] = 0

        self.pbg = PortfolioBarGenerator(
            self.on_bars, 2, self.on_2hour_bars, Interval.HOUR
        )

    def on_init(self) -> None:
        """策略初始化回调"""
        self.write_log("策略初始化")

        self.load_bars(10)

    def on_start(self) -> None:
        """策略启动回调"""
        self.write_log("策略启动")

    def on_stop(self) -> None:
        """策略停止回调"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData) -> None:
        """行情推送回调"""
        self.pbg.update_tick(tick)

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """K线切片回调"""
        for _symbol, bar in bars.items():
            self.pbg.update_bar(bar)

    def on_2hour_bars(self, bars: dict[str, BarData]) -> None:
        """2小时K线回调"""
        self.cancel_all()

        # 更新到缓存序列
        for symbol, bar in bars.items():
            am: ArrayManager = self.ams[symbol]
            am.update_bar(bar)

        for symbol, bar in bars.items():
            am: ArrayManager = self.ams[symbol]
            if not am.inited:
                return

            self.boll_up[symbol], self.boll_down[symbol] = am.boll(
                self.boll_window, self.boll_dev
            )
            self.cci_value[symbol] = am.cci(self.cci_window)
            self.atr_value[symbol] = am.atr(self.atr_window)

            # 计算目标仓位
            current_pos = self.get_pos(symbol)
            if current_pos == 0:
                self.intra_trade_high[symbol] = bar.high_price
                self.intra_trade_low[symbol] = bar.low_price

                if self.cci_value[symbol] > 0:
                    self.targets[symbol] = self.fixed_size
                elif self.cci_value[symbol] < 0:
                    self.targets[symbol] = -self.fixed_size

            elif current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )
                self.intra_trade_low[symbol] = bar.low_price

                long_stop = (
                    self.intra_trade_high[symbol]
                    - self.atr_value[symbol] * self.sl_multiplier
                )

                if bar.close_price <= long_stop:
                    self.targets[symbol] = 0

            elif current_pos < 0:
                self.intra_trade_low[symbol] = min(
                    self.intra_trade_low[symbol], bar.low_price
                )
                self.intra_trade_high[symbol] = bar.high_price

                short_stop = (
                    self.intra_trade_low[symbol]
                    + self.atr_value[symbol] * self.sl_multiplier
                )

                if bar.close_price >= short_stop:
                    self.targets[symbol] = 0

        # 基于目标仓位进行委托
        for symbol in self.symbols:
            target_pos = self.targets[symbol]
            current_pos = self.get_pos(symbol)

            pos_diff = target_pos - current_pos
            volume = abs(pos_diff)
            bar = bars[symbol]
            boll_up = self.boll_up[symbol]
            boll_down = self.boll_down[symbol]

            if pos_diff > 0:
                price = bar.close_price + self.price_add

                if current_pos < 0:
                    self.cover(symbol, price, volume)
                else:
                    self.buy(symbol, boll_up, volume)

            elif pos_diff < 0:
                price = bar.close_price - self.price_add

                if current_pos > 0:
                    self.sell(symbol, price, volume)
                else:
                    self.short(symbol, boll_down, volume)

        # 推送界面更新
        self.put_event()
