"""
Test script for backtesting functionality using local data.
"""

from datetime import datetime
from typing import Any, ClassVar

import numpy as np

from apilot.engine.backtest import BacktestingEngine
from apilot.strategy.template import PATemplate
from apilot.trader.constant import Direction, Interval
from apilot.trader.object import BarData, TickData
from apilot.trader.utility import BarGenerator


class TestStrategy(PATemplate):
    # Strategy parameters
    fast_window = 20  # 调整为更短周期,增加交易频率
    slow_window = 200  # 保持中等长度的慢速均线
    fast_ma = 0.0
    slow_ma = 0.0

    fast_ma_array: ClassVar[list] = []
    slow_ma_array: ClassVar[list] = []

    parameters: ClassVar[list] = ["fast_window", "slow_window"]
    variables: ClassVar[list] = ["fast_ma", "slow_ma"]

    def __init__(
        self,
        pa_engine: Any,
        strategy_name: str,
        symbol: str,
        setting: dict,
    ) -> None:
        super().__init__(pa_engine, strategy_name, symbol, setting)

        self.fast_ma_array = []
        self.slow_ma_array = []
        self.bg = BarGenerator(self.on_bar)
        self.inited = True
        self.trading = True
        self.pos = 0

    def on_tick(self, tick: TickData):
        """K线推送"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """K线更新"""
        # 添加当前bar到数组
        self.fast_ma_array.append(bar.close_price)
        self.slow_ma_array.append(bar.close_price)

        # 如果数组长度超过窗口大小,则移除最早的元素
        if len(self.fast_ma_array) > self.fast_window:
            self.fast_ma_array.pop(0)

        if len(self.slow_ma_array) > self.slow_window:
            self.slow_ma_array.pop(0)

        # 至少需要两个窗口才能计算均线
        if (
            len(self.fast_ma_array) >= self.fast_window
            and len(self.slow_ma_array) >= self.slow_window
        ):
            # 计算快速均线
            self.fast_ma = np.mean(self.fast_ma_array)
            # 计算慢速均线
            self.slow_ma = np.mean(self.slow_ma_array)

        # 均线值计算完成后,才允许交易
        if not self.fast_ma or not self.slow_ma:
            return

        # 使用账户价值计算交易量
        if self.fast_ma > self.slow_ma and not self.pos:  # 金叉做多
            # 获取账户可用资金
            # 注意:在回测中,我们通过pa_engine访问资金状态
            capital = 0
            if hasattr(self.pa_engine, "capital"):
                capital = self.pa_engine.capital

            # 如果没有可用资金信息,则使用默认值(初始100万)
            if not capital:
                capital = 1000000

            # 计算可以买入的最大数量 (全仓)
            # 预留1%资金作为缓冲,避免因计算精度问题导致的下单失败
            max_volume = capital * 0.5 / bar.close_price
            max_volume = int(max_volume * 0.99)  # 取整

            # 买入
            self.buy(bar.close_price, max_volume)
            self.write_log(
                f"全仓买入: {max_volume:.2f} 手,价格: {bar.close_price:.2f}, 金额: {max_volume * bar.close_price:.2f}"
            )

        elif self.fast_ma < self.slow_ma and self.pos > 0:  # 死叉卖出
            # 全部卖出
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(
                f"全仓卖出: {abs(self.pos):.2f} 手,价格: {bar.close_price:.2f}, 金额: {abs(self.pos) * bar.close_price:.2f}"
            )

    def on_order(self, order):
        """
        订单更新回调
        """
        pass

    def on_trade(self, trade):
        """
        成交更新回调
        """
        self.pos = self.pos_dict[trade.symbol]  # 更新持仓

    def on_stop(self):
        """
        策略停止回调
        """
        self.write_log("策略停止")

    def write_log(self, msg):
        """输出日志"""
        print(f"[{datetime.now()}] {msg}")


def run_backtest():
    """运行回测"""
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="SOL-USDT.BINANCE",
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 31),
        rate=0.0001,
        size=1,
        pricetick=0.01,
        capital=1000000,
    )

    engine.add_strategy(TestStrategy, {"fast_window": 5, "slow_window": 20})

    # 执行回测
    engine.load_data()
    engine.run_backtesting()

    # 统计和绘图
    engine.calculate_result()
    stats = engine.calculate_statistics()

    print("回测完成!")
    print("统计结果:")
    if stats:
        for key, value in stats.items():
            if isinstance(value, int | float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    else:
        print("无统计结果")

    print("\n=== 交易记录 ===")
    if engine.trades:
        for i, (_tradeid, trade) in enumerate(engine.trades.items()):
            print(
                f"交易 #{i + 1}: {'买入' if trade.direction == Direction.LONG else '卖出'} {trade.volume} 手, 价格: {trade.price}, 时间: {trade.datetime}"
            )
    else:
        print("无交易")


if __name__ == "__main__":
    run_backtest()
