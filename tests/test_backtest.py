"""
Test script for backtesting functionality using local data.
"""

import os
from datetime import datetime
import pandas as pd
from typing import Any, List

from apilot.trader.constant import Interval, Direction, Exchange
from apilot.trader.object import BarData, TickData
from apilot.trader.utility import BarGenerator
from apilot.cta_strategy.template import CtaTemplate
from apilot.cta_strategy.backtesting import BacktestingEngine


class TestStrategy(CtaTemplate):

    # Strategy parameters
    fast_window = 20     # 调整为更短周期，增加交易频率
    slow_window = 200    # 保持中等长度的慢速均线
    
    # Variables
    fast_ma = 0.0
    slow_ma = 0.0
    
    fast_ma_array = []
    slow_ma_array = []
    
    parameters = ["fast_window", "slow_window"]
    variables = ["fast_ma", "slow_ma"]
    
    def __init__(
        self,
        cta_engine: Any,
        strategy_name: str,
        vt_symbol: str,
        setting: dict,
    ):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        
        self.fast_ma_array = []
        self.slow_ma_array = []
        self.bg = BarGenerator(self.on_bar)  # 添加BarGenerator初始化
    
    def on_init(self):
        """Called when strategy is initialized."""
        self.write_log("策略初始化")
        
    def on_start(self):
        """Called when strategy is started."""
        self.write_log("策略启动")
    
    def on_stop(self):
        """Called when strategy is stopped."""
        self.write_log("策略停止")
    
    def on_tick(self, tick: TickData):
        """Called on every tick update."""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """Called on every bar update."""
        # Update fast MA
        self.fast_ma_array.append(bar.close_price)
        if len(self.fast_ma_array) > self.fast_window:
            self.fast_ma_array.pop(0)
        self.fast_ma = sum(self.fast_ma_array) / len(self.fast_ma_array)

        # Update slow MA
        self.slow_ma_array.append(bar.close_price)
        if len(self.slow_ma_array) > self.slow_window:
            self.slow_ma_array.pop(0)
        self.slow_ma = sum(self.slow_ma_array) / len(self.slow_ma_array)

        # No trading until both MAs are calculated
        if len(self.fast_ma_array) < self.fast_window or len(self.slow_ma_array) < self.slow_window:
            return

        # 全仓交易策略 - 先检查引擎状态
        if hasattr(self, 'inited') and not self.inited:
            return

        # 使用账户价值计算交易量
        if self.fast_ma > self.slow_ma and not self.pos:  # 金叉做多
            # 获取账户可用资金
            # 注意：在回测中，我们通过cta_engine访问资金状态
            capital = 0
            if hasattr(self.cta_engine, 'capital'):
                capital = self.cta_engine.capital
            
            # 如果没有可用资金信息，则使用默认值(初始100万)
            if not capital:
                capital = 1000000
                
            # 计算可以买入的最大数量 (全仓)
            # 预留1%资金作为缓冲，避免因计算精度问题导致的下单失败
            max_volume = capital * 0.5 / bar.close_price
            
            # 买入
            self.buy(bar.close_price, max_volume)
            self.write_log(f"全仓买入: {max_volume:.2f} 手，价格: {bar.close_price:.2f}, 金额: {max_volume * bar.close_price:.2f}")
        
        elif self.fast_ma < self.slow_ma and self.pos > 0:  # 死叉卖出
            # 全部卖出
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(f"全仓卖出: {abs(self.pos):.2f} 手，价格: {bar.close_price:.2f}, 金额: {abs(self.pos) * bar.close_price:.2f}")



    
# 创建回测引擎
engine = BacktestingEngine()

# 设置回测参数
engine.set_parameters(
    vt_symbol="SOL-USDT.BINANCE",
    interval=Interval.MINUTE,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 1, 31),
    rate=0.0001,
    slippage=0.0001,
    size=1,
    pricetick=0.01,
    capital=1000000
)

# 添加策略
engine.add_strategy(TestStrategy, {
    "fast_window": 5,
    "slow_window": 20
})

# 加载数据并运行回测
engine.load_data()  # 现在会自动从CSV文件读取数据

# 运行回测
engine.run_backtesting()

# 计算结果
engine.calculate_result()
stats = engine.calculate_statistics()

# 获取结果
df = engine.daily_df

print("回测完成！")
print("统计结果:")
if stats:
    for key, value in stats.items():
        print(f"{key}: {value}")
else:
    print("没有统计数据 - 可能没有产生交易")

# 输出详细的交易记录
print("\n=== 交易记录 ===")
if engine.trades:
    for i, (vt_tradeid, trade) in enumerate(engine.trades.items()):
        print(f"交易 #{i+1}: {'买入' if trade.direction == Direction.LONG else '卖出'} {trade.volume} 手, 价格: {trade.price}, 时间: {trade.datetime}")
else:
    print("没有交易记录")