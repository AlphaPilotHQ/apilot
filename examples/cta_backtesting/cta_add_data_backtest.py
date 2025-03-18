"""
使用Backtrader风格的add_data API进行CTA策略回测
这个示例展示了如何使用新的API来进行回测
"""
from init_env import *
from datetime import datetime

from vnpy.trader.constant import Direction, Interval, Exchange
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.cta_strategy.base import StopOrder
from vnpy.cta_strategy.template import CtaTemplate
from vnpy.cta_strategy.backtesting import BacktestingEngine
from vnpy.trader.utility import BarGenerator, ArrayManager
import numpy as np
import pandas as pd

class StdMomentumStrategy(CtaTemplate):
    """
    标准动量策略，与cta_database_backtest.py中相同
    """

    # 策略参数
    std_period = 20       
    mom_threshold = 0.05       
    trailing_std_scale = 4    

    parameters = ["std_period", "mom_threshold", "trailing_std_scale"]
    variables = ["momentum", "intra_trade_high", "intra_trade_low"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am = ArrayManager(size=200)  

        # 初始化指标
        self.momentum = 0.0        
        self.std_value = 0.0       
        
        # 追踪最高/最低价
        self.intra_trade_high = 0
        self.intra_trade_low = 0

    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(self.std_period * 2)
        
    def on_start(self):
        self.write_log("策略启动")

    def on_stop(self):
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData):
        self.cancel_all()

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return
            
        self.std_value = am.std(self.std_period)
        
        old_price = am.close_array[-self.std_period - 1]
        current_price = am.close_array[-1]
        if old_price != 0:
            self.momentum = (current_price / old_price) - 1
        else:
            self.momentum = 0.0

        if self.pos > 0:
            self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
        elif self.pos < 0:
            self.intra_trade_low = min(self.intra_trade_low, bar.low_price)

        if self.pos == 0:
            self.intra_trade_high = bar.high_price
            self.intra_trade_low = bar.low_price
            
            size = max(1, int(self.cta_engine.capital / bar.close_price))
            
            if self.momentum > self.mom_threshold:
                self.buy(bar.close_price, size)
            elif self.momentum < -self.mom_threshold:
                self.short(bar.close_price, size)

        elif self.pos > 0:
            long_stop = self.intra_trade_high - self.trailing_std_scale * self.std_value
            self.sell(long_stop, abs(self.pos), stop=True)

        elif self.pos < 0:
            short_stop = self.intra_trade_low + self.trailing_std_scale * self.std_value
            self.cover(short_stop, abs(self.pos), stop=True)

    def on_order(self, order: OrderData):
        pass

    def on_trade(self, trade: TradeData):
        self.write_log(f"成交: {trade.direction} {trade.offset} {trade.volume}@{trade.price}")

    def on_stop_order(self, stop_order: StopOrder):
        pass


def run_backtest_with_add_data():
    """使用新的add_data API运行回测"""
    print("正在使用Backtrader风格的add_data API运行回测...")
    
    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数
    engine.set_parameters(
        vt_symbol="SOL-USDT.LOCAL",  # 仍然需要设置此参数
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 6, 30),
        rate=0.0001,
        slippage=0,
        size=1,
        pricetick=0.001,
        capital=100000,
    )
    
    # 使用新的add_data API添加数据源 - 默认使用CSV
    engine.add_data(
        database_type="csv"
        # 也可以指定其他参数如data_path等
    )
    
    # 添加策略
    engine.add_strategy(
        StdMomentumStrategy, 
        {
            "std_period": 35,
            "mom_threshold": 0.01,
            "trailing_std_scale": 10
        }
    )
    
    # 仍需调用load_data来加载数据
    engine.load_data()
    
    # 运行回测
    engine.run_backtesting()
    
    # 计算结果
    df = engine.calculate_result()
    
    # 显示图表
    engine.calculate_statistics()
    engine.show_chart()
    
    return df, engine


def run_mongodb_backtest():
    """使用MongoDB数据源的示例"""
    print("正在使用MongoDB数据源运行回测...")
    
    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数 - 修改为实际的交易对名称
    engine.set_parameters(
        vt_symbol="SOLUSDT.BINANCE",  # 修改为正确的格式，去掉了中间的连字符
        interval=Interval.MINUTE,
        start=datetime(2025, 2, 1),    # 修改为当前可用的数据范围
        end=datetime(2025, 3, 15),
        rate=0.0001,
        slippage=0,
        size=1,
        pricetick=0.001,
        capital=100000,
    )
    
    # 使用用户提供的MongoDB数据源
    engine.add_data(
        database_type="mongodb",
        host="47.237.74.9",
        port=27017,
        database="alphapilot",
        username="alphapilot",
        password="123456",
        authentication_source="admin",
        collection="symbol_trade"
    )
    
    # 添加策略
    engine.add_strategy(
        StdMomentumStrategy, 
        {
            "std_period": 35,
            "mom_threshold": 0.01,
            "trailing_std_scale": 10
        }
    )
    
    # 加载数据
    engine.load_data()
    
    # 运行回测
    engine.run_backtesting()
    
    # 计算结果
    df = engine.calculate_result()
    
    # 显示图表
    engine.calculate_statistics()
    engine.show_chart()
    
    return df, engine


if __name__ == "__main__":
    # 使用CSV数据源(默认)
    # df, engine = run_backtest_with_add_data()
    
    # 使用MongoDB数据源
    df, engine = run_mongodb_backtest()
