from init_env import *
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from vnpy.trader.constant import Direction, Interval
from vnpy.trader.object import BarData, TickData, OrderData, TradeData
from vnpy.cta_strategy.base import StopOrder
from vnpy.cta_strategy.template import CtaTemplate
from vnpy.cta_strategy.backtesting import BacktestingEngine
from vnpy.trader.utility import BarGenerator, ArrayManager
from vnpy.trader.optimize import OptimizationSetting, run_ga_optimization, run_bf_optimization
import numpy as np
import pandas as pd

class StdMomentumStrategy(CtaTemplate):
    """标准差+动量+标准差追踪止损示例策略"""

    # 策略参数
    std_period = 20             # 标准差/动量计算周期
    std_threshold = 0.015       # 标准差相对值过滤倍数 (相对标准差阈值，如1.5%)
    mom_threshold = 0.03        # 动量阈值 (3%)
    trailing_std_scale = 1.0    # 追踪止损倍数

    parameters = ["std_period", "std_threshold", "mom_threshold", "trailing_std_scale"]
    variables = ["relative_std", "momentum", "intra_trade_high", "intra_trade_low"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am = ArrayManager()

        # 初始化指标
        self.relative_std = 0.0    # 相对标准差 (标准差/均值)
        self.momentum = 0.0        # 动量
        
        # 追踪最高/最低价
        self.intra_trade_high = 0
        self.intra_trade_low = 0

    def on_init(self):
        self.write_log("策略初始化")
        self.load_bar(self.std_period * 2)  # 加载足够的历史数据确保指标计算准确
        
    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")

    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")

    def on_tick(self, tick: TickData):
        """Tick数据更新"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """1分钟K线数据更新"""
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData):
        """5分钟K线数据更新，包含交易逻辑"""
        self.cancel_all()  # 取消之前的所有订单

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            return

        # 计算标准差和均值
        close_array = am.close_array
        if len(close_array) < self.std_period:
            return  # 确保有足够的数据用于计算
        
        recent_closes = close_array[-self.std_period:]
        self.std_value = float(np.std(recent_closes))
        std_mean = float(np.mean(recent_closes))  
        self.relative_std = self.std_value / std_mean if std_mean else 0 
        
        # 计算动量因子
        old_price = close_array[-(self.std_period + 1)]
        current_price = close_array[-1]
        if old_price != 0:
            self.momentum = (current_price / old_price) - 1
        else:
            self.momentum = 0.0

        # 持仓状态下更新跟踪止损价格
        if self.pos > 0:
            self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
        elif self.pos < 0:
            self.intra_trade_low = min(self.intra_trade_low, bar.low_price)

        # 交易逻辑：相对标准差过滤 & 动量信号
        if self.pos == 0:
            if self.relative_std > self.std_threshold:  # 使用相对标准差判断波动率
                # 初始化追踪价格
                self.intra_trade_high = bar.high_price
                self.intra_trade_low = bar.low_price
                
                # 在VNPy回测中，资金通过cta_engine的capital属性获取
                # BacktestingEngine类在初始化时设置了capital并会随交易更新
                size = max(1, int(self.cta_engine.capital / bar.close_price))
                
                if self.momentum > self.mom_threshold:
                    self.buy(bar.close_price, size)
                elif self.momentum < -self.mom_threshold:
                    self.short(bar.close_price, size)

        elif self.pos > 0:  # 多头持仓 → 标准差追踪止损
            # 计算移动止损价格 - 每次都更新以实现追踪止损
            long_stop = self.intra_trade_high - self.trailing_std_scale * self.std_value
            self.sell(long_stop, abs(self.pos), stop=True)

        elif self.pos < 0:  # 空头持仓 → 标准差追踪止损
            # 计算移动止损价格 - 每次都更新以实现追踪止损
            short_stop = self.intra_trade_low + self.trailing_std_scale * self.std_value
            self.cover(short_stop, abs(self.pos), stop=True)

    def on_order(self, order: OrderData):
        """委托回调"""
        pass

    def on_trade(self, trade: TradeData):
        """成交回调"""
        self.write_log(f"成交: {trade.direction} {trade.offset} {trade.volume}@{trade.price}")

    def on_stop_order(self, stop_order: StopOrder):
        """停止单回调"""
        pass

def run_backtest(
    std_period=20,           # 标准差/动量计算周期
    std_threshold=0.015,     # 标准差相对值过滤倍数 (相对标准差阈值，如1.5%)
    mom_threshold=0.03,      # 动量阈值 (3%)
    trailing_std_scale=1.0,  # 追踪止损倍数
):
    """创建并设置回测引擎"""
    engine = BacktestingEngine()
    
    # 设置回测参数
    engine.set_parameters(
        vt_symbol="SOL-USDT.LOCAL",   # 交易对
        interval="1m",                # 时间周期
        start=datetime(2023, 1, 1),   # 开始日期
        end=datetime(2023, 6, 30),    # 结束日期
        rate=0.0001,                  # 手续费
        size=10000,                   # 合约乘数
        pricetick=0.1,                # 最小价格变动
        capital=100_000,              # 初始资金
    )
    
    # 添加策略，传入动态参数
    strategy_params = {
        "std_period": std_period,
        "std_threshold": std_threshold,
        "mom_threshold": mom_threshold,
        "trailing_std_scale": trailing_std_scale
    }
    engine.add_strategy(StdMomentumStrategy, strategy_params)
    
    # 运行回测
    engine.load_data()
    engine.run_backtesting()
    df = engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()
    
    return df


def evaluate_with_parameters(params: Dict) -> Tuple[Dict, Dict]:
    """
    使用指定参数评估策略表现
    
    Args:
        params: 策略参数字典
        
    Returns:
        Tuple[Dict, Dict]: 参数字典和统计结果字典
    """
    # 提取参数
    std_period = params.get("std_period", 20)
    std_threshold = params.get("std_threshold", 0.015)
    mom_threshold = params.get("mom_threshold", 0.03)
    trailing_std_scale = params.get("trailing_std_scale", 1.0)
    
    # 创建回测引擎
    engine = BacktestingEngine()
    
    # 设置回测参数
    engine.set_parameters(
        vt_symbol="SOL-USDT.LOCAL",   # 交易对
        interval="1m",                # 时间周期
        start=datetime(2023, 1, 1),   # 开始日期
        end=datetime(2023, 6, 30),    # 结束日期
        rate=0.0001,                  # 手续费
        size=1,                   # 合约乘数
        pricetick=0.1,                # 最小价格变动
        capital=100_000,              # 初始资金
    )
    
    # 添加策略
    engine.add_strategy(
        StdMomentumStrategy, 
        {
            "std_period": std_period,
            "std_threshold": std_threshold,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale
        }
    )
    
    # 运行回测
    engine.load_data()
    engine.run_backtesting()
    
    # 计算结果
    statistics = engine.calculate_statistics(output=False)
    
    # 返回参数和统计结果
    return params, statistics


def run_optimization_ga():
    """
    使用遗传算法优化策略参数
    """
    print("使用遗传算法优化参数...")
    
    # 创建优化设置对象
    setting = OptimizationSetting()
    
    # 添加需要优化的参数，设置参数范围
    setting.add_parameter("std_period", 10, 30, 5)                # 10-30，步长5
    setting.add_parameter("std_threshold", 0.005, 0.025, 0.005)   # 0.5%-2.5%，步长0.5%
    setting.add_parameter("mom_threshold", 0.01, 0.05, 0.01)      # 0.01-0.05，步长0.01
    setting.add_parameter("trailing_std_scale", 0.5, 3.0, 0.5)    # 0.5-3.0，步长0.5
    
    # 设置优化目标
    setting.set_target("sharpe_ratio")  # 以夏普比率为优化目标
    
    # 运行遗传算法优化
    results = run_ga_optimization(
        evaluate_func=evaluate_with_parameters,     # 评估函数
        optimization_setting=setting,               # 优化设置
        key_func=lambda result: result[1].get("sharpe_ratio", 0),  # 排序键函数
        max_workers=8,                              # 最大工作进程数
        population_size=50,                         # 种群大小
        ngen_size=20                                # 世代数量
    )
    
    print("优化完成，最优参数组合:")
    for result in results[:5]:  # 打印前5个最佳结果
        params, stats = result
        print(f"参数: {params}")
        print(f"夏普比率: {stats.get('sharpe_ratio', 0):.4f}")
        print(f"总收益率: {stats.get('total_return', 0):.2%}")
        print(f"最大回撤: {stats.get('max_drawdown', 0):.2%}")
        print("-" * 50)
    
    return results


if __name__ == "__main__":
    # 使用默认参数运行回测
    # run_backtest()
    
    # 使用遗传算法优化参数
    # best_params_ga = run_optimization_ga()