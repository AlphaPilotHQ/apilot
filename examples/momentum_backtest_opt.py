"""
动量策略回测与优化示例

此模块实现了一个基于动量指标的趋势跟踪策略，结合标准差动态止损。
包含单次回测和参数优化功能，支持使用遗传算法寻找最优参数组合。
"""

import os
import sys

import setup_path

from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

import apilot as ap


class StdMomentumStrategy(ap.CtaTemplate):
    """
    策略逻辑：

    1. 核心思路：结合动量信号与标准差动态止损的中长期趋势跟踪策略

    2. 入场信号：
       - 基于动量指标(当前价格/N周期前价格-1)生成交易信号
       - 动量 > 阈值(5%)时做多
       - 动量 < -阈值(-5%)时做空
       - 使用全部账户资金进行头寸管理

    3. 风险管理：
       - 使用基于标准差的动态追踪止损
       - 多头持仓：止损设置在最高价-4倍标准差
       - 空头持仓：止损设置在最低价+4倍标准差
    """

    # 策略参数
    std_period = 20
    mom_threshold = 0.05
    trailing_std_scale = 4

    parameters = ["std_period", "mom_threshold", "trailing_std_scale"]
    variables = ["momentum", "intra_trade_high", "intra_trade_low"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """
        初始化策略
        """
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = ap.BarGenerator(self.on_bar, 5, self.on_5min_bar)
        self.am = ap.ArrayManager(size=200)  # 增加大小以支持长周期计算

        # 初始化指标
        self.momentum = 0.0  # 动量
        self.std_value = 0.0  # 标准差

        # 追踪最高/最低价
        self.intra_trade_high = 0
        self.intra_trade_low = 0

    def on_init(self):
        """
        策略初始化
        """
        self.load_bar(self.std_period * 2)  # 加载足够的历史数据确保指标计算准确

    def on_bar(self, bar: ap.BarData):
        """1分钟K线数据更新"""
        self.cta_engine.write_log(
            f"收到1分钟K线: {bar.datetime} O:{bar.open_price} "
            f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}",
            self
        )
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: ap.BarData):
        """5分钟K线数据更新，包含交易逻辑"""
        self.cta_engine.write_log(
            f"生成5分钟K线: {bar.datetime} O:{bar.open_price} "
            f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}"
        )
        self.cancel_all()  # 取消之前的所有订单

        am = self.am
        am.update_bar(bar)

        # 计算标准差
        # 使用ArrayManager的内置std函数计算标准差
        self.std_value = am.std(self.std_period)

        # 计算动量因子
        old_price = am.close_array[-self.std_period - 1]
        current_price = am.close_array[-1]
        if old_price != 0:
            self.momentum = (current_price / old_price) - 1
        else:
            self.momentum = 0.0


        # 持仓状态下更新跟踪止损价格
        if self.pos > 0:
            self.intra_trade_high = max(self.intra_trade_high, bar.high_price)
        elif self.pos < 0:
            self.intra_trade_low = min(self.intra_trade_low, bar.low_price)

        # 交易逻辑：仅基于动量信号
        if self.pos == 0:
            # 初始化追踪价格
            self.intra_trade_high = bar.high_price
            self.intra_trade_low = bar.low_price

            # BacktestingEngine类在初始化时设置了capital并会随交易更新
            size = max(1, int(self.cta_engine.capital / bar.close_price))

            if self.momentum > self.mom_threshold:

                self.buy(bar.close_price, size)
            elif self.momentum < -self.mom_threshold:

                self.short(bar.close_price, size)

        elif self.pos > 0:  # 多头持仓 → 标准差追踪止损
            # 计算移动止损价格
            long_stop = self.intra_trade_high - self.trailing_std_scale * self.std_value

            # 当价格跌破止损线时平仓
            if bar.close_price < long_stop:
                self.sell(bar.close_price, abs(self.pos))

        elif self.pos < 0:  # 空头持仓 → 标准差追踪止损
            # 计算移动止损价格
            short_stop = self.intra_trade_low + self.trailing_std_scale * self.std_value

            # 当价格突破止损线时平仓
            if bar.close_price > short_stop:
                self.cover(bar.close_price, abs(self.pos))

    def on_order(self, order: ap.OrderData):
        """委托回调"""
        return

    def on_trade(self, trade: ap.TradeData):
        """成交回调"""


def run_backtesting(
    strategy_class,
    init_cash=100000,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 6, 30),
    std_period=20,
    mom_threshold=0.05,
    trailing_std_scale=4.0,
):
    print("运行回测 - 参数:", std_period, mom_threshold, trailing_std_scale)

    # 创建回测引擎
    engine = ap.BacktestingEngine()
    engine.log_info("步骤1: 创建回测引擎实例")

    # 设置引擎参数
    engine.log_info(
        f"步骤2: 设置引擎参数 - 初始资金:{init_cash}, 时间范围:{start} 至 {end}"
    )
    engine.set_parameters(
        vt_symbols=["SOL-USDT.LOCAL"],
        interval="1m",
        start=start,
        end=end,
        rates={"SOL-USDT.LOCAL": 0.00075},
        sizes={"SOL-USDT.LOCAL": 1},
        priceticks={"SOL-USDT.LOCAL": 0.001},
        capital=init_cash,
    )

    # 验证CSV文件是否存在
    csv_path = "/Users/bobbyding/Documents/GitHub/apilot/data/SOL-USDT_LOCAL_1m.csv"
    if not os.path.exists(csv_path):
        engine.log_error(f"CSV文件不存在: {csv_path}")
        return None, None

    # 检查CSV文件内容
    try:
        df = pd.read_csv(csv_path)
        engine.log_info(f"CSV文件行数: {len(df)} 文件列名: {df.columns.tolist()}")
    except Exception as e:
        engine.log_error(f"读取CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    # 添加策略
    engine.log_info(
        f"步骤3: 添加策略 - {strategy_class.__name__} 参数: std_period={std_period}, "
        f"mom_threshold={mom_threshold}, trailing_std_scale={trailing_std_scale}"
    )
    engine.add_strategy(
        strategy_class,
        {
            "std_period": std_period,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale,
        },
    )

    # 添加CSV数据
    engine.log_info(f"步骤4: 添加数据源 - CSV数据路径: {csv_path}")
    engine.add_data(
        database_type="csv",
        data_path=csv_path,
        datetime="candle_begin_time",  # 映射时间字段
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    # 运行回测
    engine.log_info("步骤5: 开始执行回测")
    engine.run_backtesting()

    # 计算结果和统计指标
    engine.log_info("步骤6: 计算回测结果和绩效统计")
    df = engine.calculate_result()
    stats = engine.calculate_statistics()

    # 打印关键绩效指标
    if stats:
        engine.log_info(
            f"回测结果 - 收益率: {stats['return']:,.2%}, 夏普比率: {stats['sharpe_ratio']:.2f}, "
            f"最大回撤: {stats['max_drawdown']:,.2%}, 交易次数: {stats['total_trades']}"
        )
    else:
        engine.log_warning("未能生成回测统计数据")

    return df, stats


if __name__ == "__main__":
    # 单次回测
    df, stats = run_backtesting(
        StdMomentumStrategy,
        init_cash=100000,
        start=datetime(2023, 4, 20),
        end=datetime(2023, 6, 30),
        std_period=35,
        mom_threshold=0.01,
        trailing_std_scale=10,
    )
