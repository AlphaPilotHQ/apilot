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
        self.write_log("策略初始化")
        self.write_log(f"加载历史数据: {self.std_period * 2}根K线")
        self.load_bar(self.std_period * 2)  # 加载足够的历史数据确保指标计算准确

    def on_start(self):
        """策略启动"""
        self.write_log("策略启动")

    def on_stop(self):
        """策略停止"""
        self.write_log("策略停止")

    def on_tick(self, tick: ap.TickData):
        """Tick数据更新"""
        self.bg.update_tick(tick)

    def on_bar(self, bar: ap.BarData):
        """1分钟K线数据更新"""
        self.write_log(
            f"收到1分钟K线: {bar.datetime} O:{bar.open_price} "
            f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}"
        )
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: ap.BarData):
        """5分钟K线数据更新，包含交易逻辑"""
        self.write_log(
            f"生成5分钟K线: {bar.datetime} O:{bar.open_price} "
            f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}"
        )
        self.cancel_all()  # 取消之前的所有订单

        am = self.am
        am.update_bar(bar)
        if not am.inited:
            self.write_log(f"ArrayManager未初始化，当前数据量: {len(am.close_array)}")
            return

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

        self.write_log(
            f"指标计算: std={self.std_value:.6f}, momentum={self.momentum:.6f}, "
            f"threshold={self.mom_threshold}"
        )

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
                self.write_log(
                    f"多头开仓信号: momentum={self.momentum:.6f} > "
                    f"threshold={self.mom_threshold}"
                )
                self.buy(bar.close_price, size)
            elif self.momentum < -self.mom_threshold:
                self.write_log(
                    f"空头开仓信号: momentum={self.momentum:.6f} < "
                    f"-threshold={-self.mom_threshold}"
                )
                self.short(bar.close_price, size)

        elif self.pos > 0:  # 多头持仓 → 标准差追踪止损
            # 计算移动止损价格
            long_stop = self.intra_trade_high - self.trailing_std_scale * self.std_value

            # 当价格跌破止损线时平仓
            if bar.close_price < long_stop:
                self.write_log(
                    f"触发多头止损: 当前价={bar.close_price:.4f}, "
                    f"止损线={long_stop:.4f}"
                )
                self.sell(bar.close_price, abs(self.pos))

        elif self.pos < 0:  # 空头持仓 → 标准差追踪止损
            # 计算移动止损价格
            short_stop = self.intra_trade_low + self.trailing_std_scale * self.std_value

            # 当价格突破止损线时平仓
            if bar.close_price > short_stop:
                self.write_log(
                    f"触发空头止损: 当前价={bar.close_price:.4f}, "
                    f"止损线={short_stop:.4f}"
                )
                self.cover(bar.close_price, abs(self.pos))

    def on_order(self, order: ap.OrderData):
        """委托回调"""
        return

    def on_trade(self, trade: ap.TradeData):
        """成交回调"""
        self.write_log(
            f"成交: {trade.direction} {trade.offset} {trade.volume}@{trade.price}"
        )


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

    # 验证CSV文件是否存在
    csv_path = "/Users/bobbyding/Documents/GitHub/apilot/data/SOL-USDT_LOCAL_1m.csv"
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return None, None

    # 检查CSV文件内容
    print(f"读取CSV文件: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV文件行数: {len(df)}")
        print(f"CSV文件列名: {df.columns.tolist()}")
        print(f"第一行数据: {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        import traceback

        traceback.print_exc()
        return None, None

    # 创建回测引擎
    engine = ap.BacktestingEngine()

    # 设置引擎参数
    engine.set_parameters(
        vt_symbols=["SOL-USDT.LOCAL"],
        interval="1m",
        start=start,
        end=end,
        rates={"SOL-USDT.LOCAL": 0.00075},  # 万7.5手续费
        sizes={"SOL-USDT.LOCAL": 1},
        priceticks={"SOL-USDT.LOCAL": 0.001},
        capital=init_cash,
    )

    # 3 添加策略
    engine.add_strategy(
        strategy_class,
        {
            "std_period": std_period,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale,
        },
    )

    # 3 添加CSV数据
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

    # 4 运行回测
    engine.run_backtesting()

    # 5 计算结果和统计指标
    df = engine.calculate_result()
    stats = engine.calculate_statistics()

    return df, stats


def evaluate_with_parameters(params: Dict) -> Tuple[Dict, Dict]:
    """
    用于遗传算法优化
    """
    # 确保参数为正确类型
    std_period = int(params.get("std_period", 20))
    mom_threshold = float(params.get("mom_threshold", 0.02))
    trailing_std_scale = float(params.get("trailing_std_scale", 2.0))

    # 运行回测
    df, stats = run_backtesting(
        StdMomentumStrategy,
        init_cash=100000,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 6, 30),
        std_period=std_period,
        mom_threshold=mom_threshold,
        trailing_std_scale=trailing_std_scale,
    )

    # 处理空结果
    if df is None or df.empty:
        metrics = {
            "sharpe_ratio": -999,
            "total_return": -999,
            "max_drawdown": 1,
        }
        return params, metrics

    # 调用回测引擎获取统计数据
    statistics = stats

    # 提取关键指标
    metrics = {
        "sharpe_ratio": statistics["sharpe_ratio"],
        "total_return": statistics["total_return"],
        "max_drawdown": abs(statistics["max_ddpercent"]),  # 使用正数便于排序
        "total_trade_count": statistics["total_trade_count"],
    }

    return params, metrics


def get_sharpe_ratio(result):
    """提取夏普比率作为优化排序依据"""
    try:
        return result[1]["sharpe_ratio"]
    except (TypeError, KeyError, IndexError):
        return -999


def run_optimization_ga():
    """使用遗传算法优化策略参数"""
    # 设置优化参数范围
    setting = ap.OptimizationSetting()
    setting.add_parameter("std_period", 10, 100, 5)
    setting.add_parameter("mom_threshold", 0.01, 0.15, 0.01)
    setting.add_parameter("trailing_std_scale", 1, 10, 1)

    # 运行遗传算法优化
    results = ap.run_ga_optimization(
        evaluate_func=evaluate_with_parameters,
        optimization_setting=setting,
        key_func=get_sharpe_ratio,
        max_workers=8,
        population_size=16,
        ngen_size=5,
    )

    # 获取并打印最佳结果
    if not results:
        return {}

    best_result = results[0]
    best_params, best_metrics = best_result

    # 打印最佳参数和性能指标
    print("\n最佳参数组合:")
    for name, value in best_params.items():
        if name == "std_period":
            print(f"{name}: {int(value)}")
        else:
            print(f"{name}: {float(value):.4f}")

    print("\n关键性能指标:")
    metrics_names = {
        "sharpe_ratio": "夏普比率",
        "total_return": "总收益率",
        "max_drawdown": "最大回撤",
    }
    for key, label in metrics_names.items():
        if key in best_metrics:
            print(f"{label}: {best_metrics[key]:.4f}")

    return best_params


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

    # 输出结果
    if df is not None:
        print(df)
        print(stats)
