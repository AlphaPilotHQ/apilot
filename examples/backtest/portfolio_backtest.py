"""
加密货币投资组合回测示例 - 使用BTC和SOL
特别适合在crypto市场的量化交易分析
"""
from init_env import *
from datetime import datetime
import os
import shutil
import pandas as pd

# 导入所需模块和图形库
from apilot.trader.constant import Interval, Exchange

from apilot.portfolio_strategy.backtesting import BacktestingEngine

# 导入CTA模板类
from apilot.trader.object import TickData, BarData, TradeData, OrderData
from apilot.trader.utility import BarGenerator, ArrayManager
from apilot.portfolio_strategy.template import StrategyTemplate

class StdMomentumStrategy(StrategyTemplate):
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
       - 市场波动大时止损距离更远，波动小时止损更紧
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
        self.am = ArrayManager(size=200)  # 增加大小以支持长周期计算

        # 初始化指标
        self.momentum = 0.0        # 动量
        self.std_value = 0.0       # 标准差

        # 追踪最高/最低价
        self.intra_trade_high = 0
        self.intra_trade_low = 0

    def on_init(self):
        self.write_log("策略初始化")
        self.write_log(f"加载历史数据: {self.std_period * 2}根K线")
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
        self.write_log(f"收到1分钟K线: {bar.datetime} O:{bar.open_price} H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}")
        self.bg.update_bar(bar)

    def on_5min_bar(self, bar: BarData):
        """5分钟K线数据更新，包含交易逻辑"""
        self.write_log(f"生成5分钟K线: {bar.datetime} O:{bar.open_price} H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}")
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

        self.write_log(f"指标计算: std={self.std_value:.6f}, momentum={self.momentum:.6f}, threshold={self.mom_threshold}")

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
                self.write_log(f"多头开仓信号: momentum={self.momentum:.6f} > threshold={self.mom_threshold}")
                self.buy(bar.close_price, size)
            elif self.momentum < -self.mom_threshold:
                self.write_log(f"空头开仓信号: momentum={self.momentum:.6f} < -threshold={-self.mom_threshold}")
                self.short(bar.close_price, size)

        elif self.pos > 0:  # 多头持仓 → 标准差追踪止损 TODO：这里stop=Ture删了，止损没用了
            # 计算移动止损价格
            long_stop = self.intra_trade_high - self.trailing_std_scale * self.std_value

            # 当价格跌破止损线时平仓
            if bar.close_price < long_stop:
                self.write_log(f"触发多头止损: 当前价={bar.close_price:.4f}, 止损线={long_stop:.4f}")
                self.sell(bar.close_price, abs(self.pos))
                # self.write_log(f"触发多头止损: 当前价={bar.close_price:.4f}, 止损线={long_stop:.4f}")

        elif self.pos < 0:  # 空头持仓 → 标准差追踪止损
            # 计算移动止损价格
            short_stop = self.intra_trade_low + self.trailing_std_scale * self.std_value

            # 当价格突破止损线时平仓
            if bar.close_price > short_stop:
                self.write_log(f"触发空头止损: 当前价={bar.close_price:.4f}, 止损线={short_stop:.4f}")
                self.cover(bar.close_price, abs(self.pos))
                # self.write_log(f"触发空头止损: 当前价={bar.close_price:.4f}, 止损线={short_stop:.4f}")

    def on_order(self, order: OrderData):
        """委托回调"""
        pass

    def on_trade(self, trade: TradeData):
        """成交回调"""
        self.write_log(f"成交: {trade.direction} {trade.offset} {trade.volume}@{trade.price}")


def run_backtesting(
    strategy_class,
    init_cash=100000,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 6, 30),
    std_period=20,
    mom_threshold=0.05,
    trailing_std_scale=4.0
    ):

    # 1 创建回测引擎
    engine = BacktestingEngine()

    # 准备数据文件 - 将数据文件复制到系统预期的位置
    # 读取源数据文件
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    src_btc_path = os.path.join(ROOT_DIR, "data", "BTC-USDT_LOCAL_1m.csv")
    src_sol_path = os.path.join(ROOT_DIR, "data", "SOL-USDT_LOCAL_1m.csv")

    # 创建CSV数据库目录结构 (按照预期的目录结构组织)
    csv_database_dir = os.path.join(ROOT_DIR, "csv_database")
    btc_data_dir = os.path.join(csv_database_dir, "bar", "BTC-USDT", "LOCAL")
    sol_data_dir = os.path.join(csv_database_dir, "bar", "SOL-USDT", "LOCAL")

    os.makedirs(btc_data_dir, exist_ok=True)
    os.makedirs(sol_data_dir, exist_ok=True)

    # 目标文件路径 (按照预期的目录结构)
    dst_btc_path = os.path.join(btc_data_dir, "1m.csv")
    dst_sol_path = os.path.join(sol_data_dir, "1m.csv")

    # 检查并处理BTC数据
    if os.path.exists(src_btc_path):
        # 读取数据并按照适当的格式保存
        try:
            df_btc = pd.read_csv(src_btc_path)
            # 确保列名正确
            df_btc = df_btc.rename(columns={
                'candle_begin_time': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            # 仅保留需要的列
            df_btc = df_btc[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            # 保存为正确格式的CSV
            df_btc.to_csv(dst_btc_path, index=False)
            print(f"已处理BTC数据文件并保存到: {dst_btc_path}")
        except Exception as e:
            print(f"处理BTC数据文件时出错: {e}")
            return
    else:
        print(f"错误: 找不到BTC数据文件: {src_btc_path}")
        return

    # 检查并处理SOL数据
    if os.path.exists(src_sol_path):
        # 读取数据并按照适当的格式保存
        try:
            df_sol = pd.read_csv(src_sol_path)
            # 确保列名正确
            df_sol = df_sol.rename(columns={
                'candle_begin_time': 'datetime',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            # 仅保留需要的列
            df_sol = df_sol[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            # 保存为正确格式的CSV
            df_sol.to_csv(dst_sol_path, index=False)
            print(f"已处理SOL数据文件并保存到: {dst_sol_path}")
        except Exception as e:
            print(f"处理SOL数据文件时出错: {e}")
            return
    else:
        print(f"错误: 找不到SOL数据文件: {src_sol_path}")
        return

    # 2 设置引擎参数
    engine.set_parameters(
        vt_symbols=["BTC-USDT.LOCAL", "SOL-USDT.LOCAL"],  # BTC和SOL使用LOCAL交易所
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 6, 30),
        rates={                    # 手续费率
            "BTC-USDT.LOCAL": 0.0005,
            "SOL-USDT.LOCAL": 0.0005,  # 添加SOL手续费率
        },
        sizes={                    # 合约乘数
            "BTC-USDT.LOCAL": 1,
            "SOL-USDT.LOCAL": 100
        },
        priceticks={               # 价格最小变动
            "BTC-USDT.LOCAL": 0.01,
            "SOL-USDT.LOCAL": 0.001
        },
        capital=100000,            # 初始资金
    )

    # 添加动量策略，使用已定义的StdMomentumStrategy
    setting = {
        "std_period": 20,          # 标准差周期
        "mom_threshold": 0.02,     # 动量阈值
    }

    engine.add_strategy(StdMomentumStrategy, setting)

    # 加载历史数据
    engine.load_data()

    # 运行回测
    engine.run_backtesting()

    # 计算并输出回测结果
    df = engine.calculate_result()
    engine.calculate_statistics()

    # 显示回测图表
    engine.show_chart()

    # 参数优化示例 (注释掉，需要时可以解开使用)
    """
    # 设置优化参数
    setting = OptimizationSetting()
    setting.set_target("sharpe_ratio")   # 优化目标 - 夏普比率
    setting.add_parameter("std_period", 10, 30, 5)        # 参数范围
    setting.add_parameter("mom_threshold", 0.02, 0.1, 0.01)

    # 运行优化
    result = engine.run_optimization(setting, 20)

    # 输出优化结果
    for strategy_setting in result:
        print(f"参数: {strategy_setting}")
    """


if __name__ == "__main__":
    run_backtesting()
