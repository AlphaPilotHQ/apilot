"""
加密货币投资组合回测示例 - 使用BTC和SOL
特别适合在crypto市场的量化交易分析
"""
from init_env import *
from datetime import datetime

# 导入所需模块和图形库
from apilot.trader.constant import Interval, Exchange

# 导入portfolio_strategy模块 - 使用已移植的模块
from apilot.portfolio_strategy.backtesting import BacktestingEngine
from apilot.portfolio_strategy.strategies.trend_following_strategy import TrendFollowingStrategy
from apilot.trader.optimize import OptimizationSetting

class StdMomentumStrategy(CtaTemplate):
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

        # 计算标准差
        if not am.inited:
            return

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




def run_crypto_portfolio_backtest():
    """
    运行加密货币投资组合回测
    """
    print("开始运行加密货币投资组合回测...")

    # 创建回测引擎
    engine = BacktestingEngine()

    # 设置回测参数
    engine.set_parameters(
        vt_symbols=["BTC-USDT.LOCAL", "SOL-USDT.LOCAL"],  # BTC和SOL使用LOCAL交易所
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 6, 30),
        rates={                    # 手续费率
            "BTC-USDT.LOCAL": 0.0005,
            "SOL-USDT.LOCAL": 0.0005
        },
        sizes={                    # 合约乘数
            "BTC-USDT.LOCAL": 1,
            "SOL-USDT.LOCAL": 1
        },
        priceticks={               # 价格最小变动
            "BTC-USDT.LOCAL": 0.01,
            "SOL-USDT.LOCAL": 0.001
        },
        capital=100000,            # 初始资金
    )

    # 添加趋势跟踪策略
    setting = {
        "window": 20,              # 趋势窗口
        "open_std": 2.0,           # 开仓标准差
        "close_std": 1.0,          # 平仓标准差
        "fixed_size": 1            # 固定头寸
    }

    engine.add_strategy(TrendFollowingStrategy, setting)

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
    setting.add_parameter("window", 10, 30, 5)  # 参数范围
    setting.add_parameter("open_std", 1.0, 3.0, 0.5)
    setting.add_parameter("close_std", 0.5, 2.0, 0.5)

    # 执行参数优化
    result = engine.run_optimization(setting)

    # 输出优化结果
    for strategy_setting, target_value in result:
        print(f"参数: {strategy_setting}, 目标值: {target_value}")
    """


if __name__ == "__main__":
    run_crypto_portfolio_backtest()
