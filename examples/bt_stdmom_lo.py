"""
动量策略回测与优化示例

此模块实现了一个基于动量指标的趋势跟踪策略,结合标准差动态止损.
包含单次回测和参数优化功能,支持使用网格搜索寻找最优参数组合.
"""

from datetime import datetime
from typing import ClassVar

import apilot as ap
from apilot.utils.logger import get_logger, set_level

# 获取日志记录器
logger = get_logger("momentum_strategy")
set_level("info", "momentum_strategy")


class StdMomentumStrategy(ap.PATemplate):
    """
    策略逻辑:
    1. 核心思路:结合动量信号与标准差动态止损的中长期趋势跟踪策略
    2. 入场信号:
       - 基于动量指标(当前价格/N周期前价格-1)生成交易信号
       - 动量 > 阈值(5%)时做多
       - 使用全部账户资金进行头寸管理
    3. 风险管理:
       - 使用基于标准差的动态追踪止损
       - 多头持仓:止损设置在最高价-4倍标准差
    """

    # 策略参数
    std_period = 48
    mom_threshold = 0.05
    trailing_std_scale = 4

    parameters: ClassVar[list[str]] = [
        "std_period",
        "mom_threshold",
        "trailing_std_scale",
    ]
    variables: ClassVar[list[str]] = [
        "momentum",
        "intra_trade_high",
        "pos",
    ]

    def __init__(self, pa_engine, strategy_name, symbols, setting):
        super().__init__(pa_engine, strategy_name, symbols, setting)
        # 使用增强版BarGenerator实例处理所有交易标的，添加symbols支持多标的同步
        self.bg = ap.BarGenerator(
            self.on_bar,
            5,
            self.on_5min_bar,
            symbols=self.symbols,  # 传入交易标的列表，启用多标的同步
        )

        # 为每个交易对创建ArrayManager
        self.ams = {}
        for symbol in self.symbols:
            self.ams[symbol] = ap.ArrayManager(
                size=200
            )  # 保留最长200bar TODO：改成动态

        # 为每个交易对创建状态跟踪字典
        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.pos = {}

        # 初始化每个交易对的状态
        for symbol in self.symbols:
            self.momentum[symbol] = 0.0
            self.std_value[symbol] = 0.0
            self.intra_trade_high[symbol] = 0
            self.pos[symbol] = 0

    def on_init(self):
        self.load_bar(self.std_period)

    def on_bar(self, bar: ap.BarData):
        """
        处理K线数据
        """
        # 将K线传递给BarGenerator进行聚合
        self.bg.update_bar(bar)

    def on_5min_bar(self, bars: dict):
        self.cancel_all()

        # 记录收到的多标的K线数据
        logger.debug(f"on_5min_bar收到完整的多标的数据: {list(bars.keys())}")

        # 对每个交易品种执行数据更新和交易逻辑
        for symbol, bar in bars.items():
            if symbol not in self.ams:
                logger.debug(f"忽略标的 {symbol}, 因为它不在ams中")
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            # 如果数据不足，跳过交易逻辑
            if not am.inited:
                continue

            # 计算技术指标
            self.std_value[symbol] = am.std(self.std_period)

            # 计算动量因子
            if len(am.close_array) > self.std_period + 1:
                old_price = am.close_array[-self.std_period - 1]
                current_price = am.close_array[-1]
                self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1

            # 获取当前持仓
            current_pos = self.pos.get(symbol, 0)

            # 持仓状态下更新跟踪止损价格
            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )

            # 交易逻辑
            if current_pos == 0:
                # 初始化追踪价格
                self.intra_trade_high[symbol] = bar.high_price

                # 平均分配资金给所有标的（全仓交易）
                risk_percent = 1 / len(self.symbols)
                # 使用当前账户价值而非初始资本
                current_capital = self.pa_engine.get_current_capital()
                capital_to_use = current_capital * risk_percent
                size = max(1, int(capital_to_use / bar.close_price))

                # 基于动量信号开仓
                logger.debug(
                    f"{bar.datetime}: {symbol} 动量值 {self.momentum[symbol]:.4f}, 阈值 {self.mom_threshold:.4f}"
                )

                if self.momentum[symbol] > self.mom_threshold:
                    logger.debug(
                        f"{bar.datetime}: {symbol} 发出多头信号: 动量 {self.momentum[symbol]:.4f} > 阈值 {self.mom_threshold}"
                    )
                    self.buy(symbol=symbol, price=bar.close_price, volume=size)

            elif current_pos > 0:  # 多头持仓 → 标准差追踪止损
                # 计算移动止损价格
                long_stop = (
                    self.intra_trade_high[symbol]
                    - self.trailing_std_scale * self.std_value[symbol]
                )

                # 当价格跌破止损线时平仓
                if bar.close_price < long_stop:
                    self.sell(
                        symbol=symbol, price=bar.close_price, volume=abs(current_pos)
                    )

            # 已删除空头持仓逻辑

    def on_order(self, order: ap.OrderData):
        """委托回调"""
        pass

    def on_trade(self, trade: ap.TradeData):
        """成交回调"""
        symbol = trade.symbol

        # 更新持仓
        position_change = (
            trade.volume if trade.direction == ap.Direction.LONG else -trade.volume
        )
        self.pos[symbol] = self.pos.get(symbol, 0) + position_change

        # 更新最高/最低价追踪
        current_pos = self.pos[symbol]

        # 只在仓位方向存在时更新对应的跟踪价格
        if current_pos > 0:
            # 多头仓位,更新最高价
            self.intra_trade_high[symbol] = max(
                self.intra_trade_high.get(symbol, trade.price), trade.price
            )

        logger.debug(
            f"Trade: {symbol} {trade.orderid} {trade.direction} "
            f"{trade.volume}@{trade.price}, current_pos: {current_pos}"
        )


@ap.log_exceptions()
def run_backtesting(
    strategy_class=StdMomentumStrategy,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 6, 29),
    std_period=48,
    mom_threshold=0.02,
    trailing_std_scale=2.0,
):
    # 1 创建回测引擎
    engine = ap.BacktestingEngine()
    logger.info("1 创建回测引擎完成")

    # 2 设置引擎参数
    symbols = ["SOL-USDT.LOCAL", "BTC-USDT.LOCAL"]

    engine.set_parameters(
        symbols=symbols,
        interval=ap.Interval.MINUTE,
        start=start,
        end=end,
    )
    logger.info("2 设置引擎参数完成")

    # 3 添加策略
    engine.add_strategy(
        strategy_class,
        {
            "std_period": std_period,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale,
        },
    )
    logger.info("3 添加策略完成")

    # 4 添加数据
    engine.add_csv_data(
        symbol="SOL-USDT.LOCAL",
        filepath="data/SOL-USDT_LOCAL_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )
    engine.add_csv_data(
        symbol="BTC-USDT.LOCAL",
        filepath="data/BTC-USDT_LOCAL_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )
    logger.info("4 添加数据完成")

    # 6 运行回测
    engine.run_backtesting()
    logger.info("5 运行回测完成")

    # 7 计算结果并生成报告
    engine.report()
    logger.info("6 计算结果和报告生成完成")

    return engine


if __name__ == "__main__":
    # 单次回测模式
    run_backtesting()
