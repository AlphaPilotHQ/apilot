"""
改造成这样子:
监控全部的标的,出现信号买入1%,可以继续加仓,最多到4%.
可以一直选币,并给出
但是这里仓位管理比较复杂,没有想好怎么做
"""

from datetime import datetime
from typing import ClassVar

import numpy as np

import apilot as ap
from apilot.utils.logger import get_logger, set_level

# 获取日志记录器
logger = get_logger("turtle_strategy")
set_level("info", "turtle_strategy")


class TurtleSignalStrategy(ap.PATemplate):
    """
    海龟交易信号策略

    基于唐奇安通道(Donchian Channel)的趋势跟踪系统,是经典海龟交易法则的实现.
    该策略使用长短两个周期的通道,结合ATR进行仓位管理和止损设置.
    """

    # 策略参数
    entry_window: ClassVar[int] = 20  # 入场通道周期,20天
    exit_window: ClassVar[int] = 10  # 出场通道周期,10天
    atr_window: ClassVar[int] = 20  # ATR计算周期,20天
    fixed_size: ClassVar[int] = 1  # 每次交易的基础单位

    # 策略变量
    entry_up: ClassVar[float] = 0  # 入场通道上轨(最高价)
    entry_down: ClassVar[float] = 0  # 入场通道下轨(最低价)
    exit_up: ClassVar[float] = 0  # 出场通道上轨
    exit_down: ClassVar[float] = 0  # 出场通道下轨
    atr_value: ClassVar[float] = 0  # ATR值,用于仓位管理和设置止损

    long_entry: ClassVar[float] = 0  # 多头入场价
    short_entry: ClassVar[float] = 0  # 空头入场价
    long_stop: ClassVar[float] = 0  # 多头止损价
    short_stop: ClassVar[float] = 0  # 空头止损价

    # 参数和变量列表,用于UI显示和参数优化
    parameters: ClassVar[list[str]] = [
        "entry_window",
        "exit_window",
        "atr_window",
        "fixed_size",
    ]
    variables: ClassVar[list[str]] = [
        "entry_up",
        "entry_down",
        "exit_up",
        "exit_down",
        "atr_value",
    ]

    def __init__(self, pa_engine, strategy_name, symbol, setting):
        """初始化"""
        super().__init__(pa_engine, strategy_name, symbol, setting)

        # 创建K线生成器和数据管理器
        self.bg = ap.BarGenerator(self.on_bar)
        self.am = ap.ArrayManager()

    def on_init(self):
        """
        策略初始化回调函数
        """
        # 使用引擎的日志方法记录初始化信息

    def on_start(self):
        """
        策略启动回调函数
        """

    def on_stop(self):
        """
        策略停止回调函数
        """

    def on_tick(self, tick: ap.TickData):
        """
        Tick数据更新回调函数
        """
        # 将TICK数据更新至K线生成器
        self.bg.update_tick(tick)

    def on_bar(self, bar: ap.BarData):
        """
        K线数据更新回调函数 - 策略的核心交易逻辑
        """
        # 更新K线数据到数组管理器
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 撤销之前发出的所有订单
        self.cancel_all()

        # 获取当前交易品种的持仓
        current_pos = self.get_pos(self.symbols[0])

        # 只有在没有持仓时才计算新的入场通道
        if self.is_pos_equal(current_pos, 0):  # 使用辅助方法
            # 计算唐奇安通道上下轨(N日最高价和最低价)
            self.entry_up, self.entry_down = self.am.donchian(self.entry_window)

        # 始终计算短周期出场通道
        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

        # 计算ATR值用于风险管理 - 在每个bar都更新ATR，确保风险管理计算准确
        self.atr_value = self.am.atr(self.atr_window)

        # 交易逻辑:根据持仓情况分别处理
        symbol = self.symbols[0]  # 获取当前交易的品种
        current_pos = self.get_pos(symbol)

        # 无持仓状态
        if self.is_pos_equal(current_pos, 0):
            # 重置入场价和止损价
            self.long_entry = 0
            self.short_entry = 0
            self.long_stop = 0
            self.short_stop = 0

            # 发送多头和空头突破订单
            self.send_buy_orders(self.entry_up)  # 上轨突破做多
            self.send_short_orders(self.entry_down)  # 下轨突破做空

        # 持有多头仓位
        elif self.is_pos_greater(current_pos, 0):
            # 检查是否需要出场 - 价格低于出场通道下轨或者跌破止损价
            if bar.close_price < self.exit_down or bar.close_price < self.long_stop:
                self.sell(symbol, bar.close_price, abs(current_pos))
                return
            # 继续加仓逻辑,突破新高继续做多
            self.send_buy_orders(self.entry_up)

            # 多头止损逻辑:取ATR止损价和10日最低价的较大值
            sell_price = max(self.long_stop, self.exit_down)
            self.sell(symbol, sell_price, abs(current_pos), True)  # 平多仓位

        # 持有空头仓位
        elif self.is_pos_less(current_pos, 0):
            # 检查是否需要出场 - 价格高于出场通道上轨或者突破止损价
            if bar.close_price > self.exit_up or bar.close_price > self.short_stop:
                self.buy(symbol, bar.close_price, abs(current_pos))
                return
            # 继续加仓逻辑,突破新低继续做空
            self.send_short_orders(self.entry_down)

            # 空头止损逻辑:取ATR止损价和10日最高价的较小值
            cover_price = min(self.short_stop, self.exit_up)
            self.buy(symbol, cover_price, abs(current_pos), True)  # 平空仓位

    def on_trade(self, trade: ap.TradeData):
        """
        成交回调函数:记录成交价并设置止损价
        """
        if trade.direction == ap.Direction.LONG:
            # 多头成交,设置多头止损
            self.long_entry = trade.price  # 记录多头入场价格
            self.long_stop = (
                self.long_entry - 2 * self.atr_value
            )  # 止损设置在入场价格下方2倍ATR处
            logger.info(
                f"多头成交: {trade.symbol} {trade.orderid} {trade.volume}@{trade.price}, 止损价: {self.long_stop}"
            )
        else:
            # 空头成交,设置空头止损
            self.short_entry = trade.price  # 记录空头入场价格
            self.short_stop = (
                self.short_entry + 2 * self.atr_value
            )  # 止损设置在入场价格上方2倍ATR处
            logger.info(
                f"空头成交: {trade.symbol} {trade.orderid} {trade.volume}@{trade.price}, 止损价: {self.short_stop}"
            )

    def on_order(self, order: ap.OrderData):
        """
        委托回调函数
        """
        # 委托回调函数不需要实现,但将来可能需要添加逻辑

    def send_buy_orders(self, price):
        """
        发送多头委托,包括首次入场和金字塔式加仓

        海龟系统的特点之一是金字塔式逐步加仓,最多加仓至4个单位
        """
        symbol = self.symbols[0]  # 获取当前交易的品种

        # 计算当前持仓的单位数
        pos = self.get_pos(symbol)
        # 处理可能的numpy数组
        if isinstance(pos, np.ndarray):
            pos = float(pos)
        t = pos / self.fixed_size

        # 第一个单位:在通道突破点入场
        if isinstance(t, np.ndarray):
            t = float(t)  # 确保t是标量值

        if t < 1:
            self.buy(symbol, price, self.fixed_size, True)

        # 第二个单位:在第一个单位价格基础上加0.5个ATR
        if t < 2:
            self.buy(symbol, price + self.atr_value * 0.5, self.fixed_size, True)

        # 第三个单位:在第一个单位价格基础上加1个ATR
        if t < 3:
            self.buy(symbol, price + self.atr_value, self.fixed_size, True)

        # 第四个单位:在第一个单位价格基础上加1.5个ATR
        if t < 4:
            self.buy(symbol, price + self.atr_value * 1.5, self.fixed_size, True)

    def send_short_orders(self, price):
        """
        发送空头委托,包括首次入场和金字塔式加仓

        与多头逻辑相反,价格逐步下降
        """
        symbol = self.symbols[0]  # 获取当前交易的品种

        # 计算当前持仓的单位数
        pos = self.get_pos(symbol)
        # 处理可能的numpy数组
        if isinstance(pos, np.ndarray):
            pos = float(pos)
        t = pos / self.fixed_size

        # 第一个单位:在通道突破点入场
        if isinstance(t, np.ndarray):
            t = float(t)  # 确保t是标量值

        if t > -1:
            self.sell(symbol, price, self.fixed_size, True)

        # 第二个单位:在第一个单位价格基础上减0.5个ATR
        if t > -2:
            self.sell(symbol, price - self.atr_value * 0.5, self.fixed_size, True)

        # 第三个单位:在第一个单位价格基础上减1个ATR
        if t > -3:
            self.sell(symbol, price - self.atr_value, self.fixed_size, True)

        # 第四个单位:在第一个单位价格基础上减1.5个ATR
        if t > -4:
            self.sell(symbol, price - self.atr_value * 1.5, self.fixed_size, True)


@ap.log_exceptions()
def run_backtesting(show_chart=True):
    """
    运行海龟信号策略回测
    """
    # 初始化回测引擎
    bt_engine = ap.BacktestingEngine()
    logger.info("1 创建回测引擎完成")

    # 设置回测参数
    bt_engine.set_parameters(
        symbols=["SOL-USDT.LOCAL"],  # 需要使用列表形式
        interval=ap.Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 6, 30),  # 使用半年数据回测
    )
    logger.info("2 设置引擎参数完成")

    # 添加策略
    bt_engine.add_strategy(
        TurtleSignalStrategy,
        {"entry_window": 20, "exit_window": 10, "atr_window": 20, "fixed_size": 1},
    )

    # 添加数据 - 使用基于索引的数据加载方法
    bt_engine.add_csv_data(
        symbol="SOL-USDT.LOCAL",
        filepath="data/SOL-USDT_LOCAL_1m.csv",
        datetime_index=0,
        open_index=1,
        high_index=2,
        low_index=3,
        close_index=4,
        volume_index=5,
    )

    # 运行回测
    bt_engine.run_backtesting()
    logger.info("5 运行回测完成")

    # 计算结果并生成报告
    bt_engine.report()
    logger.info("6 计算结果和报告生成完成")

    # 如果不显示图表，记录日志
    if not show_chart:
        logger.info("图表显示已跳过")
    logger.info("7 显示图表完成")

    # 计算并获取数据用于返回
    df = bt_engine.daily_df
    stats = bt_engine.calculate_statistics(output=False)
    return df, stats, bt_engine


if __name__ == "__main__":
    run_backtesting(show_chart=True)
