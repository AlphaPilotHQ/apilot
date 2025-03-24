"""
改造成这样子：
监控全部的标的，出现信号买入1%，可以继续加仓，最多到4%。
可以一直选币，并给出
但是这里仓位管理比较复杂，没有想好怎么做
"""

from init_env import *
from datetime import datetime

from apilot.core.constant import Direction, Interval
from apilot.core.object import BarData, TickData, OrderData, TradeData
from apilot.engine import CtaTemplate, BacktestingEngine
from apilot.core.utility import BarGenerator, ArrayManager


class TurtleSignalStrategy(CtaTemplate):
    """
    海龟交易信号策略

    基于唐奇安通道(Donchian Channel)的趋势跟踪系统，是经典海龟交易法则的实现。
    该策略使用长短两个周期的通道，结合ATR进行仓位管理和止损设置。
    """

    # 策略参数
    entry_window = 20  # 入场通道周期，20天
    exit_window = 10  # 出场通道周期，10天
    atr_window = 20  # ATR计算周期，20天
    fixed_size = 1  # 每次交易的基础单位

    # 策略变量
    entry_up = 0  # 入场通道上轨（最高价）
    entry_down = 0  # 入场通道下轨（最低价）
    exit_up = 0  # 出场通道上轨
    exit_down = 0  # 出场通道下轨
    atr_value = 0  # ATR值，用于仓位管理和设置止损

    long_entry = 0  # 多头入场价
    short_entry = 0  # 空头入场价
    long_stop = 0  # 多头止损价
    short_stop = 0  # 空头止损价

    # 参数和变量列表，用于UI显示和参数优化
    parameters = ["entry_window", "exit_window", "atr_window", "fixed_size"]
    variables = ["entry_up", "entry_down", "exit_up", "exit_down", "atr_value"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """初始化策略"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        # 创建K线生成器和数据管理器
        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_init(self):
        """
        策略初始化回调函数
        """
        # 使用引擎的日志方法记录初始化信息
        self.cta_engine.write_log("海龟信号策略初始化")

    def on_start(self):
        """
        策略启动回调函数
        """
        self.cta_engine.write_log("海龟信号策略启动")

    def on_stop(self):
        """
        策略停止回调函数
        """
        self.cta_engine.write_log("海龟信号策略停止")

    def on_tick(self, tick: TickData):
        """
        Tick数据更新回调函数
        """
        # 将TICK数据更新至K线生成器
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        K线数据更新回调函数 - 策略的核心交易逻辑
        """
        # 撤销之前发出的所有订单
        self.cancel_all()

        # 更新K线数据到数组管理器
        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 只有在没有持仓时才计算新的入场通道
        if not self.pos:
            # 计算唐奇安通道上下轨（N日最高价和最低价）
            self.entry_up, self.entry_down = self.am.donchian(self.entry_window)

        # 始终计算短周期出场通道
        self.exit_up, self.exit_down = self.am.donchian(self.exit_window)

        # 交易逻辑：根据持仓情况分别处理
        if not self.pos:  # 无持仓状态
            # 计算ATR值用于风险管理
            self.atr_value = self.am.atr(self.atr_window)

            # 重置入场价和止损价
            self.long_entry = 0
            self.short_entry = 0
            self.long_stop = 0
            self.short_stop = 0

            # 发送多头和空头突破订单
            self.send_buy_orders(self.entry_up)  # 上轨突破做多
            self.send_short_orders(self.entry_down)  # 下轨突破做空

        elif self.pos > 0:  # 持有多头仓位
            # 继续加仓逻辑，突破新高继续做多
            self.send_buy_orders(self.entry_up)

            # 多头止损逻辑：取ATR止损价和10日最低价的较大值
            sell_price = max(self.long_stop, self.exit_down)
            self.sell(sell_price, abs(self.pos), True)  # 平多仓位

        elif self.pos < 0:  # 持有空头仓位
            # 继续加仓逻辑，突破新低继续做空
            self.send_short_orders(self.entry_down)

            # 空头止损逻辑：取ATR止损价和10日最高价的较小值
            cover_price = min(self.short_stop, self.exit_up)
            self.cover(cover_price, abs(self.pos), True)  # 平空仓位

        # 移除对put_event的调用，该方法在回测环境下不可用
        # self.put_event()

    def on_trade(self, trade: TradeData):
        """
        成交回调函数：记录成交价并设置止损价
        """
        if trade.direction == Direction.LONG:  # 多头成交
            # 记录多头入场价
            self.long_entry = trade.price
            # 设置多头止损价为入场价减去2倍ATR
            self.long_stop = self.long_entry - 2 * self.atr_value
        else:  # 空头成交
            # 记录空头入场价
            self.short_entry = trade.price
            # 设置空头止损价为入场价加上2倍ATR
            self.short_stop = self.short_entry + 2 * self.atr_value

    def on_order(self, order: OrderData):
        """
        委托回调函数
        """
        pass

    def send_buy_orders(self, price):
        """
        发送多头委托，包括首次入场和金字塔式加仓

        海龟系统的特点之一是金字塔式逐步加仓，最多加仓至4个单位
        """
        # 计算当前持仓的单位数
        t = self.pos / self.fixed_size

        # 第一个单位：在通道突破点入场
        if t < 1:
            self.buy(price, self.fixed_size, True)

        # 第二个单位：在第一个单位价格基础上加0.5个ATR
        if t < 2:
            self.buy(price + self.atr_value * 0.5, self.fixed_size, True)

        # 第三个单位：在第一个单位价格基础上加1个ATR
        if t < 3:
            self.buy(price + self.atr_value, self.fixed_size, True)

        # 第四个单位：在第一个单位价格基础上加1.5个ATR
        if t < 4:
            self.buy(price + self.atr_value * 1.5, self.fixed_size, True)

    def send_short_orders(self, price):
        """
        发送空头委托，包括首次入场和金字塔式加仓

        与多头逻辑相反，价格逐步下降
        """
        # 计算当前持仓的单位数
        t = self.pos / self.fixed_size

        # 第一个单位：在通道突破点入场
        if t > -1:
            self.short(price, self.fixed_size, True)

        # 第二个单位：在第一个单位价格基础上减0.5个ATR
        if t > -2:
            self.short(price - self.atr_value * 0.5, self.fixed_size, True)

        # 第三个单位：在第一个单位价格基础上减1个ATR
        if t > -3:
            self.short(price - self.atr_value, self.fixed_size, True)

        # 第四个单位：在第一个单位价格基础上减1.5个ATR
        if t > -4:
            self.short(price - self.atr_value * 1.5, self.fixed_size, True)


def run_backtesting(show_chart=True):
    """
    运行海龟信号策略回测
    """
    # 初始化回测引擎
    engine = BacktestingEngine()

    # 设置回测参数
    engine.set_parameters(
        vt_symbol="SOL-USDT.LOCAL",  # 修改为与CSV文件名匹配的交易对
        interval=Interval.MINUTE,
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        rate=0.0001,
        size=1,
        pricetick=0.01,
        capital=100000,
    )

    # 添加策略
    engine.add_strategy(
        TurtleSignalStrategy,
        {"entry_window": 20, "exit_window": 10, "atr_window": 20, "fixed_size": 1},
    )

    # 添加数据 - 确保文件路径正确
    engine.add_data(
        database_type="csv",
        data_path="data/SOL-USDT_LOCAL_1m.csv",
        datetime="candle_begin_time",  # 修改为与CSV文件列名匹配
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    # 运行回测
    engine.run_backtesting()

    # 计算结果和统计指标
    df = engine.calculate_result()
    stats = engine.calculate_statistics()

    # 打印统计结果
    print(f"起始资金: {100000:.2f}")
    print(f"结束资金: {stats.get('end_balance', 0):.2f}")
    print(f"总收益率: {stats.get('total_return', 0)*100:.2f}%")
    print(f"年化收益: {stats.get('annual_return', 0)*100:.2f}%")

    # 添加错误处理，避免某些指标不存在
    max_drawdown = stats.get("max_drawdown", 0)
    if isinstance(max_drawdown, (int, float)):
        print(f"最大回撤: {max_drawdown*100:.2f}%")
    else:
        print(f"最大回撤: 0.00%")

    print(f"夏普比率: {stats.get('sharpe_ratio', 0):.2f}")

    # 显示图表
    if show_chart:
        engine.show_chart()

    return df, stats, engine


if __name__ == "__main__":
    # 运行回测
    df, stats, engine = run_backtesting(show_chart=True)
