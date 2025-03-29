"""
动量策略回测与优化示例

此模块实现了一个基于动量指标的趋势跟踪策略,结合标准差动态止损.
包含单次回测和参数优化功能,支持使用遗传算法寻找最优参数组合.
"""

from datetime import datetime
from typing import ClassVar

import apilot as ap
from apilot.utils.logger import get_logger, set_level

# 获取日志记录器
logger = get_logger("momentum_strategy")
set_level("debug", "momentum_strategy")


class StdMomentumStrategy(ap.CtaTemplate):
    """
    策略逻辑:

    1. 核心思路:结合动量信号与标准差动态止损的中长期趋势跟踪策略

    2. 入场信号:
       - 基于动量指标(当前价格/N周期前价格-1)生成交易信号
       - 动量 > 阈值(5%)时做多
       - 动量 < -阈值(-5%)时做空
       - 使用全部账户资金进行头寸管理

    3. 风险管理:
       - 使用基于标准差的动态追踪止损
       - 多头持仓:止损设置在最高价-4倍标准差
       - 空头持仓:止损设置在最低价+4倍标准差
    """

    # 策略参数
    std_period = 30
    mom_threshold = 0.5
    trailing_std_scale = 4

    parameters: ClassVar[list[str]] = ["std_period", "mom_threshold", "trailing_std_scale"]
    variables: ClassVar[list[str]] = ["momentum", "intra_trade_high", "intra_trade_low", "pos"]

    def __init__(self, cta_engine, strategy_name, symbols, setting):
        super().__init__(cta_engine, strategy_name, symbols, setting)

        # 为每个交易对创建数据生成器和管理器
        self.bgs = {}
        self.ams = {}

        for symbol in self.symbols:
            self.bgs[symbol] = ap.BarGenerator(self.on_bar, 5, self.on_5min_bar)
            self.ams[symbol] = ap.ArrayManager(size=200)

        # 为每个交易对创建状态跟踪字典
        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.intra_trade_low = {}
        self.pos = {}

        # 初始化每个交易对的状态
        for symbol in self.symbols:
            self.momentum[symbol] = 0.0
            self.std_value[symbol] = 0.0
            self.intra_trade_high[symbol] = 0
            self.intra_trade_low[symbol] = 0
            self.pos[symbol] = 0

    def on_init(self):
        self.load_bar(self.std_period * 2)

    def on_bar(self, bar: ap.BarData):
        """原始K线数据更新"""
        symbol = bar.symbol
        if symbol in self.bgs:
            # 创建正确的字典格式:{symbol: bar}
            bars_dict = {symbol: bar}
            self.bgs[symbol].update_bars(bars_dict)

    def on_5min_bar(self, bars: dict):
        """5分钟K线数据更新,包含交易逻辑"""
        self.cancel_all()  # 取消之前的所有订单

        # 首先更新所有交易对的数据
        for symbol, bar in bars.items():
            if symbol not in self.ams:
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            # 计算标准差
            self.std_value[symbol] = am.std(self.std_period)

            # 计算动量因子
            if len(am.close_array) > self.std_period + 1:
                old_price = am.close_array[-self.std_period - 1]
                current_price = am.close_array[-1]
                if old_price != 0:
                    self.momentum[symbol] = (current_price / old_price) - 1
                else:
                    self.momentum[symbol] = 0.0

            logger.debug(
                f"{symbol} 生成5分钟K线: {bar.datetime} O:{bar.open_price} "
                f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}"
                f" 动量: {self.momentum.get(symbol, 0):.4f}"
            )

        # 然后执行交易逻辑
        for symbol, bar in bars.items():
            if symbol not in self.ams or not self.ams[symbol].inited:
                continue

            # 获取当前持仓
            current_pos = self.pos.get(symbol, 0)

            # 持仓状态下更新跟踪止损价格
            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )
            elif current_pos < 0:
                self.intra_trade_low[symbol] = min(
                    self.intra_trade_low[symbol], bar.low_price
                )

            # 交易逻辑
            if current_pos == 0:
                # 初始化追踪价格
                self.intra_trade_high[symbol] = bar.high_price
                self.intra_trade_low[symbol] = bar.low_price

                # 限制每次交易的资金比例,最多使用资金的10%
                risk_percent = 0.1
                capital_to_use = self.cta_engine.capital * risk_percent
                size = max(1, int(capital_to_use / bar.close_price))

                logger.debug(
                    f"{symbol} 资金情况: 可用 {self.cta_engine.capital}, 使用 {capital_to_use}, 数量 {size}"
                )

                # 基于动量信号开仓
                if self.momentum[symbol] > self.mom_threshold:
                    logger.debug(
                        f"{symbol} 发出多头信号: 动量 {self.momentum[symbol]:.4f} > 阈值 {self.mom_threshold}"
                    )
                    self.buy(symbol=symbol, price=bar.close_price, volume=size)
                elif self.momentum[symbol] < -self.mom_threshold:
                    logger.debug(
                        f"{symbol} 发出空头信号: 动量 {self.momentum[symbol]:.4f} < 阈值 {-self.mom_threshold}"
                    )
                    self.short(symbol=symbol, price=bar.close_price, volume=size)

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

            elif current_pos < 0:  # 空头持仓 → 标准差追踪止损
                # 计算移动止损价格
                short_stop = (
                    self.intra_trade_low[symbol]
                    + self.trailing_std_scale * self.std_value[symbol]
                )

                # 当价格突破止损线时平仓
                if bar.close_price > short_stop:
                    self.cover(
                        symbol=symbol, price=bar.close_price, volume=abs(current_pos)
                    )

    def on_order(self, order: ap.OrderData):
        """委托回调"""
        # 记录委托状态变化
        logger.info(f"Order {order.vt_orderid} status: {order.status}")

    def on_trade(self, trade: ap.TradeData):
        """成交回调"""
        symbol = trade.symbol

        # 更新持仓
        if trade.direction == ap.Direction.LONG:
            # 买入或平空
            if trade.offset == ap.Offset.OPEN:
                # 买入开仓
                self.pos[symbol] = self.pos.get(symbol, 0) + trade.volume
            else:
                # 买入平仓
                self.pos[symbol] = self.pos.get(symbol, 0) + trade.volume
        else:
            # 卖出或平多
            if trade.offset == ap.Offset.OPEN:
                # 卖出开仓
                self.pos[symbol] = self.pos.get(symbol, 0) - trade.volume
            else:
                # 卖出平仓
                self.pos[symbol] = self.pos.get(symbol, 0) - trade.volume

        # 更新最高/最低价追踪
        current_pos = self.pos.get(symbol, 0)
        if current_pos > 0:
            # 多头仓位,更新最高价
            self.intra_trade_high[symbol] = max(
                self.intra_trade_high.get(symbol, trade.price), trade.price
            )
        elif current_pos < 0:
            # 空头仓位,更新最低价
            self.intra_trade_low[symbol] = min(
                self.intra_trade_low.get(symbol, trade.price), trade.price
            )

        logger.info(
            f"Trade: {symbol} {trade.vt_orderid} {trade.direction} "
            f"{trade.offset} {trade.volume}@{trade.price}, pos: {current_pos}"
        )


@ap.log_exceptions()
def run_backtesting(
    strategy_class=StdMomentumStrategy,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 1, 30),
    std_period=20,
    mom_threshold=0.005,  # 降低动量阈值到0.5%以便更容易触发信号
    trailing_std_scale=2.0,
):
    logger.debug(
        f"运行回测 - 参数: {std_period}, {mom_threshold}, {trailing_std_scale}"
    )

    # 1 创建回测引擎
    engine = ap.BacktestingEngine()

    # 2 设置引擎参数
    symbols = ["SOL-USDT.LOCAL", "BTC-USDT.LOCAL"]
    engine.set_parameters(
        symbols=symbols,
        interval="1m",
        start=start,
        end=end,
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

    # 4 添加数据
    # 一个简单的回调函数占位符
    def on_bar_data(bar):
        pass

    # 为SOL-USDT添加数据
    engine.load_bar(
        symbol="SOL-USDT.LOCAL",
        days=30,  # 足够覆盖回测时间段
        interval=ap.Interval.MINUTE,
        callback=on_bar_data,
        use_database=True,
    )

    # 为BTC-USDT添加数据
    engine.load_bar(
        symbol="BTC-USDT.LOCAL",
        days=30,  # 足够覆盖回测时间段
        interval=ap.Interval.MINUTE,
        callback=on_bar_data,
        use_database=True,
    )

    # 5 运行回测
    engine.run_backtesting()

    # 6 计算和输出结果
    engine.calculate_result()
    stats = engine.calculate_statistics()

    # 打印统计结果字典键
    print(f"统计结果字典键: {list(stats.keys())}")

    # 7 显示图表 - 添加条件判断,仅当有交易数据时才尝试显示图表
    if len(engine.trades) > 0:
        try:
            # 直接调用show_chart,它会使用engine.daily_df
            engine.show_chart()
            print("图表已生成!")
        except Exception as e:
            print(f"图表显示错误: {e}")
            import traceback

            traceback.print_exc()

    # return engine, df
    # 参数优化示例 (注释掉,需要时可以解开使用)
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
    # 单次回测 - 利用所有函数默认参数
    run_backtesting()
