"""
动量策略回测与优化示例

此模块实现了一个基于动量指标的趋势跟踪策略，结合标准差动态止损。
包含单次回测和参数优化功能，支持使用遗传算法寻找最优参数组合。
"""


import setup_path
from datetime import datetime

import apilot as ap
from apilot.utils.logger import get_logger, set_level

# 获取日志记录器
logger = get_logger("momentum_strategy")
set_level("debug", "momentum_strategy")


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
    variables = ["momentum", "intra_trade_high", "intra_trade_low", "pos"]

    def __init__(self, cta_engine, strategy_name, vt_symbols, setting):
        super().__init__(cta_engine, strategy_name, vt_symbols, setting)

        # 为每个交易对创建数据生成器和管理器
        self.bgs = {}
        self.ams = {}

        for vt_symbol in self.vt_symbols:
            self.bgs[vt_symbol] = ap.BarGenerator(self.on_bar, 5, self.on_5min_bar)
            self.ams[vt_symbol] = ap.ArrayManager(size=200)

        # 为每个交易对创建状态跟踪字典
        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.intra_trade_low = {}
        self.pos = {}

        # 初始化每个交易对的状态
        for vt_symbol in self.vt_symbols:
            self.momentum[vt_symbol] = 0.0
            self.std_value[vt_symbol] = 0.0
            self.intra_trade_high[vt_symbol] = 0
            self.intra_trade_low[vt_symbol] = 0
            self.pos[vt_symbol] = 0

    def on_init(self):
        self.load_bar(self.std_period * 2)

    def on_bar(self, bar: ap.BarData):
        """原始K线数据更新"""
        vt_symbol = f"{bar.symbol}.{bar.exchange.value}"
        if vt_symbol in self.bgs:
            # 创建正确的字典格式：{vt_symbol: bar}
            bars_dict = {vt_symbol: bar}
            self.bgs[vt_symbol].update_bars(bars_dict)

    def on_5min_bar(self, bars: dict):
        """5分钟K线数据更新，包含交易逻辑"""
        self.cancel_all()  # 取消之前的所有订单

        # 首先更新所有交易对的数据
        for vt_symbol, bar in bars.items():
            if vt_symbol not in self.ams:
                continue

            am = self.ams[vt_symbol]
            am.update_bar(bar)

            # 计算标准差
            self.std_value[vt_symbol] = am.std(self.std_period)

            # 计算动量因子
            if len(am.close_array) > self.std_period + 1:
                old_price = am.close_array[-self.std_period - 1]
                current_price = am.close_array[-1]
                if old_price != 0:
                    self.momentum[vt_symbol] = (current_price / old_price) - 1
                else:
                    self.momentum[vt_symbol] = 0.0

            logger.debug(
                f"{vt_symbol} 生成5分钟K线: {bar.datetime} O:{bar.open_price} "
                f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}"
                f" 动量: {self.momentum.get(vt_symbol, 0):.4f}"
            )

        # 然后执行交易逻辑
        for vt_symbol, bar in bars.items():
            if vt_symbol not in self.ams or not self.ams[vt_symbol].inited:
                continue

            # 获取当前持仓
            current_pos = self.pos.get(vt_symbol, 0)

            # 持仓状态下更新跟踪止损价格
            if current_pos > 0:
                self.intra_trade_high[vt_symbol] = max(
                    self.intra_trade_high[vt_symbol], bar.high_price
                )
            elif current_pos < 0:
                self.intra_trade_low[vt_symbol] = min(
                    self.intra_trade_low[vt_symbol], bar.low_price
                )

            # 交易逻辑
            if current_pos == 0:
                # 初始化追踪价格
                self.intra_trade_high[vt_symbol] = bar.high_price
                self.intra_trade_low[vt_symbol] = bar.low_price

                # 限制每次交易的资金比例，最多使用资金的10%
                risk_percent = 0.1
                capital_to_use = self.cta_engine.capital * risk_percent
                size = max(1, int(capital_to_use / bar.close_price))

                # 基于动量信号开仓
                if self.momentum[vt_symbol] > self.mom_threshold:
                    self.buy(vt_symbol=vt_symbol, price=bar.close_price, volume=size)
                elif self.momentum[vt_symbol] < -self.mom_threshold:
                    self.short(vt_symbol=vt_symbol, price=bar.close_price, volume=size)

            elif current_pos > 0:  # 多头持仓 → 标准差追踪止损
                # 计算移动止损价格
                long_stop = self.intra_trade_high[vt_symbol] - self.trailing_std_scale * self.std_value[vt_symbol]

                # 当价格跌破止损线时平仓
                if bar.close_price < long_stop:
                    self.sell(vt_symbol=vt_symbol, price=bar.close_price, volume=abs(current_pos))

            elif current_pos < 0:  # 空头持仓 → 标准差追踪止损
                # 计算移动止损价格
                short_stop = self.intra_trade_low[vt_symbol] + self.trailing_std_scale * self.std_value[vt_symbol]

                # 当价格突破止损线时平仓
                if bar.close_price > short_stop:
                    self.cover(vt_symbol=vt_symbol, price=bar.close_price, volume=abs(current_pos))

    def on_order(self, order: ap.OrderData):
        """委托回调"""
        # 记录委托状态变化
        logger.info(f"Order {order.vt_orderid} status: {order.status}")

    def on_trade(self, trade: ap.TradeData):
        """成交回调"""
        vt_symbol = f"{trade.symbol}.{trade.exchange.value}"

        # 更新持仓
        if trade.direction == ap.Direction.LONG:
            # 买入或平空
            if trade.offset == ap.Offset.OPEN:
                # 买入开仓
                self.pos[vt_symbol] = self.pos.get(vt_symbol, 0) + trade.volume
            else:
                # 买入平仓
                self.pos[vt_symbol] = self.pos.get(vt_symbol, 0) + trade.volume
        else:
            # 卖出或平多
            if trade.offset == ap.Offset.OPEN:
                # 卖出开仓
                self.pos[vt_symbol] = self.pos.get(vt_symbol, 0) - trade.volume
            else:
                # 卖出平仓
                self.pos[vt_symbol] = self.pos.get(vt_symbol, 0) - trade.volume

        # 更新最高/最低价追踪
        current_pos = self.pos.get(vt_symbol, 0)
        if current_pos > 0:
            # 多头仓位，更新最高价
            self.intra_trade_high[vt_symbol] = max(
                self.intra_trade_high.get(vt_symbol, trade.price),
                trade.price
            )
        elif current_pos < 0:
            # 空头仓位，更新最低价
            self.intra_trade_low[vt_symbol] = min(
                self.intra_trade_low.get(vt_symbol, trade.price),
                trade.price
            )

        logger.info(
            f"Trade: {vt_symbol} {trade.vt_orderid} {trade.direction} "
            f"{trade.offset} {trade.volume}@{trade.price}, pos: {current_pos}"
        )


@ap.log_exceptions()
def run_backtesting(
    strategy_class=StdMomentumStrategy,
    init_cash=100000,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 1, 30),
    std_period=20,
    mom_threshold=0.01,
    trailing_std_scale=2.0,
):
    logger.debug(f"运行回测 - 参数: {std_period}, {mom_threshold}, {trailing_std_scale}")

    # 1 创建回测引擎
    engine = ap.BacktestingEngine()

    # 2 设置引擎参数
    vt_symbols = ["SOL-USDT.LOCAL", "BTC-USDT.LOCAL"]
    engine.set_parameters(
        vt_symbols=vt_symbols,
        interval="1m",
        start=start,
        end=end,
        sizes={
            "SOL-USDT.LOCAL": 1,
            "BTC-USDT.LOCAL": 1
        },
        capital=init_cash,
    )
    # 3 添加策略
    engine.add_strategy(
        strategy_class,
        {
            "std_period": std_period,
            "mom_threshold": mom_threshold,
            "trailing_std_scale": trailing_std_scale,
        }
    )

    # 4 添加数据
    # 数据路径
    data_dir = "/Users/bobbyding/Documents/GitHub/apilot/data"
    # 为SOL-USDT添加数据
    sol_data_path = f"{data_dir}/SOL-USDT_LOCAL_1m.csv"
    engine.add_data(
        database_type="csv",
        data_path=sol_data_path,
        specific_symbol="SOL-USDT",
        datetime="candle_begin_time",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume"
    )

    # 为BTC-USDT添加数据
    btc_data_path = f"{data_dir}/BTC-USDT_LOCAL_1m.csv"
    engine.add_data(
        database_type="csv",
        data_path=btc_data_path,
        specific_symbol="BTC-USDT",
        datetime="candle_begin_time",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume"
    )

    engine.load_data()

    # 5 运行回测
    engine.run_backtesting()


    # 6 计算和输出结果
    df = engine.calculate_result()
    engine.calculate_statistics()
    logger.info("步骤7: 计算回测结果和绩效统计")

    engine.show_chart()


    # return engine, df
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
    # 单次回测 - 利用所有函数默认参数
    run_backtesting()
