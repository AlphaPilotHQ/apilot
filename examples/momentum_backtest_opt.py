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
from apilot.utils.logger import get_logger

# 获取日志记录器
logger = get_logger("momentum_strategy")


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
        
        # 初始化仓位
        self.pos = 0

    def on_init(self):
        """
        策略初始化
        """
        self.load_bar(self.std_period * 2)  # 加载足够的历史数据确保指标计算准确

    def on_bar(self, bar: ap.BarData):
        """1分钟K线数据更新"""
        logger.debug(
            f"收到1分钟K线: {bar.datetime} O:{bar.open_price} "
            f"H:{bar.high_price} L:{bar.low_price} C:{bar.close_price} V:{bar.volume}"
        )
        # 创建正确的字典格式：{vt_symbol: bar}
        bars_dict = {f"{bar.symbol}.{bar.exchange.value}": bar}
        self.bg.update_bars(bars_dict)

    def on_5min_bar(self, bars: dict):
        """5分钟K线数据更新，包含交易逻辑"""
        # 获取我们关注的交易对的K线数据
        vt_symbol = f"SOL-USDT.{ap.Exchange.LOCAL.value}"
        if vt_symbol not in bars:
            return
        
        bar = bars[vt_symbol]
        
        logger.debug(
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

            # 限制每次交易的资金比例，最多使用资金的10%
            risk_percent = 0.1
            capital_to_use = self.cta_engine.capital * risk_percent
            size = max(1, int(capital_to_use / bar.close_price))

            if self.momentum > self.mom_threshold:
                # 使用self.vt_symbols[0]获取交易符号
                self.buy(vt_symbol=self.vt_symbols[0], price=bar.close_price, volume=size)
            elif self.momentum < -self.mom_threshold:
                # 使用self.vt_symbols[0]获取交易符号
                self.short(vt_symbol=self.vt_symbols[0], price=bar.close_price, volume=size)

        elif self.pos > 0:  # 多头持仓 → 标准差追踪止损
            # 计算移动止损价格
            long_stop = self.intra_trade_high - self.trailing_std_scale * self.std_value

            # 当价格跌破止损线时平仓
            if bar.close_price < long_stop:
                self.sell(vt_symbol=self.vt_symbols[0], price=bar.close_price, volume=abs(self.pos))

        elif self.pos < 0:  # 空头持仓 → 标准差追踪止损
            # 计算移动止损价格
            short_stop = self.intra_trade_low + self.trailing_std_scale * self.std_value

            # 当价格突破止损线时平仓
            if bar.close_price > short_stop:
                self.cover(vt_symbol=self.vt_symbols[0], price=bar.close_price, volume=abs(self.pos))

    def on_order(self, order: ap.OrderData):
        """委托回调"""
        # 记录委托状态变化
        logger.info(f"Order {order.vt_orderid} status: {order.status}")
        return

    def on_trade(self, trade: ap.TradeData):
        """成交回调"""
        # 更新持仓
        if trade.direction == ap.Direction.LONG:
            # 买入或平空
            if trade.offset == ap.Offset.OPEN:
                # 买入开仓
                self.pos += trade.volume
            else:
                # 买入平仓
                self.pos += trade.volume
        else:
            # 卖出或平多
            if trade.offset == ap.Offset.OPEN:
                # 卖出开仓
                self.pos -= trade.volume
            else:
                # 卖出平仓
                self.pos -= trade.volume
                
        # 更新最高/最低价追踪
        if self.pos > 0:
            # 多头仓位，更新最高价
            self.intra_trade_high = max(self.intra_trade_high, trade.price)
        elif self.pos < 0:
            # 空头仓位，更新最低价
            self.intra_trade_low = min(self.intra_trade_low, trade.price)
        
        logger.info(f"Trade: {trade.vt_orderid} {trade.direction} {trade.offset} {trade.volume}@{trade.price}, pos: {self.pos}")


@ap.log_exceptions()
def run_backtesting(
    strategy_class,
    init_cash=100000,
    start=datetime(2023, 1, 1),
    end=datetime(2023, 6, 30),
    std_period=20,
    mom_threshold=0.05,
    trailing_std_scale=4.0,
):
    logger.info(f"运行回测 - 参数: {std_period}, {mom_threshold}, {trailing_std_scale}")

    # 创建回测引擎
    engine = ap.BacktestingEngine()
    logger.info("步骤1: 创建回测引擎实例")

    # 验证CSV文件是否存在
    csv_path = os.path.join("/Users/bobbyding/Documents/GitHub/apilot/data", "SOL-USDT_LOCAL_1m.csv")
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在: {csv_path}")
        return None, None

    # 读取并验证CSV数据的日期范围
    try:
        df = pd.read_csv(csv_path)
        df['candle_begin_time'] = pd.to_datetime(df['candle_begin_time'])
        logger.info(f"CSV文件前5行: \n{df.head()}")
        logger.info(f"CSV文件行数: {len(df)} 文件列名: {df.columns.tolist()}")
        
        # 打印数据的日期范围，确认与回测时间段有重叠
        min_date = df['candle_begin_time'].min()
        max_date = df['candle_begin_time'].max()
        logger.info(f"CSV数据日期范围: {min_date} 至 {max_date}")
        logger.info(f"回测日期范围: {start} 至 {end}")
        
        # 检查回测区间内有多少数据
        in_range_data = df[(df['candle_begin_time'] >= start) & (df['candle_begin_time'] <= end)]
        logger.info(f"回测区间内数据量: {len(in_range_data)}")
        
        if len(in_range_data) == 0:
            logger.error("回测区间内没有数据！请调整回测时间范围")
            return None, None
    except Exception as e:
        logger.exception(f"读取CSV文件失败: {e}")
        return None, None

    # 设置引擎参数
    logger.info(
        f"步骤2: 设置引擎参数 - 初始资金:{init_cash}, 时间范围:{start} 至 {end}"
    )
    # 直接使用符号SOL-USDT.LOCAL，确保与CSV中的symbol对应
    symbol = "SOL-USDT.LOCAL"
    engine.set_parameters(
        vt_symbols=[symbol],
        interval="1m",
        start=start,
        end=end,
        rates={symbol: 0.00075},
        sizes={symbol: 1},
        priceticks={symbol: 0.001},
        capital=init_cash,
    )

    # 添加策略
    logger.info(
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
    logger.info(f"步骤4: 添加数据源 - CSV数据路径: {csv_path}")
    
    # 简化CSV数据添加方式
    engine.add_data(
        database_type="csv",
        data_path=csv_path,
        datetime="candle_begin_time",  # 更新为实际CSV文件中的列名
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume"
    )

    # 加载历史数据
    logger.info("步骤5: 加载历史数据")
    engine.load_data()
    
    # 检查是否成功加载了数据
    if not engine.dts:
        logger.error("历史数据加载失败，dts列表为空")
        
        # 调试CSV解析
        from apilot.datafeed import CsvDatabase
        # 直接使用CSV数据库类尝试加载数据
        csv_db = CsvDatabase()
        csv_db.data_path = csv_path 
        csv_db.is_direct_file = True
        
        # 测试直接加载
        for vt_symbol in engine.vt_symbols:
            symbol_str, exchange_str = vt_symbol.split(".")
            exchange = engine.exchanges[vt_symbol]
            
            logger.info(f"尝试直接从CSV读取 {symbol_str}.{exchange}，区间 {engine.interval}，时间 {start} 至 {end}")
            bars = csv_db.load_bar_data(
                symbol=symbol_str,
                exchange=exchange,
                interval=engine.interval,
                start=start,
                end=end
            )
            
            logger.info(f"直接加载结果: 成功加载 {len(bars)} 个bar")
            if bars:
                logger.info(f"第一个bar: {bars[0]}")
                logger.info(f"最后一个bar: {bars[-1]}")
    else:
        logger.info(f"历史数据加载成功, 加载了 {len(engine.dts)} 个时间点")
    
    # 运行回测
    logger.info("步骤6: 开始执行回测")
    engine.run_backtesting()

    # 计算结果和统计指标
    logger.info("步骤7: 计算回测结果和绩效统计")
    df = engine.calculate_result()
    stats = engine.calculate_statistics()

    # 打印关键绩效指标
    if stats and stats.get('total_return') is not None:
        logger.info(
            f"回测结果 - 收益率: {stats['total_return']:.2f}%, 夏普比率: {stats['sharpe_ratio']:.2f}, "
            f"最大回撤: {stats['max_ddpercent']:.2f}%, 交易次数: {stats['total_trade_count']}"
        )
    else:
        logger.warning("未能生成有效的回测统计数据或无交易记录")

    return df, stats


if __name__ == "__main__":
    # 单次回测 - 使用更宽的日期范围
    df, stats = run_backtesting(
        StdMomentumStrategy,
        init_cash=100000,
        start=datetime(2023, 1, 1),  # 从1月1日开始，确保有足够数据
        end=datetime(2023, 6, 30),
        std_period=20,
        mom_threshold=0.05,
        trailing_std_scale=4.0,
    )
