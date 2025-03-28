"""
回测引擎模块

提供策略的历史回测、性能分析和参数优化功能
"""

import traceback
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Callable, Dict, List, Type, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from apilot.core.constant import (
    INTERVAL_DELTA_MAP,
    BacktestingMode,
    Direction,
    EngineType,
    Exchange,
    Interval,
    Offset,
    Status,
)
from apilot.core.database import get_database
from apilot.core.object import (
    BarData,
    OrderData,
    TickData,
    TradeData,
)
from apilot.core.utility import extract_symbol, round_to
from apilot.datafeed.data_manager import DataManager
from apilot.optimizer import OptimizationSetting, run_ga_optimization
from apilot.strategy.template import CtaTemplate
from apilot.utils.logger import get_logger, set_level
from apilot.plotting.chart import plot_backtest_results
from apilot.analysis.performance import DailyResult, PerformanceCalculator, calculate_statistics

# 回测默认设置
BACKTEST_CONFIG = {
    "risk_free": 0.0,
    "size": 1,
    "pricetick": 0.0,
    # "capital": 1000000,
}

# 设置文件名
# SETTING_FILENAME: str = "apilot_setting.json"

# # 从JSON文件加载配置
# json_config = load_json(SETTING_FILENAME)

# 获取回测专用日志器
logger = get_logger("backtest")
# 设置为INFO级别，减少调试信息输出
set_level("info", "backtest")


class BacktestingEngine:
    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self, main_engine=None) -> None:
        """
        初始化回测引擎
        """
        self.main_engine = main_engine
        self.symbols: List[str] = []  # Full trading symbols (e.g. "BTC.BINANCE")
        self.exchanges: Dict[str, Exchange] = {}  # 可选缓存，加速访问
        self.start: datetime = None
        self.end: datetime = None
        self.sizes: Dict[str, float] = {}
        self.priceticks: Dict[str, float] = {}
        self.capital: int = 1_000_000
        self.annual_days: int = 240
        self.mode: BacktestingMode = BacktestingMode.BAR

        self.strategy_class: Type[CtaTemplate] = None
        self.strategy: CtaTemplate = None
        self.tick: TickData = None
        self.bars: Dict[str, BarData] = {}
        self.datetime: datetime = None

        self.interval: Interval = None
        self.days: int = 0
        self.callback: Callable = None
        self.history_data = {}  # 使用字典来存储历史数据
        self.dts: List[datetime] = []

        self.limit_order_count: int = 0
        self.limit_orders: Dict[str, OrderData] = {}
        self.active_limit_orders: Dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: Dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: Dict[date, DailyResult] = {}
        self.daily_df: DataFrame = None

        # 添加数据管理器
        self.add_data = DataManager(self)

    def clear_data(self) -> None:
        """
        清空回测数据
        """
        self.strategy = None
        self.tick = None
        self.bars = {}
        self.datetime = None

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        symbols: List[str],
        interval: Interval,
        start: datetime,
        sizes: Dict[str, float] = None,
        priceticks: Dict[str, float] = None,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        annual_days: int = 240,
    ) -> None:
        """
        设置回测参数
        """
        self.mode = mode
        self.symbols = symbols
        self.interval = Interval(interval)
        self.sizes = sizes or {}
        self.priceticks = priceticks or {}
        self.start = start

        # 缓存交易所对象以提高性能
        for symbol in symbols:
            from apilot.utils import get_exchange
            self.exchanges[symbol] = get_exchange(symbol)

        self.capital = capital
        self.annual_days = annual_days

        if not end:
            end = datetime.now()
        self.end = end.replace(hour=23, minute=59, second=59)

        if self.start >= self.end:
            logger.warning(f"错误：起始日期({self.start})必须小于结束日期({self.end})")

    def add_strategy(
        self, strategy_class: Type[CtaTemplate], setting: dict = None
    ) -> None:
        """
        添加策略
        """
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.symbols, setting
        )

    def run_backtesting(self) -> None:
        """
        开始回测
        """
        logger.debug("=== 开始回测过程 ===")
        self.strategy.on_init()
        logger.debug("策略on_init()调用完成")

        # 使用指定时间的历史数据初始化策略
        day_count: int = 0
        ix: int = 0

        logger.debug(f"准备初始化，时间点数量: {len(self.dts)}")
        for ix, dt in enumerate(self.dts):
            if self.datetime and dt.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            try:
                logger.debug(f"初始化阶段处理时间点: {dt}")
                self.new_bars(dt)
            except Exception as e:
                logger.error(f"触发异常，回测终止: {e}")
                logger.error(traceback.format_exc())
                return

        self.strategy.inited = True
        logger.info("策略初始化结束，处理了 {ix} 个时间点")

        self.strategy.on_start()
        self.strategy.trading = True
        logger.info("开始回放历史数据")

        # 使用剩余历史数据进行策略回测
        logger.debug(f"开始回测阶段，起始索引: {ix}, 结束索引: {len(self.dts)}")
        tick_count = 0
        for dt in self.dts[ix:]:
            try:
                tick_count += 1
                if tick_count % 1000 == 0:
                    logger.debug(
                        f"回测进度: 已处理 {tick_count} 个时间点, 当前时间: {dt}"
                    )
                self.new_bars(dt)
            except Exception as e:
                logger.error(f"触发异常，回测终止: {e}")
                logger.error(traceback.format_exc())
                return

        logger.info("历史数据回放结束")
        logger.debug(
            f"回测完成统计: 总交易笔数={self.trade_count}, 活跃订单数={len(self.active_limit_orders)}, 总订单数={len(self.limit_orders)}"
        )

    def new_bars(self, dt: datetime) -> None:
        """
        创建新的bar数据
        """
        self.datetime = dt

        # 获取当前时间点上所有交易品种的bar
        bars = self.history_data.get(dt, {})

        if bars:
            logger.debug(f"时间点 {dt} 获取到K线数据: {list(bars.keys())}")
        else:
            logger.debug(f"时间点 {dt} 没有可用的K线数据")
            return

        # 更新策略的多个bar数据
        self.bars = bars

        logger.debug(f"开始撮合订单，活跃订单数: {len(self.active_limit_orders)}")
        self.cross_limit_order()
        logger.debug("调用策略on_bars处理K线数据")
        self.strategy.on_bars(bars)

        # 更新每个品种的收盘价
        for symbol, bar in bars.items():
            self.update_daily_close(bar.close_price, symbol)

    def calculate_result(self) -> DataFrame:
        """
        计算回测结果
        """
        calculator = PerformanceCalculator(self.capital, self.annual_days)
        self.daily_df = calculator.calculate_result(self.trades, self.daily_results, self.sizes)
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        """
        计算统计数据
        """
        if df is None:
            df = self.daily_df

        # 调用底层函数，现在它返回(stats, df)元组
        stats, updated_df = calculate_statistics(df, self.capital, self.annual_days, output)

        # 保存更新后的DataFrame用于图表展示
        if updated_df is not None and not updated_df.empty:
            self.daily_df = updated_df

        return stats

    def show_chart(self, df: DataFrame = None) -> None:
        """
        显示图表
        """
        if not df:
            df = self.daily_df

        if df is None:
            return

        # 使用新的绘图模块
        plot_backtest_results(df)

    def update_daily_close(self, price: float, symbol: str) -> None:
        """
        更新每日收盘价
        """
        d: date = self.datetime.date()

        daily_result: DailyResult = self.daily_results.get(d, None)
        if daily_result:
            daily_result.add_close_price(symbol, price)
        else:
            self.daily_results[d] = DailyResult(d)
            self.daily_results[d].add_close_price(symbol, price)

    def new_bar(self, bar: BarData) -> None:
        """
        处理新bar数据
        """
        self.bars[bar.symbol] = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price, bar.symbol)

    def new_tick(self, tick: TickData) -> None:
        """
        处理新tick数据
        """
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price, tick.symbol)

    def cross_limit_order(self) -> None:
        """
        撮合限价单
        """
        for order in list(self.active_limit_orders.values()):
            logger.debug(
                f"检查订单: {order.orderid}, 方向: {order.direction}, 价格: {order.price}"
            )

            # 更新订单状态
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # 根据订单的交易品种获取对应的价格
            symbol = order.symbol

            # 根据模式设置触发价格
            if self.mode == BacktestingMode.BAR:
                bar = self.bars.get(symbol)
                if not bar:
                    logger.debug(f"找不到订单对应的K线数据: {symbol}")
                    continue
                buy_price = bar.low_price
                sell_price = bar.high_price
                logger.debug(
                    f"Bar模式下的价格 - 买入价: {buy_price}, 卖出价: {sell_price}"
                )
            else:
                if self.tick.symbol != symbol:
                    continue
                buy_price = self.tick.ask_price_1
                sell_price = self.tick.bid_price_1
                logger.debug(
                    f"Tick模式下的价格 - 买入价: {buy_price}, 卖出价: {sell_price}"
                )

            # 判断是否满足成交条件
            buy_cross = (
                order.direction == Direction.LONG
                and order.price >= buy_price
                and buy_price > 0
            )
            sell_cross = (
                order.direction == Direction.SHORT
                and order.price <= sell_price
                and sell_price > 0
            )

            if not buy_cross and not sell_cross:
                continue

            # 设置成交
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            if order.orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.orderid)

            # 创建成交记录
            self.trade_count += 1

            # 确定成交价格和持仓变化
            trade_price = buy_price if buy_cross else sell_price
            pos_change = order.volume if buy_cross else -order.volume

            # 创建成交对象
            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            trade.orderid = order.orderid
            trade.tradeid = f"{self.gateway_name}.{trade.tradeid}"

            self.trades[trade.tradeid] = trade
            logger.debug(
                f"成交记录创建: {trade.tradeid}, 方向: {trade.direction}, 价格: {trade.price}, 数量: {trade.volume}"
            )

            self.strategy.on_trade(trade)

    def load_bar(
        self,
        symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool = False,
    ) -> List[BarData]:
        """
        加载K线数据
        """
        logger.debug(f"加载K线数据: {symbol}, 天数: {days}")

        init_end = self.start - INTERVAL_DELTA_MAP[interval]
        init_start = self.start - timedelta(days=days)

        from apilot.utils import split_symbol
        base_symbol, exchange_str = split_symbol(symbol)
        self.exchanges[symbol] = Exchange(exchange_str)

        bars: List[BarData] = load_bar_data(
            symbol,
            exchange_str,
            interval,
            init_start,
            init_end,
            gateway_name=self.gateway_name,
        )

        return bars

    def load_tick(
        self, symbol: str, days: int, callback: Callable
    ) -> List[TickData]:
        """
        加载tick数据
        """
        init_end = self.start - timedelta(seconds=1)
        init_start = self.start - timedelta(days=days)

        from apilot.utils import split_symbol
        base_symbol, exchange_str = split_symbol(symbol)
        self.exchanges[symbol] = Exchange(exchange_str)

        ticks: List[TickData] = load_tick_data(
            symbol,
            exchange_str,
            init_start,
            init_end,
            database_settings=getattr(self, "database_settings", None),
        )

        return ticks

    def send_order(
        self,
        strategy: CtaTemplate,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False,
    ) -> list:
        """
        发送订单
        """
        price_tick = self.priceticks.get(symbol, 0.001)
        price: float = round_to(price, price_tick)
        orderid: str = self.send_limit_order(
            symbol, direction, offset, price, volume
        )
        return [orderid]

    def send_limit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> str:
        """
        发送限价单
        """
        self.limit_order_count += 1

        logger.debug(
            f"创建订单 - 合约: {symbol}, 方向: {direction}, 价格: {price}, 数量: {volume}"
        )

        order: OrderData = OrderData(
            symbol=symbol,  # 直接使用完整交易符号
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime,
        )

        self.active_limit_orders[order.orderid] = order
        self.limit_orders[order.orderid] = order

        logger.debug(f"订单已创建: {order.orderid}")
        return order.orderid

    def cancel_order(self, strategy: CtaTemplate, orderid: str) -> None:
        """
        撤销订单
        """
        if orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        """
        撤销所有订单
        """
        orderids: list = list(self.active_limit_orders.keys())
        for orderid in orderids:
            self.cancel_order(strategy, orderid)

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        """
        同步策略数据
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        获取引擎类型
        """
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate, symbol: str) -> float:
        """
        获取价格Tick
        """
        return self.priceticks.get(symbol, 0.0001)

    def get_size(self, strategy: CtaTemplate, symbol: str) -> int:
        """
        获取合约大小
        """
        # 如果交易对不在sizes字典中，则返回默认值1
        return self.sizes.get(symbol, 1)

    def get_all_trades(self) -> list:
        """
        获取所有交易
        """
        return list(self.trades.values())

    def get_all_orders(self) -> list:
        """
        获取所有订单
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list:
        """
        获取所有日结果
        """
        return list(self.daily_results.values())

@lru_cache(maxsize=1024)
def load_bar_data(
    symbol: str,
    exchange: Union[Exchange, str],
    interval: Interval,
    start: datetime,
    end: datetime,
    gateway_name: str = ""
) -> List[BarData]:
    """
    加载K线数据
    """

    # 获取数据库实例
    database = get_database()

    # 设置为索引模式并使用默认索引映射
    if hasattr(database, "set_index_mapping"):
        database.set_index_mapping(
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5
        )

    # 加载数据
    return database.load_bar_data(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start=start,
        end=end
    )


@lru_cache(maxsize=1024)
def load_tick_data(
    symbol: str,
    exchange: Union[Exchange, str],
    start: datetime,
    end: datetime,
    database_settings: dict = None,
) -> List[TickData]:
    """
    加载Tick数据

    参数：
        symbol: 交易对名称
        exchange: 交易所或交易所名称字符串
        start: 开始时间
        end: 结束时间
        database_settings: 数据库设置

    返回：
        List[TickData]: Tick数据列表
    """
    # 获取数据库实例
    database = get_database()

    # 加载数据
    ticks = database.load_tick_data(
        symbol=symbol,
        exchange=exchange,
        start=start,
        end=end
    )

    return ticks


def optimize(
    target_name: str,
    strategy_class: Type[CtaTemplate],
    symbols: List[str],
    interval: str,
    start: datetime,
    end: datetime,
    sizes: Dict[str, float],
    priceticks: Dict[str, float],
    capital: int,
    setting: dict,
    optimization_setting: OptimizationSetting,
    use_ga: bool = True,
    max_workers: int = None,
) -> DataFrame:
    """
    多币种回测优化
    """
    engine = BacktestingEngine()

    engine.set_parameters(
        symbols=symbols,
        interval=interval,
        start=start,
        end=end,
        sizes=sizes,
        priceticks=priceticks,
        capital=capital,
    )

    engine.add_strategy(strategy_class, setting)

    result = run_ga_optimization(
        target_name=target_name,
        evaluator=engine,
        optimization_setting=optimization_setting,
        max_workers=max_workers,
    )
    return result
