"""
回测引擎模块

实现了回测引擎,用于策略回测和优化.
"""

from collections.abc import Callable
from datetime import date, datetime

from pandas import DataFrame

from apilot.analysis.performance import (
    DailyResult,
    PerformanceCalculator,
    calculate_statistics,
)
from apilot.core.constant import (
    BacktestingMode,
    Direction,
    EngineType,
    Exchange,
    Interval,
    Offset,
    Status,
)
from apilot.core.object import (
    BarData,
    OrderData,
    TickData,
    TradeData,
)
from apilot.core.utility import round_to
from apilot.plotting.chart import plot_backtest_results
from apilot.strategy.template import PATemplate
from apilot.utils.logger import get_logger, set_level

# 获取日志记录器
logger = get_logger("BacktestEngine")
set_level("debug", "BacktestEngine")


# 回测默认设置
BACKTEST_CONFIG = {
    "risk_free": 0.0,
    "size": 1,
    "pricetick": 0.0,
}


class BacktestingEngine:
    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self, main_engine=None) -> None:
        self.main_engine = main_engine
        self.symbols: list[str] = []  # Full trading symbols (e.g. "BTC.BINANCE")
        self.exchanges: dict[str, Exchange] = {}  # 可选缓存,加速访问
        self.start: datetime = None
        self.end: datetime = None
        self.sizes: dict[str, float] | None = None
        self.priceticks: dict[str, float] | None = None
        self.capital: int = 100_000
        self.annual_days: int = 240
        self.mode: BacktestingMode = BacktestingMode.BAR

        self.strategy_class: type[PATemplate] = None
        self.strategy: PATemplate = None
        self.tick: TickData = None
        self.bars: dict[str, BarData] = {}
        self.datetime: datetime = None

        self.interval: Interval = None
        self.callback: Callable | None = None
        self.history_data = {}
        self.dts: list[datetime] = []

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: dict[date, DailyResult] = {}
        self.daily_df: DataFrame = None

    def clear_data(self) -> None:
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
        symbols: list[str],
        interval: Interval,
        start: datetime,
        sizes: dict[str, float] | None = None,
        priceticks: dict[str, float] | None = None,
        capital: int = 100_000,
        end: datetime | None = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        annual_days: int = 240,
    ) -> None:
        self.mode = mode
        self.symbols = symbols
        self.interval = Interval(interval)
        self.sizes = sizes if sizes is not None else {}
        self.priceticks = priceticks if priceticks is not None else {}
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
            logger.warning(f"错误:起始日期({self.start})必须小于结束日期({self.end})")

    def add_strategy(
        self, strategy_class: type[PATemplate], setting: dict | None = None
    ) -> None:
        """
        添加策略
        """
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.symbols, setting
        )

    def add_data(self, provider_type, symbol, **kwargs):
        from apilot.datafeed import DATA_PROVIDERS

        # 获取数据提供者类
        if provider_type not in DATA_PROVIDERS:
            raise ValueError(f"未知的数据提供者类型: {provider_type}")

        provider_class = DATA_PROVIDERS[provider_type]

        # 创建数据提供者实例
        provider = provider_class(**kwargs)

        # 确保symbol在symbols列表中
        if symbol not in self.symbols:
            self.symbols.append(symbol)

        # 加载数据
        bars = provider.load_bar_data(
            symbol=symbol,
            interval=self.interval,
            start=self.start,
            end=self.end,
        )

        # 处理数据
        for bar in bars:
            bar.symbol = symbol
            self.dts.append(bar.datetime)
            self.history_data.setdefault(bar.datetime, {})[symbol] = bar

        # 排序时间点
        self.dts = sorted(set(self.dts))
        return self

    def add_csv_data(self, symbol, filepath, **kwargs):
        return self.add_data("csv", symbol, filepath=filepath, **kwargs)

    def add_mongodb_data(self, symbol, database, collection, **kwargs):
        return self.add_data(
            "mongodb", symbol, database=database, collection=collection, **kwargs
        )

    def run_backtesting(self) -> None:
        self.strategy.on_init()
        logger.debug("策略on_init()调用完成")

        # 固定预热100个bar，TODO：改成真实情况
        warmup_bars = 100
        for i in range(warmup_bars):
            try:
                self.new_bars(self.dts[i])
            except Exception as e:
                logger.error(f"预热阶段出错: {e}")
                return
        logger.info(f"策略初始化结束,处理了 {warmup_bars} 个时间点,TODO")

        # 设置为交易模式
        self.strategy.inited = True
        self.strategy.trading = True
        self.strategy.on_start()

        # 使用剩余历史数据进行策略回测
        logger.info(
            f"Strat Backtesting,起始索引: {warmup_bars}, 结束索引: {len(self.dts)}"
        )
        for dt in self.dts[warmup_bars:]:
            try:
                self.new_bars(dt)
            except Exception as e:
                logger.error(f"回测阶段出错: {e}")
                return
        logger.info(
            f"Backtest Finished: "
            f"trade_count: {self.trade_count}, "
            f"active_limit_orders: {len(self.active_limit_orders)}, "
            f"limit_orders: {len(self.limit_orders)}"
        )

    def new_bars(self, dt: datetime) -> None:
        self.datetime = dt

        # 获取当前时间点上所有交易品种的bar
        bars = self.history_data.get(dt, {})

        # 更新策略的多个bar数据
        self.bars = bars

        self.cross_limit_order()
        self.strategy.on_bars(bars)

        # 更新每个品种的收盘价
        for symbol, bar in bars.items():
            self.update_daily_close(bar.close_price, symbol)

    def calculate_result(self) -> DataFrame:
        """
        计算回测结果
        """
        calculator = PerformanceCalculator(self.capital, self.annual_days)
        self.daily_df = calculator.calculate_result(
            self.trades, self.daily_results, self.sizes
        )
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        """
        计算统计数据
        """
        if df is None:
            df = self.daily_df

        # 调用底层函数,现在它返回(stats, df)元组
        stats, updated_df = calculate_statistics(
            df, self.capital, self.annual_days, output
        )

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

        plot_backtest_results(df)

    def update_daily_close(self, price: float, symbol: str) -> None:
        d: date = self.datetime.date()

        daily_result: DailyResult = self.daily_results.get(d, None)
        if daily_result:
            daily_result.add_close_price(symbol, price)
        else:
            self.daily_results[d] = DailyResult(d)
            self.daily_results[d].add_close_price(symbol, price)

    def load_bar(
        self,
        symbol: str,
        count: int,
        interval: Interval = Interval.MINUTE,
        callback: Callable | None = None,
        use_database: bool = False,
    ) -> list[BarData]:
        """回测环境中的占位方法,实际数据由add_csv_data等方法预先加载"""
        logger.debug(f"回测引擎load_bar调用: {symbol}, {count} count bar")
        return []

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
            # pos_change = order.volume if buy_cross else -order.volume

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

    def send_order(
        self,
        strategy: PATemplate,
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
        orderid: str = self.send_limit_order(symbol, direction, offset, price, volume)
        return [orderid]

    def send_limit_order(
        self,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> str:
        self.limit_order_count += 1

        logger.debug(
            f"创建订单 - 合约: {symbol}, 方向: {direction}, 价格: {price}, 数量: {volume}"
        )

        order: OrderData = OrderData(
            symbol=symbol,
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

    def cancel_order(self, strategy: PATemplate, orderid: str) -> None:
        """
        撤销订单
        """
        if orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: PATemplate) -> None:
        """
        撤销所有订单
        """
        orderids: list = list(self.active_limit_orders.keys())
        for orderid in orderids:
            self.cancel_order(strategy, orderid)

    def sync_strategy_data(self, strategy: PATemplate) -> None:
        """
        同步策略数据
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        获取引擎类型
        """
        return self.engine_type

    def get_pricetick(self, strategy: PATemplate, symbol: str) -> float:
        """
        获取价格Tick
        """
        return self.priceticks.get(symbol, 0.0001)

    def get_size(self, strategy: PATemplate, symbol: str) -> int:
        """
        获取合约大小
        """
        # 如果交易对不在sizes字典中,则返回默认值1
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
