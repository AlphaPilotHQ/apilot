"""
回测引擎模块

实现了回测引擎,用于策略回测和优化.
"""

from collections.abc import Callable
from datetime import date, datetime

from pandas import DataFrame

from apilot.performance.calculator import calculate_statistics
# 临时定义DailyResult类，后续完整重构时应移除
class DailyResult:
    def __init__(self, date):
        self.date = date
        self.close_prices = {}
        self.net_pnl = 0.0     # 每日净盈亏
        self.turnover = 0.0    # 每日成交额
        self.trade_count = 0   # 每日交易次数
        
    def add_close_price(self, symbol, price):
        self.close_prices[symbol] = price
        
    def add_trade(self, trade, profit=0.0):
        """添加交易记录及其利润"""
        self.turnover += trade.price * trade.volume
        self.trade_count += 1
        self.net_pnl += profit

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
from apilot.performance.report import PerformanceReport, create_performance_report
from apilot.strategy.template import PATemplate
from apilot.utils.logger import get_logger, set_level

# 获取日志记录器
logger = get_logger("BacktestEngine")
set_level("info", "BacktestEngine")


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
        self.accounts = {"balance": self.capital}

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

        # 固定预热100个bar,TODO:改成真实情况
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

        # 调试日志：记录当前时间点上的标的
        if not bars:
            logger.debug(f"当前时间点 {dt} 没有数据")
        else:
            if "SOL-USDT.LOCAL" not in bars:
                logger.debug(f"当前时间点 {dt} 没有SOL-USDT.LOCAL数据")

        # 更新策略的多个bar数据
        self.bars = bars

        self.cross_limit_order()
        self.strategy.on_bars(bars)

        # 更新每个品种的收盘价
        for symbol, bar in bars.items():
            self.update_daily_close(bar.close_price, symbol)

    def update_daily_close(self, price: float, symbol: str) -> None:
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
            # 更新订单状态
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)
                logger.debug(f"Order {order.orderid} status: {order.status}")

            # 根据订单的交易品种获取对应的价格
            symbol = order.symbol

            # 根据模式设置触发价格
            if self.mode == BacktestingMode.BAR:
                bar = self.bars.get(symbol)
                if not bar:
                    logger.info(
                        f"找不到订单对应的K线数据: {symbol}, 当前时间: {self.datetime}, 订单ID: {order.orderid}"
                    )
                    continue
                buy_price = bar.low_price
                sell_price = bar.high_price

            else:
                if self.tick.symbol != symbol:
                    continue
                buy_price = self.tick.ask_price_1
                sell_price = self.tick.bid_price_1

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
            logger.debug(f"Order {order.orderid} status: {order.status}")

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

            # 更新当前持仓和账户余额
            self.update_account_balance(trade)

            self.strategy.on_trade(trade)

    def update_account_balance(self, trade: TradeData) -> None:
        """
        更新账户余额, 在每笔交易后调用
        简单实现: 跟踪每个交易对的持仓成本和数量
        """
        # 如果持仓字典不存在, 初始化它
        if not hasattr(self, "positions"):
            self.positions = {}  # 格式: {symbol: {"long": {"volume": 0, "cost": 0}, "short": {"volume": 0, "cost": 0}}}

        symbol = trade.symbol
        if symbol not in self.positions:
            self.positions[symbol] = {
                "long": {"volume": 0, "cost": 0},
                "short": {"volume": 0, "cost": 0},
            }

        # 确定方向
        pos_type = "long" if trade.direction == Direction.LONG else "short"
        opposite_type = "short" if pos_type == "long" else "long"

        # 根据开平标志处理持仓
        if trade.offset == Offset.OPEN:  # 开仓
            # 增加持仓量和成本
            old_cost = self.positions[symbol][pos_type]["cost"]
            old_volume = self.positions[symbol][pos_type]["volume"]

            # 更新持仓均价和数量
            new_volume = old_volume + trade.volume
            new_cost = old_cost + (trade.price * trade.volume)

            self.positions[symbol][pos_type]["volume"] = new_volume
            self.positions[symbol][pos_type]["cost"] = new_cost

        else:  # 平仓
            # 确定要平仓的持仓数据
            opposite_pos = self.positions[symbol][opposite_type]

            # 计算平仓盈亏
            if opposite_pos["volume"] > 0:
                # 计算平均开仓价
                avg_price = opposite_pos["cost"] / opposite_pos["volume"]

                # 计算盈亏
                if opposite_type == "long":  # 平多仓
                    profit = (trade.price - avg_price) * min(
                        trade.volume, opposite_pos["volume"]
                    )
                else:  # 平空仓
                    profit = (avg_price - trade.price) * min(
                        trade.volume, opposite_pos["volume"]
                    )

                # 更新持仓
                opposite_pos["volume"] -= trade.volume
                if opposite_pos["volume"] <= 0:
                    # 持仓已全部平仓
                    opposite_pos["volume"] = 0
                    opposite_pos["cost"] = 0
                else:
                    # 持仓部分平仓,成本等比例减少
                    cost_ratio = 1 - (
                        trade.volume / (opposite_pos["volume"] + trade.volume)
                    )
                    opposite_pos["cost"] *= cost_ratio

                # 更新账户余额
                self.accounts["balance"] += profit
                logger.debug(
                    f"平仓盈亏: {profit:.2f}, 当前账户余额: {self.accounts['balance']:.2f}"
                )
                
                # 更新每日结果
                trade_date = trade.datetime.date()
                if trade_date in self.daily_results:
                    self.daily_results[trade_date].add_trade(trade, profit)
                else:
                    # 如果该日期还没有记录，创建新的日结果
                    self.daily_results[trade_date] = DailyResult(trade_date)
                    self.daily_results[trade_date].add_trade(trade, profit)

    def send_order(
        self,
        strategy: PATemplate,
        symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
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

        logger.debug(
            f"创建订单: {order.orderid}, 标的: {symbol}, 方向: {direction}, 价格: {price}, 数量: {volume}"
        )
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

    def calculate_result(self) -> DataFrame:
        """
        计算回测结果
        """
        import pandas as pd
        from datetime import date
        
        # 检查是否有交易
        if not self.trades:
            return pd.DataFrame()
            
        # 收集每日数据
        daily_results = []
        
        # 确保所有交易日期都有记录
        all_dates = sorted(self.daily_results.keys())
        
        # 初始化第一天的balance为初始资金
        current_balance = self.capital
        
        for d in all_dates:
            daily_result = self.daily_results[d]
            
            # 获取每日结果数据
            result = {
                'date': d,
                'trade_count': daily_result.trade_count,
                'turnover': daily_result.turnover,
                'net_pnl': daily_result.net_pnl
            }
            
            # 更新当前余额
            current_balance += daily_result.net_pnl
            result['balance'] = current_balance
            
            daily_results.append(result)
            
        # 创建DataFrame
        self.daily_df = pd.DataFrame(daily_results)
        
        if not self.daily_df.empty:
            self.daily_df.set_index('date', inplace=True)
            
            # 计算回撤
            self.daily_df['highlevel'] = self.daily_df['balance'].cummax()
            self.daily_df['ddpercent'] = (self.daily_df['balance'] - self.daily_df['highlevel']) / self.daily_df['highlevel'] * 100
            
            # 计算收益率
            pre_balance = self.daily_df['balance'].shift(1)
            pre_balance.iloc[0] = self.capital
            
            # 安全计算收益率
            self.daily_df['return'] = self.daily_df['balance'].pct_change().fillna(0) * 100
            self.daily_df.loc[self.daily_df.index[0], 'return'] = ((self.daily_df['balance'].iloc[0] / self.capital) - 1) * 100
        
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        """
        计算统计数据
        """
        if df is None:
            df = self.daily_df
            
        # 如果DataFrame为空，返回空结果
        if df is None or df.empty:
            return {}

        # 使用新的统计函数
        stats = calculate_statistics(
            df=df, 
            trades=list(self.trades.values()),
            capital=self.capital, 
            annual_days=self.annual_days
        )
        
        # 打印结果
        if output:
            self._print_statistics(stats)

        return stats
        
    def _print_statistics(self, stats):
        """打印统计结果"""
        logger.info(f"Trade day:\t{stats.get('start_date', '')} - {stats.get('end_date', '')}")
        logger.info(f"Profit days:\t{stats.get('profit_days', 0)}")
        logger.info(f"Loss days:\t{stats.get('loss_days', 0)}")
        logger.info(f"Initial capital:\t{self.capital:.2f}")
        logger.info(f"Final capital:\t{stats.get('final_capital', 0):.2f}")
        logger.info(f"Total return:\t{stats.get('total_return', 0):.2f}%")
        logger.info(f"Annual return:\t{stats.get('annual_return', 0):.2f}%")
        logger.info(f"Max drawdown:\t{stats.get('max_drawdown', 0):.2f}%")
        logger.info(f"Total turnover:\t{stats.get('total_turnover', 0):.2f}")
        logger.info(f"Total trades:\t{stats.get('total_trade_count', 0)}")
        logger.info(f"Sharpe ratio:\t{stats.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Return/Drawdown:\t{stats.get('return_drawdown_ratio', 0):.2f}")

    def show_chart(self, df: DataFrame = None) -> None:
        """
        显示图表
        
        Args:
            df: 回测结果数据框，默认使用引擎的daily_df
        """
        if not df:
            df = self.daily_df

        if df is None:
            return
        
        # 使用新的性能报告
        create_performance_report(
            df=df,
            trades=list(self.trades.values()),
            capital=self.capital,
            annual_days=self.annual_days
        )

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

    def get_current_capital(self) -> float:
        """
        获取当前账户价值(初始资本+已实现盈亏)

        简洁实现:直接使用账户余额
        """
        return self.accounts.get("balance", self.capital)
