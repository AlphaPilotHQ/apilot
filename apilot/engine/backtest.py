"""
回测引擎模块

提供策略的历史回测、性能分析和参数优化功能
"""

import os
import os.path
import traceback
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.subplots import make_subplots

from apilot.core.constant import (
    Direction,
    EngineType,
    Exchange,
    INTERVAL_DELTA_MAP,
    Interval,
    Offset,
    Status,
    BacktestingMode,
)
from apilot.core.object import (
    BarData,
    OrderData,
    TickData,
    TradeData,
)
from apilot.core.utility import (
    extract_vt_symbol,
    round_to
)
from apilot.datafeed import get_database
from apilot.optimizer import (
    OptimizationSetting,
    run_ga_optimization
)
from apilot.strategy.template import CtaTemplate


class BacktestingEngine:

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
        """
        初始化回测引擎
        """
        self.vt_symbols: List[str] = []
        self.symbols: Dict[str, str] = {}
        self.exchanges: Dict[str, Exchange] = {}
        self.start: datetime = None
        self.end: datetime = None
        self.rates: Dict[str, float] = {}
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
        self.history_data: Dict[Tuple[datetime, str], BarData] = {}
        self.dts: List[datetime] = []

        self.limit_order_count: int = 0
        self.limit_orders: Dict[str, OrderData] = {}
        self.active_limit_orders: Dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: Dict[str, TradeData] = {}

        self.logs: list = []

        self.daily_results: Dict[date, DailyResult] = {}
        self.daily_df: DataFrame = None

        # 添加数据源配置
        self.database_config = None
        self.specific_data_file = None

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
        vt_symbols: List[str],
        interval: Interval,
        start: datetime,
        rates: Dict[str, float],
        sizes: Dict[str, float],
        priceticks: Dict[str, float],
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        annual_days: int = 240,
    ) -> None:
        """
        设置回测参数
        """
        self.mode = mode
        self.vt_symbols = vt_symbols
        self.interval = Interval(interval)
        self.rates = rates
        self.sizes = sizes
        self.priceticks = priceticks
        self.start = start

        for vt_symbol in vt_symbols:
            symbol, exchange_str = vt_symbol.split(".")
            self.symbols[vt_symbol] = symbol
            self.exchanges[vt_symbol] = Exchange(exchange_str)

        self.capital = capital

        if not end:
            end = datetime.now()
        self.end = end.replace(hour=23, minute=59, second=59)

        if self.start >= self.end:
            raise ValueError(
                f"错误：起始日期({self.start})必须小于结束日期({self.end})"
            )

        self.annual_days = annual_days

    def add_strategy(
        self, strategy_class: Type[CtaTemplate], setting: dict = None
    ) -> None:
        """
        添加策略
        """
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbols, setting
        )

    def add_data(self, database_type: Optional[str] = None, **kwargs) -> None:
        """
        添加数据
        """
        # 检查是否为直接CSV文件路径
        data_path = kwargs.get("data_path", "")
        if (
            database_type == "csv"
            and data_path
            and data_path.endswith(".csv")
            and os.path.isfile(data_path)
        ):
            # 如果是直接指定的CSV文件，我们需要保存字段映射信息
            # 将字段映射信息直接添加到kwargs中，后续CSV数据库会使用
            from apilot.core.setting import SETTINGS

            # 将字段映射添加到SETTINGS字典
            if "datetime" in kwargs:
                SETTINGS["csv_datetime_field"] = kwargs["datetime"]
            if "open" in kwargs:
                SETTINGS["csv_open_field"] = kwargs["open"]
            if "high" in kwargs:
                SETTINGS["csv_high_field"] = kwargs["high"]
            if "low" in kwargs:
                SETTINGS["csv_low_field"] = kwargs["low"]
            if "close" in kwargs:
                SETTINGS["csv_close_field"] = kwargs["close"]
            if "volume" in kwargs:
                SETTINGS["csv_volume_field"] = kwargs["volume"]

        self.database_type = database_type
        self.database_config = kwargs

        return self

    def load_data(self) -> None:
        """
        加载历史数据
        """
        self.output("开始加载历史数据")
        self.output(f"数据库类型: {self.database_type}")
        self.output(f"数据库配置: {self.database_config}")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        # 清理上次加载的历史数据
        self.history_data.clear()
        self.dts.clear()

        # 每次加载30天历史数据
        progress_delta: timedelta = timedelta(days=30)
        total_delta: timedelta = self.end - self.start
        interval_delta: timedelta = INTERVAL_DELTA_MAP[self.interval]

        for vt_symbol in self.vt_symbols:
            if self.interval == Interval.MINUTE:
                start: datetime = self.start
                end: datetime = self.start + progress_delta
                progress = 0

                data_count = 0
                while start < self.end:
                    end = min(end, self.end)

                    data: list[BarData] = load_bar_data(
                        self.symbols[vt_symbol],
                        self.exchanges[vt_symbol],
                        self.interval,
                        start,
                        end,
                        database_settings=getattr(self, "database_settings", None),
                    )

                    for bar in data:
                        bar.vt_symbol = vt_symbol
                        self.dts.append(bar.datetime)
                        self.history_data[(bar.datetime, vt_symbol)] = bar
                        data_count += 1

                    progress += progress_delta / total_delta
                    progress = min(progress, 1)
                    progress_bar = "#" * int(progress * 10)
                    self.output(f"{vt_symbol}加载进度：{progress_bar} [{progress:.0%}]")

                    start = end + interval_delta
                    end += progress_delta + interval_delta
            else:
                data: list[BarData] = load_bar_data(
                    self.symbols[vt_symbol],
                    self.exchanges[vt_symbol],
                    self.interval,
                    self.start,
                    self.end,
                    database_settings=getattr(self, "database_settings", None),
                )

                for bar in data:
                    bar.vt_symbol = vt_symbol
                    self.dts.append(bar.datetime)
                    self.history_data[(bar.datetime, vt_symbol)] = bar

                data_count = len(data)

            self.output(f"{vt_symbol}历史数据加载完成，数据量：{data_count}")

        # 对时间序列进行排序，去重
        self.dts = list(set(self.dts))
        self.dts.sort()

        self.output("所有历史数据加载完成")

    def run_backtesting(self) -> None:
        """
        开始回测
        """
        self.strategy.on_init()

        # 使用指定时间的历史数据初始化策略
        day_count: int = 0
        ix: int = 0

        for ix, dt in enumerate(self.dts):
            if self.datetime and dt.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            try:
                self.new_bars(dt)
            except Exception as e:
                self.output(f"触发异常，回测终止: {e}")
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        # 使用剩余历史数据进行策略回测
        for dt in self.dts[ix:]:
            try:
                self.new_bars(dt)
            except Exception as e:
                self.output(f"触发异常，回测终止: {e}")
                self.output(traceback.format_exc())
                return

        self.output("历史数据回放结束")

    def new_bars(self, dt: datetime) -> None:
        """
        创建新的bar数据
        """
        self.datetime = dt

        # 获取当前时间点上所有交易品种的bar
        bars = {}
        for vt_symbol in self.vt_symbols:
            bar = self.history_data.get((dt, vt_symbol), None)
            if bar:
                bars[vt_symbol] = bar

        if not bars:
            return

        # 更新策略的多个bar数据
        self.bars = bars

        self.cross_limit_order()
        self.strategy.on_bars(bars)

        # 更新每个品种的收盘价
        for vt_symbol, bar in bars.items():
            self.update_daily_close(bar.close_price, vt_symbol)

    def calculate_result(self) -> DataFrame:
        """
        计算回测结果
        """
        if not self.trades:
            self.output("回测成交记录为空")
            return DataFrame()

        # 将成交数据添加到每日结果中
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results.get(d, None)
            if daily_result:
                daily_result.add_trade(trade)

        # 迭代计算每日结果
        pre_closes = {}
        start_poses = {}
        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                self.sizes,
                self.rates,
            )
            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        # 生成DataFrame
        first_result = next(iter(self.daily_results.values()))
        results = {
            key: [getattr(dr, key) for dr in self.daily_results.values()]
            for key in first_result.__dict__
        }

        self.daily_df = DataFrame.from_dict(results).set_index("date")
        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:
        """
        计算统计数据
        """
        if df is None:
            df = self.daily_df

        stats = {
            "start_date": "",
            "end_date": "",
            "total_days": 0,
            "profit_days": 0,
            "loss_days": 0,
            "capital": self.capital,
            "end_balance": 0,
            "max_ddpercent": 0,
            "total_commission": 0,
            "total_turnover": 0,
            "total_trade_count": 0,
            "total_return": 0,
            "annual_return": 0,
            "sharpe_ratio": 0,
            "return_drawdown_ratio": 0,
        }

        # Return early if no data
        if df is None or df.empty:
            self.output("No trading data available")
            return stats

        # Make a copy to avoid modifying original data
        df = df.copy()

        df["balance"] = df["net_pnl"].cumsum() + self.capital

        # Calculate daily returns
        pre_balance = df["balance"].shift(1)
        pre_balance.iloc[0] = self.capital
        x = df["balance"] / pre_balance
        x[x <= 0] = np.nan
        df["return"] = np.log(x).fillna(0)

        # Calculate drawdown
        df["highlevel"] = df["balance"].cummax()
        df["ddpercent"] = (df["balance"] - df["highlevel"]) / df["highlevel"] * 100

        # Save dataframe for charting
        self.daily_df = df

        # Check for bankruptcy
        if not (df["balance"] > 0).all():
            self.output("Bankruptcy detected during backtest")
            return stats

        # Calculate basic statistics
        stats.update(
            {
                "start_date": df.index[0],
                "end_date": df.index[-1],
                "total_days": len(df),
                "profit_days": (df["net_pnl"] > 0).sum(),
                "loss_days": (df["net_pnl"] < 0).sum(),
                "end_balance": df["balance"].iloc[-1],
                "max_ddpercent": df["ddpercent"].min(),
                "total_commission": df["commission"].sum(),
                "total_turnover": df["turnover"].sum(),
                "total_trade_count": df["trade_count"].sum(),
            }
        )

        # Calculate return metrics
        stats["total_return"] = (stats["end_balance"] / self.capital - 1) * 100
        stats["annual_return"] = (
            stats["total_return"] / stats["total_days"] * self.annual_days
        )

        # Calculate risk-adjusted metrics
        daily_returns = df["return"].values
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            stats["sharpe_ratio"] = (
                np.mean(daily_returns)
                / np.std(daily_returns)
                * np.sqrt(self.annual_days)
            )

        if stats["max_ddpercent"] < 0:
            stats["return_drawdown_ratio"] = (
                -stats["total_return"] / stats["max_ddpercent"]
            )

        # Clean up invalid values
        stats = {
            k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            for k, v in stats.items()
        }

        if output:
            self.output(f"Trade day:\t{stats['start_date']} - {stats['end_date']}")
            self.output(f"Profit days:\t{stats['profit_days']}")
            self.output(f"Loss days:\t{stats['loss_days']}")
            self.output(f"Initial capital:\t{self.capital:.2f}")
            self.output(f"Final capital:\t{stats['end_balance']:.2f}")
            self.output(f"Total return:\t{stats['total_return']:.2f}%")
            self.output(f"Annual return:\t{stats['annual_return']:.2f}%")
            self.output(f"Max drawdown:\t{stats['max_ddpercent']:.2f}%")
            self.output(f"Total commission:\t{stats['total_commission']:.2f}")
            self.output(f"Total turnover:\t{stats['total_turnover']:.2f}")
            self.output(f"Total trades:\t{stats['total_trade_count']}")
            self.output(f"Sharpe ratio:\t{stats['sharpe_ratio']:.2f}")
            self.output(f"Return/Drawdown:\t{stats['return_drawdown_ratio']:.2f}")
        return stats

    def show_chart(self, df: DataFrame = None) -> None:
        """
        显示图表
        """
        if not df:
            df = self.daily_df

        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Pnl", "Pnl Distribution"],
            vertical_spacing=0.06,
        )

        balance_line = go.Scatter(
            x=df.index, y=df["balance"], mode="lines", name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["ddpercent"],
            fillcolor="red",
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def update_daily_close(self, price: float, vt_symbol: str) -> None:
        """
        更新每日收盘价
        """
        d: date = self.datetime.date()

        daily_result: DailyResult = self.daily_results.get(d, None)
        if daily_result:
            daily_result.add_close_price(vt_symbol, price)
        else:
            self.daily_results[d] = DailyResult(d)
            self.daily_results[d].add_close_price(vt_symbol, price)

    def new_bar(self, bar: BarData) -> None:
        """
        处理新bar数据
        """
        self.bars[bar.vt_symbol] = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price, bar.vt_symbol)

    def new_tick(self, tick: TickData) -> None:
        """
        处理新tick数据
        """
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price, tick.vt_symbol)

    def cross_limit_order(self) -> None:
        """
        撮合限价单
        """
        for order in list(self.active_limit_orders.values()):
            # 更新订单状态
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # 根据订单的交易品种获取对应的价格
            vt_symbol = f"{order.symbol}.{order.exchange.value}"

            # 根据模式设置触发价格
            if self.mode == BacktestingMode.BAR:
                bar = self.bars.get(vt_symbol)
                if not bar:
                    continue
                buy_price = bar.low_price
                sell_price = bar.high_price
            else:
                if self.tick.vt_symbol != vt_symbol:
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

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

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

            # 设置vt_symbol
            trade.vt_symbol = vt_symbol

            trade.vt_orderid = order.vt_orderid
            trade.vt_tradeid = f"{self.gateway_name}.{trade.tradeid}"

            self.trades[trade.vt_tradeid] = trade
            self.strategy.on_trade(trade)

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool,
    ) -> List[BarData]:
        """
        加载bar数据
        """
        self.callback = callback

        init_end = self.start - INTERVAL_DELTA_MAP[interval]
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        bars: List[BarData] = load_bar_data(
            symbol,
            exchange,
            interval,
            init_start,
            init_end,
            database_settings=getattr(self, "database_settings", None),
        )

        return bars

    def load_tick(
        self, vt_symbol: str, days: int, callback: Callable
    ) -> List[TickData]:
        """
        加载tick数据
        """
        self.callback = callback

        init_end = self.start - timedelta(seconds=1)
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        ticks: List[TickData] = load_tick_data(
            symbol,
            exchange,
            init_start,
            init_end,
            database_settings=getattr(self, "database_settings", None),
        )

        return ticks

    def send_order(
        self,
        strategy: CtaTemplate,
        vt_symbol: str,
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
        price: float = round_to(price, self.priceticks[vt_symbol])
        vt_orderid: str = self.send_limit_order(
            vt_symbol, direction, offset, price, volume
        )
        return [vt_orderid]

    def send_limit_order(
        self,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> str:
        """
        发送限价单
        """
        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=self.symbols[vt_symbol],
            exchange=self.exchanges[vt_symbol],
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """
        撤销订单
        """
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        """
        撤销所有订单
        """
        vt_orderids: list = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        写入日志
        """
        msg: str = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        发送邮件
        """
        pass

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

    def get_pricetick(self, strategy: CtaTemplate, vt_symbol: str) -> float:
        """
        获取价格Tick
        """
        return self.priceticks[vt_symbol]

    def get_size(self, strategy: CtaTemplate, vt_symbol: str) -> int:
        """
        获取合约大小
        """
        return self.sizes[vt_symbol]

    def output(self, msg) -> None:
        """
        输出信息
        """
        print(f"{datetime.now()}\t{msg}")

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

    def run_single_backtest(
        self,
        strategy_class: Type[CtaTemplate],
        vt_symbols: List[str],
        interval: str,
        start: datetime,
        end: datetime,
        rates: Dict[str, float],
        sizes: Dict[str, float],
        priceticks: Dict[str, float],
        capital: int,
        setting: dict,
    ) -> Tuple[DataFrame, dict]:
        """
        运行单一回测
        """
        self.clear_data()

        if interval == Interval.TICK.value:
            mode = BacktestingMode.TICK
        else:
            mode = BacktestingMode.BAR

        self.set_parameters(
            vt_symbols=vt_symbols,
            interval=interval,
            start=start,
            end=end,
            rates=rates,
            sizes=sizes,
            priceticks=priceticks,
            capital=capital,
            mode=mode,
        )

        self.add_strategy(strategy_class, setting)
        self.load_data()
        self.run_backtesting()
        self.calculate_result()

        statistics = self.calculate_statistics()
        df = self.daily_df.copy()

        return df, statistics


class DailyResult:

    def __init__(self, date: date) -> None:
        """
        初始化日结果
        """
        self.date: date = date
        self.close_prices: Dict[str, float] = {}
        self.pre_closes: Dict[str, float] = {}

        self.trades: List[TradeData] = []
        self.trade_count: int = 0

        self.start_poses: Dict[str, float] = {}
        self.end_poses: Dict[str, float] = {}

        self.turnover: float = 0
        self.commission: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """
        添加交易
        """
        self.trades.append(trade)

    def add_close_price(self, vt_symbol: str, price: float) -> None:
        """
        添加收盘价
        """
        self.close_prices[vt_symbol] = price

    def calculate_pnl(
        self,
        pre_closes: Dict[str, float],
        start_poses: Dict[str, float],
        sizes: Dict[str, float],
        rates: Dict[str, float],
    ) -> None:
        """
        计算盈亏
        """
        self.pre_closes = pre_closes
        self.start_poses = start_poses.copy()
        self.end_poses = start_poses.copy()

        # 计算持仓盈亏
        self.holding_pnl = 0
        for vt_symbol in self.pre_closes.keys():
            pre_close = self.pre_closes.get(vt_symbol, 0)
            if not pre_close:
                pre_close = 1  # 避免除零错误

            start_pos = self.start_poses.get(vt_symbol, 0)
            size = sizes.get(vt_symbol, 1)
            close_price = self.close_prices.get(vt_symbol, pre_close)

            symbol_holding_pnl = start_pos * (close_price - pre_close) * size
            self.holding_pnl += symbol_holding_pnl

        # 计算交易盈亏
        self.trade_count = len(self.trades)
        self.trading_pnl = 0
        self.turnover = 0
        self.commission = 0

        for trade in self.trades:
            vt_symbol = trade.vt_symbol
            pos_change = (
                trade.volume if trade.direction == Direction.LONG else -trade.volume
            )

            if vt_symbol in self.end_poses:
                self.end_poses[vt_symbol] += pos_change
            else:
                self.end_poses[vt_symbol] = pos_change

            size = sizes.get(vt_symbol, 1)
            rate = rates.get(vt_symbol, 0)
            close_price = self.close_prices.get(vt_symbol, trade.price)

            turnover = trade.volume * size * trade.price
            self.trading_pnl += pos_change * (close_price - trade.price) * size

            self.turnover += turnover
            self.commission += turnover * rate

        # 计算净盈亏
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission


@lru_cache(maxsize=1024)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime,
    database_settings: dict = None,
) -> List[BarData]:
    """
    加载K线数据
    """

    # 获取数据库实例，如果提供了特殊设置，则使用这些设置
    if database_settings:
        # 用于支持MongoDB等特殊数据库的实现
        # 注意：实际的MongoDB数据库处理应该在具体的数据库实现类中处理
        from apilot.core.setting import SETTINGS

        old_settings = {}

        # 暂存原始设置
        for key, value in database_settings.items():
            if key in SETTINGS:
                old_settings[key] = SETTINGS[key]
                SETTINGS[key] = value

        # 获取数据库连接
        database = get_database()

        # 加载数据
        bars = database.load_bar_data(symbol, exchange, interval, start, end)

        # 恢复原始设置
        for key, value in old_settings.items():
            SETTINGS[key] = value

        return bars
    else:
        # 使用默认数据库连接
        database = get_database()
        return database.load_bar_data(symbol, exchange, interval, start, end)


@lru_cache(maxsize=1024)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime,
    database_settings: dict = None,
) -> List[TickData]:
    """
    加载Tick数据
    """
    # 获取数据库实例，如果提供了特殊设置，则使用这些设置
    if database_settings:
        # 用于支持MongoDB等特殊数据库的实现
        # 注意：实际的MongoDB数据库处理应该在具体的数据库实现类中处理
        from apilot.core.setting import SETTINGS

        old_settings = {}

        # 暂存原始设置
        for key, value in database_settings.items():
            if key in SETTINGS:
                old_settings[key] = SETTINGS[key]
                SETTINGS[key] = value

        # 获取数据库连接
        database = get_database()

        # 加载数据
        ticks = database.load_tick_data(symbol, exchange, start, end)

        # 恢复原始设置
        for key, value in old_settings.items():
            SETTINGS[key] = value

        return ticks
    else:
        # 使用默认数据库连接
        database = get_database()
        return database.load_tick_data(symbol, exchange, start, end)


def optimize(
    target_name: str,
    strategy_class: Type[CtaTemplate],
    vt_symbols: List[str],
    interval: str,
    start: datetime,
    end: datetime,
    rates: Dict[str, float],
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
        vt_symbols=vt_symbols,
        interval=interval,
        start=start,
        end=end,
        rates=rates,
        sizes=sizes,
        priceticks=priceticks,
        capital=capital,
    )

    engine.add_strategy(strategy_class, setting)

    if use_ga:
        result = run_ga_optimization(
            target_name=target_name,
            evaluator=engine,
            optimization_setting=optimization_setting,
            max_workers=max_workers,
        )
    else:
        result = run_optimization(
            target_name=target_name,
            evaluator=engine,
            optimization_setting=optimization_setting,
            max_workers=max_workers,
        )

    return result
