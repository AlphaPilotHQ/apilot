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
    round_to,
    load_json,
    save_json
)
from apilot.core.database import get_database
from apilot.optimizer import (
    OptimizationSetting,
    run_ga_optimization
)
from apilot.strategy.template import CtaTemplate
from apilot.utils.logger import get_logger, set_level
from apilot.datafeed.csv_database import CsvDatabase

# 回测默认设置
BACKTEST_CONFIG = {
    "risk_free": 0.0,
    "size": 1,
    "pricetick": 0.0,
    "capital": 1000000,
}

# 设置文件名
SETTING_FILENAME: str = "apilot_setting.json"

# 从JSON文件加载配置
json_config = load_json(SETTING_FILENAME)

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
        self.vt_symbols: List[str] = []
        self.symbols: Dict[str, str] = {}
        self.exchanges: Dict[str, Exchange] = {}
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

        # 添加数据源配置
        self.database_type = None
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
        sizes: Dict[str, float],
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
        self.vt_symbols = vt_symbols
        self.interval = Interval(interval)
        self.sizes = sizes
        self.priceticks = priceticks or {}  # 使用空字典作为默认值
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
            logger.warning(f"错误：起始日期({self.start})必须小于结束日期({self.end})")

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

    def add_data(self, database_type: Optional[str] = None, **kwargs) -> "BacktestingEngine":
        """
        添加数据
        """
        # 检查是否指定了特定交易对
        specific_symbol = kwargs.pop("specific_symbol", None)

        # 检查是否为直接CSV文件路径
        data_path = kwargs.get("data_path", "")
        if (
            database_type == "csv"
            and data_path
            and data_path.endswith(".csv")
            and os.path.isfile(data_path)
        ):
            # 记录日志，帮助调试
            logger.info(f"设置CSV数据源: {data_path}")

            # 如果指定了CSV字段映射，记录到日志
            if "datetime" in kwargs:
                dt_field = kwargs["datetime"]
                open_field = kwargs.get("open", "open")
                high_field = kwargs.get("high", "high")
                low_field = kwargs.get("low", "low")
                close_field = kwargs.get("close", "close")
                volume_field = kwargs.get("volume", "volume")

                logger.info(f"字段映射: datetime={dt_field}, open={open_field}, high={high_field}, low={low_field}, close={close_field}, volume={volume_field}")

            # 如果指定了特定交易对，为该交易对创建单独的数据配置
            if specific_symbol:
                # 记录日志
                logger.info(f"为特定交易对 {specific_symbol} 指定数据源: {data_path}")

                # 找到匹配的vt_symbol
                matched_symbols = [s for s in self.vt_symbols if specific_symbol in s]
                if matched_symbols:
                    # 为该交易对创建特定的数据配置
                    for vt_symbol in matched_symbols:
                        if not hasattr(self, "symbol_data_configs"):
                            self.symbol_data_configs = {}
                        self.symbol_data_configs[vt_symbol] = {
                            "database_type": database_type,
                            "data_path": data_path,
                            **{k: v for k, v in kwargs.items() if k in ["datetime", "open", "high", "low", "close", "volume"]}
                        }
                else:
                    logger.warning(f"未找到匹配的交易对: {specific_symbol}")

            # 默认情况下，设置为全局数据源
            self.database_type = database_type
            self.database_config = {
                "data_path": data_path,
                **{k: v for k, v in kwargs.items() if k in ["datetime", "open", "high", "low", "close", "volume"]}
            }
            self.specific_data_file = data_path

            logger.info(f"步骤4: 数据添加完成: {data_path}")
            return self

        # 设置普通数据库
        self.database_type = database_type
        self.database_config = kwargs
        logger.info("步骤4: 数据添加完成")
        return self

    def load_data(self) -> None:
        """
        加载数据
        """
        # 初始化历史数据存储
        logger.info("开始加载历史数据")
        self.history_data = {}
        self.dts = []

        # 检查是否有交易对特定的数据配置
        if hasattr(self, "symbol_data_configs") and self.symbol_data_configs:
            # 遍历每个交易对的特定数据配置
            for vt_symbol, config in self.symbol_data_configs.items():
                if config["database_type"] == "csv":
                    # 获取数据路径
                    data_path = config["data_path"]
                    if not os.path.isfile(data_path):
                        logger.error(f"CSV文件不存在: {data_path}")
                        continue

                    # 从vt_symbol中提取基础交易对名称
                    base_symbol = vt_symbol.split(".")[0]  # 例如从"SOL-USDT.LOCAL"提取"SOL-USDT"

                    logger.info(f"从CSV直接加载 {base_symbol} 数据，时间: {self.start} - {self.end}")

                    # 创建CsvDatabase实例
                    database = CsvDatabase(data_path)

                    # 设置数据字段映射
                    field_config = {k: v for k, v in config.items() if k in ["datetime", "open", "high", "low", "close", "volume"]}
                    if field_config:
                        for field_name, csv_column in field_config.items():
                            if hasattr(database, f"{field_name}_field"):
                                setattr(database, f"{field_name}_field", csv_column)

                    # 提取交易对信息
                    symbol, exchange = extract_vt_symbol(vt_symbol)

                    # 加载数据
                    bars = database.load_bar_data(
                        symbol=symbol,
                        exchange=Exchange.LOCAL,  # 使用枚举类型
                        interval=self.interval,
                        start=self.start,
                        end=self.end
                    )

                    # 处理数据
                    data = []
                    for bar in bars:
                        bar.vt_symbol = vt_symbol  # 确保vt_symbol正确
                        data.append(bar)
                        self.dts.append(bar.datetime)
                        self.history_data.setdefault(bar.datetime, {})[vt_symbol] = bar

                    logger.info(f"加载了 {len(data)} 条 {vt_symbol} 的历史数据")

            # 对时间点从小到大排序
            self.dts = sorted(list(set(self.dts)))
            logger.info(f"历史数据加载完成，数据量：{len(self.dts)}")
            return

        # 如果没有特定交易对配置，使用默认方式加载数据
        logger.info("使用默认数据源加载数据")

        # 处理默认数据源
        if self.database_type == "csv" and isinstance(self.database_config, dict) and "data_path" in self.database_config:
            data_path = self.database_config["data_path"]
            field_config = {k: v for k, v in self.database_config.items() if k in ["datetime", "open", "high", "low", "close", "volume"]}

            for vt_symbol in self.vt_symbols:
                base_symbol = vt_symbol.split(".")[0]
                symbol, exchange = extract_vt_symbol(vt_symbol)

                logger.info(f"从默认CSV加载 {symbol} 数据，时间: {self.start} - {self.end}")

                database = CsvDatabase(data_path)

                # 设置字段映射
                if field_config:
                    for field_name, csv_column in field_config.items():
                        if hasattr(database, f"{field_name}_field"):
                            setattr(database, f"{field_name}_field", csv_column)

                # 加载数据
                bars = database.load_bar_data(
                    symbol=symbol,
                    exchange=Exchange.LOCAL,
                    interval=self.interval,
                    start=self.start,
                    end=self.end
                )

                data = []
                for bar in bars:
                    bar.vt_symbol = vt_symbol
                    data.append(bar)
                    self.dts.append(bar.datetime)
                    self.history_data.setdefault(bar.datetime, {})[vt_symbol] = bar

                logger.info(f"加载了 {len(data)} 条 {vt_symbol} 的历史数据")

        # 对时间点从小到大排序
        self.dts = sorted(list(set(self.dts)))
        logger.info(f"历史数据加载完成，数据量：{len(self.dts)}")

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
                    logger.debug(f"回测进度: 已处理 {tick_count} 个时间点, 当前时间: {dt}")
                self.new_bars(dt)
            except Exception as e:
                logger.error(f"触发异常，回测终止: {e}")
                logger.error(traceback.format_exc())
                return

        logger.info("历史数据回放结束")
        logger.debug(f"回测完成统计: 总交易笔数={self.trade_count}, 活跃订单数={len(self.active_limit_orders)}, 总订单数={len(self.limit_orders)}")

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
        for vt_symbol, bar in bars.items():
            self.update_daily_close(bar.close_price, vt_symbol)

    def calculate_result(self) -> DataFrame:
        """
        计算回测结果
        """
        if not self.trades:
            logger.info("回测成交记录为空")
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
        logger.info("逐日盯市盈亏计算完成")
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
            "total_turnover": 0,
            "total_trade_count": 0,
            "total_return": 0,
            "annual_return": 0,
            "sharpe_ratio": 0,
            "return_drawdown_ratio": 0,
        }

        # Return early if no data
        if df is None or df.empty:
            logger.warning("No trading data available")
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
            logger.warning("Bankruptcy detected during backtest")
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
            logger.info(f"Trade day:\t{stats['start_date']} - {stats['end_date']}")
            logger.info(f"Profit days:\t{stats['profit_days']}")
            logger.info(f"Loss days:\t{stats['loss_days']}")
            logger.info(f"Initial capital:\t{self.capital:.2f}")
            logger.info(f"Final capital:\t{stats['end_balance']:.2f}")
            logger.info(f"Total return:\t{stats['total_return']:.2f}%")
            logger.info(f"Annual return:\t{stats['annual_return']:.2f}%")
            logger.info(f"Max drawdown:\t{stats['max_ddpercent']:.2f}%")
            logger.info(f"Total turnover:\t{stats['total_turnover']:.2f}")
            logger.info(f"Total trades:\t{stats['total_trade_count']}")
            logger.info(f"Sharpe ratio:\t{stats['sharpe_ratio']:.2f}")
            logger.info(f"Return/Drawdown:\t{stats['return_drawdown_ratio']:.2f}")
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
            logger.debug(f"检查订单: {order.vt_orderid}, 方向: {order.direction}, 价格: {order.price}")

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
                    logger.debug(f"找不到订单对应的K线数据: {vt_symbol}")
                    continue
                buy_price = bar.low_price
                sell_price = bar.high_price
                logger.debug(f"Bar模式下的价格 - 买入价: {buy_price}, 卖出价: {sell_price}")
            else:
                if self.tick.vt_symbol != vt_symbol:
                    continue
                buy_price = self.tick.ask_price_1
                sell_price = self.tick.bid_price_1
                logger.debug(f"Tick模式下的价格 - 买入价: {buy_price}, 卖出价: {sell_price}")

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
            logger.debug(f"成交记录创建: {trade.vt_tradeid}, 方向: {trade.direction}, 价格: {trade.price}, 数量: {trade.volume}")

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

        logger.debug(f"创建订单 - 合约: {vt_symbol}, 方向: {direction}, 价格: {price}, 数量: {volume}")

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

        logger.debug(f"订单已创建: {order.vt_orderid}")
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
        return self.priceticks.get(vt_symbol, 0.0001)

    def get_size(self, strategy: CtaTemplate, vt_symbol: str) -> int:
        """
        获取合约大小
        """
        return self.sizes[vt_symbol]

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
            close_price = self.close_prices.get(vt_symbol, trade.price)

            turnover = trade.volume * size * trade.price
            self.trading_pnl += pos_change * (close_price - trade.price) * size

            self.turnover += turnover

        # 计算净盈亏 (无手续费)
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl


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
        available_settings = {
            "csv_data_path": "data_path",
            "csv_datetime_field": "datetime",
            "csv_open_field": "open",
            "csv_high_field": "high",
            "csv_low_field": "low",
            "csv_close_field": "close",
            "csv_volume_field": "volume",
        }
        old_settings = {}

        # 暂存原始设置
        for key, value in database_settings.items():
            setting_key = available_settings.get(key, key)
            if setting_key in BACKTEST_CONFIG:
                old_settings[setting_key] = BACKTEST_CONFIG[setting_key]
                BACKTEST_CONFIG[setting_key] = value

        # 获取数据库连接
        database = get_database()

        # 加载数据
        bars = database.load_bar_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end,
        )

        # 恢复原始设置
        for key, value in old_settings.items():
            BACKTEST_CONFIG[key] = value

        return bars
    else:
        # 使用默认数据库连接
        database = get_database()
        return database.load_bar_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end,
        )


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
        available_settings = {
            "csv_data_path": "data_path",
            "csv_datetime_field": "datetime",
            "csv_open_field": "open",
            "csv_high_field": "high",
            "csv_low_field": "low",
            "csv_close_field": "close",
            "csv_volume_field": "volume",
        }
        old_settings = {}

        # 暂存原始设置
        for key, value in database_settings.items():
            setting_key = available_settings.get(key, key)
            if setting_key in BACKTEST_CONFIG:
                old_settings[setting_key] = BACKTEST_CONFIG[setting_key]
                BACKTEST_CONFIG[setting_key] = value

        # 获取数据库连接
        database = get_database()

        # 加载数据
        ticks = database.load_tick_data(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
        )

        # 恢复原始设置
        for key, value in old_settings.items():
            BACKTEST_CONFIG[key] = value

        return ticks
    else:
        # 使用默认数据库连接
        database = get_database()
        return database.load_tick_data(
            symbol=symbol,
            exchange=exchange,
            start=start,
            end=end,
        )


def optimize(
    target_name: str,
    strategy_class: Type[CtaTemplate],
    vt_symbols: List[str],
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
        vt_symbols=vt_symbols,
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
