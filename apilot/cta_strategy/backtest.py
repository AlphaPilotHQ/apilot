from datetime import datetime, date, timedelta
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, List, Dict, Type, Tuple, Union, Optional
from functools import lru_cache
import traceback
import os

import numpy as np
from pandas import DataFrame, Series
from pandas.core.window import ExponentialMovingWindow
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from apilot.trader.constant import (
    Direction,
    Offset,
    Exchange,
    Interval,
    Status
)
from apilot.trader.database import get_database, BaseDatabase
from apilot.trader.object import OrderData, TradeData, BarData, TickData
from apilot.trader.utility import round_to, extract_vt_symbol

from .constants import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .strategy_base import CtaTemplate


class BacktestingEngine:

    engine_type: EngineType = EngineType.BACKTESTING
    gateway_name: str = "BACKTESTING"

    def __init__(self) -> None:
        self.vt_symbol: str = ""
        self.symbol: str = ""
        self.exchange: Exchange = None
        self.start: datetime = None
        self.end: datetime = None
        self.rate: float = 0
        self.slippage: float = 0
        self.size: float = 1
        self.pricetick: float = 0
        self.capital: int = 1_000_000
        self.annual_days: int = 240
        self.mode: BacktestingMode = BacktestingMode.BAR

        self.strategy_class: Type[CtaTemplate] = None
        self.strategy: CtaTemplate = None
        self.tick: TickData
        self.bar: BarData
        self.datetime: datetime = None

        self.interval: Interval = None
        self.days: int = 0
        self.callback: Callable = None
        self.history_data: list = []

        self.stop_order_count: int = 0
        self.stop_orders: Dict[str, StopOrder] = {}
        self.active_stop_orders: Dict[str, StopOrder] = {}

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
        self.strategy = None
        self.tick = None
        self.bar = None
        self.datetime = None

        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        rate: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        annual_days: int = 240,
    ) -> None:
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = Interval(interval)
        self.rate = rate
        self.slippage = 0
        self.size = size
        self.pricetick = pricetick
        self.start = start

        self.symbol, exchange_str = self.vt_symbol.split(".")
        self.exchange = Exchange(exchange_str)

        self.capital = capital

        if not end:
            end = datetime.now()
        self.end = end.replace(hour=23, minute=59, second=59)

        if self.start >= self.end:
            raise ValueError(f"错误：起始日期({self.start})必须小于结束日期({self.end})")

        self.annual_days = annual_days

    def add_strategy(self, strategy_class: Type[CtaTemplate], setting: dict) -> None:

        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbol, setting
        )

    def add_data(
        self,
        database_type: Optional[str] = None,
        **kwargs
    ) -> None:
        self.database_type = database_type
        self.database_config = kwargs

        return self 

    def load_data(self) -> None:

        self.history_data.clear()

        # 检查是否设置了数据源类型
        if not self.database_type:
            raise ValueError("错误：未设置数据源类型，请先调用add_data方法")

        # CSV数据源处理逻辑
        if self.database_type == "csv":
            self._load_from_csv()
        # MongoDB数据源处理逻辑
        elif self.database_type == "mongodb":
            self._load_from_database()
        # 其他数据源类型
        else:
            self.output(f"使用自定义数据源: {self.database_type}")
            self._load_from_database()

        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")

    def _load_from_csv(self) -> None:
        # 获取CSV文件路径
        data_path = self.database_config.get("data_path")

        # 检查是否提供了文件路径
        if not data_path:
            raise ValueError("错误：使用CSV数据源时必须提供data_path参数")

        # 检查文件是否存在
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"错误：找不到指定的CSV数据文件: {data_path}")

        self.output(f"使用数据源: csv，文件: {data_path}")

        try:
            # 读取CSV文件
            import pandas as pd
            df = pd.read_csv(data_path)

            # 获取列名配置
            datetime_col = self.database_config.get("datetime")
            if not datetime_col:
                raise ValueError("错误：必须指定datetime列名")
            
            if datetime_col not in df.columns:
                raise ValueError(f"错误：CSV文件中没有指定的时间列: {datetime_col}")

            # 确保datetime列为datetime类型
            df[datetime_col] = pd.to_datetime(df[datetime_col])

            # 过滤日期范围
            if self.start and self.end:
                df = df[(df[datetime_col] >= self.start) & (df[datetime_col] <= self.end)]
                if df.empty:
                    raise ValueError(f"错误：指定日期范围内没有数据: {self.start} 至 {self.end}")

            # 获取OHLCV列名
            open_col = self.database_config.get("open")
            high_col = self.database_config.get("high")
            low_col = self.database_config.get("low") 
            close_col = self.database_config.get("close")
            volume_col = self.database_config.get("volume")
            
            # 检查是否提供了所有必需的列名配置
            if not all([open_col, high_col, low_col, close_col, volume_col]):
                raise ValueError("错误：必须指定所有OHLCV列名")

            # 检查必需列是否存在
            missing_cols = []
            for name, col in {
                "open": open_col, 
                "high": high_col, 
                "low": low_col,
                "close": close_col, 
                "volume": volume_col
            }.items():
                if col not in df.columns:
                    missing_cols.append(f"{name}({col})")

            if missing_cols:
                raise ValueError(f"错误：CSV文件缺少必需列: {', '.join(missing_cols)}")

            # 创建BarData对象
            batch_data = []
            for _, row in df.iterrows():
                bar = BarData(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    datetime=row[datetime_col],
                    interval=self.interval,
                    volume=row[volume_col],
                    open_price=row[open_col],
                    high_price=row[high_col],
                    low_price=row[low_col],
                    close_price=row[close_col],
                    gateway_name=self.gateway_name
                )
                batch_data.append(bar)

            # 排序并添加
            batch_data.sort(key=lambda x: x.datetime)
            self.history_data.extend(batch_data)
            self.output(f"成功从文件加载了 {len(batch_data)} 条数据")

        except Exception as e:
            self.output(f"从CSV文件加载数据时出错: {str(e)}")
            raise

    def _load_from_database(self) -> None:
        """从数据库加载数据"""
        self.output(f"使用数据源: {self.database_type}")
        self.database_settings = self.database_config

        try:
            # 直接加载整个时间范围的数据
            if self.mode == BacktestingMode.BAR:
                self.history_data = load_bar_data(
                    self.symbol, 
                    self.exchange, 
                    self.interval,
                    self.start, 
                    self.end,
                    database_settings=self.database_settings
                )
            else:
                self.history_data = load_tick_data(
                    self.symbol, 
                    self.exchange,
                    self.start, 
                    self.end,
                    database_settings=self.database_settings
                )

        except Exception as e:
            self.output(f"从数据库加载数据时出错: {str(e)}")
            raise

    def run_backtesting(self) -> None:
        if not self.history_data:
            self.load_data()

        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()
        self.strategy.inited = True

        self.strategy.on_start()
        self.strategy.trading = True

        try:
            for data in self.history_data:
                func(data)
        except Exception as e:
            self.output(f"回测终止: {str(e)}\n{traceback.format_exc()}")
            return

        self.strategy.on_stop()
        self.output("历史数据回放结束")

    def calculate_result(self) -> DataFrame:
        """计算回测结果"""
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
        pre_close = 0
        start_pos = 0

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                self.size,
                self.rate,
                0  # 忽略slippage计算
            )
            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

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
        """计算统计指标"""
        # 使用当前结果或传入的DataFrame
        if df is None:
            df = self.daily_df

        # 初始化统计值
        stats = {
            "start_date": "",
            "end_date": "",
            "total_days": 0,
            "profit_days": 0,
            "loss_days": 0,
            "capital": self.capital,
            "end_balance": 0,
            "max_ddpercent": 0,
            "total_net_pnl": 0,
            "total_commission": 0,
            "total_turnover": 0,
            "total_trade_count": 0,
            "total_return": 0,
            "annual_return": 0,
            "return_std": 0,
            "sharpe_ratio": 0,
            "return_drawdown_ratio": 0,
        }

        positive_balance = False

        if df is not None and not df.empty:
            # 计算资金曲线相关数据
            df["balance"] = df["net_pnl"].cumsum() + self.capital

            # 计算日收益率，当资金为负时设为0
            pre_balance = df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital
            x = df["balance"] / pre_balance
            x[x <= 0] = np.nan
            df["return"] = np.log(x).fillna(0)

            # 计算高点和回撤
            df["highlevel"] = df["balance"].cummax()
            df["ddpercent"] = (df["balance"] - df["highlevel"]) / df["highlevel"] * 100

            # 所有资金值必须为正
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

        # 计算统计值
        if positive_balance:
            stats.update({
                "start_date": df.index[0],
                "end_date": df.index[-1],
                "total_days": len(df),
                "profit_days": (df["net_pnl"] > 0).sum(),
                "loss_days": (df["net_pnl"] < 0).sum(),
                "end_balance": df["balance"].iloc[-1],
                "max_ddpercent": df["ddpercent"].min(),
                "total_net_pnl": df["net_pnl"].sum(),
                "total_commission": df["commission"].sum(),
                "total_turnover": df["turnover"].sum(),
                "total_trade_count": df["trade_count"].sum(),
            })

            # 计算收益相关指标
            stats["total_return"] = (stats["end_balance"] / self.capital - 1) * 100
            stats["annual_return"] = stats["total_return"] / stats["total_days"] * self.annual_days
            stats["return_std"] = df["return"].std() * 100

            if stats["return_std"]:
                stats["sharpe_ratio"] = (df["return"].mean() * 100) / stats["return_std"] * np.sqrt(self.annual_days)

            if stats["max_ddpercent"]:
                stats["return_drawdown_ratio"] = -stats["total_return"] / stats["max_ddpercent"]

        # 输出结果
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{stats['start_date']}")
            self.output(f"最后交易日：\t{stats['end_date']}")
            self.output(f"总交易日：\t{stats['total_days']}")
            self.output(f"盈利交易日：\t{stats['profit_days']}")
            self.output(f"亏损交易日：\t{stats['loss_days']}")
            self.output(f"起始资金：\t{self.capital:.2f}")
            self.output(f"结束资金：\t{stats['end_balance']:.2f}")
            self.output(f"总收益率：\t{stats['total_return']:.2f}%")
            self.output(f"年化收益：\t{stats['annual_return']:.2f}%")
            self.output(f"百分比最大回撤: {stats['max_ddpercent']:.2f}%")
            self.output(f"总盈亏：\t{stats['total_net_pnl']:.2f}")
            self.output(f"总手续费：\t{stats['total_commission']:.2f}")
            self.output(f"总成交金额：\t{stats['total_turnover']:.2f}")
            self.output(f"总成交笔数：\t{stats['total_trade_count']}")
            self.output(f"收益标准差：\t{stats['return_std']:.2f}%")
            self.output(f"Sharpe Ratio：\t{stats['sharpe_ratio']:.2f}")
            self.output(f"收益回撤比：\t{stats['return_drawdown_ratio']:.2f}")

        # 处理无限值
        stats = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in stats.items()}

        self.output("策略统计指标计算完成")
        return stats

    def show_chart(self, df: DataFrame = None) -> None:
        if not df:
            df = self.daily_df

        if df is None:
            return

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df.index,
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df.index,
            y=df["ddpercent"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def update_daily_close(self, price: float) -> None:
        """"""
        d: date = self.datetime.date()

        daily_result: DailyResult = self.daily_results.get(d, None)
        if daily_result:
            daily_result.close_price = price
        else:
            self.daily_results[d] = DailyResult(d, price)

    def new_bar(self, bar: BarData) -> None:
        self.bar = bar
        self.datetime = bar.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_bar(bar)

        self.update_daily_close(bar.close_price)

    def new_tick(self, tick: TickData) -> None:
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self) -> None:
        """撮合限价单"""
        # 根据模式设置触发价格
        if self.mode == BacktestingMode.BAR:
            buy_price = self.bar.low_price
            sell_price = self.bar.high_price
        else:
            buy_price = self.tick.ask_price_1
            sell_price = self.tick.bid_price_1

        for order in list(self.active_limit_orders.values()):
            # 更新订单状态
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

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

            # 更新策略持仓和记录
            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)
            self.trades[trade.vt_tradeid] = trade

    def cross_stop_order(self) -> None:
        """撮合止损单"""
        # 根据模式设置触发价格
        if self.mode == BacktestingMode.BAR:
            buy_trigger_price = self.bar.high_price
            sell_trigger_price = self.bar.low_price
            buy_price = buy_trigger_price
            sell_price = sell_trigger_price
        else:
            buy_trigger_price = self.tick.last_price
            sell_trigger_price = self.tick.last_price
            buy_price = getattr(self.tick, "ask_price_1", self.tick.last_price)
            sell_price = getattr(self.tick, "bid_price_1", self.tick.last_price)

        for stop_order in list(self.active_stop_orders.values()):
            # 检查是否触发止损
            buy_triggered = (
                stop_order.direction == Direction.LONG
                and stop_order.price <= buy_trigger_price
            )
            sell_triggered = (
                stop_order.direction == Direction.SHORT
                and stop_order.price >= sell_trigger_price
            )

            if not buy_triggered and not sell_triggered:
                continue

            # 创建订单数据
            self.limit_order_count += 1
            order = OrderData(
                symbol=self.symbol,
                exchange=self.exchange,
                orderid=str(self.limit_order_count),
                direction=stop_order.direction,
                offset=stop_order.offset,
                price=stop_order.price,
                volume=stop_order.volume,
                traded=stop_order.volume,
                status=Status.ALLTRADED,
                gateway_name=self.gateway_name,
                datetime=self.datetime
            )
            self.limit_orders[order.vt_orderid] = order

            # 创建成交数据
            self.trade_count += 1
            trade_price = buy_price if buy_triggered else sell_price
            pos_change = order.volume if buy_triggered else -order.volume

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
            self.trades[trade.vt_tradeid] = trade

            # 更新止损单状态
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # 推送更新给策略
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)
            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool
    ) -> List[BarData]:
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
            database_settings=getattr(self, "database_settings", None)
        )

        return bars

    def load_tick(self, vt_symbol: str, days: int, callback: Callable) -> List[TickData]:
        self.callback = callback

        init_end = self.start - timedelta(seconds=1)
        init_start = self.start - timedelta(days=days)

        symbol, exchange = extract_vt_symbol(vt_symbol)

        ticks: List[TickData] = load_tick_data(
            symbol,
            exchange,
            init_start,
            init_end,
            database_settings=getattr(self, "database_settings", None)
        )

        return ticks

    def send_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool = False,
        net: bool = False
    ) -> list:
        price: float = round_to(price, self.pricetick)
        if stop:
            vt_orderid: str = self.send_stop_order(direction, offset, price, volume)
        else:
            vt_orderid: str = self.send_limit_order(direction, offset, price, volume)
        return [vt_orderid]

    def send_stop_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ) -> str:
        self.stop_order_count += 1

        stop_order: StopOrder = StopOrder(
            vt_symbol=self.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            datetime=self.datetime,
            stop_orderid=f"{STOPORDER_PREFIX}.{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )

        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order

        return stop_order.stop_orderid

    def send_limit_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ) -> str:
        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=self.symbol,
            exchange=self.exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
            datetime=self.datetime
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        if vt_orderid not in self.active_stop_orders:
            return
        stop_order: StopOrder = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        vt_orderids: list = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids: list = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None) -> None:
        msg: str = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None) -> None:
        pass

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        pass

    def get_engine_type(self) -> EngineType:
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate) -> float:
        return self.pricetick

    def get_size(self, strategy: CtaTemplate) -> int:
        return self.size

    def output(self, msg) -> None:
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self) -> list:
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self) -> list:
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list:
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())

    def run_single_backtest(
        self,
        strategy_class: Type[CtaTemplate],
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        size: int,
        pricetick: float,
        capital: int,
        setting: dict
    ) -> Tuple[DataFrame, dict]:
        """
        Run a single strategy backtest and return the results.

        This is a high-level method that combines multiple steps into one
        for easier usage.
        """
        self.clear_data()

        if interval == Interval.TICK.value:
            mode = BacktestingMode.TICK
        else:
            mode = BacktestingMode.BAR

        self.set_parameters(
            vt_symbol=vt_symbol,
            interval=interval,
            start=start,
            end=end,
            rate=rate,
            size=size,
            pricetick=pricetick,
            capital=capital,
            mode=mode
        )

        self.add_strategy(strategy_class, setting)
        self.load_data()
        self.run_backtesting()
        self.calculate_result()

        statistics = self.calculate_statistics()
        df = self.daily_df.copy()

        return df, statistics


class DailyResult:

    def __init__(self, date: date, close_price: float) -> None:
        self.date: date = date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: List[TradeData] = []
        self.trade_count: int = 0

        self.start_pos = 0
        self.end_pos = 0

        self.turnover: float = 0
        self.commission: float = 0
        self.slippage: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int,
        rate: float,
        slippage: float = 0  # 保持兼容，但忽略此参数
    ) -> None:
        # 如果第一天没有提供pre_close，使用1避免除零错误
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # 持仓盈亏是开始持仓的盈亏
        self.start_pos = start_pos
        self.end_pos = start_pos
        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # 交易盈亏是当天新交易产生的盈亏
        self.trade_count = len(self.trades)

        for trade in self.trades:
            pos_change = trade.volume if trade.direction == Direction.LONG else -trade.volume
            self.end_pos += pos_change

            turnover = trade.volume * size * trade.price
            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.slippage = 0

            self.turnover += turnover
            self.commission += turnover * rate

        # 净盈亏只考虑手续费（滑点已移除）
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission


@lru_cache(maxsize=1024)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime,
    database_settings: dict = None
) -> List[BarData]:
    """加载K线数据"""
    # 获取数据库实例，如果提供了特殊设置，则使用这些设置
    if database_settings:
        # 用于支持MongoDB等特殊数据库的实现
        # 注意：实际的MongoDB数据库处理应该在具体的数据库实现类中处理
        from apilot.trader.setting import SETTINGS
        old_settings = {}

        # 暂存原始设置
        for key, value in database_settings.items():
            if hasattr(SETTINGS, key):
                old_settings[key] = getattr(SETTINGS, key)
                setattr(SETTINGS, key, value)

        # 获取数据库连接
        database = get_database()

        # 加载数据
        bars = database.load_bar_data(symbol, exchange, interval, start, end)

        # 恢复原始设置
        for key, value in old_settings.items():
            setattr(SETTINGS, key, value)

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
    database_settings: dict = None
) -> List[TickData]:
    """加载Tick数据"""
    # 获取数据库实例，如果提供了特殊设置，则使用这些设置
    if database_settings:
        # 用于支持MongoDB等特殊数据库的实现
        # 注意：实际的MongoDB数据库处理应该在具体的数据库实现类中处理
        from apilot.trader.setting import SETTINGS
        old_settings = {}

        # 暂存原始设置
        for key, value in database_settings.items():
            if hasattr(SETTINGS, key):
                old_settings[key] = getattr(SETTINGS, key)
                setattr(SETTINGS, key, value)

        # 获取数据库连接
        database = get_database()

        # 加载数据
        ticks = database.load_tick_data(symbol, exchange, start, end)

        # 恢复原始设置
        for key, value in old_settings.items():
            setattr(SETTINGS, key, value)

        return ticks
    else:
        # 使用默认数据库连接
        database = get_database()
        return database.load_tick_data(symbol, exchange, start, end)
