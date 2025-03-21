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

from .base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .template import CtaTemplate


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
        self.half_life: int = 120
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
        half_life: int = 120,
        slippage: float = 0,  
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
        self.half_life = half_life

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
        """"""
        self.database_type = database_type
        self.database_config = kwargs
        
        return self  # 支持链式调用

    def load_data(self) -> None:
        """加载历史数据"""
        # 清空之前的数据
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
        """从CSV文件加载数据"""
        # 获取CSV文件路径
        data_path = self.database_config.get("data_path")
        
        # 检查是否提供了文件路径
        if not data_path:
            raise ValueError("错误：使用CSV数据源时必须提供data_path参数")
            
        # 检查文件是否存在
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"错误：找不到指定的CSV数据文件: {data_path}")
            
        self.output(f"找到指定数据文件: {data_path}")
        self.output(f"使用数据源: csv")
        
        try:
            # 读取CSV文件
            import pandas as pd
            self.output(f"从指定文件加载数据: {data_path}")
            df = pd.read_csv(data_path)
            
            # 处理时间列
            datetime_col = self.database_config.get("datetime")
            if not datetime_col or datetime_col not in df.columns:
                # 如果没有指定或指定的列不存在，尝试自动检测
                for col in ["datetime", "candle_begin_time", "date", "time", "Date", "Time"]:
                    if col in df.columns:
                        datetime_col = col
                        break
            
            if not datetime_col:
                raise ValueError("错误：CSV文件中未找到时间列")
            
            # 确保datetime列为datetime类型
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            
            # 过滤日期范围
            if self.start and self.end:
                df = df[(df[datetime_col] >= self.start) & (df[datetime_col] <= self.end)]
                
                # 检查过滤后是否还有数据
                if df.empty:
                    raise ValueError(f"错误：指定日期范围内没有数据: {self.start} 至 {self.end}")
            
            # 获取OHLCV列名
            open_col = self.database_config.get("open", "open")
            high_col = self.database_config.get("high", "high") 
            low_col = self.database_config.get("low", "low")
            close_col = self.database_config.get("close", "close")
            volume_col = self.database_config.get("volume", "volume")
            
            # 检查列是否存在
            required_cols = {
                "open": open_col,
                "high": high_col,
                "low": low_col,
                "close": close_col,
                "volume": volume_col
            }
            
            # 检查是否所有必需列都找到了
            missing_cols = []
            for name, col in required_cols.items():
                if col not in df.columns:
                    missing_cols.append(f"{name}({col})")
            
            if missing_cols:
                raise ValueError(f"错误：CSV文件缺少必需列: {', '.join(missing_cols)}")
            
            # 创建BarData对象
            batch_data = []
            for _, row in df.iterrows():
                dt = row[datetime_col]
                
                # 创建K线数据对象
                bar = BarData(
                    symbol=self.symbol,
                    exchange=self.exchange,
                    datetime=dt,
                    interval=self.interval,
                    volume=row[volume_col],
                    open_price=row[open_col],
                    high_price=row[high_col],
                    low_price=row[low_col],
                    close_price=row[close_col],
                    gateway_name=self.gateway_name
                )
                
                batch_data.append(bar)
            
            # 排序
            batch_data.sort(key=lambda x: x.datetime)
            
            self.history_data.extend(batch_data)
            self.output(f"成功从文件加载了 {len(batch_data)} 条数据")
            
        except Exception as e:
            self.output(f"从CSV文件加载数据时出错: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _load_from_database(self) -> None:
        """从数据库加载数据"""
        self.output(f"使用数据源: {self.database_type}")
        
        # 设置数据库配置
        self.database_settings = self.database_config
        
        # 计算进度分块
        total_days = (self.end - self.start).days
        progress_days = max(int(total_days / 10), 1)
        progress_delta = timedelta(days=progress_days)
        interval_delta = INTERVAL_DELTA_MAP[self.interval]
        
        # 分块加载数据
        current_start = self.start
        
        try:
            while current_start < self.end:
                current_end = min(current_start + progress_delta, self.end)
                
                # 根据模式加载适当的数据类型
                if self.mode == BacktestingMode.BAR:
                    batch_data = load_bar_data(
                        self.symbol, self.exchange, self.interval, current_start, current_end, 
                        database_settings=self.database_settings
                    )
                else:
                    batch_data = load_tick_data(
                        self.symbol, self.exchange, current_start, current_end,
                        database_settings=self.database_settings
                    )
                
                self.history_data.extend(batch_data)
                current_start = current_end + interval_delta
                
        except Exception as e:
            self.output(f"从数据库加载数据时出错: {str(e)}\n{traceback.format_exc()}")
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

        # 简化批处理逻辑
        try:
            for data in self.history_data:
                func(data)
        except Exception as e:
            self.output(f"回测终止: {str(e)}\n{traceback.format_exc()}")
            return

        self.strategy.on_stop()
        self.output("历史数据回放结束")

    def calculate_result(self) -> DataFrame:

        if not self.trades:
            self.output("回测成交记录为空")
            return DataFrame()  

        # Add trade data into daily result
        for trade in self.trades.values():
            d: date = trade.datetime.date()
            daily_result: DailyResult = self.daily_results.get(d, None)
            if daily_result:
                daily_result.add_trade(trade)

        # Calculate daily result by iteration
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

        # Generate dataframe 
        first_result = next(iter(self.daily_results.values()))
        results = {
            key: [getattr(dr, key) for dr in self.daily_results.values()]
            for key in first_result.__dict__
        }

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True) -> dict:

        # Check DataFrame input exterior
        if df is None:
            df: DataFrame = self.daily_df

        # Init all statistics default value
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_ddpercent: float = 0
        total_net_pnl: float = 0
        total_commission: float = 0
        total_turnover: float = 0
        total_trade_count: int = 0
        total_return: float = 0
        annual_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        ewm_sharpe: float = 0
        return_drawdown_ratio: float = 0

        # Check if balance is always positive
        positive_balance: bool = False

        if df is not None:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital

            # When balance falls below 0, set daily return to 0
            pre_balance: Series = df["balance"].shift(1)
            pre_balance.iloc[0] = self.capital
            x = df["balance"] / pre_balance
            x[x <= 0] = np.nan
            df["return"] = np.log(x).fillna(0)

            df["highlevel"] = df["balance"].rolling(min_periods=1, window=len(df), center=False).max()
            df["ddpercent"] = (df["balance"] - df["highlevel"]) / df["highlevel"] * 100

            # All balance value needs to be positive
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                self.output("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

        # Calculate statistics value
        if positive_balance:
            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = (df["net_pnl"] > 0).sum()
            loss_days = (df["net_pnl"] < 0).sum()

            end_balance = df["balance"].iloc[-1]

            max_ddpercent = df["ddpercent"].min()

            total_net_pnl = df["net_pnl"].sum()
            total_commission = df["commission"].sum()
            total_turnover = df["turnover"].sum()
            total_trade_count = df["trade_count"].sum()

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            return_std = df["return"].std() * 100

            if return_std:
                sharpe_ratio = (df["return"].mean() * 100) / return_std * np.sqrt(self.annual_days)

                ewm_window = df["return"].ewm(halflife=self.half_life)
                ewm_mean = ewm_window.mean() * 100
                ewm_std = ewm_window.std() * 100
                ewm_sharpe = (ewm_mean / ewm_std)[-1] * np.sqrt(self.annual_days)
            else:
                sharpe_ratio = 0
                ewm_sharpe = 0

            return_drawdown_ratio = -total_return / max_ddpercent if max_ddpercent else 0

        # Output
        if output:
            self.output("-" * 30)
            self.output("首个交易日：\t{}".format(start_date))
            self.output("最后交易日：\t{}".format(end_date))

            self.output("总交易日：\t{}".format(total_days))
            self.output("盈利交易日：\t{}".format(profit_days))
            self.output("亏损交易日：\t{}".format(loss_days))

            self.output("起始资金：\t{:.2f}".format(self.capital))
            self.output("结束资金：\t{:.2f}".format(end_balance))

            self.output("总收益率：\t{:.2f}%".format(total_return))
            self.output("年化收益：\t{:.2f}%".format(annual_return))
            self.output("百分比最大回撤: {:.2f}%".format(max_ddpercent))

            self.output("总盈亏：\t{:.2f}".format(total_net_pnl))
            self.output("总手续费：\t{:.2f}".format(total_commission))
            self.output("总成交金额：\t{:.2f}".format(total_turnover))
            self.output("总成交笔数：\t{}".format(total_trade_count))

            self.output("收益标准差：\t{:.2f}%".format(return_std))
            self.output("Sharpe Ratio：\t{:.2f}".format(sharpe_ratio))
            self.output("EWM Sharpe：\t{:.2f}".format(ewm_sharpe))
            self.output("收益回撤比：\t{:.2f}".format(return_drawdown_ratio))

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_ddpercent": max_ddpercent,
            "total_net_pnl": total_net_pnl,
            "total_commission": total_commission,
            "total_turnover": total_turnover,
            "total_trade_count": total_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "ewm_sharpe": ewm_sharpe,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # 简化无限值处理
        statistics = {k: np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for k, v in statistics.items()}

        self.output("策略统计指标计算完成")
        return statistics

    def show_chart(self, df: DataFrame = None) -> None:
        """"""
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
        """"""
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self) -> None:
        """
        Cross limit order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.low_price
            short_cross_price = self.bar.high_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.ask_price_1
            short_cross_price = self.tick.bid_price_1
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for order in list(self.active_limit_orders.values()):
            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            # Check whether limit orders can be filled.
            long_cross: bool = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross: bool = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = max(order.price, short_best_price)
                pos_change = -order.volume

            trade: TradeData = TradeData(
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

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

            self.trades[trade.vt_tradeid] = trade

    def cross_stop_order(self) -> None:
        """
        Cross stop order with last bar/tick data.
        """
        if self.mode == BacktestingMode.BAR:
            long_cross_price = self.bar.high_price
            short_cross_price = self.bar.low_price
            long_best_price = self.bar.open_price
            short_best_price = self.bar.open_price
        else:
            long_cross_price = self.tick.last_price
            short_cross_price = self.tick.last_price
            long_best_price = long_cross_price
            short_best_price = short_cross_price

        for stop_order in list(self.active_stop_orders.values()):
            # Check whether stop order can be triggered.
            long_cross: bool = (
                stop_order.direction == Direction.LONG
                and stop_order.price <= long_cross_price
            )

            short_cross: bool = (
                stop_order.direction == Direction.SHORT
                and stop_order.price >= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Create order data.
            self.limit_order_count += 1

            order: OrderData = OrderData(
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

            # Create trade data.
            if long_cross:
                trade_price = max(stop_order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = min(stop_order.price, short_best_price)
                pos_change = -order.volume

            self.trade_count += 1

            trade: TradeData = TradeData(
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

            # Update stop order.
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # Push update to strategy.
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
        stop: bool,
        lock: bool,
        net: bool
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
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_stop_orders:
            return
        stop_order: StopOrder = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        """
        Cancel all orders, both limit and stop.
        """
        vt_orderids: list = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids: list = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        Write log message.
        """
        msg: str = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None) -> None:
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        pass

    def get_engine_type(self) -> EngineType:
        """
        Return engine type.
        """
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate) -> float:
        """
        Return contract pricetick data.
        """
        return self.pricetick

    def get_size(self, strategy: CtaTemplate) -> int:
        """
        Return contract size data.
        """
        return self.size

    def put_strategy_event(self, strategy: CtaTemplate) -> None:
        """
        Put an event to update strategy status.
        """
        pass

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
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            pos_change = trade.volume if trade.direction == Direction.LONG else -trade.volume
            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price
            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.slippage = 0

            self.turnover += turnover
            self.commission += turnover * rate

        # Net pnl takes account of commission only (slippage已移除)
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
