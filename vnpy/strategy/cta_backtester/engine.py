"""
CTA Strategy Backtester Engine.
"""

from datetime import datetime
from pathlib import Path
import importlib
import traceback
from types import ModuleType
from typing import Type, Dict, List, Tuple, Set, Any
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from itertools import product

import numpy as np
from pandas import DataFrame
from copy import copy

from vnpy.event import Event, EventEngine
from vnpy.trader.engine import BaseEngine, MainEngine
from vnpy.trader.object import (
    HistoryRequest,
    TickData,
    BarData,
    ContractData,
)
from vnpy.trader.constant import (
    Direction,
    Offset,
    Exchange,
    Interval,
    Status
)
from vnpy.trader.utility import extract_vt_symbol, load_json, save_json, BarGenerator
from vnpy.trader.database import get_database, BaseDatabase
from vnpy.trader.datafeed import get_datafeed, BaseDatafeed
from vnpy.vnpy_ctastrategy import CtaTemplate, StopOrder
from vnpy.vnpy_ctastrategy.backtesting import BacktestingEngine, BacktestingMode

APP_NAME = "CtaBacktester"

# 直接定义回测事件常量
EVENT_BACKTESTER_LOG = "eBacktesterLog"
EVENT_BACKTESTER_BACKTESTING_FINISHED = "eBacktesterBacktestingFinished"
EVENT_BACKTESTER_OPTIMIZATION_FINISHED = "eBacktesterOptimizationFinished"


class OptimizationSetting:
    """
    Optimization setting.
    """

    def __init__(self):
        """Constructor"""
        self.params = {}
        self.target_name = ""

    def add_parameter(
        self, name: str, start: float, end: float = None, step: float = None
    ):
        """Add parameter"""
        if not end and not step:
            self.params[name] = [start]
            return self

        if start >= end:
            raise ValueError("开始点必须小于结束点")

        if step <= 0:
            raise ValueError("步长必须大于0")

        value = start
        value_list = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

        return self

    def set_target(self, target_name: str):
        """Set target"""
        self.target_name = target_name

        return self

    def generate_settings(self):
        """Generate settings"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p))
            settings.append(setting)

        return settings


class BacktesterEngine(BaseEngine):
    """
    For running CTA strategy backtesting.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """Constructor"""
        super().__init__(main_engine, event_engine, APP_NAME)

        self.classes: Dict[str, Type[CtaTemplate]] = {}
        self.backtesting_engine: Optional[BacktestingEngine] = None
        self.thread: Optional[Thread] = None

        # 优雅地处理datafeed和database
        try:
            self.datafeed: BaseDatafeed = get_datafeed()
        except (KeyError, ImportError):
            self.write_log("无法初始化数据源，将使用本地CSV文件回测")
            self.datafeed = None
            
        try:
            self.database: BaseDatabase = get_database()
        except (KeyError, ImportError):
            self.write_log("无法初始化数据库，将使用内存存储")
            self.database = None

        # Backtesting reuslt
        self.result_df: Optional[DataFrame] = None
        self.result_statistics: Optional[dict] = None

        # Optimization result
        self.result_values: Optional[list] = None

    def init_engine(self) -> None:
        """Initialize backtester engine."""
        self.write_log("初始化CTA回测引擎")

        self.backtesting_engine = BacktestingEngine()
        # Redirect log from backtesting engine outside.
        self.backtesting_engine.output = self.write_log

        self.load_strategy_class()
        self.write_log("策略文件加载完成")

        self.init_datafeed()

    def init_datafeed(self) -> None:
        """
        Init datafeed client.
        """
        if self.datafeed:
            result: bool = self.datafeed.init(self.write_log)
            if result:
                self.write_log("数据服务初始化成功")
        else:
            self.write_log("数据服务未配置，将使用本地CSV文件回测")

    def write_log(self, msg: str) -> None:
        """
        Output log message.
        """
        event: Event = Event(EVENT_BACKTESTER_LOG)
        event.data = msg
        self.event_engine.put(event)

    def load_strategy_class(self) -> None:
        """
        Load strategy class from source code.
        """
        # 从vnpy_ctastrategy模块加载策略
        try:
            app_path: Path = Path(importlib.import_module("vnpy.vnpy_ctastrategy").__file__).parent
            path1: Path = app_path.joinpath("strategies")
            self.load_strategy_class_from_folder(path1, "vnpy.vnpy_ctastrategy.strategies")
        except Exception as e:
            self.write_log(f"从vnpy_ctastrategy加载策略失败: {str(e)}")

        # 从用户目录加载策略
        try:
            path2: Path = Path.cwd().joinpath("strategies")
            if path2.exists():
                self.load_strategy_class_from_folder(path2, "strategies")
        except Exception as e:
            self.write_log(f"从用户目录加载策略失败: {str(e)}")
            
        # 如果自定义策略目录不存在，就创建一个
        try:
            if not Path.cwd().joinpath("strategies").exists():
                Path.cwd().joinpath("strategies").mkdir(exist_ok=True)
                self.write_log("创建strategies目录")
        except Exception as e:
            self.write_log(f"创建strategies目录失败: {str(e)}")
            
        # 如果还没有策略，就从SimpleMaStrategy.py文件加载
        if not self.classes and Path.cwd().joinpath("strategies/simple_ma_strategy.py").exists():
            self.load_strategy_class_from_module("strategies.simple_ma_strategy")
            self.write_log("从strategies/simple_ma_strategy.py加载策略")
        
        # 打印所有加载的策略类
        strategy_names = list(self.classes.keys())
        if strategy_names:
            self.write_log(f"已加载的策略类: {', '.join(strategy_names)}")
        else:
            self.write_log("没有找到任何策略类")

    def load_strategy_class_from_folder(self, path: Path, module_name: str = "") -> None:
        """
        Load strategy class from certain folder.
        """
        for suffix in ["py", "pyd", "so"]:
            pathname: str = str(path.joinpath(f"*.{suffix}"))
            for filepath in glob(pathname):
                filename: str = Path(filepath).stem
                name: str = f"{module_name}.{filename}"
                self.load_strategy_class_from_module(name)

    def load_strategy_class_from_module(self, module_name: str) -> None:
        """
        Load strategy class from module file.
        """
        try:
            module: ModuleType = importlib.import_module(module_name)

            # 重载模块，确保如果策略文件中有任何修改，能够立即生效。
            importlib.reload(module)

            for name in dir(module):
                value = getattr(module, name)
                if (
                    isinstance(value, type)
                    and issubclass(value, CtaTemplate)
                    and value not in {CtaTemplate}
                ):
                    self.classes[value.__name__] = value
        except:  # noqa
            msg: str = f"策略文件{module_name}加载失败，触发异常：\n{traceback.format_exc()}"
            self.write_log(msg)

    def reload_strategy_class(self) -> None:
        """
        Reload strategy classes.
        """
        self.classes.clear()
        self.load_strategy_class()
        self.write_log("策略文件重载刷新完成")

    def get_strategy_class_names(self) -> List[str]:
        """
        Get names of strategy classes.
        """
        return list(self.classes.keys())

    def run_backtesting(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        setting: dict
    ) -> None:
        """
        Run backtesting.
        """
        self.result_df = None
        self.result_statistics = None

        engine: BacktestingEngine = self.backtesting_engine
        engine.clear_data()

        if interval == Interval.TICK.value:
            mode: BacktestingMode = BacktestingMode.TICK
        else:
            mode: BacktestingMode = BacktestingMode.BAR

        engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=interval,
            start=start,
            end=end,
            rate=rate,
            slippage=slippage,
            size=size,
            pricetick=pricetick,
            capital=capital,
            mode=mode
        )

        # 确保策略类已经加载
        if class_name not in self.classes:
            self.write_log(f"策略类 {class_name} 未找到，尝试重新加载策略")
            self.reload_strategy_class()
            
            if class_name not in self.classes:
                self.write_log(f"无法找到策略类 {class_name}，回测失败")
                return
        
        strategy_class: Type[CtaTemplate] = self.classes[class_name]
        self.write_log(f"使用策略类 {class_name} 开始回测")
        
        try:
            engine.add_strategy(
                strategy_class,
                setting
            )
        except Exception as e:
            self.write_log(f"添加策略失败: {str(e)}\n{traceback.format_exc()}")
            return

        try:
            self.write_log(f"开始加载数据: {vt_symbol}, {interval}, {start} - {end}")
            engine.load_data()
            if not engine.history_data:
                self.write_log("策略回测失败，历史数据为空")
                self.thread = None
                return
            self.write_log(f"成功加载历史数据: {len(engine.history_data)}条记录")
        except Exception as e:
            self.write_log(f"加载数据失败: {str(e)}\n{traceback.format_exc()}")
            self.thread = None
            return

        try:
            self.write_log("开始运行回测")
            engine.run_backtesting()
            self.write_log("回测完成")
        except Exception as e:
            msg: str = f"策略回测失败，触发异常：{str(e)}\n{traceback.format_exc()}"
            self.write_log(msg)
            self.thread = None
            return

        try:
            self.write_log("开始计算回测结果")
            self.result_df = engine.calculate_result()
            self.result_statistics = engine.calculate_statistics(output=False)
            self.write_log("回测结果计算完成")
        except Exception as e:
            self.write_log(f"计算结果失败: {str(e)}\n{traceback.format_exc()}")
            self.thread = None
            return

        # Clear thread object handler.
        self.thread = None

        # Put backtesting done event
        event: Event = Event(EVENT_BACKTESTER_BACKTESTING_FINISHED)
        self.event_engine.put(event)

    def start_backtesting(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        setting: dict
    ) -> bool:
        """
        Start backtesting in thread.
        """
        if self.thread:
            self.write_log("已有任务在运行，请等待完成")
            return False

        self.write_log("-" * 40)
        self.thread = Thread(
            target=self.run_backtesting,
            args=(
                class_name,
                vt_symbol,
                interval,
                start,
                end,
                rate,
                slippage,
                size,
                pricetick,
                capital,
                setting
            )
        )
        self.thread.start()

        return True

    def get_result_df(self) -> DataFrame:
        """
        Get result dataframe.
        """
        return self.result_df

    def get_result_statistics(self) -> dict:
        """
        Get result statistics.
        """
        return self.result_statistics

    def get_result_values(self) -> list:
        """
        Get result values.
        """
        return self.result_values

    def get_default_setting(self, class_name: str) -> dict:
        """
        Get strategy default setting.
        """
        strategy_class: Type[CtaTemplate] = self.classes[class_name]
        return strategy_class.get_class_parameters()

    def run_optimization(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        optimization_setting: OptimizationSetting,
        use_ga: bool
    ) -> None:
        """
        Run optimization.
        """
        self.result_values = None

        engine: BacktestingEngine = self.backtesting_engine
        engine.clear_data()

        if interval == Interval.TICK.value:
            mode: BacktestingMode = BacktestingMode.TICK
        else:
            mode: BacktestingMode = BacktestingMode.BAR

        engine.set_parameters(
            vt_symbol=vt_symbol,
            interval=interval,
            start=start,
            end=end,
            rate=rate,
            slippage=slippage,
            size=size,
            pricetick=pricetick,
            capital=capital,
            mode=mode
        )

        strategy_class: Type[CtaTemplate] = self.classes[class_name]

        if use_ga:
            self.result_values = engine.run_ga_optimization(
                strategy_class,
                optimization_setting,
                output=False
            )
        else:
            self.result_values = engine.run_bf_optimization(
                strategy_class,
                optimization_setting,
                output=False
            )

        # Clear thread object handler.
        self.thread = None

        # Put optimization done event
        event: Event = Event(EVENT_BACKTESTER_OPTIMIZATION_FINISHED)
        self.event_engine.put(event)

    def start_optimization(
        self,
        class_name: str,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        rate: float,
        slippage: float,
        size: int,
        pricetick: float,
        capital: int,
        optimization_setting: OptimizationSetting,
        use_ga: bool
    ) -> bool:
        """
        Start optimization in thread.
        """
        if self.thread:
            self.write_log("已有任务在运行，请等待完成")
            return False

        self.write_log("-" * 40)
        self.write_log("开始多进程参数优化")
        self.write_log("使用算法: " + ("遗传算法" if use_ga else "暴力算法"))
        self.write_log("参数优化空间：")
        for name, values in optimization_setting.params.items():
            self.write_log(f"{name}: {values}")

        self.thread = Thread(
            target=self.run_optimization,
            args=(
                class_name,
                vt_symbol,
                interval,
                start,
                end,
                rate,
                slippage,
                size,
                pricetick,
                capital,
                optimization_setting,
                use_ga
            )
        )
        self.thread.start()

        return True

    def run_downloading(
        self,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> None:
        """
        Run downloading.
        """
        self.write_log(f"开始下载历史数据：{vt_symbol} - {interval}")
        self.write_log(f"时间范围：{start} - {end}")

        symbol, exchange = extract_vt_symbol(vt_symbol)

        req = HistoryRequest(
            symbol=symbol,
            exchange=exchange,
            interval=Interval(interval),
            start=start,
            end=end
        )

        contract = self.main_engine.get_contract(vt_symbol)
        if contract:
            req.gateway_name = contract.gateway_name

        # 查询数据库是否已有数据
        records = []
        if self.database:
            if interval == "tick":
                records = self.database.get_tick_history(req)
            else:
                records = self.database.get_bar_history(req)

        if records:
            self.write_log(f"历史数据已存在，共{len(records)}条")
            return

        # 数据下载
        records = self.datafeed.query_history(req, self.write_log)

        if not records:
            self.write_log("数据下载失败")
            return

        # 保存数据到数据库
        if self.database:
            if interval == "tick":
                for tick in records:
                    self.database.save_tick_data([tick])
            else:
                for bar in records:
                    self.database.save_bar_data([bar])

            self.write_log(f"历史数据下载完成，共{len(records)}条")

        # Clear thread object handler.
        self.thread = None

    def start_downloading(
        self,
        vt_symbol: str,
        interval: str,
        start: datetime,
        end: datetime
    ) -> bool:
        """
        Start downloading in thread.
        """
        if self.thread:
            self.write_log("已有任务在运行，请等待完成")
            return False

        self.write_log("-" * 40)
        self.thread = Thread(
            target=self.run_downloading,
            args=(
                vt_symbol,
                interval,
                start,
                end
            )
        )
        self.thread.start()

        return True

    def load_bar_data_from_csv(self, csv_path: str, symbol: str, exchange: Exchange, interval: str, datetime_head: str = "candle_begin_time"):
        """
        直接从CSV文件加载K线数据
        """
        import pandas as pd
        from vnpy.trader.object import BarData

        self.write_log(f"从CSV文件加载数据: {csv_path}")
        
        # 检查文件是否存在
        import os
        if not os.path.exists(csv_path):
            self.write_log(f"CSV文件不存在: {csv_path}")
            return False

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            self.write_log(f"CSV数据行数: {len(df)}")
            
            # 转换时间列
            df[datetime_head] = pd.to_datetime(df[datetime_head])
            
            # 创建BarData列表
            bars = []
            total = len(df)
            
            # 添加进度日志
            self.write_log(f"开始转换CSV数据为K线数据，共{total}条记录")
            
            vt_symbol = f"{symbol}.{exchange.value}"
            
            # 将DataFrame转换为BarData对象
            for _, row in df.iterrows():
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=row[datetime_head],
                    interval=interval,
                    volume=float(row["volume"]),
                    open_price=float(row["open"]),
                    high_price=float(row["high"]),
                    low_price=float(row["low"]),
                    close_price=float(row["close"]),
                    gateway_name="BACKTESTING"
                )
                bars.append(bar)
            
            self.write_log(f"CSV数据转换完成，共{len(bars)}条K线记录")
            return bars
        except Exception as e:
            self.write_log(f"CSV数据加载失败: {str(e)}")
            import traceback
            self.write_log(traceback.format_exc())
            return []
