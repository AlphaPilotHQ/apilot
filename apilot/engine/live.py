"""
实时交易引擎模块

实现交易策略的实时运行与管理，包括信号处理、订单执行与风控
"""

import copy
import csv
import glob
import importlib
import logging
import os
import re
import sys
import traceback
from collections import defaultdict, OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

import numpy as np

from apilot.core import (
    # 核心类
    BaseEngine,
    Event,
    EventEngine,
    MainEngine,

    # 数据类和常量
    BarData,
    ContractData,
    Direction,
    Exchange,
    Interval,
    LogData,
    Offset,
    OrderData,
    OrderRequest,
    OrderType,
    Status,
    SubscribeRequest,
    TickData,
    TradeData,

    # 常量和工具函数
    APP_NAME,
    EngineType,
    extract_vt_symbol,
    load_json,
    round_to,
    save_json,

    # 事件常量
    EVENT_CTA_LOG,
    EVENT_CTA_STRATEGY,
    EVENT_ORDER,
    EVENT_TICK,
    EVENT_TRADE,
)
from apilot.datafeed import BaseDatabase, get_database
from apilot.strategy import CtaTemplate, TargetPosTemplate
from apilot.utils.logger import get_logger

# 模块级初始化日志器
logger = get_logger("LiveTrading")

class CtaEngine(BaseEngine):
    engine_type: EngineType = EngineType.LIVE
    setting_filename: str = "cta_strategy_setting.json"
    data_filename: str = "cta_strategy_data.json"

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        super().__init__(main_engine, event_engine, APP_NAME)

        self.strategy_setting = {}
        self.strategy_data = {}
        self.classes = {}
        self.strategies = {}
        self.symbol_strategy_map = defaultdict(list)
        self.orderid_strategy_map = {}
        self.strategy_orderid_map = defaultdict(set)
        self.init_executor = ThreadPoolExecutor(max_workers=1)
        self.vt_tradeids = set()
        self.database: BaseDatabase = get_database()

    def init_engine(self) -> None:
        """
        初始化引擎
        """
        self.load_strategy_setting()
        self.load_strategy_data()
        # 脚本环境中不需要动态加载策略类
        # self.load_strategy_class()
        self.register_event()
        logger.info("CTA策略引擎初始化成功")

    def close(self) -> None:
        self.stop_all_strategies()

    def register_event(self) -> None:
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)

    def process_tick_event(self, event: Event) -> None:
        tick = event.data

        strategies = self.symbol_strategy_map[tick.vt_symbol]
        if not strategies:
            return

        for strategy in strategies:
            if strategy.inited:
                self.call_strategy_func(strategy, strategy.on_tick, tick)

    def process_order_event(self, event: Event) -> None:
        order = event.data

        strategy: Optional[type] = self.orderid_strategy_map.get(order.vt_orderid, None)
        if not strategy:
            return

        # Remove vt_orderid if order is no longer active.
        vt_orderids: set = self.strategy_orderid_map[strategy.strategy_name]
        if order.vt_orderid in vt_orderids and not order.is_active():
            vt_orderids.remove(order.vt_orderid)

        # Call strategy on_order function
        self.call_strategy_func(strategy, strategy.on_order, order)

    def process_trade_event(self, event: Event) -> None:
        trade: TradeData = event.data

        # Filter duplicate trade push
        if trade.vt_tradeid in self.vt_tradeids:
            return
        self.vt_tradeids.add(trade.vt_tradeid)

        strategy: Optional[type] = self.orderid_strategy_map.get(trade.vt_orderid, None)
        if not strategy:
            return

        # Update strategy pos before calling on_trade method
        if trade.direction == Direction.LONG:
            strategy.pos += trade.volume
        else:
            strategy.pos -= trade.volume

        self.call_strategy_func(strategy, strategy.on_trade, trade)

        # Sync strategy variables to data file
        self.sync_strategy_data(strategy)

    def send_server_order(
        self,
        strategy: CtaTemplate,
        contract: ContractData,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        type: OrderType,
        net: bool
    ) -> list:
        # Create request and send order.
        original_req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            offset=offset,
            type=type,
            price=price,
            volume=volume,
            reference=f"{APP_NAME}_{strategy.strategy_name}"
        )

        # Convert with offset converter
        req_list: List[OrderRequest] = self.main_engine.convert_order_request(
            original_req,
            contract.gateway_name,
            net
        )

        # Send Orders
        vt_orderids: list = []

        for req in req_list:
            vt_orderid: str = self.main_engine.send_order(req, contract.gateway_name)

            # Check if sending order successful
            if not vt_orderid:
                continue

            vt_orderids.append(vt_orderid)

            self.main_engine.update_order_request(req, vt_orderid, contract.gateway_name)

            # Save relationship between orderid and strategy.
            self.orderid_strategy_map[vt_orderid] = strategy
            self.strategy_orderid_map[strategy.strategy_name].add(vt_orderid)

        return vt_orderids

    def send_limit_order(
        self,
        strategy: CtaTemplate,
        contract: ContractData,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        net: bool
    ) -> list:
        return self.send_server_order(
            strategy,
            contract,
            direction,
            offset,
            price,
            volume,
            OrderType.LIMIT,
            net
        )

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
        contract: Optional[ContractData] = self.main_engine.get_contract(strategy.vt_symbol)
        if not contract:
            msg = f"[{strategy.strategy_name}] 委托失败，找不到合约：{strategy.vt_symbol}"
            self.main_engine.log_error(msg, source=APP_NAME)
            return ""

        # Round order price and volume to nearest incremental value
        price: float = round_to(price, contract.pricetick)
        volume: float = round_to(volume, contract.min_volume)

        return self.send_limit_order(
            strategy, contract, direction, offset, price, volume, net
        )

    def cancel_server_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        """
        Cancel existing order by vt_orderid.
        """
        order: Optional[OrderData] = self.main_engine.get_order(vt_orderid)
        if not order:
            if strategy:
                logger.error(f"[{strategy.strategy_name}] 撤单失败，找不到委托{vt_orderid}")
            else:
                logger.error(f"撤单失败，找不到委托{vt_orderid}")
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str) -> None:
        self.cancel_server_order(strategy, vt_orderid)

    def cancel_all(self, strategy: CtaTemplate) -> None:
        vt_orderids: set = self.strategy_orderid_map[strategy.strategy_name]
        if not vt_orderids:
            return

        for vt_orderid in copy(vt_orderids):
            self.cancel_order(strategy, vt_orderid)

    def get_engine_type(self) -> EngineType:
        return self.engine_type

    def get_pricetick(self, strategy: CtaTemplate) -> float:
        contract: Optional[ContractData] = self.main_engine.get_contract(strategy.vt_symbol)

        if contract:
            return contract.pricetick
        else:
            return None

    def get_size(self, strategy: CtaTemplate) -> int:
        contract: Optional[ContractData] = self.main_engine.get_contract(strategy.vt_symbol)

        if contract:
            return contract.size
        else:
            return None

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable[[BarData], None],
        use_database: bool
    ) -> List[BarData]:
        symbol, exchange = extract_vt_symbol(vt_symbol)
        end: datetime = datetime.now()
        start: datetime = end - timedelta(days)
        bars: List[BarData] = []

        # Try to query bars from database, if not found, load from database.
        bars: List[BarData] = self.database.load_bar_data(
            symbol, exchange, interval, start, end
        )

        return bars

    def load_tick(
        self,
        vt_symbol: str,
        days: int,
        callback: Callable[[TickData], None]
    ) -> List[TickData]:
        symbol, exchange = extract_vt_symbol(vt_symbol)
        end: datetime = datetime.now()
        start: datetime = end - timedelta(days)

        ticks: List[TickData] = self.database.load_tick_data(
            symbol, exchange, start, end
        )

        return ticks

    def call_strategy_func(self, strategy: CtaTemplate, func: Callable, params: Any = None) -> None:
        try:
            func(params) if params is not None else func()
        except Exception:
            strategy.trading = strategy.inited = False
            msg = f"[{strategy.strategy_name}] 触发异常已停止\n{traceback.format_exc()}"
            self.main_engine.log_critical(msg, source=APP_NAME)

    def add_strategy(
        self, class_name: str, strategy_name: str, vt_symbol: str, setting: dict
    ) -> None:
        if strategy_name in self.strategies:
            self.main_engine.log_error(f"创建策略失败，存在重名{strategy_name}", source=APP_NAME)
            return

        strategy_class: Optional[Type[CtaTemplate]] = self.classes.get(class_name, None)
        if not strategy_class:
            self.main_engine.log_error(f"创建策略失败，找不到策略类{class_name}", source=APP_NAME)
            return

        if "." not in vt_symbol:
            self.main_engine.log_error("创建策略失败，本地代码缺失交易所后缀", source=APP_NAME)
            return

        __, exchange_str = vt_symbol.split(".")
        if exchange_str not in Exchange.__members__:
            self.main_engine.log_error("创建策略失败，本地代码的交易所后缀不正确", source=APP_NAME)
            return

        strategy: CtaTemplate = strategy_class(self, strategy_name, vt_symbol, setting)
        self.strategies[strategy_name] = strategy

        # Add vt_symbol to strategy map.
        strategies: list = self.symbol_strategy_map[vt_symbol]
        strategies.append(strategy)

        # Update to setting file.
        self.update_strategy_setting(strategy_name, setting)

    def init_strategy(self, strategy_name: str) -> Future:
        return self.init_executor.submit(self._init_strategy, strategy_name)

    def _init_strategy(self, strategy_name: str) -> None:
        """
        Init strategies in queue.
        """
        strategy: CtaTemplate = self.strategies[strategy_name]

        if strategy.inited:
            self.main_engine.log_error(f"{strategy_name}已经完成初始化，禁止重复操作", source=APP_NAME)
            return

        self.main_engine.log_info(f"{strategy_name}开始执行初始化", source=APP_NAME)

        # Call on_init function of strategy
        self.call_strategy_func(strategy, strategy.on_init)

        # Restore strategy data(variables)
        data: Optional[dict] = self.strategy_data.get(strategy_name, None)
        if data:
            for name in strategy.variables:
                value = data.get(name, None)
                if value is not None:
                    setattr(strategy, name, value)

        # Subscribe market data
        contract: Optional[ContractData] = self.main_engine.get_contract(strategy.vt_symbol)
        if contract:
            req: SubscribeRequest = SubscribeRequest(
                symbol=contract.symbol, exchange=contract.exchange)
            self.main_engine.subscribe(req, contract.gateway_name)
        else:
            self.main_engine.log_error(f"行情订阅失败，找不到合约{strategy.vt_symbol}", source=APP_NAME)

        # Put event to update init completed status.
        strategy.inited = True
        self.main_engine.log_info(f"{strategy_name}初始化完成", source=APP_NAME)

    def start_strategy(self, strategy_name: str) -> None:
        strategy: CtaTemplate = self.strategies[strategy_name]
        if not strategy.inited:
            self.main_engine.log_error(f"策略{strategy_name}启动失败，请先初始化", source=APP_NAME)
            return

        if strategy.trading:
            self.main_engine.log_error(f"{strategy_name}已经启动，请勿重复操作", source=APP_NAME)
            return
        self.call_strategy_func(strategy, strategy.on_start)
        strategy.trading = True

    def stop_strategy(self, strategy_name: str) -> None:
        strategy: CtaTemplate = self.strategies[strategy_name]
        if not strategy.trading:
            return

        # Call on_stop function of the strategy
        self.call_strategy_func(strategy, strategy.on_stop)

        # Change trading status of strategy to False
        strategy.trading = False

        # Cancel all orders of the strategy
        self.cancel_all(strategy)

        # Sync strategy variables to data file
        self.sync_strategy_data(strategy)

    def edit_strategy(self, strategy_name: str, setting: dict) -> None:
        strategy: CtaTemplate = self.strategies[strategy_name]
        strategy.update_setting(setting)

        self.update_strategy_setting(strategy_name, setting)

    def remove_strategy(self, strategy_name: str) -> bool:
        strategy: CtaTemplate = self.strategies[strategy_name]
        if strategy.trading:
            self.main_engine.log_error(f"策略{strategy_name}移除失败，请先停止", source=APP_NAME)
            return

        # Remove setting
        self.remove_strategy_setting(strategy_name)

        # Remove from symbol strategy map
        strategies: list = self.symbol_strategy_map[strategy.vt_symbol]
        strategies.remove(strategy)

        # Remove from active orderid map
        if strategy_name in self.strategy_orderid_map:
            vt_orderids: set = self.strategy_orderid_map.pop(strategy_name)

            # Remove vt_orderid strategy map
            for vt_orderid in vt_orderids:
                if vt_orderid in self.orderid_strategy_map:
                    self.orderid_strategy_map.pop(vt_orderid)

        # Remove from strategies
        self.strategies.pop(strategy_name)

        self.main_engine.log_info(f"策略{strategy_name}移除成功", source=APP_NAME)
        return True

    def load_strategy_class(self) -> None:
        """
        加载策略类（简化版本，脚本环境不需要动态加载）
        """
        # 在脚本环境中，策略类通常通过直接导入获取，无需动态加载
        pass

    def load_strategy_class_from_folder(self, path: Path, module_name: str = "") -> None:
        """
        从特定文件夹加载策略类（简化版本，脚本环境不需要）
        """
        # 在脚本环境中，通常直接导入策略类，此方法可删除
        pass

    def load_strategy_class_from_module(self, module_name: str) -> None:
        """
        从模块加载策略类（简化版本，脚本环境不需要）
        """
        # 在脚本环境中，通常直接导入策略类，此方法可删除
        pass

    def load_strategy_data(self) -> None:
        self.strategy_data = load_json(self.data_filename)

    def sync_strategy_data(self, strategy: CtaTemplate) -> None:
        """
        Sync strategy data into json file.
        """
        data: dict = strategy.get_variables()
        data.pop("inited")      # Strategy status (inited, trading) should not be synced.
        data.pop("trading")

        self.strategy_data[strategy.strategy_name] = data
        save_json(self.data_filename, self.strategy_data)

    def get_all_strategy_class_names(self) -> list:
        """获取所有已加载的策略类名（简化版）"""
        # 在脚本环境中，可以直接引用策略类，无需此方法
        return list(self.classes.keys())

    def get_strategy_class_parameters(self, class_name: str) -> dict:
        """获取策略类默认参数（简化版）"""
        # 在脚本环境中，可以直接从策略类获取默认参数
        strategy_class: Type[CtaTemplate] = self.classes[class_name]
        return {name: getattr(strategy_class, name) for name in strategy_class.parameters}

    def get_strategy_parameters(self, strategy_name) -> dict:
        """获取策略实例参数（简化版）"""
        # 在脚本环境中，可以直接访问策略实例获取参数
        strategy: CtaTemplate = self.strategies[strategy_name]
        return strategy.get_parameters()

    def init_all_strategies(self) -> Dict[str, Future]:
        """初始化所有策略（简化版）"""
        # 在脚本环境中通常会显式初始化每个策略
        # 此方法主要用于GUI批量操作
        futures: Dict[str, Future] = {}
        for strategy_name in self.strategies.keys():
            futures[strategy_name] = self.init_strategy(strategy_name)
        return futures

    def start_all_strategies(self) -> None:
        """启动所有策略（简化版）"""
        # 在脚本环境中通常会显式启动每个策略
        for strategy_name in self.strategies.keys():
            self.start_strategy(strategy_name)

    def stop_all_strategies(self) -> None:
        """停止所有策略（简化版）"""
        # 在关闭引擎时仍然需要调用此方法
        for strategy_name in self.strategies.keys():
            self.stop_strategy(strategy_name)

    def load_strategy_setting(self) -> None:
        """
        加载策略设置（简化版）
        """
        # 在脚本环境中，策略设置通常直接在代码中定义，而不是从文件加载
        # 但保留此方法可以支持从配置文件恢复
        self.strategy_setting = load_json(self.setting_filename)

        # 在脚本环境中通常不自动添加策略，注释掉自动加载的代码
        # for strategy_name, strategy_config in self.strategy_setting.items():
        #     self.add_strategy(
        #         strategy_config["class_name"],
        #         strategy_name,
        #         strategy_config["vt_symbol"],
        #         strategy_config["setting"]
        #     )

    def update_strategy_setting(self, strategy_name: str, setting: dict) -> None:
        """
        更新策略设置（简化版）
        """
        # 在脚本环境中，策略参数通常直接在代码中修改，而不是动态更新
        # 但保留文件保存功能以便记录
        strategy: CtaTemplate = self.strategies[strategy_name]

        self.strategy_setting[strategy_name] = {
            "class_name": strategy.__class__.__name__,
            "vt_symbol": strategy.vt_symbol,
            "setting": setting,
        }
        save_json(self.setting_filename, self.strategy_setting)

    def remove_strategy_setting(self, strategy_name: str) -> None:
        """
        移除策略设置（简化版）
        """
        # 在脚本环境中，通常不需要动态移除策略
        # 但保留此方法以便在需要时清理配置文件
        if strategy_name not in self.strategy_setting:
            return

        self.strategy_setting.pop(strategy_name)
        save_json(self.setting_filename, self.strategy_setting)

        self.strategy_data.pop(strategy_name, None)
        save_json(self.data_filename, self.strategy_data)

    def put_strategy_event(self, strategy: CtaTemplate) -> None:
        """发送策略状态更新事件（GUI用，可删除）"""
        # 在无GUI环境下不需要此方法
        pass

    def send_email(self, msg: str, strategy: CtaTemplate = None) -> None:
        """发送邮件通知（可以删除）"""
        # 在脚本环境中可以直接使用主引擎的邮件功能
        if strategy:
            subject: str = "{}".format(strategy.strategy_name)
        else:
            subject: str = "CTA策略引擎"

        self.main_engine.send_email(subject, msg)
