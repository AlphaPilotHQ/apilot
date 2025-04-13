"""
标准动量策略实盘交易脚本 - Binance版本

本脚本实现了一个基于动量指标的趋势跟踪策略，使用Binance交易所API进行实时交易。
策略逻辑:
1. 核心思路:结合动量信号与标准差动态止损的中长期趋势跟踪策略
2. 入场信号:
   - 基于动量指标(当前价格/N周期前价格-1)生成交易信号
   - 动量 > 阈值时做多
   - 使用全部账户资金进行头寸管理
3. 风险管理:
   - 使用基于标准差的动态追踪止损
   - 多头持仓:止损设置在最高价-4倍标准差
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import ClassVar

import apilot as ap
from apilot.core.event import EventEngine
from apilot.core.constant import Direction, Exchange, Interval
from apilot.core.gateway import BaseGateway
from apilot.core.engine import MainEngine
from apilot.engine.live import PAEngine
from apilot.execution.gateway.binance import BinanceGateway
from apilot.utils.logger import get_logger, set_level
from dotenv import load_dotenv

# 配置日志
logger = get_logger("StdMomentumLive")
set_level("info", "StdMomentumLive")


class StdMomentumStrategy(ap.PATemplate):
    """
    动量指标跟踪策略实盘版本
    
    策略逻辑:
    1. 核心思路:结合动量信号与标准差动态止损的中长期趋势跟踪策略
    2. 入场信号:
       - 基于动量指标(当前价格/N周期前价格-1)生成交易信号
       - 动量 > 阈值时做多
       - 使用全部账户资金进行头寸管理
    3. 风险管理:
       - 使用基于标准差的动态追踪止损
       - 多头持仓:止损设置在最高价-4倍标准差
    """

    # 策略参数
    std_period = 48
    mom_threshold = 0.05
    trailing_std_scale = 4
    
    # 资金管理参数
    risk_percent = 0.3  # 单个标的使用的资金比例

    parameters: ClassVar[list[str]] = [
        "std_period",
        "mom_threshold",
        "trailing_std_scale",
        "risk_percent",
    ]
    variables: ClassVar[list[str]] = [
        "momentum",
        "intra_trade_high",
        "pos",
    ]

    def __init__(self, pa_engine, strategy_name, symbols, setting):
        super().__init__(pa_engine, strategy_name, symbols, setting)
        
        # 使用增强版BarGenerator实例处理所有交易标的
        self.bg = ap.BarGenerator(
            self.on_bar,
            5,
            self.on_5min_bar,
            symbols=self.symbols,
        )

        # 为每个交易对创建ArrayManager
        self.ams = {}
        for symbol in self.symbols:
            self.ams[symbol] = ap.ArrayManager(size=200)

        # 为每个交易对创建状态跟踪字典
        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.pos = {}

        # 初始化每个交易对的状态
        for symbol in self.symbols:
            self.momentum[symbol] = 0.0
            self.std_value[symbol] = 0.0
            self.intra_trade_high[symbol] = 0
            self.pos[symbol] = 0

    def on_init(self):
        """策略初始化"""
        logger.info("策略初始化中，加载历史数据...")
        self.load_bar(self.std_period)
        logger.info("策略初始化完成")

    def on_start(self):
        """策略启动"""
        logger.info("策略启动，开始实时交易")
        
        # 打印当前策略参数
        logger.info(f"策略参数: 周期={self.std_period}, 动量阈值={self.mom_threshold}, 止损系数={self.trailing_std_scale}")
        logger.info(f"资金管理: 单标的资金比例={self.risk_percent}")

    def on_stop(self):
        """策略停止"""
        logger.info("策略停止，结束实时交易")

    def on_tick(self, tick):
        """
        处理TICK行情数据
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar):
        """
        处理K线数据
        """
        self.bg.update_bar(bar)

    def on_5min_bar(self, bars):
        """
        处理5分钟K线数据
        """
        self.cancel_all()

        # 记录收到的多标的K线数据
        logger.debug(f"收到完整的多标的数据: {list(bars.keys())}")

        # 对每个交易品种执行数据更新和交易逻辑
        for symbol, bar in bars.items():
            if symbol not in self.ams:
                logger.debug(f"忽略标的 {symbol}, 因为它不在ams中")
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            # 如果数据不足，跳过交易逻辑
            if not am.inited:
                continue

            # 计算技术指标
            self.std_value[symbol] = am.std(self.std_period)

            # 计算动量因子
            if len(am.close_array) > self.std_period + 1:
                old_price = am.close_array[-self.std_period - 1]
                current_price = am.close_array[-1]
                self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1

            # 获取当前持仓
            current_pos = self.pos.get(symbol, 0)

            # 持仓状态下更新跟踪止损价格
            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )

            # 交易逻辑
            if current_pos == 0:
                # 初始化追踪价格
                self.intra_trade_high[symbol] = bar.high_price

                # 使用风险系数配置仓位大小
                capital_to_use = self.risk_percent * self.pa_engine.get_current_capital()
                size = max(1, int(capital_to_use / bar.close_price))

                # 基于动量信号开仓
                logger.info(
                    f"{bar.datetime}: {symbol} 动量值 {self.momentum[symbol]:.4f}, 阈值 {self.mom_threshold:.4f}"
                )

                if self.momentum[symbol] > self.mom_threshold:
                    logger.info(
                        f"{bar.datetime}: {symbol} 发出多头信号: 动量 {self.momentum[symbol]:.4f} > 阈值 {self.mom_threshold}"
                    )
                    self.buy(symbol=symbol, price=bar.close_price, volume=size)

            elif current_pos > 0:  # 多头持仓 → 标准差追踪止损
                # 计算移动止损价格
                long_stop = (
                    self.intra_trade_high[symbol]
                    - self.trailing_std_scale * self.std_value[symbol]
                )

                # 当价格跌破止损线时平仓
                if bar.close_price < long_stop:
                    logger.info(
                        f"{bar.datetime}: {symbol} 触发止损: 价格 {bar.close_price:.4f} < 止损线 {long_stop:.4f}"
                    )
                    self.sell(
                        symbol=symbol, price=bar.close_price, volume=abs(current_pos)
                    )

    def on_order(self, order):
        """委托回调"""
        logger.info(f"委托状态更新: {order.orderid}, 状态: {order.status}, 成交量: {order.traded}/{order.volume}")

    def on_trade(self, trade):
        """成交回调"""
        symbol = trade.symbol

        # 更新持仓
        position_change = (
            trade.volume if trade.direction == Direction.LONG else -trade.volume
        )
        self.pos[symbol] = self.pos.get(symbol, 0) + position_change

        # 更新最高/最低价追踪
        current_pos = self.pos[symbol]

        # 只在仓位方向存在时更新对应的跟踪价格
        if current_pos > 0:
            # 多头仓位,更新最高价
            self.intra_trade_high[symbol] = max(
                self.intra_trade_high.get(symbol, trade.price), trade.price
            )

        logger.info(
            f"成交: {symbol} {trade.orderid} {trade.direction} "
            f"{trade.volume}@{trade.price}, 当前持仓: {current_pos}"
        )


def run_live_trading(api_key="", api_secret="", proxy_host="", proxy_port=0):
    """
    启动实时交易
    
    Args:
        api_key: Binance API Key
        api_secret: Binance API Secret
        proxy_host: 代理服务器地址
        proxy_port: 代理服务器端口
    """
    try:
        logger.info("======= 开始实时交易 =======")
        
        # 创建事件引擎
        event_engine = EventEngine()
        
        # 创建主引擎
        main_engine = MainEngine(event_engine)
        
        # 添加Binance网关
        main_engine.add_gateway(BinanceGateway)
        
        # 创建交易引擎
        pa_engine = PAEngine(main_engine, event_engine)
        
        # 连接到Binance
        setting = {
            "API Key": api_key,
            "Secret Key": api_secret,
            "Session Number": 3,
            "Proxy Host": proxy_host,
            "Proxy Port": int(proxy_port),
        }
        
        logger.info("正在连接到Binance...")
        # 通过网关正确连接
        main_engine.get_gateway("BINANCE").connect(setting)
        
        # 等待网关连接
        sleep(5)
        
        # 查询账户余额
        logger.info("查询账户余额")
        accounts = main_engine.get_all_accounts()
        for account in accounts:
            logger.info(f"账户 {account.accountid}: 余额 = {account.balance}, 可用 = {account.available}")
        
        # 添加策略
        strategy_name = "StdMomentum"
        symbols = ["BTC/USDT.BINANCE"]  # 使用Binance标准格式
        
        # 设置策略参数
        strategy_setting = {
            "std_period": 48,             # 标准差周期
            "mom_threshold": 0.05,        # 动量阈值
            "trailing_std_scale": 4.0,    # 追踪止损系数 
            "risk_percent": 0.3,          # 单标的资金比例
        }
        
        # 添加策略
        logger.info(f"添加策略: {strategy_name}, 交易品种: {symbols}")
        pa_engine.add_strategy(StdMomentumStrategy, strategy_name, symbols[0], strategy_setting)
        
        # 初始化策略
        logger.info("初始化策略...")
        future = pa_engine.init_strategy(strategy_name)
        future.result()  # 等待初始化完成
        
        # 启动策略
        logger.info("启动策略...")
        pa_engine.start_strategy(strategy_name)
        
        # 保持主线程运行
        logger.info("策略已启动，按Ctrl+C停止")
        while True:
            sleep(10)
            
    except KeyboardInterrupt:
        logger.info("用户中断，正在停止策略...")
        pa_engine.stop_strategy(strategy_name)
        main_engine.close()
        logger.info("策略已停止")
        
    except Exception as e:
        logger.error(f"发生错误: {e}")
        if 'main_engine' in locals():
            main_engine.close()
        logger.error("系统已关闭")
        raise


if __name__ == "__main__":
    # 加载环境变量
    dotenv_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=dotenv_path)
    
    # 获取API密钥和代理设置
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_SECRET_KEY", "")
    proxy_host = os.environ.get("BINANCE_PROXY_HOST", "")
    proxy_port = int(os.environ.get("BINANCE_PROXY_PORT", 0))
    
    # 检查API密钥
    if not api_key or not api_secret:
        logger.error("API密钥未设置。请在.env文件中设置BINANCE_API_KEY和BINANCE_SECRET_KEY。")
        sys.exit(1)
    
    # 运行交易
    run_live_trading(api_key, api_secret, proxy_host, proxy_port)