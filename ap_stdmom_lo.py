import json
import logging
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, ClassVar

from dotenv import load_dotenv

import apilot as ap
from apilot.utils.logger import setup_logging

setup_logging("ap_stdmom_lo", level=logging.INFO)
logger = logging.getLogger("ap_stdmom_lo")

dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

API_URL = os.environ.get("API_URL", "https://dev-api.alphapilot.tech")
API_KEY = os.environ.get("API_KEY", "")


class StdMomentumStrategy(ap.PATemplate):
    # 策略参数
    std_period = 48
    mom_threshold = 0.01
    trailing_std_scale = 2

    parameters: ClassVar[list[str]] = [
        "std_period",
        "mom_threshold",
        "trailing_std_scale",
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
            1,
            self.on_1min_bar,
            symbols=self.symbols,
        )

        # 为每个交易对创建ArrayManager - 使用更小的size以便更快初始化
        self.ams = {}
        # 确保size比std_period大一点，但不要太大
        self.am_size = max(self.std_period + 10, 60)
        for symbol in self.symbols:
            self.ams[symbol] = ap.ArrayManager(size=self.am_size)
            logger.info(
                f"为 {symbol} 创建ArrayManager, 容量={self.am_size}, 策略周期={self.std_period}"
            )

        logger.info(f"策略构造函数初始化完成，将监控以下交易对: {self.symbols}")

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
        logger.info(
            f"[{self.strategy_name}] on_init called, ArrayManager size={getattr(self, 'am_size', 'unknown')}"
        )
        self.load_bar(self.am_size)
        logger.info(f"[{self.strategy_name}] 历史K线已自动推进on_bar")

    def on_start(self):
        logger.info(f"[{self.strategy_name}] on_start called")
        logger.info(
            f"策略参数: 周期={self.std_period}, 动量阈值={self.mom_threshold}, 止损系数={self.trailing_std_scale}"
        )

    def on_stop(self):
        # 可以加日志，或者直接pass
        logger.info(f"策略 {self.strategy_name} 已停止")
        pass

    def on_bar(self, bar):
        logger.info(f"[{self.strategy_name}] on_bar called, bar time: {bar.datetime}")
        # 记录收到的K线数据详细信息
        logger.info(
            f"策略收到K线数据: {bar.symbol} @ {bar.datetime}, 价格: O={bar.open_price:.2f} H={bar.high_price:.2f} L={bar.low_price:.2f} C={bar.close_price:.2f}"
        )

        # 记录ArrayManager中的数据量和初始化状态
        symbol = bar.symbol
        if symbol in self.ams:
            am = self.ams[symbol]
            logger.info(
                f"ArrayManager推进: {symbol}, 当前数据量={getattr(am, 'count', '未知')}, inited={getattr(am, 'inited', '未知')}"
            )
            data_count = (
                len(am.close_array)
                if hasattr(am, "close_array") and am.close_array is not None
                else 0
            )
            am_count = am.count if hasattr(am, "count") else 0
            am_status = "已初始化" if am.inited else "未初始化"
            logger.info(
                f"当前 {symbol} 的ArrayManager: 数据量={data_count}, 计数={am_count}/{am.size}, 状态={am_status}"
            )

        # 传递给BarGenerator处理
        try:
            logger.info(f"传递K线数据给BarGenerator: {bar.symbol} @ {bar.datetime}")
            self.bg.update_bar(bar)
            logger.info(f"BarGenerator处理完成: {bar.symbol} @ {bar.datetime}")
        except Exception as e:
            logger.error(f"BarGenerator处理出错: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def on_1min_bar(self, bars):
        logger.info(f"on_1min_bar被调用，收到 {len(bars)} 个交易对的K线数据")
        self.cancel_all()

        # 对每个交易品种执行数据更新和交易逻辑
        for symbol, bar in bars.items():
            # 记录每个K线数据
            logger.info(
                f"处理1分钟K线: {symbol}, 时间={bar.datetime}, 开={bar.open_price:.2f}, 高={bar.high_price:.2f}, 低={bar.low_price:.2f}, 收={bar.close_price:.2f}"
            )

            if symbol not in self.ams:
                logger.info(f"忽略标的 {symbol}, 因为它不在ams中")
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            # 如果数据不足，跳过交易逻辑
            if not am.inited:
                data_count = (
                    len(am.close_array)
                    if hasattr(am, "close_array") and am.close_array is not None
                    else 0
                )
                required_count = self.am_size  # 使用策略中保存的ArrayManager大小
                logger.info(
                    f"数据不足，等待更多K线: {symbol}, 当前数据量={data_count}, 需要的数据量={required_count}"
                )
                continue

            # 计算技术指标
            try:
                self.std_value[symbol] = am.std(self.std_period)
                logger.info(
                    f"计算标准差: {symbol}, 周期={self.std_period}, 结果={self.std_value[symbol]:.4f}"
                )

                # 计算动量因子
                if len(am.close_array) > self.std_period + 1:
                    old_price = am.close_array[-self.std_period - 1]
                    current_price = am.close_array[-1]
                    self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1
                    logger.info(
                        f"计算动量: {symbol}, 旧价格={old_price:.2f}, 当前价格={current_price:.2f}, 动量={self.momentum[symbol]:.4f}, 标准差={self.std_value[symbol]:.4f}"
                    )
                else:
                    logger.info(
                        f"数据不足以计算动量: {symbol}, 需要至少 {self.std_period + 1} 个周期的数据"
                    )
            except Exception as e:
                logger.error(f"计算指标出错: {symbol}, 错误: {e!s}")
                import traceback

                logger.error(traceback.format_exc())

            # 获取当前持仓
            current_pos = self.pos.get(symbol, 0)

            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )

            if current_pos == 0:
                self.intra_trade_high[symbol] = bar.high_price
                size = 1

                # 详细记录判断过程
                logger.info(
                    f"判断入场条件: {symbol}, 动量={self.momentum[symbol]:.4f}, 阈值={self.mom_threshold}"
                )

                if self.momentum[symbol] > self.mom_threshold:
                    logger.info(
                        f"{bar.datetime}: {symbol} 发出多头信号: 动量 {self.momentum[symbol]:.4f} > 阈值 {self.mom_threshold}"
                    )

                    # 发送入场信号到API
                    extra_info = {
                        "indicator": {
                            "momentum": self.momentum[symbol],
                            "std": self.std_value[symbol],
                        }
                    }
                    if self.send_signal(
                        symbol=symbol,
                        direction=ap.Direction.LONG,
                        price=bar.close_price,
                        volume=size,
                        timestamp=bar.datetime,
                        signal_type="entry",
                        extra_info=extra_info,
                    ):
                        # 更新内部持仓状态
                        self.pos[symbol] = size
                        logger.info(
                            f"已发送多头信号到API: {symbol}, 价格={bar.close_price}, 数量={size}"
                        )

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

                    # 发送平仓信号到API
                    extra_info = {
                        "indicator": {
                            "momentum": self.momentum[symbol],
                            "std": self.std_value[symbol],
                            "stop_price": long_stop,
                        },
                        "reason": "trailing_stop_loss",
                    }
                    if self.send_signal(
                        symbol=symbol,
                        direction=ap.Direction.SHORT,  # 平仓方向
                        price=bar.close_price,
                        volume=abs(current_pos),
                        timestamp=bar.datetime,
                        signal_type="exit",
                        extra_info=extra_info,
                    ):
                        # 更新内部持仓状态
                        self.pos[symbol] = 0
                        logger.info(
                            f"已发送平仓信号到API: {symbol}, 价格={bar.close_price}, 数量={abs(current_pos)}"
                        )

    def on_order(self, order):
        pass

    def on_trade(self, trade):
        pass

    def send_signal(
        self,
        symbol: str,
        direction: ap.Direction,
        price: float,
        volume: float,
        timestamp: datetime,
        signal_type: str = "entry",
        extra_info: dict[str, Any] | None = None,
    ) -> bool:
        try:
            # 使用全局API配置
            global API_URL, API_KEY

            # 准备数据
            data = {
                "strategy_name": self.strategy_name,
                "symbol": symbol,
                "direction": direction.value,
                "price": price,
                "volume": volume,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "signal_type": signal_type,
            }

            if extra_info:
                data.update(extra_info)

            # 转换为JSON
            json_data = json.dumps(data).encode("utf-8")

            # 准备请求头
            headers = {
                "Content-Type": "application/json",
            }

            if API_KEY:
                headers["X-AP-API-Key"] = API_KEY

            # 创建请求
            req = urllib.request.Request(
                url=API_URL,
                data=json_data,
                headers=headers,
                method="POST",
            )

            # 发送请求
            response = urllib.request.urlopen(req)
            if response.status in (200, 201, 202):
                logger.info(
                    f"交易信号已发送到API: {self.strategy_name}, {symbol}, {direction.value}, {price}, {volume}"
                )
                return True
            else:
                logger.error(f"发送交易信号失败状态码: {response.status}")
                return False

        except urllib.error.URLError as e:
            logger.error(f"API请求错误: {e.reason}")
            return False
        except Exception as e:
            logger.error(f"发送交易信号失败 {e}")
            return False


def run_signal_service(proxy_host="127.0.0.1", proxy_port=7890):
    main_engine = ap.MainEngine()
    logger.info("1. EventEngine MainEngine Ready")

    # 添加Binance网关
    main_engine.add_gateway(ap.BinanceGateway)
    logger.info("2. Binance Gateway Added")

    # 创建交易引擎
    pa_engine = main_engine.add_engine(ap.LiveEngine)
    logger.info("3 PAEngine Ready")

    setting = {
        "API Key": "",
        "Secret Key": "",
        "Proxy Host": proxy_host,
        "Proxy Port": int(proxy_port),
    }
    main_engine.get_gateway("BINANCE").connect(setting)
    logger.info("等待网关连接和合约初始化 (5秒)...")
    sleep(5)
    logger.info("已连接到Binance行情接口")

    # 添加策略
    strategy_name = "StdMomentum"
    symbols = ["SOL/USDT"]
    strategy_setting = {
        "std_period": 48,
        "mom_threshold": 0.005,
        "trailing_std_scale": 1.0,
    }
    logger.info(f"添加信号策略: {strategy_name}, 监控品种: {symbols}")
    pa_engine.add_strategy(
        StdMomentumStrategy, strategy_name, symbols, strategy_setting
    )
    logger.info("4. Strategy Added")

    future = pa_engine.init_strategy(strategy_name)
    future.result()  # 阻塞直到初始化线程真正完成
    pa_engine.start_strategy(strategy_name)
    logger.info("5 strategy strated")

    # 保持主线程运行
    count = 0
    while True:
        count += 1
        # 每10秒输出一次策略状态摘要
        if count % 10 == 0:
            for symbol in symbols:
                strategy = pa_engine.strategies[strategy_name]

                # 获取指标和状态
                mom_value = strategy.momentum.get(symbol, 0)
                std_value = strategy.std_value.get(symbol, 0)
                pos_value = strategy.pos.get(symbol, 0)

                # 检查ArrayManager的初始化状态
                am_initialized = (
                    "已初始化"
                    if symbol in strategy.ams and strategy.ams[symbol].inited
                    else "未初始化"
                )

                # 尝试获取当前数据量
                data_count = "未知"
                if (
                    symbol in strategy.ams
                    and hasattr(strategy.ams[symbol], "close_array")
                    and strategy.ams[symbol].close_array is not None
                ):
                    data_count = len(strategy.ams[symbol].close_array)

                # 输出更详细的状态摘要
                logger.info(
                    f"状态摘要: {symbol}, 动量={mom_value:.4f}, 标准差={std_value:.4f}, 持仓={pos_value}, "
                    f"ArrayManager状态: {am_initialized}, 数据量: {data_count}"
                )
        sleep(1)


if __name__ == "__main__":
    run_signal_service()
