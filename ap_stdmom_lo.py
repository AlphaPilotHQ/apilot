import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, ClassVar

from dotenv import load_dotenv

import apilot as ap

# 加载环境变量
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# 获取API设置
API_URL = os.environ.get("API_URL", "https://dev-api.alphapilot.tech")
API_KEY = os.environ.get("API_KEY", "")

# 配置日志
logger = ap.get_logger("StdMomentumLive")
ap.set_level("info", "StdMomentumLive")


class StdMomentumStrategy(ap.PATemplate):
    # 策略参数
    std_period = 48
    mom_threshold = 0.05
    trailing_std_scale = 4

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
        self.load_bar(self.std_period)
        logger.info("策略初始化完成")

    def on_start(self):
        logger.info(
            f"策略参数: 周期={self.std_period}, 动量阈值={self.mom_threshold}, 止损系数={self.trailing_std_scale}"
        )

    def on_bar(self, bar):
        self.bg.update_bar(bar)

    def on_5min_bar(self, bars):
        self.cancel_all()

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

            if current_pos > 0:
                self.intra_trade_high[symbol] = max(
                    self.intra_trade_high[symbol], bar.high_price
                )

            if current_pos == 0:
                self.intra_trade_high[symbol] = bar.high_price
                size = 1

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
    event_engine = ap.EventEngine()
    main_engine = ap.MainEngine(event_engine)

    # 添加Binance网关
    main_engine.add_gateway(ap.BinanceGateway)

    # 创建交易引擎
    pa_engine = ap.PAEngine(main_engine, event_engine)

    setting = {
        "API Key": "",
        "Secret Key": "",
        "Session Number": 3,
        "Proxy Host": proxy_host,
        "Proxy Port": int(proxy_port),
    }

    logger.info("正在连接到Binance公共行情接口...")
    # 通过网关连接
    main_engine.get_gateway("BINANCE").connect(setting)

    # 等待网关连接
    sleep(5)

    logger.info("已连接到Binance行情接口")

    # 添加策略
    strategy_name = "StdMomentum"
    symbols = ["BTCUSDT.BINANCE"]
    # 设置策略参数
    strategy_setting = {
        "std_period": 48,
        "mom_threshold": 0.05,
        "trailing_std_scale": 4.0,
    }

    # 添加策略
    logger.info(f"添加信号策略: {strategy_name}, 监控品种: {symbols}")
    pa_engine.add_strategy(
        StdMomentumStrategy, strategy_name, symbols, strategy_setting
    )

    # 初始化策略
    logger.info("初始化信号策略...")
    future = pa_engine.init_strategy(strategy_name)
    future.result()  # 等待初始化完成

    # 启动策略
    logger.info("启动信号策略...")
    pa_engine.start_strategy(strategy_name)

    # 保持主线程运行
    logger.info("信号服务已启动，按Ctrl+C停止")
    while True:
        sleep(10)


if __name__ == "__main__":
    run_signal_service()
