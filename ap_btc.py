import logging
import os
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, ClassVar

import requests
from dotenv import load_dotenv

import apilot as ap
from apilot.utils.logger import setup_logging

setup_logging("StdMomentum", level=logging.INFO)
logger = logging.getLogger("StdMomentum")

dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# API配置
API_HOST = os.environ.get("API_HOST", "dev-api.alphapilot.tech")
API_URL = f"https://{API_HOST}/inner/signal/push"
API_KEY = os.environ.get("API_KEY", "")


class StdMomentumStrategy(ap.PATemplate):
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

        self.bg = ap.BarGenerator(
            self.on_bar,
            1,
            self.on_1min_bar,
            symbols=self.symbols,
        )

        self.ams = {}
        self.am_size = max(self.std_period + 10, 60)
        for symbol in self.symbols:
            self.ams[symbol] = ap.ArrayManager(size=self.am_size)

        self.momentum = {}
        self.std_value = {}
        self.intra_trade_high = {}
        self.pos = {}

        for symbol in self.symbols:
            self.momentum[symbol] = 0.0
            self.std_value[symbol] = 0.0
            self.intra_trade_high[symbol] = 0
            self.pos[symbol] = 0

    def on_init(self):
        self.load_bar(self.am_size)
        logger.info("on_init")

    def on_start(self):
        logger.info("on_start")

    def on_stop(self):
        logger.info("on_stop")

    def on_bar(self, bar):
        try:
            self.bg.update_bar(bar)
        except Exception as e:
            logger.error(f"BarGenerator处理出错: {e}")

    def on_1min_bar(self, bars):
        self.cancel_all()

        for symbol, bar in bars.items():
            if symbol not in self.ams:
                logger.info(f"忽略标的 {symbol}, 因为它不在ams中")
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            if not am.inited:
                continue

            try:
                self.std_value[symbol] = am.std(self.std_period)
                if len(am.close_array) > self.std_period + 1:
                    old_price = am.close_array[-self.std_period - 1]
                    current_price = am.close_array[-1]
                    self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1
                else:
                    logger.info(
                        f"数据不足以计算动量: {symbol}, 需要至少 {self.std_period + 1} 个周期的数据"
                    )
            except Exception as e:
                logger.error(f"计算指标出错: {symbol}, 错误: {e!s}")

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
                        signal_type="entry",
                        extra_info=extra_info,
                    ):
                        self.pos[symbol] = size
                        logger.info(
                            f"已发送多头信号到API: {symbol}, 价格={bar.close_price}, 数量={size}"
                        )

            elif current_pos > 0:
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
        signal_type: str = "entry",
        extra_info: dict[str, Any] | None = None,
    ) -> bool:
        try:
            data = {
                "symbol": symbol,
                "direction": direction.value,
                "price": price,
                "signal_type": signal_type,
            }
            if extra_info:
                data.update(extra_info)

            payload = {
                "data": data,
                "strategyId": "6807515645011a07107ba029",
                "time": int(datetime.now().timestamp() * 1000),
            }

            # 设置请求头并发送请求
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["X-AP-API-Key"] = API_KEY
            else:
                logger.warning("未设置API密钥，请检查环境变量API_KEY是否已配置")

            response = requests.post(
                url=API_URL,
                json=payload,
                headers=headers,
                timeout=10,
            )

            if response.status_code in (200, 201, 202):
                logger.info(
                    f"signal sent successfully: {self.strategy_name}, {symbol}, {direction.value}, {price}"
                )
                return True

            logger.error(f"Signal failed to send: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Signal failed to send: {e}")
            return False


def run_signal_service(proxy_host="127.0.0.1", proxy_port=7890):
    main_engine = ap.MainEngine()
    logger.info("1 EventEngine MainEngine Ready")

    main_engine.add_gateway(ap.BinanceGateway)
    logger.info("2 Binance Gateway Added")

    pa_engine = main_engine.add_engine(ap.LiveEngine)
    logger.info("3 PAEngine Ready")

    setting = {
        "API Key": "",
        "Secret Key": "",
        "Proxy Host": proxy_host,
        "Proxy Port": int(proxy_port),
    }
    main_engine.get_gateway("BINANCE").connect(setting)
    sleep(5)

    strategy_name = "StdMomentum"
    symbols = ["SOL/USDT"]
    strategy_setting = {
        "std_period": 20,
        "mom_threshold": -0.005,
        "trailing_std_scale": 1.0,
    }
    pa_engine.add_strategy(
        StdMomentumStrategy, strategy_name, symbols, strategy_setting
    )
    logger.info("4. Strategy Added")

    future = pa_engine.init_strategy(strategy_name)
    future.result()
    pa_engine.start_strategy(strategy_name)
    logger.info("5 strategy strated")
    while True:
        # 每20秒打印一次当前状态
        for _ in range(20):
            sleep(1)

        # 从引擎获取策略对象
        strategy = pa_engine.strategies.get(strategy_name)
        if not strategy:
            logger.error(f"找不到策略: {strategy_name}")
            continue

        for symbol in symbols:
            am = strategy.ams.get(symbol)
            if am and am.inited:
                momentum = strategy.momentum.get(symbol, 0)
                std_value = strategy.std_value.get(symbol, 0)
                threshold = strategy.mom_threshold
                current_pos = strategy.pos.get(symbol, 0)
                logger.info(
                    f"状态: {symbol} momentum={momentum:.6f} (阈值={threshold}), std={std_value:.6f}"
                )
                logger.info(f"条件检查: {momentum > threshold}")

                # 如果有持仓，计算并显示止损线
                if current_pos > 0:
                    intra_high = strategy.intra_trade_high.get(symbol, 0)
                    trailing_scale = strategy.trailing_std_scale
                    stop_price = intra_high - trailing_scale * std_value
                    latest_price = am.close_array[-1] if len(am.close_array) > 0 else 0
                    logger.info(
                        f"止损线: {stop_price:.4f}, 当前价格: {latest_price:.4f}, 最高价: {intra_high:.4f}"
                    )
            else:
                logger.info(f"状态: {symbol} 数据管理器尚未初始化或不存在")


if __name__ == "__main__":
    run_signal_service()
