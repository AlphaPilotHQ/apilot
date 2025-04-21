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

        # 使用增强版BarGenerator实例处理所有交易标的
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
        self.load_bar(self.am_size)
        logger.info(f"[{self.strategy_name}] 历史K线已自动推进on_bar")

    def on_start(self):
        logger.info(f"[{self.strategy_name}] on_start called")
        logger.info(
            f"策略参数: 周期={self.std_period}, 动量阈值={self.mom_threshold}, 止损系数={self.trailing_std_scale}"
        )

    def on_stop(self):
        logger.info(f"策略 {self.strategy_name} 已停止")
        pass

    def on_bar(self, bar):
        symbol = bar.symbol
        if symbol in self.ams:
            am = self.ams[symbol]
            am_status = "已初始化" if am.inited else "未初始化"
            logger.info(f"ArrayManager状态: {symbol}, 状态={am_status}")

        try:
            self.bg.update_bar(bar)
        except Exception as e:
            logger.error(f"BarGenerator处理出错: {e}")

    def on_1min_bar(self, bars):
        logger.info(f"on_1min_bar被调用，收到 {len(bars)} 个交易对的K线数据")
        self.cancel_all()

        # 对每个交易品种执行数据更新和交易逻辑
        for symbol, bar in bars.items():
            if symbol not in self.ams:
                logger.info(f"忽略标的 {symbol}, 因为它不在ams中")
                continue

            am = self.ams[symbol]
            am.update_bar(bar)

            # 如果数据不足，跳过交易逻辑
            if not am.inited:
                continue

            try:
                self.std_value[symbol] = am.std(self.std_period)

                # 计算动量因子
                if len(am.close_array) > self.std_period + 1:
                    old_price = am.close_array[-self.std_period - 1]
                    current_price = am.close_array[-1]
                    self.momentum[symbol] = (current_price / max(old_price, 1e-6)) - 1
                    logger.info(
                        f"指标计算: {symbol}, 动量={self.momentum[symbol]:.4f}, 标准差={self.std_value[symbol]:.4f}"
                    )
                else:
                    logger.info(
                        f"数据不足以计算动量: {symbol}, 需要至少 {self.std_period + 1} 个周期的数据"
                    )
            except Exception as e:
                logger.error(f"计算指标出错: {symbol}, 错误: {e!s}")

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
                        # 更新内部持仓状态
                        self.pos[symbol] = size
                        logger.info(
                            f"已发送多头信号到API: {symbol}, 价格={bar.close_price}, 数量={size}"
                        )

            elif current_pos > 0:
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
            # 使用全局API配置
            global API_URL, API_KEY

            # 构建通知内容 - 添加emoji
            notification_text = f"🚀 {self.strategy_name}: Your Next Big Trade Starts Here\n📊 Signal type: {signal_type}\n💱 Symbol: {symbol}\n💰 Price: ${price}"
            title_text = "Your Next Big Trade Starts Here"

            data = {
                "symbol": symbol,
                "direction": direction.value,
                "price": price,
                "signal_type": signal_type,
                "notification": notification_text,
                "title": title_text,
            }
            if extra_info:
                data.update(extra_info)

            # 使用当前时间戳
            current_time = datetime.now()
            payload = {
                "data": data,
                "strategyId": "6800c11f7d8349638b37b3af",
                "time": int(current_time.timestamp() * 1000),
            }

            # 设置请求头
            headers = {"Content-Type": "application/json"}
            if API_KEY:
                headers["X-AP-API-Key"] = API_KEY
            else:
                logger.warning("未设置API密钥，请检查环境变量API_KEY是否已配置")

            logger.info(f"发送信号请求: {API_URL}, strategyId={payload['strategyId']}")

            # 发送HTTP请求
            response = requests.post(
                url=API_URL,
                json=payload,
                headers=headers,
                timeout=10,  # 设置超时时间
            )

            # 检查响应
            logger.info(
                f"响应状态码: {response.status_code}，响应内容: {response.text}"
            )

            if response.status_code in (200, 201, 202):
                logger.info(
                    f"✅ 交易信号已发送到API: {self.strategy_name}, {symbol}, {direction.value}, {price}"
                )
                return True
            else:
                logger.error(f"❌ 发送交易信号失败状态码: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API请求错误: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 发送交易信号失败: {e}")
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
        "mom_threshold": 0.005,
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

    # 保持主线程运行
    while True:
        sleep(1)


if __name__ == "__main__":
    run_signal_service()
