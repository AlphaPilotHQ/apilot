from time import time
from typing import ClassVar

import apilot as ap
from apilot.utils.logger import get_logger

# Initialize logger for this strategy
logger = get_logger(__name__)


class TestStrategy(ap.PATemplate):
    """
    测试策略
    """

    # 类变量
    test_trigger: ClassVar[int] = 1
    tick_count: ClassVar[int] = 0
    test_all_done: ClassVar[bool] = False

    parameters: ClassVar[list[str]] = ["test_trigger"]
    variables: ClassVar[list[str]] = ["tick_count", "test_all_done"]

    def __init__(self, pa_engine, strategy_name, symbol, setting):
        """"""
        super().__init__(pa_engine, strategy_name, symbol, setting)

        self.test_funcs = [
            self.test_market_order,
            self.test_limit_order,
            self.test_cancel_all,
        ]
        self.last_tick = None

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        logger.info(f"[{self.strategy_name}] Strategy initialized")

    def on_start(self):
        """
        Callback when strategy is started.
        """
        logger.info(f"[{self.strategy_name}] Strategy started")

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        logger.info(f"[{self.strategy_name}] Strategy stopped")

    def on_tick(self, tick: ap.TickData):
        """
        Callback of new tick data update.
        """
        if self.test_all_done:
            return

        self.last_tick = tick

        self.tick_count += 1
        if self.tick_count >= self.test_trigger:
            self.tick_count = 0

            if self.test_funcs:
                test_func = self.test_funcs.pop(0)

                start = time()
                test_func()
                time_cost = (time() - start) * 1000
                logger.info(f"[{self.strategy_name}] Time cost: {time_cost} ms")
            else:
                logger.info(f"[{self.strategy_name}] All tests completed")
                self.test_all_done = True

        self.put_event()

    def on_bar(self, bar: ap.BarData):
        """
        Callback of new bar data update.
        """
        pass

    def on_order(self, order: ap.OrderData):
        """
        Callback of new order data update.
        """
        self.put_event()

    def on_trade(self, trade: ap.TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def test_market_order(self):
        """"""
        self.buy(self.last_tick.limit_up, 1)
        logger.info(f"[{self.strategy_name}] Market order test executed")

    def test_limit_order(self):
        """"""
        self.buy(self.last_tick.limit_down, 1)
        logger.info(f"[{self.strategy_name}] Limit order test executed")

    def test_cancel_all(self):
        """"""
        self.cancel_all()
        logger.info(f"[{self.strategy_name}] Cancel all test executed")
