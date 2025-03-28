from apilot_ctastrategy import (
    CtaTemplate,
    TickData,
    BarData,
    TradeData,
    OrderData
)

from time import time
from apilot.utils.logger import get_logger

# Initialize logger for this strategy
logger = get_logger("TestStrategy")


class TestStrategy(CtaTemplate):
    """"""
    test_trigger = 10

    tick_count = 0
    test_all_done = False

    parameters = ["test_trigger"]
    variables = ["tick_count", "test_all_done"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

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

    def on_tick(self, tick: TickData):
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

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """
        pass

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        self.put_event()

    def on_trade(self, trade: TradeData):
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
