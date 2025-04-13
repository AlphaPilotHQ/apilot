"""
Execution Algorithm Engine

Manages execution and monitoring of algorithm instances, processes market data and distributes execution results.
"""

from collections import defaultdict

from apilot.core.constant import AlgoStatus, Direction, Exchange, OrderType
from apilot.core.engine import (
    BaseEngine,
    EventEngine,
    MainEngine,
)
from apilot.core.event import (
    EVENT_ALGO_UPDATE,
    EVENT_ORDER,
    EVENT_TICK,
    EVENT_TIMER,
    EVENT_TRADE,
    Event,
)
from apilot.core.object import (
    CancelRequest,
    ContractData,
    OrderData,
    OrderRequest,
    SubscribeRequest,
    TickData,
    TradeData,
)
from apilot.utils.logger import get_logger

from .algo_template import AlgoTemplate

ENGINE_NAME = "AlgoTrading"

# Initialize logger at module level
logger = get_logger("AlgoEngine")


class AlgoEngine(BaseEngine):
    """Algorithm Engine"""

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """Constructor"""
        super().__init__(main_engine, event_engine, ENGINE_NAME)

        self.algo_templates: dict[str, type[AlgoTemplate]] = {}

        self.algos: dict[str, AlgoTemplate] = {}
        self.symbol_algo_map: dict[str, set[AlgoTemplate]] = defaultdict(set)
        self.orderid_algo_map: dict[str, AlgoTemplate] = {}

        self.load_algo_template()
        self.register_event()

    def init_engine(self) -> None:
        """Initialize engine"""
        logger.info("Algorithm trading engine started")

    def close(self) -> None:
        """Close engine"""
        self.stop_all()

    def load_algo_template(self) -> None:
        """Load algorithm classes"""
        from .best_limit_algo import BestLimitAlgo
        from .iceberg_algo import IcebergAlgo
        from .sniper_algo import SniperAlgo
        from .stop_algo import StopAlgo
        from .twap_algo import TwapAlgo

        self.add_algo_template(TwapAlgo)
        self.add_algo_template(IcebergAlgo)
        self.add_algo_template(SniperAlgo)
        self.add_algo_template(StopAlgo)
        self.add_algo_template(BestLimitAlgo)

    def add_algo_template(self, template: AlgoTemplate) -> None:
        """Add algorithm class"""
        self.algo_templates[template.__name__] = template

    def get_algo_template(self) -> dict:
        """Get algorithm class"""
        return self.algo_templates

    def register_event(self) -> None:
        """Register event listeners"""
        self.event_engine.register(EVENT_TICK, self.process_tick_event)
        self.event_engine.register(EVENT_TIMER, self.process_timer_event)
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)

    def process_tick_event(self, event: Event) -> None:
        """Process tick event"""
        tick: TickData = event.data
        algos: set[AlgoTemplate] = self.symbol_algo_map[tick.symbol]

        for algo in algos:
            algo.update_tick(tick)

    def process_timer_event(self, event: Event) -> None:
        """Process timer event"""
        # Create list to avoid dictionary changes
        algos: list[AlgoTemplate] = list(self.algos.values())

        for algo in algos:
            algo.update_timer()

    def process_trade_event(self, event: Event) -> None:
        """Process trade event"""
        trade: TradeData = event.data

        algo: AlgoTemplate | None = self.orderid_algo_map.get(trade.orderid, None)

        if algo and algo.status in {AlgoStatus.RUNNING, AlgoStatus.PAUSED}:
            algo.update_trade(trade)

    def process_order_event(self, event: Event) -> None:
        """Process order event"""
        order: OrderData = event.data

        algo: AlgoTemplate | None = self.orderid_algo_map.get(order.orderid, None)

        if algo and algo.status in {AlgoStatus.RUNNING, AlgoStatus.PAUSED}:
            algo.update_order(order)

    def start_algo(
        self,
        template_name: str,
        symbol: str,
        direction: Direction,
        price: float,
        volume: int,
        setting: dict,
    ) -> str:
        """Start algorithm"""
        contract: ContractData | None = self.main_engine.get_contract(symbol)
        if not contract:
            logger.warning(f"Algorithm start failed, contract not found: {symbol}")
            return ""

        algo_template: AlgoTemplate = self.algo_templates[template_name]

        # Create algorithm instance
        algo_template._count += 1
        algo_name: str = f"{algo_template.__name__}_{algo_template._count}"
        algo: AlgoTemplate = algo_template(
            self, algo_name, symbol, direction, price, volume, setting
        )

        # Subscribe to market data
        algos: set = self.symbol_algo_map[algo.symbol]
        if not algos:
            self.subscribe(contract.symbol, contract.exchange, contract.gateway_name)
        algos.add(algo)

        # Start algorithm
        algo.start()
        self.algos[algo_name] = algo

        return algo_name

    def pause_algo(self, algo_name: str) -> None:
        """Pause algorithm"""
        algo: AlgoTemplate | None = self.algos.get(algo_name, None)
        if algo:
            algo.pause()

    def resume_algo(self, algo_name: str) -> None:
        """Resume algorithm"""
        algo: AlgoTemplate | None = self.algos.get(algo_name, None)
        if algo:
            algo.resume()

    def stop_algo(self, algo_name: str) -> None:
        """Stop algorithm"""
        algo: AlgoTemplate | None = self.algos.get(algo_name, None)
        if algo:
            algo.stop()

    def stop_all(self) -> None:
        """Stop all algorithms"""
        for algo_name in list(self.algos.keys()):
            self.stop_algo(algo_name)

    def subscribe(self, symbol: str, exchange: Exchange, gateway_name: str) -> None:
        """Subscribe to market data"""
        req: SubscribeRequest = SubscribeRequest(symbol=symbol, exchange=exchange)
        self.main_engine.subscribe(req, gateway_name)

    def send_order(
        self,
        algo: AlgoTemplate,
        direction: Direction,
        price: float,
        volume: float,
        order_type: OrderType,
    ) -> str:
        """Send order"""
        contract: ContractData | None = self.main_engine.get_contract(algo.symbol)
        volume: float = float(round(volume / contract.min_volume) * contract.min_volume)
        if not volume:
            return ""

        req: OrderRequest = OrderRequest(
            symbol=contract.symbol,
            exchange=contract.exchange,
            direction=direction,
            type=order_type,
            volume=volume,
            price=price,
            reference=f"{ENGINE_NAME}_{algo.algo_name}",
        )
        orderid: str = self.main_engine.send_order(req, contract.gateway_name)

        self.orderid_algo_map[orderid] = algo
        return orderid

    def cancel_order(self, algo: AlgoTemplate, orderid: str) -> None:
        """Cancel order"""
        order: OrderData | None = self.main_engine.get_order(orderid)

        if not order:
            logger.warning(
                f"[{ENGINE_NAME}:{algo.algo_name}] Cancel order failed, order not found: {orderid}"
            )
            return

        req: CancelRequest = order.create_cancel_request()
        self.main_engine.cancel_order(req, order.gateway_name)

    def get_tick(self, algo: AlgoTemplate) -> TickData | None:
        """Get market data"""
        tick: TickData | None = self.main_engine.get_tick(algo.symbol)

        if not tick:
            logger.warning(
                f"[{ENGINE_NAME}:{algo.algo_name}] Get market data failed, data not found: {algo.symbol}"
            )

        return tick

    def get_contract(self, algo: AlgoTemplate) -> ContractData | None:
        """Get contract"""
        contract: ContractData | None = self.main_engine.get_contract(algo.symbol)

        if not contract:
            source = (
                f"{ENGINE_NAME}:{algo.algo_name}" if algo.algo_name else ENGINE_NAME
            )
            logger.warning(
                f"[{source}] Get contract failed, contract not found: {algo.symbol}"
            )

        return contract

    def put_algo_event(self, algo: AlgoTemplate, data: dict) -> None:
        """Push update"""
        # Remove finished algorithm instances
        if algo in self.algos.values() and algo.status in {
            AlgoStatus.STOPPED,
            AlgoStatus.FINISHED,
        }:
            self.algos.pop(algo.algo_name)

            for algos in self.symbol_algo_map.values():
                if algo in algos:
                    algos.remove(algo)

        event: Event = Event(EVENT_ALGO_UPDATE, data)
        self.event_engine.put(event)
