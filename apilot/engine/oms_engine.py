"""
订单管理系统引擎模块

负责订单全生命周期管理、仓位跟踪和交易事件处理
"""

from apilot.core import (
    # Event types
    EVENT_ACCOUNT,
    EVENT_CONTRACT,
    EVENT_ORDER,
    EVENT_POSITION,
    EVENT_QUOTE,
    EVENT_TRADE,
    # 数据类
    AccountData,
    # 引擎类
    ContractData,
    # 事件类
    Event,
    EventEngine,
    # 组件类
    OrderData,
    OrderRequest,
    PositionData,
    QuoteData,
    TradeData,
)
from apilot.engine.base_engine import register_engine

from .base_engine import BaseEngine
from .main_engine import MainEngine


@register_engine
class OmsEngine(BaseEngine):
    """
    Provides order management system function.
    """

    def __init__(self, main_engine: MainEngine, event_engine: EventEngine) -> None:
        """"""
        super().__init__(main_engine, event_engine, "oms")

        self.orders: dict[str, OrderData] = {}
        self.trades: dict[str, TradeData] = {}
        self.positions: dict[str, PositionData] = {}
        self.accounts: dict[str, AccountData] = {}
        self.contracts: dict[str, ContractData] = {}
        self.quotes: dict[str, QuoteData] = {}

        self.active_orders: dict[str, OrderData] = {}
        self.active_quotes: dict[str, QuoteData] = {}

        self.add_function()
        self.register_event()

    def add_function(self) -> None:
        """Add query function to main engine."""
        self.main_engine.get_order = self.get_order
        self.main_engine.get_trade = self.get_trade
        self.main_engine.get_position = self.get_position
        self.main_engine.get_account = self.get_account
        self.main_engine.get_contract = self.get_contract
        self.main_engine.get_quote = self.get_quote

        self.main_engine.get_all_orders = self.get_all_orders
        self.main_engine.get_all_trades = self.get_all_trades
        self.main_engine.get_all_positions = self.get_all_positions
        self.main_engine.get_all_accounts = self.get_all_accounts
        self.main_engine.get_all_contracts = self.get_all_contracts
        self.main_engine.get_all_quotes = self.get_all_quotes
        self.main_engine.get_all_active_orders = self.get_all_active_orders
        self.main_engine.get_all_active_quotes = self.get_all_active_quotes

        self.main_engine.update_order_request = self.update_order_request
        self.main_engine.convert_order_request = self.convert_order_request
        self.main_engine.get_converter = self.get_converter

    def register_event(self) -> None:
        """"""
        self.event_engine.register(EVENT_ORDER, self.process_order_event)
        self.event_engine.register(EVENT_TRADE, self.process_trade_event)
        self.event_engine.register(EVENT_POSITION, self.process_position_event)
        self.event_engine.register(EVENT_ACCOUNT, self.process_account_event)
        self.event_engine.register(EVENT_CONTRACT, self.process_contract_event)
        self.event_engine.register(EVENT_QUOTE, self.process_quote_event)

    def process_order_event(self, event: Event) -> None:
        """"""
        order: OrderData = event.data
        self.orders[order.orderid] = order

        # If order is active, then update data in dict.
        if order.is_active():
            self.active_orders[order.orderid] = order
        # Otherwise, pop inactive order from in dict
        elif order.orderid in self.active_orders:
            self.active_orders.pop(order.orderid)

    def process_trade_event(self, event: Event) -> None:
        """"""
        trade: TradeData = event.data
        self.trades[trade.tradeid] = trade

    def process_position_event(self, event: Event) -> None:
        """"""
        position: PositionData = event.data
        self.positions[position.positionid] = position

    def process_account_event(self, event: Event) -> None:
        """"""
        account: AccountData = event.data
        self.accounts[account.accountid] = account

    def process_contract_event(self, event: Event) -> None:
        """"""
        contract: ContractData = event.data
        self.contracts[contract.symbol] = contract

    def process_quote_event(self, event: Event) -> None:
        """处理行情事件，可以是QuoteData或BarData"""
        data = event.data
        if hasattr(data, "quoteid"):
            quote: QuoteData = data
            self.quotes[quote.quoteid] = quote

            # If quote is active, then update data in dict.
            if quote.is_active():
                self.active_quotes[quote.quoteid] = quote
            # Otherwise, pop inactive quote from in dict
            elif quote.quoteid in self.active_quotes:
                self.active_quotes.pop(quote.quoteid)

    def get_order(self, orderid: str) -> OrderData | None:
        """
        Get latest order data by orderid.
        """
        return self.orders.get(orderid, None)

    def get_trade(self, tradeid: str) -> TradeData | None:
        """
        Get trade data by tradeid.
        """
        return self.trades.get(tradeid, None)

    def get_position(self, positionid: str) -> PositionData | None:
        """
        Get latest position data by positionid.
        """
        return self.positions.get(positionid, None)

    def get_account(self, accountid: str) -> AccountData | None:
        """
        Get latest account data by accountid.
        """
        return self.accounts.get(accountid, None)

    def get_contract(self, symbol: str) -> ContractData | None:
        """
        Get contract data by symbol.
        """
        return self.contracts.get(symbol, None)

    def get_quote(self, quoteid: str) -> QuoteData | None:
        """
        Get latest quote data by quoteid.
        """
        return self.quotes.get(quoteid, None)

    def get_all_orders(self) -> list[OrderData]:
        """
        Get all order data.
        """
        return list(self.orders.values())

    def get_all_trades(self) -> list[TradeData]:
        """
        Get all trade data.
        """
        return list(self.trades.values())

    def get_all_positions(self) -> list[PositionData]:
        """
        Get all position data.
        """
        return list(self.positions.values())

    def get_all_accounts(self) -> list[AccountData]:
        """
        Get all account data.
        """
        return list(self.accounts.values())

    def get_all_contracts(self) -> list[ContractData]:
        """
        Get all contract data.
        """
        return list(self.contracts.values())

    def get_all_quotes(self) -> list[QuoteData]:
        """
        Get all quote data.
        """
        return list(self.quotes.values())

    def get_all_active_orders(self, symbol: str = "") -> list[OrderData]:
        """
        Get all active orders by symbol.

        If symbol is empty, return all active orders.
        """
        if not symbol:
            return list(self.active_orders.values())
        else:
            active_orders: list[OrderData] = [
                order for order in self.active_orders.values() if order.symbol == symbol
            ]
            return active_orders

    def get_all_active_quotes(self, symbol: str = "") -> list[QuoteData]:
        """
        Get all active quotes by symbol.
        If symbol is empty, return all active qutoes.
        """
        if not symbol:
            return list(self.active_quotes.values())
        else:
            active_quotes: list[QuoteData] = [
                quote for quote in self.active_quotes.values() if quote.symbol == symbol
            ]
            return active_quotes

    def update_order_request(
        self, req: OrderRequest, orderid: str, gateway_name: str
    ) -> None:
        """
        Update order request (simple version for crypto/US markets without offset conversion)
        """
        pass

    def convert_order_request(
        self, req: OrderRequest, gateway_name: str
    ) -> list[OrderRequest]:
        """
        Simple version for crypto/US markets without offset conversion
        """
        return [req]

    def get_converter(self, gateway_name: str) -> None:
        """
        Simple stub for crypto/US markets
        """
        return None

    def close(self) -> None:
        """
        Close the engine and release resources.
        """
        self.orders.clear()
        self.trades.clear()
        self.positions.clear()
        self.accounts.clear()
        self.contracts.clear()
        self.quotes.clear()
        self.active_orders.clear()
        self.active_quotes.clear()
