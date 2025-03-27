"""
Order management utilities.

This module provides tools for managing orders and order IDs
in trading systems.
"""

from copy import copy
from typing import Callable, Dict, Optional

from apilot.core.gateway import BaseGateway
from apilot.core.object import CancelRequest, OrderData


class LocalOrderManager:
    """
    Management tool to support use local order id for trading.
    
    This utility helps manage the mapping between local order IDs and
    exchange/broker system order IDs. It provides buffering capability
    for handling asynchronous order operations.
    """

    def __init__(self, gateway: BaseGateway, order_prefix: str = "") -> None:
        """
        Initialize order manager.
        
        Args:
            gateway: Trading gateway instance
            order_prefix: Prefix for local order IDs
        """
        self.gateway: BaseGateway = gateway

        # For generating local orderid
        self.order_prefix = order_prefix
        self.order_count: int = 0
        self.orders: Dict[str, OrderData] = {}        # local_orderid: order

        # Map between local and system orderid
        self.local_sys_orderid_map: Dict[str, str] = {}
        self.sys_local_orderid_map: Dict[str, str] = {}

        # Push order data buf
        self.push_data_buf: Dict[str, Dict] = {}  # sys_orderid: data

        # Callback for processing push order data
        self.push_data_callback: Callable = None

        # Cancel request buf
        self.cancel_request_buf: Dict[str, CancelRequest] = {}    # local_orderid: req

        # Hook cancel order function
        self._cancel_order: Callable = gateway.cancel_order
        gateway.cancel_order = self.cancel_order

    def new_local_orderid(self) -> str:
        """
        Generate a new local orderid.
        """
        self.order_count += 1
        local_orderid: str = self.order_prefix + str(self.order_count).rjust(8, "0")
        return local_orderid

    def get_local_orderid(self, sys_orderid: str) -> str:
        """
        Get local orderid with sys orderid.
        """
        local_orderid: str = self.sys_local_orderid_map.get(sys_orderid, "")

        if not local_orderid:
            local_orderid = self.new_local_orderid()
            self.update_orderid_map(local_orderid, sys_orderid)

        return local_orderid

    def get_sys_orderid(self, local_orderid: str) -> str:
        """
        Get sys orderid with local orderid.
        """
        sys_orderid: str = self.local_sys_orderid_map.get(local_orderid, "")
        return sys_orderid

    def update_orderid_map(self, local_orderid: str, sys_orderid: str) -> None:
        """
        Update orderid map.
        """
        self.sys_local_orderid_map[sys_orderid] = local_orderid
        self.local_sys_orderid_map[local_orderid] = sys_orderid

        self.check_cancel_request(local_orderid)
        self.check_push_data(sys_orderid)

    def check_push_data(self, sys_orderid: str) -> None:
        """
        Check if any order push data waiting.
        """
        if sys_orderid not in self.push_data_buf:
            return

        data: dict = self.push_data_buf.pop(sys_orderid)
        if self.push_data_callback:
            self.push_data_callback(data)

    def add_push_data(self, sys_orderid: str, data: dict) -> None:
        """
        Add push data into buf.
        """
        self.push_data_buf[sys_orderid] = data

    def get_order_with_sys_orderid(self, sys_orderid: str) -> Optional[OrderData]:
        """
        Get order by system order ID.
        
        Args:
            sys_orderid: System order ID
            
        Returns:
            Optional[OrderData]: Order data or None if not found
        """
        local_orderid: str = self.sys_local_orderid_map.get(sys_orderid, None)
        if not local_orderid:
            return None
        else:
            return self.get_order_with_local_orderid(local_orderid)

    def get_order_with_local_orderid(self, local_orderid: str) -> OrderData:
        """
        Get order by local order ID.
        
        Args:
            local_orderid: Local order ID
            
        Returns:
            OrderData: Order data
        """
        order: OrderData = self.orders[local_orderid]
        return copy(order)

    def on_order(self, order: OrderData) -> None:
        """
        Keep an order buf before pushing it to gateway.
        
        Args:
            order: Order data
        """
        self.orders[order.orderid] = copy(order)
        self.gateway.on_order(order)

    def cancel_order(self, req: CancelRequest) -> None:
        """
        Cancel order with request.
        
        Handles cases where system order ID is not yet available
        by buffering the cancel request.
        
        Args:
            req: Cancel request
        """
        sys_orderid: str = self.get_sys_orderid(req.orderid)
        if not sys_orderid:
            self.cancel_request_buf[req.orderid] = req
            return

        self._cancel_order(req)

    def check_cancel_request(self, local_orderid: str) -> None:
        """
        Check if there's pending cancel request for an order ID.
        
        Args:
            local_orderid: Local order ID
        """
        if local_orderid not in self.cancel_request_buf:
            return

        req: CancelRequest = self.cancel_request_buf.pop(local_orderid)
        self.gateway.cancel_order(req)
