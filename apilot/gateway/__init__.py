"""
Trading Gateway Module

Contains gateway implementations for connecting to different exchanges and trading APIs.
"""

from .binance import BinanceGateway
from .gateway import BaseGateway

# Define public API
__all__ = [
    "BaseGateway",
    "BinanceGateway",
]
