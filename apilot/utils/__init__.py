"""
Utility functions module.

Contains general helper functions for data processing, calculations, and other common operations.

This module includes tools for:
- Mathematical calculations (e.g., Sharpe ratio, max drawdown)
- Date handling (e.g., time conversion, backtest period splitting)
- Data processing (e.g., format conversion, filtering)
- Visualization (e.g., plotting K-lines, equity curves)
- Order management (e.g., local order ID management)
- Symbol handling (e.g., parsing and processing trading symbols)

Recommended usage:
    from apilot.utils import specific_utility_function
"""

# Import symbol handling tools
# Logging system
# Technical indicators
from .indicators import ArrayManager
from .logger import get_logger, log_exceptions, set_level

# Order management tools
from .order_manager import LocalOrderManager
from .symbol import get_base_symbol, get_exchange, split_symbol

# Define public API
__all__ = [
    "ArrayManager",
    "LocalOrderManager",
    "get_base_symbol",
    "get_exchange",
    "get_logger",
    "log_exceptions",
    "set_level",
    "split_symbol",
]
