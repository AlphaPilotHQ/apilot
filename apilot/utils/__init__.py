"""
Utility functions module.

Contains general helper functions for data processing, calculations, and other common operations.

This module includes tools for:
- Mathematical calculations (e.g., Sharpe ratio, max drawdown)
- Date handling (e.g., time conversion, backtest period splitting)
- Data processing (e.g., format conversion, filtering)
- Visualization (e.g., plotting K-lines, equity curves)
- Order management (e.g., local order ID management)

Recommended usage:
    from apilot.utils import specific_utility_function
"""

# Logging system
# Technical indicators
from .array_manager import ArrayManager
from .logger import setup_logging

# Define public API
__all__ = [
    "ArrayManager",
    "setup_logging",
]
