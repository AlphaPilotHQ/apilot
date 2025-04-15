"""
Strategy Module

Contains templates and implementations for various trading strategies,
supporting both PA (Portfolio Allocation) and target position strategies.

Main Components:
- PATemplate: Base class for PA strategies, providing a standard framework.

"""

# Import base class from the template module
from .pa_template import PATemplate

# Define public API
__all__ = [
    "PATemplate",
]
