"""
Data Source Module

Contains database interfaces and data source implementations for storing and loading market data.
"""

from apilot.core.database import BaseDatabase

from .providers.csv_provider import CsvDatabase

DATA_PROVIDERS = {}


def register_provider(name, provider_class):
    DATA_PROVIDERS[name] = provider_class


try:
    from .providers.csv_provider import CsvDatabase

    register_provider("csv", CsvDatabase)
except ImportError:
    pass

# MongoDB provider has been moved to examples directory, no longer auto-registered
# If you need to use it, please refer to examples/data_providers/mongodb_provider.py

# Define public API
__all__ = [
    "DATA_PROVIDERS",
    "CsvDatabase",
    "register_provider",
]
