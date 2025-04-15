"""
Data Source Module

Contains database interfaces and data source implementations for storing and loading market data.
"""

from .csv_provider import CsvDatabase
from .database import BarOverview, BaseDatabase, use_database

DATA_PROVIDERS = {}


def register_provider(name, provider_class):
    DATA_PROVIDERS[name] = provider_class


try:
    register_provider("csv", CsvDatabase)
except ImportError:
    pass


# Define public API
__all__ = [
    "DATA_PROVIDERS",
    "CsvDatabase",
    "register_provider",
    "BaseDatabase",
    "BarOverview",
    "use_database",
]
