"""
Data Provider Module

Contains implementations of all database providers, responsible for data storage and loading.
"""

# Import CSV database provider
from .csv_provider import CsvDatabase

# Try to import MongoDB database provider (optional dependency)
try:
    from .mongodb_provider import MongoDBDatabase

    _HAS_MONGODB = True
except ImportError:
    _HAS_MONGODB = False

# Register database providers based on availability
from apilot.core.database import register_database

# Register CSV provider by default
register_database("csv", CsvDatabase)

__all__ = ["CsvDatabase"]

# If MongoDB is available, add it to the public API
if _HAS_MONGODB:
    __all__.append("MongoDBDatabase")
    register_database("mongodb", MongoDBDatabase)
