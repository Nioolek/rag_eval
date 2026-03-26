"""
Storage module: local file, SQLite, and MySQL storage backends.
"""

from .base import StorageBackend
from .local_storage import LocalStorage
from .sqlite_storage import SQLiteStorage
from .storage_factory import StorageFactory, get_storage

# Optional MySQL storage - only available if aiomysql is installed
try:
    from .mysql_storage import MySQLStorage
    __all__ = [
        "StorageBackend",
        "LocalStorage",
        "SQLiteStorage",
        "MySQLStorage",
        "StorageFactory",
        "get_storage",
    ]
except ImportError:
    __all__ = [
        "StorageBackend",
        "LocalStorage",
        "SQLiteStorage",
        "StorageFactory",
        "get_storage",
    ]