"""
Data migration utilities for moving data between storage backends.
Supports migration from Local/SQLite to MySQL.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional

from .base import StorageBackend
from .local_storage import LocalStorage
from .sqlite_storage import SQLiteStorage
from .mysql_storage import MySQLStorage
from ..core.logging import logger


@dataclass
class MigrationResult:
    """Result of a migration operation."""
    collection: str
    total_records: int
    migrated_records: int
    failed_records: int
    errors: list[str]


class StorageMigrator:
    """
    Migrate data between storage backends.

    Supports:
    - LocalStorage -> MySQLStorage
    - SQLiteStorage -> MySQLStorage
    - LocalStorage -> SQLiteStorage
    """

    def __init__(
        self,
        source: StorageBackend,
        target: StorageBackend,
        batch_size: int = 100,
    ):
        """
        Initialize migrator.

        Args:
            source: Source storage backend
            target: Target storage backend
            batch_size: Number of records to migrate per batch
        """
        self.source = source
        self.target = target
        self.batch_size = batch_size

    async def migrate_collection(
        self,
        collection: str,
        overwrite: bool = False,
    ) -> MigrationResult:
        """
        Migrate a single collection.

        Args:
            collection: Collection name to migrate
            overwrite: If True, overwrite existing records in target

        Returns:
            MigrationResult with statistics
        """
        result = MigrationResult(
            collection=collection,
            total_records=0,
            migrated_records=0,
            failed_records=0,
            errors=[],
        )

        try:
            # Count total records
            result.total_records = await self.source.count(collection)
            logger.info(
                f"Starting migration of {collection}: {result.total_records} records"
            )

            # Check if target has records
            target_count = await self.target.count(collection)
            if target_count > 0 and not overwrite:
                logger.warning(
                    f"Target already has {target_count} records in {collection}. "
                    "Use overwrite=True to replace them."
                )
                result.errors.append(
                    f"Target already has {target_count} records. Use overwrite=True."
                )
                return result

            # Migrate in batches
            async for record in self.source.iterate(collection, batch_size=self.batch_size):
                try:
                    record_id = record.get("id")
                    if not record_id:
                        result.failed_records += 1
                        result.errors.append("Record missing ID field")
                        continue

                    if overwrite:
                        # Delete existing record in target if exists
                        await self.target.delete(collection, record_id)

                    # Save to target
                    await self.target.save(collection, record)
                    result.migrated_records += 1

                    if result.migrated_records % 100 == 0:
                        logger.info(
                            f"Migrated {result.migrated_records}/{result.total_records} "
                            f"records in {collection}"
                        )

                except Exception as e:
                    result.failed_records += 1
                    error_msg = f"Failed to migrate record {record.get('id', 'unknown')}: {e}"
                    result.errors.append(error_msg)
                    logger.error(error_msg)

            logger.info(
                f"Completed migration of {collection}: "
                f"{result.migrated_records} migrated, {result.failed_records} failed"
            )

        except Exception as e:
            error_msg = f"Migration failed for {collection}: {e}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        return result

    async def migrate_all(
        self,
        collections: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> list[MigrationResult]:
        """
        Migrate all specified collections.

        Args:
            collections: List of collections to migrate. If None, migrates default collections.
            overwrite: If True, overwrite existing records

        Returns:
            List of MigrationResult for each collection
        """
        if collections is None:
            collections = [
                "annotations",
                "evaluation_results",
                "evaluation_runs",
            ]

        results = []
        for collection in collections:
            result = await self.migrate_collection(collection, overwrite=overwrite)
            results.append(result)

        return results

    async def migrate_versions(
        self,
        collection: str,
        record_id: str,
    ) -> int:
        """
        Migrate version history for a specific record.

        Args:
            collection: Collection name
            record_id: Record ID

        Returns:
            Number of versions migrated
        """
        versions = await self.source.get_versions(collection, record_id)
        migrated = 0

        for version in versions:
            try:
                version_number = version.pop("version_number", None)
                versioned_at = version.pop("versioned_at", None)

                await self.target.save_version(collection, record_id, version)
                migrated += 1
            except Exception as e:
                logger.error(f"Failed to migrate version: {e}")

        return migrated


async def create_storage(
    storage_type: str,
    database_url: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> StorageBackend:
    """
    Create a storage backend instance.

    Args:
        storage_type: "local", "sqlite", or "mysql"
        database_url: Database URL for SQLite/MySQL
        data_dir: Data directory for local storage

    Returns:
        Initialized storage backend
    """
    if storage_type == "local":
        from pathlib import Path
        storage = LocalStorage(data_dir=Path(data_dir or "./data"))
    elif storage_type == "sqlite":
        if not database_url:
            database_url = f"sqlite:///{data_dir or './data'}/rag_eval.db"
        storage = SQLiteStorage(database_url=database_url)
    elif storage_type == "mysql":
        if not database_url:
            raise ValueError("DATABASE_URL is required for MySQL storage")
        storage = MySQLStorage(database_url=database_url)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

    await storage.initialize()
    return storage


async def migrate_from_sqlite_to_mysql(
    sqlite_url: str,
    mysql_url: str,
    collections: Optional[list[str]] = None,
    overwrite: bool = False,
    batch_size: int = 100,
) -> list[MigrationResult]:
    """
    Convenience function for SQLite to MySQL migration.

    Args:
        sqlite_url: SQLite database URL
        mysql_url: MySQL database URL
        collections: Collections to migrate
        overwrite: Overwrite existing records
        batch_size: Batch size for migration

    Returns:
        List of migration results
    """
    # Create source and target storage
    source = SQLiteStorage(database_url=sqlite_url)
    target = MySQLStorage(database_url=mysql_url)

    try:
        await source.initialize()
        await target.initialize()

        migrator = StorageMigrator(
            source=source,
            target=target,
            batch_size=batch_size,
        )

        results = await migrator.migrate_all(collections=collections, overwrite=overwrite)
        return results

    finally:
        await source.close()
        await target.close()


async def migrate_from_local_to_mysql(
    data_dir: str,
    mysql_url: str,
    collections: Optional[list[str]] = None,
    overwrite: bool = False,
    batch_size: int = 100,
) -> list[MigrationResult]:
    """
    Convenience function for Local storage to MySQL migration.

    Args:
        data_dir: Local data directory
        mysql_url: MySQL database URL
        collections: Collections to migrate
        overwrite: Overwrite existing records
        batch_size: Batch size for migration

    Returns:
        List of migration results
    """
    from pathlib import Path

    # Create source and target storage
    source = LocalStorage(data_dir=Path(data_dir))
    target = MySQLStorage(database_url=mysql_url)

    try:
        await source.initialize()
        await target.initialize()

        migrator = StorageMigrator(
            source=source,
            target=target,
            batch_size=batch_size,
        )

        results = await migrator.migrate_all(collections=collections, overwrite=overwrite)
        return results

    finally:
        await source.close()
        await target.close()


async def migrate_from_local_to_sqlite(
    data_dir: str,
    sqlite_url: str,
    collections: Optional[list[str]] = None,
    overwrite: bool = False,
    batch_size: int = 100,
) -> list[MigrationResult]:
    """
    Convenience function for Local storage to SQLite migration.

    Args:
        data_dir: Local data directory
        sqlite_url: SQLite database URL
        collections: Collections to migrate
        overwrite: Overwrite existing records
        batch_size: Batch size for migration

    Returns:
        List of migration results
    """
    from pathlib import Path

    # Create source and target storage
    source = LocalStorage(data_dir=Path(data_dir))
    target = SQLiteStorage(database_url=sqlite_url)

    try:
        await source.initialize()
        await target.initialize()

        migrator = StorageMigrator(
            source=source,
            target=target,
            batch_size=batch_size,
        )

        results = await migrator.migrate_all(collections=collections, overwrite=overwrite)
        return results

    finally:
        await source.close()
        await target.close()


def print_migration_report(results: list[MigrationResult]) -> None:
    """Print a formatted migration report."""
    print("\n" + "=" * 60)
    print("Migration Report")
    print("=" * 60)

    total_records = 0
    total_migrated = 0
    total_failed = 0

    for result in results:
        print(f"\nCollection: {result.collection}")
        print(f"  Total records:    {result.total_records}")
        print(f"  Migrated:         {result.migrated_records}")
        print(f"  Failed:           {result.failed_records}")

        if result.errors:
            print(f"  Errors ({len(result.errors)}):")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(result.errors) > 5:
                print(f"    ... and {len(result.errors) - 5} more errors")

        total_records += result.total_records
        total_migrated += result.migrated_records
        total_failed += result.failed_records

    print("\n" + "-" * 60)
    print(f"Summary:")
    print(f"  Total records:    {total_records}")
    print(f"  Total migrated:   {total_migrated}")
    print(f"  Total failed:     {total_failed}")
    print("=" * 60 + "\n")