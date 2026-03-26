#!/usr/bin/env python
"""
Migration CLI script for moving data between storage backends.

Usage:
    python scripts/migrate_to_mysql.py --source sqlite --target mysql --mysql-url "mysql+aiomysql://user:pass@host/db"
    python scripts/migrate_to_mysql.py --source local --target mysql --data-dir "./data" --mysql-url "mysql+aiomysql://user:pass@host/db"
    python scripts/migrate_to_mysql.py --source sqlite --sqlite-url "sqlite:///./data/rag_eval.db" --target mysql --mysql-url "mysql+aiomysql://user:pass@host/db"
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.migration import (
    migrate_from_local_to_mysql,
    migrate_from_local_to_sqlite,
    migrate_from_sqlite_to_mysql,
    print_migration_report,
)
from src.core.logging import logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Migrate data between storage backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate from SQLite to MySQL
  python scripts/migrate_to_mysql.py --source sqlite --target mysql \\
      --sqlite-url "sqlite:///./data/rag_eval.db" \\
      --mysql-url "mysql+aiomysql://root:password@localhost:3306/rag_eval"

  # Migrate from Local storage to MySQL
  python scripts/migrate_to_mysql.py --source local --target mysql \\
      --data-dir "./data" \\
      --mysql-url "mysql+aiomysql://root:password@localhost:3306/rag_eval"

  # Migrate from Local to SQLite
  python scripts/migrate_to_mysql.py --source local --target sqlite \\
      --data-dir "./data" \\
      --sqlite-url "sqlite:///./data/rag_eval.db"

Environment variables:
  DATABASE_URL        MySQL or SQLite connection URL
  MYSQL_POOL_SIZE     MySQL connection pool size (default: 10)
""",
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "sqlite"],
        required=True,
        help="Source storage type",
    )

    parser.add_argument(
        "--target",
        type=str,
        choices=["sqlite", "mysql"],
        required=True,
        help="Target storage type",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Local data directory (default: ./data)",
    )

    parser.add_argument(
        "--sqlite-url",
        type=str,
        help="SQLite database URL (e.g., sqlite:///./data/rag_eval.db)",
    )

    parser.add_argument(
        "--mysql-url",
        type=str,
        help="MySQL database URL (e.g., mysql+aiomysql://user:pass@host:port/db)",
    )

    parser.add_argument(
        "--collections",
        type=str,
        nargs="+",
        default=["annotations", "evaluation_results", "evaluation_runs"],
        help="Collections to migrate (default: annotations evaluation_results evaluation_runs)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing records in target",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for migration (default: 100)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating",
    )

    return parser.parse_args()


async def run_migration(args: argparse.Namespace) -> int:
    """Run the migration based on arguments."""
    # Get URLs from args or environment
    sqlite_url = args.sqlite_url or os.getenv("DATABASE_URL")
    mysql_url = args.mysql_url or os.getenv("DATABASE_URL")

    # Build URLs if not provided
    if args.source == "sqlite" and not sqlite_url:
        sqlite_url = f"sqlite:///{args.data_dir}/rag_eval.db"
    if args.target == "sqlite" and not sqlite_url:
        sqlite_url = f"sqlite:///{args.data_dir}/rag_eval.db"

    # Validate required URLs
    if args.source == "sqlite" and not sqlite_url:
        print("Error: SQLite URL is required when source is sqlite")
        return 1

    if args.target == "mysql" and not mysql_url:
        print("Error: MySQL URL is required when target is mysql")
        print("Use --mysql-url or set DATABASE_URL environment variable")
        return 1

    if args.target == "sqlite" and not sqlite_url:
        print("Error: SQLite URL is required when target is sqlite")
        return 1

    # Dry run - just show what would happen
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Source: {args.source}")
        print(f"Target: {args.target}")
        print(f"Collections: {args.collections}")
        print(f"Batch size: {args.batch_size}")
        print(f"Overwrite: {args.overwrite}")
        if args.source == "sqlite" or args.target == "sqlite":
            print(f"SQLite URL: {sqlite_url}")
        if args.target == "mysql":
            print(f"MySQL URL: {mysql_url}")
        print("\nNo changes will be made.")
        return 0

    # Run migration
    print(f"\nStarting migration from {args.source} to {args.target}...")
    print(f"Collections: {args.collections}")

    try:
        if args.source == "sqlite" and args.target == "mysql":
            results = await migrate_from_sqlite_to_mysql(
                sqlite_url=sqlite_url,
                mysql_url=mysql_url,
                collections=args.collections,
                overwrite=args.overwrite,
                batch_size=args.batch_size,
            )
        elif args.source == "local" and args.target == "mysql":
            results = await migrate_from_local_to_mysql(
                data_dir=args.data_dir,
                mysql_url=mysql_url,
                collections=args.collections,
                overwrite=args.overwrite,
                batch_size=args.batch_size,
            )
        elif args.source == "local" and args.target == "sqlite":
            results = await migrate_from_local_to_sqlite(
                data_dir=args.data_dir,
                sqlite_url=sqlite_url,
                collections=args.collections,
                overwrite=args.overwrite,
                batch_size=args.batch_size,
            )
        else:
            print(f"Error: Migration from {args.source} to {args.target} is not supported")
            return 1

        # Print report
        print_migration_report(results)

        # Check for failures
        total_failed = sum(r.failed_records for r in results)
        if total_failed > 0:
            print(f"\nWarning: {total_failed} records failed to migrate")
            return 1

        print("\nMigration completed successfully!")
        return 0

    except Exception as e:
        print(f"\nError during migration: {e}")
        logger.exception("Migration failed")
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return asyncio.run(run_migration(args))


if __name__ == "__main__":
    sys.exit(main())