"""
Migration script to add dataset_id to existing annotations.
Creates a default dataset and assigns all existing annotations to it.

Usage:
    python scripts/migrate_add_dataset.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def migrate():
    """Run the migration."""
    print("=" * 60)
    print("Dataset Migration Script")
    print("=" * 60)
    print()

    # Import here to ensure path is set up
    from src.storage.storage_factory import get_storage
    from src.models.dataset import Dataset, DatasetStatus
    from datetime import datetime

    storage = await get_storage()

    # Check if datasets collection already has entries
    existing_datasets = await storage.get_all("datasets", limit=10)

    if existing_datasets:
        print("Found existing datasets:")
        for d in existing_datasets:
            print(f"  - {d.get('name')} (ID: {d.get('id')[:8]}...)")

        # Check if there's already a default dataset
        default_datasets = [d for d in existing_datasets if d.get("is_default")]
        if default_datasets:
            default_dataset_id = default_datasets[0]["id"]
            print(f"\nUsing existing default dataset: {default_dataset_id[:8]}...")
        else:
            # Create default dataset
            default_dataset = Dataset(
                name="Default Dataset",
                description="Default dataset created during migration",
                status=DatasetStatus.ACTIVE,
                is_default=True,
            )
            await storage.save("datasets", default_dataset.to_dict())
            default_dataset_id = default_dataset.id
            print(f"\nCreated new default dataset: {default_dataset_id[:8]}...")
    else:
        # Create default dataset
        default_dataset = Dataset(
            name="Default Dataset",
            description="Default dataset for existing annotations",
            status=DatasetStatus.ACTIVE,
            is_default=True,
        )
        await storage.save("datasets", default_dataset.to_dict())
        default_dataset_id = default_dataset.id
        print(f"Created default dataset: {default_dataset_id[:8]}...")

    # Update all annotations without dataset_id
    print("\nScanning annotations...")
    updated_count = 0
    total_count = 0

    async for ann_data in storage.iterate("annotations"):
        total_count += 1
        if not ann_data.get("dataset_id"):
            await storage.update(
                "annotations",
                ann_data["id"],
                {"dataset_id": default_dataset_id}
            )
            updated_count += 1

            if updated_count % 100 == 0:
                print(f"  Updated {updated_count} annotations...")

    # Update dataset annotation count
    print("\nUpdating dataset statistics...")
    final_count = await storage.count("annotations", filters={"dataset_id": default_dataset_id})

    await storage.update(
        "datasets",
        default_dataset_id,
        {
            "annotation_count": final_count,
            "updated_at": datetime.now().isoformat(),
        }
    )

    print()
    print("=" * 60)
    print("Migration Summary:")
    print(f"  - Total annotations scanned: {total_count}")
    print(f"  - Annotations updated: {updated_count}")
    print(f"  - Final dataset annotation count: {final_count}")
    print(f"  - Default dataset ID: {default_dataset_id}")
    print("=" * 60)
    print("\nMigration complete!")


def main():
    """Main entry point."""
    try:
        asyncio.run(migrate())
    except KeyboardInterrupt:
        print("\nMigration cancelled by user.")
    except Exception as e:
        print(f"\nMigration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()