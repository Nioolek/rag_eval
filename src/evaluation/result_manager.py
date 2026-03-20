"""
Result manager for storing and retrieving evaluation results.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..models.evaluation_result import EvaluationResult, EvaluationRun
from ..storage.base import StorageBackend
from ..storage.storage_factory import get_storage
from ..core.exceptions import EvaluationError
from ..core.logging import logger


class ResultManager:
    """
    Manager for evaluation results with versioning support.
    """

    COLLECTION = "evaluation_results"
    RUNS_COLLECTION = "evaluation_runs"

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    async def save_result(self, result: EvaluationResult) -> str:
        """
        Save an evaluation result.

        Args:
            result: EvaluationResult to save

        Returns:
            Result ID
        """
        data = result.to_dict()
        result_id = await self.storage.save(self.COLLECTION, data)
        logger.debug(f"Saved evaluation result: {result_id}")
        return result_id

    async def get_result(self, result_id: str) -> Optional[EvaluationResult]:
        """Get an evaluation result by ID."""
        data = await self.storage.get(self.COLLECTION, result_id)
        if data:
            return EvaluationResult.from_dict(data)
        return None

    async def save_run(self, run: EvaluationRun) -> str:
        """
        Save an evaluation run.

        Args:
            run: EvaluationRun to save

        Returns:
            Run ID
        """
        data = run.to_dict()
        run_id = await self.storage.save(self.RUNS_COLLECTION, data)
        logger.info(f"Saved evaluation run: {run_id}")
        return run_id

    async def get_run(self, run_id: str) -> Optional[EvaluationRun]:
        """Get an evaluation run by ID."""
        data = await self.storage.get(self.RUNS_COLLECTION, run_id)
        if data:
            return EvaluationRun.from_dict(data)
        return None

    async def list_runs(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[EvaluationRun]:
        """
        List evaluation runs.

        Args:
            limit: Maximum number of runs to return
            offset: Number of runs to skip

        Returns:
            List of EvaluationRun
        """
        runs_data = await self.storage.get_all(
            self.RUNS_COLLECTION,
            limit=limit,
            offset=offset,
        )

        return [EvaluationRun.from_dict(d) for d in runs_data]

    async def get_results_by_run(
        self,
        run_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EvaluationResult]:
        """
        Get all results for a specific run.

        Args:
            run_id: Run ID
            limit: Maximum results to return
            offset: Number to skip

        Returns:
            List of EvaluationResult
        """
        results_data = await self.storage.get_all(
            self.COLLECTION,
            filters={"run_id": run_id},
            limit=limit,
            offset=offset,
        )

        return [EvaluationResult.from_dict(d) for d in results_data]

    async def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run and its results."""
        # Delete results
        results = await self.get_results_by_run(run_id, limit=10000)
        for result in results:
            await self.storage.delete(self.COLLECTION, result.id)

        # Delete run
        await self.storage.delete(self.RUNS_COLLECTION, run_id)

        logger.info(f"Deleted evaluation run: {run_id}")
        return True

    async def export_run(
        self,
        run_id: str,
        format: str = "json",
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Export evaluation run to file.

        Args:
            run_id: Run ID
            format: Export format (json, csv)
            output_path: Output file path

        Returns:
            Path to exported file
        """
        run = await self.get_run(run_id)
        if not run:
            raise EvaluationError(f"Run not found: {run_id}")

        results = await self.get_results_by_run(run_id, limit=10000)

        if output_path is None:
            output_path = Path(f"evaluation_run_{run_id}.{format}")

        if format == "json":
            import json
            export_data = {
                "run": run.to_dict(),
                "results": [r.to_dict() for r in results],
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

        elif format == "csv":
            import csv
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    "result_id", "annotation_id", "success",
                    "duration_ms", "average_score", "rag_interface"
                ])

                # Data
                for r in results:
                    writer.writerow([
                        r.id,
                        r.annotation_id,
                        r.success,
                        r.duration_ms,
                        r.metrics.average_score,
                        r.rag_interface,
                    ])

        logger.info(f"Exported run {run_id} to {output_path}")
        return output_path

    async def get_run_statistics(self, run_id: str) -> dict[str, Any]:
        """
        Get statistics for an evaluation run.

        Args:
            run_id: Run ID

        Returns:
            Statistics dictionary
        """
        run = await self.get_run(run_id)
        if not run:
            raise EvaluationError(f"Run not found: {run_id}")

        return {
            "run_id": run_id,
            "status": run.status,
            "total_annotations": run.total_annotations,
            "completed_count": run.completed_count,
            "failed_count": run.failed_count,
            "duration_seconds": run.duration_seconds,
            "summary_by_interface": run.summary_by_interface,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        }

    # ===== Tag Management =====

    async def add_result_tag(
        self,
        run_id: str,
        result_id: str,
        tag: str,
    ) -> bool:
        """
        Add a tag to a specific result.

        Args:
            run_id: Run ID
            result_id: Result ID
            tag: Tag to add

        Returns:
            True if successful
        """
        run = await self.get_run(run_id)
        if not run:
            return False

        for result in run.results:
            if result.id == result_id:
                result.add_tag(tag)
                break
        else:
            return False

        # Save updated run
        await self.storage.save(self.RUNS_COLLECTION, run.to_dict())
        logger.debug(f"Added tag '{tag}' to result {result_id}")
        return True

    async def remove_result_tag(
        self,
        run_id: str,
        result_id: str,
        tag: str,
    ) -> bool:
        """
        Remove a tag from a specific result.

        Args:
            run_id: Run ID
            result_id: Result ID
            tag: Tag to remove

        Returns:
            True if successful
        """
        run = await self.get_run(run_id)
        if not run:
            return False

        for result in run.results:
            if result.id == result_id:
                result.remove_tag(tag)
                break
        else:
            return False

        # Save updated run
        await self.storage.save(self.RUNS_COLLECTION, run.to_dict())
        logger.debug(f"Removed tag '{tag}' from result {result_id}")
        return True

    async def add_run_tag(self, run_id: str, tag: str) -> bool:
        """
        Add a tag to an evaluation run.

        Args:
            run_id: Run ID
            tag: Tag to add

        Returns:
            True if successful
        """
        run = await self.get_run(run_id)
        if not run:
            return False

        run.add_tag(tag)
        await self.storage.save(self.RUNS_COLLECTION, run.to_dict())
        logger.debug(f"Added tag '{tag}' to run {run_id}")
        return True

    async def remove_run_tag(self, run_id: str, tag: str) -> bool:
        """
        Remove a tag from an evaluation run.

        Args:
            run_id: Run ID
            tag: Tag to remove

        Returns:
            True if successful
        """
        run = await self.get_run(run_id)
        if not run:
            return False

        run.remove_tag(tag)
        await self.storage.save(self.RUNS_COLLECTION, run.to_dict())
        logger.debug(f"Removed tag '{tag}' from run {run_id}")
        return True

    async def get_results_by_tag(
        self,
        run_id: str,
        tag: str,
    ) -> list[EvaluationResult]:
        """
        Get all results with a specific tag.

        Args:
            run_id: Run ID
            tag: Tag to filter by

        Returns:
            List of EvaluationResult with the tag
        """
        run = await self.get_run(run_id)
        if not run:
            return []

        return run.get_results_by_tag(tag)

    async def get_all_tags(self, run_id: str) -> dict[str, Any]:
        """
        Get all tags used in a run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary with run_tags and result_tags
        """
        run = await self.get_run(run_id)
        if not run:
            return {"run_tags": [], "result_tags": []}

        # Get tags from all results with counts
        tag_counts: dict[str, int] = {}
        for result in run.results:
            for tag in result.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return {
            "run_tags": run.tags,
            "result_tags": [
                {"name": tag, "count": count}
                for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])
            ],
        }

    async def batch_add_tag(
        self,
        run_id: str,
        result_ids: list[str],
        tag: str,
    ) -> int:
        """
        Add a tag to multiple results at once.

        Args:
            run_id: Run ID
            result_ids: List of result IDs
            tag: Tag to add

        Returns:
            Number of results updated
        """
        run = await self.get_run(run_id)
        if not run:
            return 0

        result_id_set = set(result_ids)
        updated = 0

        for result in run.results:
            if result.id in result_id_set:
                result.add_tag(tag)
                updated += 1

        if updated > 0:
            await self.storage.save(self.RUNS_COLLECTION, run.to_dict())
            logger.debug(f"Batch added tag '{tag}' to {updated} results")

        return updated


# Singleton instance
_manager_instance: Optional[ResultManager] = None


async def get_result_manager() -> ResultManager:
    """Get singleton result manager instance."""
    global _manager_instance

    if _manager_instance is None:
        storage = await get_storage()
        _manager_instance = ResultManager(storage)

    return _manager_instance