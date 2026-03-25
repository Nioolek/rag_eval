"""
Evaluation runner with concurrent execution support.
Implements Template Method pattern for standardized evaluation workflow.
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional

from ..models.annotation import Annotation
from ..models.evaluation_result import EvaluationResult, EvaluationRun
from ..models.metric_result import MetricCategory, MetricResult
from ..rag.base_adapter import RAGAdapter
from ..rag.mock_adapter import MockRAGAdapter
from ..storage.base import StorageBackend
from ..storage.storage_factory import get_storage
from ..core.config import get_config
from ..core.exceptions import EvaluationError
from ..core.logging import logger

from .metrics.base import MetricContext
from .metrics.metric_factory import MetricFactory
from .llm_evaluator import get_llm_evaluator
from .result_manager import ResultManager, get_result_manager


# Global concurrency control for multi-user support
_max_concurrent_runs = 3  # Maximum concurrent evaluation runs
_running_count = 0
_global_semaphore: Optional[asyncio.Semaphore] = None


def _get_global_semaphore() -> asyncio.Semaphore:
    """Get or create the global semaphore for evaluation run limiting."""
    global _global_semaphore
    if _global_semaphore is None:
        _global_semaphore = asyncio.Semaphore(_max_concurrent_runs)
    return _global_semaphore


@dataclass
class EvaluationProgress:
    """Progress information for running evaluation."""
    total: int = 0
    completed: int = 0
    failed: int = 0
    current_query: str = ""
    start_time: float = 0.0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def progress_percent(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0

    @property
    def estimated_remaining_seconds(self) -> float:
        if self.completed == 0:
            return 0
        rate = self.completed / self.elapsed_seconds
        remaining = self.total - self.completed
        return remaining / rate if rate > 0 else 0


class EvaluationRunner:
    """
    Main evaluation runner with concurrent execution.
    Supports single/dual RAG interface comparison.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: int = 120,
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self._rag_adapters: dict[str, RAGAdapter] = {}
        self._metrics: list = []
        self._process_pool: Optional[ProcessPoolExecutor] = None
        self._running = False
        self._cancelled = False
        self._progress: Optional[EvaluationProgress] = None
        self._progress_callback: Optional[Callable] = None

    def set_rag_adapter(
        self,
        adapter: RAGAdapter,
        name: str = "default",
    ) -> None:
        """Set a RAG adapter for evaluation."""
        self._rag_adapters[name] = adapter
        logger.info(f"Set RAG adapter: {name}")

    def set_metrics(
        self,
        metric_names: list[str],
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Set metrics to use for evaluation."""
        self._metrics = MetricFactory.create_all(metric_names, config)
        logger.info(f"Set {len(self._metrics)} metrics: {[m.name for m in self._metrics]}")

    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    async def run(
        self,
        annotations: list[Annotation],
        run_name: str = "",
        rag_interfaces: Optional[list[str]] = None,
    ) -> EvaluationRun:
        """
        Run evaluation on annotations.

        Args:
            annotations: List of annotations to evaluate
            run_name: Name for this evaluation run
            rag_interfaces: RAG interface names to use (default: ["default"])

        Returns:
            EvaluationRun with results

        Raises:
            EvaluationError: If evaluation is already running or max concurrent runs reached
        """
        global _running_count

        if self._running:
            raise EvaluationError("Evaluation already running")

        # Acquire global semaphore to limit concurrent evaluation runs
        global_semaphore = _get_global_semaphore()
        acquired = global_semaphore.locked() and _running_count >= _max_concurrent_runs

        if acquired:
            raise EvaluationError(
                f"Maximum concurrent evaluation runs ({_max_concurrent_runs}) reached. "
                "Please wait for other evaluations to complete."
            )

        # Try to acquire the global semaphore
        async with global_semaphore:
            _running_count += 1
            self._running = True
            self._cancelled = False

            try:
                return await self._run_evaluation(
                    annotations, run_name, rag_interfaces
                )
            finally:
                _running_count -= 1
                self._running = False

    async def run_iter(
        self,
        annotations: list[Annotation],
        run_name: str = "",
        rag_interfaces: Optional[list[str]] = None,
    ) -> AsyncIterator[tuple[EvaluationProgress, Optional[EvaluationResult], dict]]:
        """
        Run evaluation and yield progress after each item completes.

        This is a generator version of run() that yields progress updates
        after each evaluation completes, enabling real-time UI updates.

        Args:
            annotations: List of annotations to evaluate
            run_name: Name for this evaluation run
            rag_interfaces: RAG interface names to use (default: ["default"])

        Yields:
            Tuple of (progress, result, stats) after each evaluation completes.
            result is None for the final yield indicating completion.
            stats contains real-time metrics statistics.

        Raises:
            EvaluationError: If evaluation is already running or max concurrent runs reached
        """
        global _running_count

        if self._running:
            raise EvaluationError("Evaluation already running")

        # Acquire global semaphore to limit concurrent evaluation runs
        global_semaphore = _get_global_semaphore()
        acquired = global_semaphore.locked() and _running_count >= _max_concurrent_runs

        if acquired:
            raise EvaluationError(
                f"Maximum concurrent evaluation runs ({_max_concurrent_runs}) reached. "
                "Please wait for other evaluations to complete."
            )

        # Try to acquire the global semaphore
        async with global_semaphore:
            _running_count += 1
            self._running = True
            self._cancelled = False

            try:
                async for progress, result, stats in self._run_evaluation_iter(
                    annotations, run_name, rag_interfaces
                ):
                    yield progress, result, stats
            finally:
                _running_count -= 1
                self._running = False

    async def _run_evaluation(
        self,
        annotations: list[Annotation],
        run_name: str,
        rag_interfaces: Optional[list[str]],
    ) -> EvaluationRun:
        """
        Internal method to run evaluation.
        Separated from run() to allow proper semaphore handling.
        """

        rag_interfaces = rag_interfaces or list(self._rag_adapters.keys()) or ["default"]

        run = EvaluationRun(
            name=run_name,
            rag_interfaces=rag_interfaces,
            selected_metrics=[m.name for m in self._metrics],
            concurrent_workers=self.max_concurrent,
            total_annotations=len(annotations),
            status="running",
        )

        self._progress = EvaluationProgress(
            total=len(annotations) * len(rag_interfaces),
            start_time=time.time(),
        )

        # Initialize process pool for CPU-intensive work
        self._process_pool = ProcessPoolExecutor(max_workers=4)

        # Get LLM evaluator
        llm_evaluator = None
        if any(m.requires_llm for m in self._metrics):
            llm_evaluator = await get_llm_evaluator()

        try:
            # Save initial run
            result_manager = await get_result_manager()
            await result_manager.save_run(run)

            # Run evaluation with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)

            tasks = []
            for annotation in annotations:
                for rag_interface in rag_interfaces:
                    if self._cancelled:
                        break
                    task = self._evaluate_single(
                        annotation=annotation,
                        rag_interface=rag_interface,
                        run_id=run.id,
                        llm_evaluator=llm_evaluator,
                        semaphore=semaphore,
                    )
                    tasks.append(task)

                if self._cancelled:
                    break

            # Execute tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Evaluation task failed: {result}")
                    run.failed_count += 1
                elif isinstance(result, EvaluationResult):
                    run.add_result(result)

            # Finish run
            run.finish()
            await result_manager.save_run(run)

            logger.info(
                f"Evaluation run completed: {run.completed_count}/{run.total_annotations} "
                f"in {run.duration_seconds:.1f}s"
            )

        except Exception as e:
            run.status = "failed"
            logger.error(f"Evaluation run failed: {e}")
            raise EvaluationError(f"Evaluation run failed: {e}")

        finally:
            if self._process_pool:
                self._process_pool.shutdown(wait=False)
                self._process_pool = None

        return run

    async def _run_evaluation_iter(
        self,
        annotations: list[Annotation],
        run_name: str,
        rag_interfaces: Optional[list[str]],
    ) -> AsyncIterator[tuple[EvaluationProgress, Optional[EvaluationResult], dict]]:
        """
        Internal generator method to run evaluation with progress updates.
        Uses asyncio.as_completed to yield results as they complete.
        Yields (progress, result, stats) after each evaluation completes.
        """
        rag_interfaces = rag_interfaces or list(self._rag_adapters.keys()) or ["default"]

        run = EvaluationRun(
            name=run_name,
            rag_interfaces=rag_interfaces,
            selected_metrics=[m.name for m in self._metrics],
            concurrent_workers=self.max_concurrent,
            total_annotations=len(annotations),
            status="running",
        )

        self._progress = EvaluationProgress(
            total=len(annotations) * len(rag_interfaces),
            start_time=time.time(),
        )

        # Initialize process pool for CPU-intensive work
        self._process_pool = ProcessPoolExecutor(max_workers=4)

        # Get LLM evaluator
        llm_evaluator = None
        if any(m.requires_llm for m in self._metrics):
            llm_evaluator = await get_llm_evaluator()

        try:
            # Save initial run
            result_manager = await get_result_manager()
            await result_manager.save_run(run)

            # Run evaluation with semaphore for concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)

            # Build task map for as_completed
            task_to_info: dict = {}
            for annotation in annotations:
                for rag_interface in rag_interfaces:
                    if self._cancelled:
                        break
                    task = asyncio.create_task(
                        self._evaluate_single(
                            annotation=annotation,
                            rag_interface=rag_interface,
                            run_id=run.id,
                            llm_evaluator=llm_evaluator,
                            semaphore=semaphore,
                        )
                    )
                    task_to_info[task] = (annotation, rag_interface)

                if self._cancelled:
                    break

            # Process tasks as they complete - enables real-time progress
            for coro in asyncio.as_completed(task_to_info.keys()):
                if self._cancelled:
                    break

                try:
                    result = await coro
                    run.add_result(result)

                    # Yield progress update with stats
                    yield self._progress, result, self._get_current_stats(run)

                except Exception as e:
                    logger.error(f"Evaluation task failed: {e}")
                    run.failed_count += 1
                    yield self._progress, None, self._get_current_stats(run)

            # Finish run
            run.finish()
            await result_manager.save_run(run)

            logger.info(
                f"Evaluation run completed: {run.completed_count}/{run.total_annotations} "
                f"in {run.duration_seconds:.1f}s"
            )

            # Final yield to signal completion
            yield self._progress, None, self._get_current_stats(run)

        except Exception as e:
            run.status = "failed"
            logger.error(f"Evaluation run failed: {e}")
            raise EvaluationError(f"Evaluation run failed: {e}")

        finally:
            if self._process_pool:
                self._process_pool.shutdown(wait=False)
                self._process_pool = None

    async def _evaluate_single(
        self,
        annotation: Annotation,
        rag_interface: str,
        run_id: str,
        llm_evaluator: Any,
        semaphore: asyncio.Semaphore,
    ) -> EvaluationResult:
        """Evaluate a single annotation."""
        async with semaphore:
            if self._cancelled:
                return EvaluationResult(
                    annotation_id=annotation.id,
                    run_id=run_id,
                    rag_interface=rag_interface,
                    success=False,
                    error_message="Evaluation cancelled",
                )

            start_time = time.time()
            result = EvaluationResult(
                annotation_id=annotation.id,
                run_id=run_id,
                rag_interface=rag_interface,
                annotation=annotation,
            )

            try:
                # Get RAG response
                adapter = self._rag_adapters.get(rag_interface)
                if adapter is None:
                    # Use mock adapter if not configured
                    adapter = MockRAGAdapter(name=rag_interface)

                rag_response = await adapter.query_from_annotation(annotation)
                result.rag_response = rag_response

                # Calculate metrics
                context = MetricContext(
                    annotation=annotation,
                    rag_response=rag_response,
                    llm_client=llm_evaluator.llm if llm_evaluator else None,
                )

                for metric in self._metrics:
                    if self._cancelled:
                        break
                    try:
                        metric_result = await metric.evaluate(context)
                        result.add_metric(metric_result)
                    except Exception as e:
                        logger.warning(f"Metric {metric.name} failed: {e}")
                        result.add_metric(
                            metric._error_result(str(e))
                        )

                result.success = True

            except Exception as e:
                result.success = False
                result.error_message = str(e)
                logger.error(f"Evaluation failed for annotation {annotation.id}: {e}")

            finally:
                result.duration_ms = (time.time() - start_time) * 1000

                # Update progress
                if self._progress:
                    self._progress.completed += 1
                    self._progress.current_query = annotation.query

                    if self._progress_callback:
                        await self._safe_callback()

            return result

    async def _safe_callback(self) -> None:
        """Safely call progress callback."""
        if self._progress_callback:
            try:
                if asyncio.iscoroutinefunction(self._progress_callback):
                    await self._progress_callback(self._progress)
                else:
                    self._progress_callback(self._progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _get_current_stats(self, run: EvaluationRun) -> dict:
        """获取当前统计数据"""
        stats = {
            "interfaces": {},
        }

        for iface, summary in run.summary_by_interface.items():
            stats["interfaces"][iface] = {
                "average_score": round(summary.get("average_score", 0), 3),
                "success_rate": round(summary.get("success_rate", 0) * 100, 1),
                "total": summary.get("total", 0),
                "successful": summary.get("successful", 0),
            }

        # 计算总体通过率 (指标级别)
        total_metrics = 0
        passed_metrics = 0
        for result in run.results:
            if result.success and result.metrics:
                total_metrics += result.metrics.total_metrics
                passed_metrics += result.metrics.passed_metrics

        if total_metrics > 0:
            stats["overall_pass_rate"] = round(passed_metrics / total_metrics * 100, 1)

        return stats

    def cancel(self) -> None:
        """Cancel the running evaluation."""
        self._cancelled = True
        logger.info("Evaluation cancellation requested")

    def get_progress(self) -> Optional[EvaluationProgress]:
        """Get current progress."""
        return self._progress


async def create_runner(
    max_concurrent: Optional[int] = None,
    timeout: Optional[int] = None,
) -> EvaluationRunner:
    """
    Create an evaluation runner with default configuration.

    Args:
        max_concurrent: Maximum concurrent evaluations
        timeout: Evaluation timeout in seconds

    Returns:
        Configured EvaluationRunner
    """
    config = get_config()

    runner = EvaluationRunner(
        max_concurrent=max_concurrent or config.evaluation.max_concurrent,
        timeout=timeout or config.evaluation.timeout,
    )

    return runner