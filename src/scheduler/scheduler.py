"""
Task scheduler with APScheduler integration.
Provides cron-based scheduling for evaluation tasks.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger

from .models import EvaluationTask, ScheduledTask, TaskStatus
from .task_queue import TaskQueue
from ..core.logging import logger
from ..core.exceptions import SchedulerError


class TaskScheduler:
    """
    Task scheduler using APScheduler.
    
    Features:
    - Cron-based scheduling
    - SQLite job persistence
    - Automatic task creation from schedules
    - Timezone support
    """
    
    def __init__(
        self,
        task_queue: TaskQueue,
        db_path: str = "./data/scheduler.db",
        timezone: str = "Asia/Shanghai",
    ):
        self.task_queue = task_queue
        self.db_path = db_path
        self.timezone = timezone
        
        # Configure APScheduler with memory job store
        # Note: Using MemoryJobStore for simplicity. For persistence,
        # install SQLAlchemy and use SQLAlchemyJobStore with SQLite URL
        jobstores = {
            "default": MemoryJobStore()
        }
        
        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            timezone=timezone,
            job_defaults={
                "misfire_grace_time": 3600,  # 1 hour grace for missed executions
                "coalesce": True,  # Coalesce missed executions
                "max_instances": 1,  # Only one instance per job
            }
        )
        
        self._running = False
    
    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            raise SchedulerError("Scheduler already running")
        
        self._scheduler.start()
        self._running = True
        
        # Load scheduled tasks from database and register jobs
        await self._load_scheduled_tasks()
        
        logger.info("Task scheduler started")
    
    async def stop(self, wait: bool = True) -> None:
        """Stop the scheduler."""
        if not self._running:
            return
        
        self._scheduler.shutdown(wait=wait)
        self._running = False
        
        logger.info("Task scheduler stopped")
    
    async def _load_scheduled_tasks(self) -> None:
        """Load scheduled tasks from database and register with APScheduler."""
        scheduled_tasks = await self.task_queue.get_enabled_scheduled_tasks()
        
        for scheduled in scheduled_tasks:
            try:
                await self._register_job(scheduled)
                logger.info(f"Registered scheduled task: {scheduled.name}")
            except Exception as e:
                logger.error(f"Failed to register scheduled task {scheduled.id}: {e}")
    
    async def _register_job(self, scheduled: ScheduledTask) -> None:
        """Register a scheduled task with APScheduler."""
        # Parse cron expression
        parts = scheduled.cron_expression.split()
        
        if len(parts) == 5:
            minute, hour, day, month, day_of_week = parts
        elif len(parts) == 6:
            minute, hour, day, month, day_of_week, year = parts
        else:
            raise SchedulerError(f"Invalid cron expression: {scheduled.cron_expression}")
        
        # Create cron trigger
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week,
            timezone=self.timezone,
        )
        
        # Create job
        self._scheduler.add_job(
            self._on_scheduled_execution,
            trigger=trigger,
            id=f"sched_{scheduled.id}",
            name=scheduled.name,
            args=[scheduled.id],
            replace_existing=True,
        )
    
    async def _on_scheduled_execution(self, scheduled_id: str) -> None:
        """Callback when a scheduled task is triggered."""
        logger.info(f"Scheduled task triggered: {scheduled_id}")
        
        start_time = time.time()
        
        try:
            # Get scheduled task config
            scheduled = await self.task_queue.get_scheduled_task(scheduled_id)
            
            if not scheduled:
                logger.error(f"Scheduled task not found: {scheduled_id}")
                return
            
            if not scheduled.enabled:
                logger.debug(f"Scheduled task disabled: {scheduled_id}")
                return
            
            # Create evaluation task from schedule
            task_config = scheduled.task_config
            
            task = EvaluationTask(
                name=f"[Scheduled] {scheduled.name}",
                priority=task_config.get("priority", 2),
                annotation_source=task_config.get("annotation_source", "all"),
                annotation_filter=task_config.get("annotation_filter"),
                metrics_config=task_config.get("metrics_config", []),
                rag_interface=task_config.get("rag_interface"),
                schedule_id=scheduled.id,
            )
            
            # Add task to queue
            await self.task_queue.add_task(task)
            
            # Update scheduled task run info
            duration_ms = int((time.time() - start_time) * 1000)
            await self.task_queue.update_scheduled_task_run(
                scheduled_id,
                status="success",
                duration_ms=duration_ms,
            )
            
            logger.info(f"Scheduled task executed successfully: {scheduled_id}")
            
        except Exception as e:
            logger.error(f"Scheduled task execution failed: {scheduled_id}: {e}")
            
            # Update scheduled task with failure
            duration_ms = int((time.time() - start_time) * 1000)
            await self.task_queue.update_scheduled_task_run(
                scheduled_id,
                status="failed",
                duration_ms=duration_ms,
            )
    
    async def create_scheduled_task(
        self,
        name: str,
        cron_expression: str,
        task_config: dict[str, Any],
        timezone: Optional[str] = None,
        enabled: bool = True,
    ) -> ScheduledTask:
        """
        Create a new scheduled task.
        
        Args:
            name: Schedule name
            cron_expression: Cron expression (e.g., "0 2 * * *" for daily at 2 AM)
            task_config: Task configuration including:
                - priority: Task priority (0-3)
                - annotation_source: "all" / "new_only" / "custom"
                - annotation_filter: Filter conditions
                - metrics_config: List of metric names
                - rag_interface: RAG interface name
            timezone: Timezone (default: scheduler timezone)
            enabled: Whether to enable immediately
        
        Returns:
            Created ScheduledTask
        """
        scheduled = ScheduledTask(
            name=name,
            cron_expression=cron_expression,
            timezone=timezone or self.timezone,
            enabled=enabled,
            task_config=task_config,
        )
        
        # Save to database
        await self.task_queue.add_scheduled_task(scheduled)
        
        # Register with scheduler if running
        if self._running and enabled:
            await self._register_job(scheduled)
        
        logger.info(f"Created scheduled task: {scheduled.id}")
        return scheduled
    
    async def update_scheduled_task(
        self,
        scheduled_id: str,
        cron_expression: Optional[str] = None,
        task_config: Optional[dict[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """
        Update an existing scheduled task.
        
        Returns:
            True if updated successfully
        """
        scheduled = await self.task_queue.get_scheduled_task(scheduled_id)
        
        if not scheduled:
            return False
        
        # Update fields
        if cron_expression:
            scheduled.cron_expression = cron_expression
        
        if task_config:
            scheduled.task_config.update(task_config)
        
        if enabled is not None:
            scheduled.enabled = enabled
        
        # Remove existing job if exists
        job_id = f"sched_{scheduled_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        
        # Re-register if enabled and scheduler is running
        if self._running and scheduled.enabled:
            await self._register_job(scheduled)
        
        # Update in database
        await self.task_queue.add_scheduled_task(scheduled)
        
        logger.info(f"Updated scheduled task: {scheduled_id}")
        return True
    
    async def delete_scheduled_task(self, scheduled_id: str) -> bool:
        """Delete a scheduled task."""
        # Remove from APScheduler
        job_id = f"sched_{scheduled_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
        
        # Delete from database
        return await self.task_queue.delete_scheduled_task(scheduled_id)
    
    async def enable_scheduled_task(self, scheduled_id: str) -> bool:
        """Enable a scheduled task."""
        return await self.update_scheduled_task(scheduled_id, enabled=True)
    
    async def disable_scheduled_task(self, scheduled_id: str) -> bool:
        """Disable a scheduled task."""
        return await self.update_scheduled_task(scheduled_id, enabled=False)
    
    def get_next_run_time(self, scheduled_id: str) -> Optional[datetime]:
        """Get next run time for a scheduled task."""
        job_id = f"sched_{scheduled_id}"
        job = self._scheduler.get_job(job_id)
        
        if job:
            return job.next_run_time
        return None
    
    def list_jobs(self) -> list[dict[str, Any]]:
        """List all scheduled jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time,
                "trigger": str(job.trigger),
            })
        return jobs


class TaskWorker:
    """
    Task worker that processes tasks from the queue.
    
    Features:
    - Concurrent task execution
    - Heartbeat updates
    - Graceful shutdown
    """
    
    def __init__(
        self,
        task_queue: TaskQueue,
        max_concurrent: int = 3,
        heartbeat_interval: int = 30,
    ):
        self.task_queue = task_queue
        self.max_concurrent = max_concurrent
        self.heartbeat_interval = heartbeat_interval
        
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def start(self) -> None:
        """Start the worker."""
        if self._running:
            raise SchedulerError("Worker already running")
        
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            task = asyncio.create_task(self._worker_loop(i))
            self._tasks.append(task)
        
        logger.info(f"Task worker started with {self.max_concurrent} workers")
    
    async def stop(self, wait: bool = True) -> None:
        """Stop the worker."""
        self._running = False
        
        if wait:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        else:
            for task in self._tasks:
                task.cancel()
        
        self._tasks.clear()
        logger.info("Task worker stopped")
    
    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop."""
        logger.debug(f"Worker {worker_id} started")
        
        while self._running:
            try:
                # Try to claim a task
                task = await self.task_queue.get_next_pending_task()
                
                if not task:
                    # No pending tasks, wait
                    await asyncio.sleep(1)
                    continue
                
                # Try to claim the task
                claimed = await self.task_queue.claim_task(task.id)
                
                if not claimed:
                    # Task was claimed by another worker
                    continue
                
                # Process the task
                async with self._semaphore:
                    await self._process_task(claimed, worker_id)
                    
            except asyncio.CancelledError:
                logger.debug(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: EvaluationTask, worker_id: int) -> None:
        """Process a single task."""
        logger.info(f"Worker {worker_id} processing task {task.id}: {task.name}")
        
        # Mark as running
        await self.task_queue.start_task(task.id)
        task.start()
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(task.id)
        )
        
        try:
            # TODO: Actually execute the evaluation
            # For now, just simulate work
            await asyncio.sleep(1)
            
            # Mark as completed
            await self.task_queue.complete_task(task.id)
            task.complete()
            
            logger.info(f"Task {task.id} completed by worker {worker_id}")
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            await self.task_queue.fail_task(task.id, str(e))
            task.fail(str(e))
            
        finally:
            # Cancel heartbeat
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    async def _heartbeat_loop(self, task_id: str) -> None:
        """Send periodic heartbeats for a task."""
        try:
            while self._running:
                await asyncio.sleep(self.heartbeat_interval)
                await self.task_queue.heartbeat(task_id)
                logger.debug(f"Heartbeat sent for task {task_id}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat failed for task {task_id}: {e}")


async def create_scheduler(
    task_queue: TaskQueue,
    db_path: str = "./data/scheduler.db",
    timezone: str = "Asia/Shanghai",
) -> TaskScheduler:
    """Create a task scheduler instance."""
    scheduler = TaskScheduler(
        task_queue=task_queue,
        db_path=db_path,
        timezone=timezone,
    )
    return scheduler


async def create_worker(
    task_queue: TaskQueue,
    max_concurrent: int = 3,
    heartbeat_interval: int = 30,
) -> TaskWorker:
    """Create a task worker instance."""
    worker = TaskWorker(
        task_queue=task_queue,
        max_concurrent=max_concurrent,
        heartbeat_interval=heartbeat_interval,
    )
    return worker
