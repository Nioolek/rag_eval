"""
Scheduler data models.
Defines TaskStatus enum and task-related data classes.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """
    Task status enumeration.
    Represents the lifecycle of an evaluation task.
    """
    PENDING = "pending"           # Waiting to be executed
    CLAIMED = "claimed"           # Claimed by a worker
    RUNNING = "running"           # Currently executing
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Execution failed
    CANCELLED = "cancelled"       # Cancelled by user


class EvaluationTask(BaseModel):
    """
    Evaluation task model.
    Represents a single evaluation job in the queue.
    """
    # Identity
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    name: str = Field(..., description="Task name")
    
    # Status tracking
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    priority: int = Field(default=2, ge=0, le=3, description="Priority 0-3 (0=highest)")
    
    # Task configuration
    annotation_source: str = Field(default="all", description="all/new_only/custom")
    annotation_filter: Optional[dict[str, Any]] = Field(default=None, description="Filter conditions")
    metrics_config: list[str] = Field(default_factory=list, description="Selected metrics")
    rag_interface: Optional[str] = Field(default=None, description="RAG interface name")
    
    # Worker assignment (for row-level locking)
    claimed_by: Optional[str] = Field(default=None, description="Worker ID that claimed this task")
    claimed_at: Optional[datetime] = Field(default=None)
    heartbeat_at: Optional[datetime] = Field(default=None, description="Last heartbeat timestamp")
    
    # Scheduling reference
    schedule_id: Optional[str] = Field(default=None, description="Reference to scheduled task if triggered by schedule")
    
    # Execution results
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    is_deleted: bool = Field(default=False)
    
    # Computed properties
    @property
    def is_claimable(self) -> bool:
        """Check if task can be claimed."""
        return self.status == TaskStatus.PENDING
    
    @property
    def is_running(self) -> bool:
        """Check if task is currently running."""
        return self.status == TaskStatus.RUNNING
    
    @property
    def is_finished(self) -> bool:
        """Check if task is finished (completed/failed/cancelled)."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
    
    def claim(self, worker_id: str) -> None:
        """Claim this task for a worker."""
        self.status = TaskStatus.CLAIMED
        self.claimed_by = worker_id
        self.claimed_at = datetime.now()
        self.heartbeat_at = datetime.now()
        self.updated_at = datetime.now()
    
    def start(self) -> None:
        """Mark task as running."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
        self.updated_at = datetime.now()
    
    def complete(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def cancel(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
        self.updated_at = datetime.now()
    
    def heartbeat(self) -> None:
        """Update heartbeat timestamp."""
        self.heartbeat_at = datetime.now()
        self.updated_at = datetime.now()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority,
            "annotation_source": self.annotation_source,
            "annotation_filter": self.annotation_filter,
            "metrics_config": self.metrics_config,
            "rag_interface": self.rag_interface,
            "claimed_by": self.claimed_by,
            "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "schedule_id": self.schedule_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_deleted": self.is_deleted,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationTask":
        """Create from dictionary."""
        # Handle datetime conversion
        for field in ["claimed_at", "heartbeat_at", "started_at", "completed_at", "created_at", "updated_at"]:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        # Handle status conversion
        if "status" in data and isinstance(data["status"], str):
            data["status"] = TaskStatus(data["status"])
        
        return cls(**data)


class ScheduledTask(BaseModel):
    """
    Scheduled task model.
    Represents a recurring evaluation task with cron-based scheduling.
    """
    # Identity
    id: str = Field(default_factory=lambda: datetime.now().strftime("sched_%Y%m%d%H%M%S%f"))
    name: str = Field(..., description="Schedule name")
    
    # Schedule configuration
    cron_expression: str = Field(..., description="Cron expression (e.g., '0 2 * * *')")
    timezone: str = Field(default="Asia/Shanghai")
    enabled: bool = Field(default=True)
    
    # Task configuration
    task_config: dict[str, Any] = Field(default_factory=dict, description="Task configuration")
    
    # Schedule status
    last_run_at: Optional[datetime] = Field(default=None)
    last_run_status: Optional[str] = Field(default=None)
    last_run_duration_ms: Optional[int] = Field(default=None)
    next_run_at: Optional[datetime] = Field(default=None)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "name": self.name,
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "enabled": self.enabled,
            "task_config": self.task_config,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_run_status": self.last_run_status,
            "last_run_duration_ms": self.last_run_duration_ms,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        """Create from dictionary."""
        # Handle datetime conversion
        for field in ["last_run_at", "next_run_at", "created_at", "updated_at"]:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)
    
    def update_last_run(self, status: str, duration_ms: int) -> None:
        """Update last run information."""
        self.last_run_at = datetime.now()
        self.last_run_status = status
        self.last_run_duration_ms = duration_ms
        self.updated_at = datetime.now()


class PerformanceBenchmark(BaseModel):
    """
    Performance benchmark result model.
    Stores performance test results for tracking and comparison.
    """
    # Identity
    id: str = Field(default_factory=lambda: datetime.now().strftime("bench_%Y%m%d%H%M%S%f"))
    test_name: str = Field(..., description="Test name")
    test_type: str = Field(..., description="benchmark/stress/load")
    version: str = Field(..., description="System version")
    
    # Performance metrics
    avg_response_time_ms: float = Field(default=0.0)
    p95_response_time_ms: float = Field(default=0.0)
    p99_response_time_ms: float = Field(default=0.0)
    throughput_qps: float = Field(default=0.0)
    error_rate: float = Field(default=0.0)
    memory_peak_mb: float = Field(default=0.0)
    cpu_peak_percent: float = Field(default=0.0)
    
    # Environment info
    environment_config: dict[str, Any] = Field(default_factory=dict)
    hardware_info: dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = Field(default=None)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "test_name": self.test_name,
            "test_type": self.test_type,
            "version": self.version,
            "avg_response_time_ms": self.avg_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "throughput_qps": self.throughput_qps,
            "error_rate": self.error_rate,
            "memory_peak_mb": self.memory_peak_mb,
            "cpu_peak_percent": self.cpu_peak_percent,
            "environment_config": self.environment_config,
            "hardware_info": self.hardware_info,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceBenchmark":
        """Create from dictionary."""
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return cls(**data)
