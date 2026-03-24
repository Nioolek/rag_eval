"""
Scheduler tab component for Gradio UI.
Provides management interface for scheduled evaluation tasks.
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

import gradio as gr

from ...scheduler.models import ScheduledTask, EvaluationTask, TaskStatus
from ...scheduler.task_queue import TaskQueue
from ...scheduler.scheduler import TaskScheduler, create_scheduler
from ...evaluation.metrics.metric_registry import get_registry
from ...models.metric_result import MetricCategory
from ...core.logging import logger


# Global scheduler instance
_scheduler: Optional[TaskScheduler] = None
_task_queue: Optional[TaskQueue] = None


async def get_scheduler() -> TaskScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler, _task_queue

    if _scheduler is None:
        _task_queue = TaskQueue()
        await _task_queue.initialize()
        _scheduler = await create_scheduler(_task_queue)
        await _scheduler.start()

    return _scheduler


async def get_task_queue() -> TaskQueue:
    """Get the global task queue instance."""
    global _task_queue

    if _task_queue is None:
        _task_queue = TaskQueue()
        await _task_queue.initialize()

    return _task_queue


def create_scheduler_tab() -> None:
    """Create the scheduler management tab."""

    # Header
    gr.Markdown("""
    ### ⏰ 定时任务管理
    配置和管理周期性评测任务，支持 Cron 表达式灵活调度
    """)

    with gr.Row():
        # Left column: Task list
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("**📋 定时任务列表**")

                with gr.Row():
                    refresh_btn = gr.Button(
                        "🔄 刷新",
                        variant="secondary",
                        size="sm",
                    )
                    start_scheduler_btn = gr.Button(
                        "▶️ 启动调度器",
                        variant="secondary",
                        size="sm",
                    )
                    stop_scheduler_btn = gr.Button(
                        "⏹️ 停止调度器",
                        variant="secondary",
                        size="sm",
                    )

                scheduler_status = gr.Markdown(
                    "**调度器状态**: 未启动",
                    elem_classes=["status-badge", "info"],
                )

                task_list = gr.Dataframe(
                    headers=["名称", "Cron 表达式", "状态", "上次执行", "下次执行"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    label="定时任务",
                )

                with gr.Row():
                    edit_task_btn = gr.Button(
                        "✏️ 编辑选中",
                        variant="secondary",
                    )
                    delete_task_btn = gr.Button(
                        "🗑️ 删除选中",
                        variant="stop",
                    )

        # Right column: Task editor
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("**➕ 创建/编辑定时任务**")

                task_name = gr.Textbox(
                    label="任务名称",
                    placeholder="例如：每日凌晨评测",
                )

                with gr.Row():
                    cron_minute = gr.Textbox(
                        label="分钟 (0-59)",
                        value="0",
                        scale=1,
                    )
                    cron_hour = gr.Textbox(
                        label="小时 (0-23)",
                        value="2",
                        scale=1,
                    )
                    cron_day = gr.Textbox(
                        label="日期 (1-31)",
                        value="*",
                        scale=1,
                    )
                    cron_month = gr.Textbox(
                        label="月份 (1-12)",
                        value="*",
                        scale=1,
                    )
                    cron_weekday = gr.Textbox(
                        label="星期 (0-6)",
                        value="*",
                        scale=1,
                    )

                cron_preview = gr.Markdown(
                    "**Cron 表达式**: `0 2 * * *` (每天凌晨2点执行)",
                    elem_classes=["info-text"],
                )

                with gr.Row():
                    preset_schedule = gr.Dropdown(
                        label="快速选择",
                        choices=[
                            ("每小时整点", "0 * * * *"),
                            ("每天凌晨2点", "0 2 * * *"),
                            ("每天中午12点", "0 12 * * *"),
                            ("每周一凌晨", "0 2 * * 1"),
                            ("工作日早上9点", "0 9 * * 1-5"),
                            ("每月1号凌晨", "0 0 1 * *"),
                        ],
                        value=None,
                        interactive=True,
                    )

                task_enabled = gr.Checkbox(
                    label="启用任务",
                    value=True,
                )

                # Task configuration
                gr.Markdown("**⚙️ 评测配置**")

                with gr.Row():
                    annotation_source = gr.Dropdown(
                        label="数据来源",
                        choices=[
                            ("全部标注", "all"),
                            ("仅新增", "new_only"),
                        ],
                        value="all",
                    )
                    task_priority = gr.Dropdown(
                        label="优先级",
                        choices=[
                            ("最高 (0)", 0),
                            ("高 (1)", 1),
                            ("普通 (2)", 2),
                            ("低 (3)", 3),
                        ],
                        value=2,
                    )

                # Metrics selection
                registry = get_registry()
                try:
                    available_metrics = (
                        registry.list_by_category(MetricCategory.RETRIEVAL) +
                        registry.list_by_category(MetricCategory.GENERATION) +
                        registry.list_by_category(MetricCategory.FAQ) +
                        registry.list_by_category(MetricCategory.COMPREHENSIVE)
                    )
                except Exception:
                    available_metrics = []

                selected_metrics = gr.CheckboxGroup(
                    choices=available_metrics,
                    value=available_metrics[:5] if available_metrics else [],
                    label="评测指标",
                )

                rag_interface = gr.Textbox(
                    label="RAG 接口名称",
                    placeholder="default（留空使用默认接口）",
                )

                with gr.Row():
                    save_task_btn = gr.Button(
                        "💾 保存任务",
                        variant="primary",
                    )
                    clear_form_btn = gr.Button(
                        "🗑️ 清空表单",
                        variant="secondary",
                    )

                task_status = gr.Markdown("")

    # Execution history
    with gr.Group():
        gr.Markdown("**📜 执行历史**")

        with gr.Row():
            history_filter = gr.Dropdown(
                label="状态筛选",
                choices=["全部", "completed", "failed", "running", "pending"],
                value="全部",
                scale=1,
            )
            refresh_history_btn = gr.Button(
                "🔄 刷新历史",
                variant="secondary",
                scale=1,
            )

        history_table = gr.Dataframe(
            headers=["任务ID", "名称", "状态", "创建时间", "开始时间", "完成时间", "错误"],
            datatype=["str", "str", "str", "str", "str", "str", "str"],
            interactive=False,
            label="执行记录",
        )

    # Quick actions
    with gr.Group():
        gr.Markdown("**⚡ 快捷操作**")

        with gr.Row():
            run_now_btn = gr.Button(
                "🚀 立即执行选中任务",
                variant="primary",
            )
            clear_completed_btn = gr.Button(
                "🧹 清理已完成任务",
                variant="secondary",
            )

        action_status = gr.Markdown("")

    # ===== Event Handlers =====

    # State
    current_task_id = gr.State(None)
    scheduler_running = gr.State(False)

    def update_cron_preview(minute, hour, day, month, weekday):
        """Update cron expression preview."""
        cron = f"{minute} {hour} {day} {month} {weekday}"

        # Parse meaning
        descriptions = []

        if minute == "0" and hour == "*":
            descriptions.append("每小时整点")
        elif minute == "0" and hour != "*":
            descriptions.append(f"每天{hour}点")
        elif minute != "*" and hour != "*":
            descriptions.append(f"每天{hour}:{minute.zfill(2)}")

        if day != "*":
            descriptions.append(f"每月{day}号")

        if weekday != "*":
            weekday_names = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]
            if "-" in weekday:
                start, end = weekday.split("-")
                descriptions.append(f"{weekday_names[int(start)]}至{weekday_names[int(end)]}")
            else:
                descriptions.append(weekday_names[int(weekday)])

        if not descriptions:
            desc = "自定义调度"
        else:
            desc = "，".join(descriptions)

        return f"**Cron 表达式**: `{cron}` ({desc})"

    def apply_preset_schedule(preset):
        """Apply preset schedule values."""
        if not preset:
            return [gr.update() for _ in range(5)]

        parts = preset.split()
        return (
            gr.update(value=parts[0] if len(parts) > 0 else "*"),
            gr.update(value=parts[1] if len(parts) > 1 else "*"),
            gr.update(value=parts[2] if len(parts) > 2 else "*"),
            gr.update(value=parts[3] if len(parts) > 3 else "*"),
            gr.update(value=parts[4] if len(parts) > 4 else "*"),
        )

    async def load_scheduled_tasks():
        """Load all scheduled tasks."""
        queue = await get_task_queue()

        try:
            tasks = await queue.get_enabled_scheduled_tasks()

            # Also get disabled tasks from raw query
            async with queue._lock:
                cursor = await queue._db.execute(
                    "SELECT * FROM scheduled_tasks ORDER BY name"
                )
                rows = await cursor.fetchall()

            data = []
            for row in rows:
                status = "✅ 启用" if row["enabled"] else "⏸️ 禁用"
                last_run = row["last_run_at"]
                next_run = row["next_run_at"]

                if last_run:
                    last_run = datetime.fromisoformat(last_run).strftime("%m-%d %H:%M")
                else:
                    last_run = "从未执行"

                if next_run:
                    next_run = datetime.fromisoformat(next_run).strftime("%m-%d %H:%M")
                else:
                    next_run = "-"

                data.append([
                    row["name"],
                    row["cron_expression"],
                    status,
                    last_run,
                    next_run,
                ])

            return gr.update(value=data)

        except Exception as e:
            logger.error(f"Failed to load scheduled tasks: {e}")
            return gr.update(value=[])

    async def save_scheduled_task(
        name: str,
        minute: str, hour: str, day: str, month: str, weekday: str,
        enabled: bool,
        annotation_src: str,
        priority: int,
        metrics: list,
        rag_iface: str,
        current_id: Optional[str],
    ):
        """Create or update a scheduled task."""
        if not name:
            return "❌ 请输入任务名称"

        # Build cron expression
        cron = f"{minute} {hour} {day} {month} {weekday}"

        # Validate cron expression
        try:
            from apscheduler.triggers.cron import CronTrigger
            CronTrigger.from_crontab(cron)
        except Exception:
            return f"❌ 无效的 Cron 表达式: `{cron}`"

        # Build task config
        task_config = {
            "annotation_source": annotation_src,
            "priority": priority,
            "metrics_config": list(metrics) if metrics else [],
            "rag_interface": rag_iface or None,
        }

        try:
            scheduler = await get_scheduler()

            if current_id:
                # Update existing
                success = await scheduler.update_scheduled_task(
                    scheduled_id=current_id,
                    cron_expression=cron,
                    task_config=task_config,
                    enabled=enabled,
                )
                if success:
                    return f"✅ 任务已更新: {name}"
                else:
                    return "❌ 更新失败，任务不存在"
            else:
                # Create new
                task = await scheduler.create_scheduled_task(
                    name=name,
                    cron_expression=cron,
                    task_config=task_config,
                    enabled=enabled,
                )
                return f"✅ 任务已创建: {task.id}"

        except Exception as e:
            logger.error(f"Failed to save scheduled task: {e}")
            return f"❌ 保存失败: {str(e)}"

    async def delete_selected_task(evt: gr.SelectData):
        """Delete the selected scheduled task."""
        if evt.index is None:
            return "❌ 请先选择要删除的任务"

        queue = await get_task_queue()

        try:
            # Get task at selected index
            tasks = await queue.get_enabled_scheduled_tasks()
            if evt.index >= len(tasks):
                return "❌ 任务不存在"

            task = tasks[evt.index]
            success = await queue.delete_scheduled_task(task.id)

            if success:
                return f"✅ 任务已删除: {task.name}"
            else:
                return "❌ 删除失败"

        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    async def toggle_scheduler(is_running: bool):
        """Start or stop the scheduler."""
        try:
            scheduler = await get_scheduler()

            if not is_running:
                await scheduler.start()
                return (
                    "**调度器状态**: ✅ 运行中",
                    True,
                )
            else:
                await scheduler.stop()
                return (
                    "**调度器状态**: ⏸️ 已停止",
                    False,
                )

        except Exception as e:
            return f"**调度器状态**: ❌ 错误 - {str(e)}", is_running

    async def load_execution_history(status_filter: str):
        """Load execution history."""
        queue = await get_task_queue()

        try:
            if status_filter == "全部":
                status = None
            else:
                status = TaskStatus(status_filter)

            # Get tasks from queue
            async with queue._lock:
                query = """
                    SELECT id, name, status, created_at, started_at, completed_at, error_message
                    FROM evaluation_tasks
                    WHERE is_deleted = 0
                    ORDER BY created_at DESC
                    LIMIT 50
                """
                cursor = await queue._db.execute(query)
                rows = await cursor.fetchall()

            data = []
            for row in rows:
                task_status = row["status"]
                if status_filter != "全部" and task_status != status_filter:
                    continue

                status_icon = {
                    "completed": "✅",
                    "failed": "❌",
                    "running": "🔄",
                    "pending": "⏳",
                    "claimed": "🔒",
                    "cancelled": "🚫",
                }.get(task_status, "❓")

                created = row["created_at"]
                started = row["started_at"]
                completed = row["completed_at"]

                if created:
                    created = datetime.fromisoformat(created).strftime("%m-%d %H:%M")
                if started:
                    started = datetime.fromisoformat(started).strftime("%m-%d %H:%M")
                if completed:
                    completed = datetime.fromisoformat(completed).strftime("%m-%d %H:%M")

                data.append([
                    row["id"][:12] + "..." if row["id"] else "-",
                    row["name"] or "-",
                    f"{status_icon} {task_status}",
                    created or "-",
                    started or "-",
                    completed or "-",
                    row["error_message"][:30] + "..." if row["error_message"] else "-",
                ])

            return gr.update(value=data)

        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return gr.update(value=[])

    async def run_task_now(evt: gr.SelectData):
        """Run a scheduled task immediately."""
        if evt.index is None:
            return "❌ 请先选择要执行的任务"

        queue = await get_task_queue()

        try:
            tasks = await queue.get_enabled_scheduled_tasks()
            if evt.index >= len(tasks):
                return "❌ 任务不存在"

            scheduled = tasks[evt.index]

            # Create immediate task
            task = EvaluationTask(
                name=f"[立即执行] {scheduled.name}",
                priority=scheduled.task_config.get("priority", 2),
                annotation_source=scheduled.task_config.get("annotation_source", "all"),
                metrics_config=scheduled.task_config.get("metrics_config", []),
                rag_interface=scheduled.task_config.get("rag_interface"),
                schedule_id=scheduled.id,
            )

            await queue.add_task(task)
            return f"✅ 任务已加入队列: {task.id}"

        except Exception as e:
            return f"❌ 执行失败: {str(e)}"

    async def clear_completed_tasks():
        """Clear completed tasks from history."""
        queue = await get_task_queue()

        try:
            async with queue._lock:
                cursor = await queue._db.execute(
                    """
                    UPDATE evaluation_tasks
                    SET is_deleted = 1
                    WHERE status IN ('completed', 'failed', 'cancelled')
                    """
                )
                count = cursor.rowcount
                await queue._db.commit()

            return f"✅ 已清理 {count} 条记录"

        except Exception as e:
            return f"❌ 清理失败: {str(e)}"

    def clear_form():
        """Clear the task form."""
        return (
            gr.update(value=""),
            gr.update(value="0"),
            gr.update(value="2"),
            gr.update(value="*"),
            gr.update(value="*"),
            gr.update(value="*"),
            gr.update(value=True),
            gr.update(value="all"),
            gr.update(value=2),
            gr.update(value=[]),
            gr.update(value=""),
            gr.update(value=None),
            "",
        )

    # ===== Connect Events =====

    # Cron preview updates
    for input_comp in [cron_minute, cron_hour, cron_day, cron_month, cron_weekday]:
        input_comp.change(
            fn=update_cron_preview,
            inputs=[cron_minute, cron_hour, cron_day, cron_month, cron_weekday],
            outputs=[cron_preview],
        )

    # Preset schedule
    preset_schedule.change(
        fn=apply_preset_schedule,
        inputs=[preset_schedule],
        outputs=[cron_minute, cron_hour, cron_day, cron_month, cron_weekday],
    )

    # Refresh task list
    refresh_btn.click(
        fn=load_scheduled_tasks,
        outputs=[task_list],
    )

    # Scheduler controls
    start_scheduler_btn.click(
        fn=toggle_scheduler,
        inputs=[scheduler_running],
        outputs=[scheduler_status, scheduler_running],
    )

    async def stop_scheduler_wrapper(r: bool):
        """停止调度器"""
        return await toggle_scheduler(not r)

    stop_scheduler_btn.click(
        fn=stop_scheduler_wrapper,
        inputs=[scheduler_running],
        outputs=[scheduler_status, scheduler_running],
    )

    # Save task
    save_task_btn.click(
        fn=save_scheduled_task,
        inputs=[
            task_name,
            cron_minute, cron_hour, cron_day, cron_month, cron_weekday,
            task_enabled,
            annotation_source,
            task_priority,
            selected_metrics,
            rag_interface,
            current_task_id,
        ],
        outputs=[task_status],
    )

    # Clear form
    clear_form_btn.click(
        fn=clear_form,
        outputs=[
            task_name,
            cron_minute, cron_hour, cron_day, cron_month, cron_weekday,
            task_enabled,
            annotation_source,
            task_priority,
            selected_metrics,
            rag_interface,
            current_task_id,
            task_status,
        ],
    )

    # Delete task
    delete_task_btn.click(
        fn=delete_selected_task,
        outputs=[task_status],
    )

    # Load history
    refresh_history_btn.click(
        fn=load_execution_history,
        inputs=[history_filter],
        outputs=[history_table],
    )

    history_filter.change(
        fn=load_execution_history,
        inputs=[history_filter],
        outputs=[history_table],
    )

    # Quick actions
    run_now_btn.click(
        fn=run_task_now,
        outputs=[action_status],
    )

    clear_completed_btn.click(
        fn=clear_completed_tasks,
        outputs=[action_status],
    )

    # 返回需要初始化加载的组件和函数
    return {
        "task_list": task_list,
        "load_scheduled_tasks": load_scheduled_tasks,
    }