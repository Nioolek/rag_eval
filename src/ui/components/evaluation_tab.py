"""
Evaluation tab component for Gradio UI.
Enhanced with modern styling and improved UX.
"""

import asyncio
from typing import Optional

import gradio as gr

from ...models.annotation import Annotation
from ...evaluation.runner import create_runner, EvaluationRunner
from ...evaluation.metrics.metric_registry import get_registry
from ...models.metric_result import MetricCategory
from ...evaluation.result_manager import get_result_manager
from ...annotation.annotation_handler import get_annotation_handler
from ...rag.mock_adapter import MockRAGAdapter
from ...core.logging import logger


def create_evaluation_tab() -> None:
    """Create the evaluation configuration and execution tab with enhanced styling."""

    # Header
    gr.Markdown("""
    ### 🚀 RAG 评测配置与执行
    配置评测参数、选择指标、执行批量评测任务
    """)

    with gr.Row():
        # Left column: Configuration
        with gr.Column(scale=2):
            # RAG Interface Configuration
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**🔗 RAG 接口配置**")

                with gr.Row():
                    dual_mode = gr.Checkbox(
                        label="⚖️ 双接口对比模式",
                        value=False,
                        info="启用后将使用两个 RAG 接口进行对比评测",
                    )

                with gr.Row():
                    rag_url_1 = gr.Textbox(
                        label="RAG 接口 1 URL",
                        placeholder="http://localhost:8000",
                        value="",
                        scale=1,
                    )
                    rag_url_2 = gr.Textbox(
                        label="RAG 接口 2 URL",
                        placeholder="http://localhost:8001",
                        value="",
                        visible=False,
                        scale=1,
                    )

            # Metrics selection
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**📊 评测指标选择**")

                with gr.Row():
                    select_all_btn = gr.Button(
                        "✅ 全选",
                        variant="secondary",
                        size="sm",
                    )
                    deselect_all_btn = gr.Button(
                        "⬜ 取消全选",
                        variant="secondary",
                        size="sm",
                    )

                # Get available metrics
                registry = get_registry()

                with gr.Row():
                    # Retrieval metrics
                    with gr.Column():
                        gr.Markdown("**🔍 检索指标**")
                        retrieval_metrics = gr.CheckboxGroup(
                            choices=registry.list_by_category(MetricCategory.RETRIEVAL),
                            value=registry.list_by_category(MetricCategory.RETRIEVAL)[:3],
                            label="",
                        )

                    # Generation metrics
                    with gr.Column():
                        gr.Markdown("**✍️ 生成指标**")
                        generation_metrics = gr.CheckboxGroup(
                            choices=registry.list_by_category(MetricCategory.GENERATION),
                            value=[],
                            label="",
                        )

                with gr.Row():
                    # FAQ metrics
                    with gr.Column():
                        gr.Markdown("**🎯 FAQ 指标**")
                        faq_metrics = gr.CheckboxGroup(
                            choices=registry.list_by_category(MetricCategory.FAQ),
                            value=[],
                            label="",
                        )

                    # Comprehensive metrics
                    with gr.Column():
                        gr.Markdown("**📈 综合指标**")
                        comprehensive_metrics = gr.CheckboxGroup(
                            choices=registry.list_by_category(MetricCategory.COMPREHENSIVE),
                            value=[],
                            label="",
                        )

            # Concurrency settings
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**⚙️ 执行配置**")

                with gr.Row():
                    concurrency = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="🔢 并发数",
                        scale=1,
                    )
                    eval_timeout = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=10,
                        label="⏱️ 超时时间 (秒)",
                        scale=1,
                    )

                run_name = gr.Textbox(
                    label="📝 评测名称",
                    placeholder="输入评测名称（可选）",
                )

        # Right column: Execution
        with gr.Column(scale=2):
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**▶️ 评测执行**")

                with gr.Row():
                    start_btn = gr.Button(
                        "🚀 开始评测",
                        variant="primary",
                        size="lg",
                        scale=2,
                    )
                    cancel_btn = gr.Button(
                        "⏹ 取消",
                        variant="stop",
                        size="lg",
                        visible=False,
                        scale=1,
                    )

                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="📊 进度",
                    interactive=False,
                )

                progress_text = gr.Markdown(
                    "🟢 准备就绪",
                    elem_classes=["progress-text"],
                )

                status_display = gr.JSON(
                    label="评测状态",
                    value={"status": "ready"},
                )

    # Results preview
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**📋 最近评测结果**")
        results_table = gr.Dataframe(
            headers=["运行 ID", "名称", "状态", "完成/总数", "耗时"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            label="评测历史记录",
        )

    # ===== Event Handlers =====

    def toggle_dual_mode(is_dual):
        """Toggle dual mode visibility."""
        # In Gradio 6.x, need to explicitly set interactive when making visible
        return gr.update(visible=is_dual, interactive=True)

    def select_all_metrics():
        """Select all metrics."""
        retrieval = registry.list_by_category(MetricCategory.RETRIEVAL)
        generation = registry.list_by_category(MetricCategory.GENERATION)
        faq = registry.list_by_category(MetricCategory.FAQ)
        comp = registry.list_by_category(MetricCategory.COMPREHENSIVE)
        return (
            gr.update(value=retrieval),
            gr.update(value=generation),
            gr.update(value=faq),
            gr.update(value=comp),
        )

    def deselect_all_metrics():
        """Deselect all metrics."""
        return (
            gr.update(value=[]),
            gr.update(value=[]),
            gr.update(value=[]),
            gr.update(value=[]),
        )

    async def start_evaluation(
        dual, url1, url2,
        ret_m, gen_m, faq_m, comp_m,
        conc, timeout, name
    ):
        """Start evaluation process."""
        # Combine selected metrics
        selected_metrics = list(ret_m) + list(gen_m) + list(faq_m) + list(comp_m)

        if not selected_metrics:
            yield (
                gr.update(value=0),
                "❌ 请至少选择一个评测指标",
                {},
                gr.update(visible=True),
                gr.update(visible=False),
            )
            return

        # Create runner
        runner = await create_runner(
            max_concurrent=int(conc),
            timeout=int(timeout),
        )

        # Set metrics
        runner.set_metrics(selected_metrics)

        # Set RAG adapters
        interfaces = ["default"]
        if url1:
            from ...rag.langgraph_adapter import LangGraphAdapter
            from ...rag.base_adapter import RAGAdapterConfig
            config = RAGAdapterConfig(service_url=url1, timeout=int(timeout))
            adapter1 = LangGraphAdapter(config)
            await adapter1.initialize()
            runner.set_rag_adapter(adapter1, "default")
        else:
            # Use mock adapter
            runner.set_rag_adapter(MockRAGAdapter(), "default")

        if dual and url2:
            from ...rag.langgraph_adapter import LangGraphAdapter
            from ...rag.base_adapter import RAGAdapterConfig
            config = RAGAdapterConfig(service_url=url2, timeout=int(timeout))
            adapter2 = LangGraphAdapter(config)
            await adapter2.initialize()
            runner.set_rag_adapter(adapter2, "interface_2")
            interfaces = ["default", "interface_2"]

        # Get annotations
        handler = await get_annotation_handler()
        ann_list = await handler.list(page=1, page_size=1000)
        annotations = ann_list.items

        if not annotations:
            yield (
                gr.update(value=0),
                "❌ 没有可评测的标注数据",
                {},
                gr.update(visible=True),
                gr.update(visible=False),
            )
            return

        # Progress callback
        def on_progress(progress):
            return (
                gr.update(value=progress.progress_percent),
                f"正在评测：{progress.current_query[:50]}... ({progress.completed}/{progress.total})",
            )

        runner.set_progress_callback(lambda p: None)

        # Run evaluation
        yield (
            gr.update(value=0),
            "🔄 开始评测...",
            {"status": "running", "total": len(annotations)},
            gr.update(visible=False),
            gr.update(visible=True),
        )

        try:
            run = await runner.run(
                annotations=annotations,
                run_name=name,
                rag_interfaces=interfaces,
            )

            status = {
                "status": run.status,
                "completed": run.completed_count,
                "failed": run.failed_count,
                "duration": f"{run.duration_seconds:.1f}s",
                "average_score": sum(
                    r.metrics.average_score for r in run.results if r.success
                ) / len([r for r in run.results if r.success]) if any(r.success for r in run.results) else 0,
            }

            yield (
                gr.update(value=100),
                f"✅ 评测完成：{run.completed_count}/{run.total_annotations}",
                status,
                gr.update(visible=True),
                gr.update(visible=False),
            )

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            yield (
                gr.update(value=0),
                f"❌ 评测失败：{str(e)}",
                {"status": "failed", "error": str(e)},
                gr.update(visible=True),
                gr.update(visible=False),
            )

    async def load_recent_results():
        """Load recent evaluation results."""
        manager = await get_result_manager()
        runs = await manager.list_runs(limit=10)

        data = []
        for run in runs:
            data.append([
                run.id[:8] + "...",
                run.name or "未命名",
                run.status,
                f"{run.completed_count}/{run.total_annotations}",
                f"{run.duration_seconds:.1f}s",
            ])

        return gr.update(value=data)

    def cancel_evaluation():
        """Cancel current evaluation."""
        return "正在取消..."

    # ===== Connect Events =====

    dual_mode.change(
        fn=toggle_dual_mode,
        inputs=[dual_mode],
        outputs=[rag_url_2],
        show_progress='hidden',
    )

    select_all_btn.click(
        fn=select_all_metrics,
        outputs=[retrieval_metrics, generation_metrics, faq_metrics, comprehensive_metrics],
    )

    deselect_all_btn.click(
        fn=deselect_all_metrics,
        outputs=[retrieval_metrics, generation_metrics, faq_metrics, comprehensive_metrics],
    )

    start_btn.click(
        fn=start_evaluation,
        inputs=[
            dual_mode, rag_url_1, rag_url_2,
            retrieval_metrics, generation_metrics, faq_metrics, comprehensive_metrics,
            concurrency, eval_timeout, run_name,
        ],
        outputs=[progress_bar, progress_text, status_display, start_btn, cancel_btn],
    )

    cancel_btn.click(
        fn=cancel_evaluation,
        outputs=[progress_text],
    )

    # 返回需要初始化加载的组件和函数
    return {
        "results_table": results_table,
        "load_recent_results": load_recent_results,
    }