"""
Single query run tab component for Gradio UI.
Supports running a single query with real-time streaming display.
"""

import asyncio
import time
from typing import Any, Optional

import gradio as gr

from ...models.annotation import Annotation
from ...models.rag_response import RAGResponse, StageTiming
from ...models.metric_result import MetricCategory
from ...evaluation.metrics.metric_registry import get_registry
from ...evaluation.metrics.base import MetricContext
from ...annotation.annotation_handler import get_annotation_handler
from ...rag.mock_adapter import MockRAGAdapter
from ...rag.langgraph_adapter import LangGraphAdapter
from ...rag.base_adapter import RAGAdapterConfig, StreamingChunk
from ...core.config import get_config
from ...core.logging import logger


# Stage name mappings for display
STAGE_NAMES = {
    "start": "🚀 开始",
    "query_rewrite": "📝 查询改写",
    "faq_match": "🎯 FAQ匹配",
    "retrieval": "📚 检索",
    "rerank": "🔄 重排序",
    "thinking": "💭 思考过程",
    "generation": "✍️ 生成答案",
    "final": "✅ 最终结果",
    "error": "❌ 错误",
}

# Timing stage name mappings
TIMING_STAGE_NAMES = {
    "query_rewrite_ms": "查询改写",
    "faq_match_ms": "FAQ匹配",
    "retrieval_ms": "检索",
    "rerank_ms": "重排序",
    "generation_ms": "生成",
}


def create_single_run_tab() -> dict:
    """Create the single query run tab with streaming display."""

    # Header
    gr.Markdown("""
    ### 🔬 单题运行
    实时运行单个查询，查看RAG处理过程和详细结果
    """)

    # ===== Input Section =====
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**📝 输入设置**")

        input_mode = gr.Radio(
            choices=["自定义输入", "测试集选择"],
            value="自定义输入",
            label="输入方式",
        )

        # Custom input mode (default visible)
        with gr.Group(visible=True) as custom_input_group:
            query_input = gr.Textbox(
                label="查询内容",
                lines=3,
                placeholder="请输入您的问题...",
            )
            with gr.Accordion("高级选项", open=False):
                conversation_history = gr.Dataframe(
                    headers=["角色", "内容"],
                    label="对话历史（可选）",
                    datatype=["str", "str"],
                    row_count=(1, 5),
                    col_count=(2, "fixed"),
                    interactive=True,
                )
                enable_thinking = gr.Checkbox(
                    label="启用思考模式",
                    value=False,
                    info="开启后将展示LLM的思考过程",
                )

        # Test set selection mode (initially hidden)
        with gr.Group(visible=False) as test_set_group:
            with gr.Row():
                annotation_selector = gr.Dropdown(
                    label="选择测试数据",
                    choices=[],
                    interactive=True,
                    scale=3,
                )
                refresh_annotations_btn = gr.Button(
                    "🔄 刷新列表",
                    variant="secondary",
                    size="sm",
                    scale=1,
                )

            with gr.Row():
                gt_documents_display = gr.JSON(
                    label="GT文档",
                    visible=False,
                )
                standard_answers_display = gr.JSON(
                    label="标准答案",
                    visible=False,
                )

            # Metric selection for test set mode
            gr.Markdown("**勾选要计算的指标**")
            registry = get_registry()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**🔍 检索指标**")
                    metric_retrieval = gr.CheckboxGroup(
                        choices=[
                            ("检索精确率", "retrieval_precision"),
                            ("检索召回率", "retrieval_recall"),
                            ("Hit Rate", "hit_rate"),
                            ("MRR", "mrr"),
                        ],
                        value=["retrieval_precision", "hit_rate"],
                        label="",
                    )

                with gr.Column():
                    gr.Markdown("**✍️ 生成指标**")
                    metric_generation = gr.CheckboxGroup(
                        choices=[
                            ("答案相关性", "answer_relevance"),
                            ("事实一致性", "factual_consistency"),
                            ("答案完整性", "answer_completeness"),
                            ("拒答准确性", "refusal_accuracy"),
                        ],
                        value=["answer_relevance", "refusal_accuracy"],
                        label="",
                    )

    # ===== RAG Interface Selection =====
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**🔧 RAG接口设置**")

        run_mode = gr.Radio(
            choices=["单接口运行", "双接口对比"],
            value="单接口运行",
            label="运行模式",
        )

        # Single interface mode (default)
        with gr.Group(visible=True) as single_interface_group:
            interface_selector = gr.Dropdown(
                label="RAG接口",
                choices=["LangGraph (配置的服务)", "Mock (模拟)"],
                value="LangGraph (配置的服务)",
                interactive=True,
            )

        # Dual interface mode (initially hidden)
        with gr.Group(visible=False) as dual_interface_group:
            with gr.Row():
                interface_1 = gr.Dropdown(
                    label="接口1",
                    choices=["LangGraph", "Mock"],
                    value="LangGraph",
                    scale=1,
                )
                interface_2 = gr.Dropdown(
                    label="接口2",
                    choices=["LangGraph", "Mock"],
                    value="Mock",
                    scale=1,
                )

    # ===== Run Button =====
    with gr.Row():
        run_btn = gr.Button(
            "▶ 运行",
            variant="primary",
            size="lg",
            scale=2,
        )
        clear_btn = gr.Button(
            "🗑️ 清空",
            variant="secondary",
            size="lg",
            scale=1,
        )

    # ===== Streaming Output Section =====
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**🔄 处理过程**")

        # Single interface output
        with gr.Group(visible=True) as single_output_group:
            streaming_output = gr.Markdown(
                value="点击'运行'开始处理...",
                elem_classes=["streaming-output"],
            )

        # Dual interface output (initially hidden)
        with gr.Group(visible=False) as dual_output_group:
            with gr.Row():
                streaming_output_1 = gr.Markdown(
                    label="接口1处理过程",
                    elem_classes=["streaming-output"],
                )
                streaming_output_2 = gr.Markdown(
                    label="接口2处理过程",
                    elem_classes=["streaming-output"],
                )

    # ===== Results Section =====
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**📊 结果详情**")

        with gr.Tabs():
            with gr.TabItem("最终答案"):
                # Single interface answer
                with gr.Group(visible=True) as single_answer_group:
                    final_answer = gr.Textbox(
                        label="最终答案",
                        lines=5,
                        interactive=False,
                    )

                # Dual interface answers (initially hidden)
                with gr.Group(visible=False) as dual_answer_group:
                    with gr.Row():
                        final_answer_1 = gr.Textbox(
                            label="接口1答案",
                            lines=5,
                            interactive=False,
                        )
                        final_answer_2 = gr.Textbox(
                            label="接口2答案",
                            lines=5,
                            interactive=False,
                        )

            with gr.TabItem("State状态"):
                # Single interface state
                with gr.Group(visible=True) as single_state_group:
                    rag_state_display = gr.JSON(
                        label="RAG State",
                        value={},
                    )

                # Dual interface state (initially hidden)
                with gr.Group(visible=False) as dual_state_group:
                    with gr.Row():
                        rag_state_1 = gr.JSON(
                            label="接口1 State",
                            value={},
                        )
                        rag_state_2 = gr.JSON(
                            label="接口2 State",
                            value={},
                        )

            with gr.TabItem("时间统计"):
                # Single interface timing
                with gr.Group(visible=True) as single_timing_group:
                    timing_display = gr.Dataframe(
                        headers=["阶段", "耗时(ms)", "占比(%)"],
                        label="时间统计",
                        value=[],
                        interactive=False,
                    )

                # Dual interface timing (initially hidden)
                with gr.Group(visible=False) as dual_timing_group:
                    with gr.Row():
                        timing_1 = gr.Dataframe(
                            headers=["阶段", "耗时(ms)", "占比(%)"],
                            label="接口1时间",
                            value=[],
                            interactive=False,
                        )
                        timing_2 = gr.Dataframe(
                            headers=["阶段", "耗时(ms)", "占比(%)"],
                            label="接口2时间",
                            value=[],
                            interactive=False,
                        )

    # ===== Metrics Result Section (test set mode only) =====
    with gr.Group(visible=False, elem_classes=["gr-box"]) as metrics_result_group:
        gr.Markdown("**📈 评测指标**")

        # Single interface metrics
        with gr.Group(visible=True) as single_metrics_group:
            metrics_display = gr.JSON(
                label="指标详情",
                value={},
            )
            metrics_summary = gr.Markdown(
                label="指标摘要",
                value="",
            )

        # Dual interface metrics (initially hidden)
        with gr.Group(visible=False) as dual_metrics_group:
            with gr.Row():
                metrics_display_1 = gr.JSON(
                    label="接口1指标",
                    value={},
                )
                metrics_display_2 = gr.JSON(
                    label="接口2指标",
                    value={},
                )
            metrics_comparison = gr.Markdown(
                label="对比分析",
                value="",
            )

    # ===== Helper Functions =====

    def toggle_input_mode(mode: str):
        """Toggle between custom input and test set mode."""
        is_custom = mode == "自定义输入"
        return (
            gr.update(visible=is_custom),
            gr.update(visible=not is_custom),
            gr.update(visible=not is_custom),  # metrics_result_group
        )

    def toggle_run_mode(mode: str):
        """Toggle between single and dual interface mode."""
        is_single = mode == "单接口运行"
        return (
            gr.update(visible=is_single),   # single_interface_group
            gr.update(visible=not is_single),  # dual_interface_group
            gr.update(visible=is_single),   # single_output_group
            gr.update(visible=not is_single),  # dual_output_group
            gr.update(visible=is_single),   # single_answer_group
            gr.update(visible=not is_single),  # dual_answer_group
            gr.update(visible=is_single),   # single_state_group
            gr.update(visible=not is_single),  # dual_state_group
            gr.update(visible=is_single),   # single_timing_group
            gr.update(visible=not is_single),  # dual_timing_group
            gr.update(visible=is_single),   # single_metrics_group
            gr.update(visible=not is_single),  # dual_metrics_group
        )

    def get_adapter_by_name(name: str):
        """Get RAG adapter by name."""
        if "Mock" in name:
            return MockRAGAdapter(name="mock")
        else:
            config = get_config()
            adapter_config = RAGAdapterConfig(
                service_url=config.rag_service_url,
                timeout=60,
            )
            return LangGraphAdapter(
                config=adapter_config,
                assistant_id="rag_agent"
            )

    def format_streaming_chunk(output: str, chunk: StreamingChunk) -> str:
        """Format streaming chunk for display."""
        stage = chunk.stage
        content = chunk.content
        metadata = chunk.metadata or {}

        stage_name = STAGE_NAMES.get(stage, stage)

        # Add stage header if not already present
        stage_header = f"### {stage_name}\n\n"
        if stage_header not in output and stage not in output:
            output += f"\n{stage_header}"

        # Format content based on stage
        if stage == "query_rewrite":
            output += f"{content}\n"
        elif stage == "faq_match":
            if metadata.get("matched"):
                output += f"✅ 匹配成功: {content}\n"
            else:
                output += f"❌ 未匹配\n"
        elif stage == "retrieval":
            doc_count = metadata.get("count", metadata.get("doc_count", "?"))
            output += f"检索到 {doc_count} 个文档\n"
            if content:
                output += f"\n{content}\n"
        elif stage == "rerank":
            doc_count = metadata.get("count", "?")
            output += f"重排序后保留 {doc_count} 个文档\n"
        elif stage == "thinking":
            output += f"```\n{content}\n```\n"
        elif stage == "generation":
            # For generation, append content progressively
            output += content
        elif stage == "final":
            output += f"\n\n{content}\n\n"
            latency = metadata.get("latency_ms", 0)
            output += f"**耗时**: {latency:.0f}ms\n"
        elif stage == "error":
            output += f"{content}\n"

        return output

    def format_timing(stage_timing: Optional[StageTiming]) -> list[list]:
        """Format timing data for display."""
        if not stage_timing:
            return []

        data = []
        total = stage_timing.total_ms or 1

        for field, name in TIMING_STAGE_NAMES.items():
            ms = getattr(stage_timing, field, 0)
            if ms > 0:
                pct = (ms / total) * 100 if total > 0 else 0
                data.append([name, round(ms, 2), round(pct, 1)])

        # Add total row
        data.append(["**总计**", round(total, 2), 100.0])

        return data

    async def format_metrics_summary(metrics: dict[str, Any]) -> str:
        """Format metrics summary for display."""
        if not metrics:
            return "无指标数据"

        summary = "| 指标 | 得分 | 结果 |\n|------|------|------|\n"
        for key, data in metrics.items():
            if "error" in data:
                summary += f"| {key} | - | ❌ 计算失败 |\n"
            else:
                score = data.get("score", 0)
                passed = data.get("passed", False)
                status = "✅ 通过" if passed else "❌ 未通过"
                summary += f"| {data.get('name', key)} | {score:.2%} | {status} |\n"
        return summary

    async def load_test_set_annotations():
        """Load annotations for test set selection."""
        handler = await get_annotation_handler()
        ann_list = await handler.list(page=1, page_size=100)

        choices = [
            (f"{a.id[:8]} - {a.query[:50]}{'...' if len(a.query) > 50 else ''}", a.id)
            for a in ann_list.items
            if not a.is_deleted
        ]

        return gr.update(choices=choices)

    async def on_annotation_selected(annotation_id: str):
        """Handle annotation selection - display GT info."""
        if not annotation_id:
            return (
                gr.update(visible=False, value={}),
                gr.update(visible=False, value={}),
            )

        handler = await get_annotation_handler()
        annotation = await handler.get(annotation_id)

        if not annotation:
            return (
                gr.update(visible=False, value={}),
                gr.update(visible=False, value={}),
            )

        gt_docs = annotation.gt_documents if annotation.gt_documents else []
        std_answers = annotation.standard_answers if annotation.standard_answers else []

        return (
            gr.update(value=gt_docs, visible=bool(gt_docs)),
            gr.update(value=std_answers, visible=bool(std_answers)),
        )

    async def calculate_selected_metrics(
        annotation: Annotation,
        rag_response: RAGResponse,
        selected_metrics: list[str],
    ) -> dict[str, Any]:
        """Calculate user-selected metrics."""
        results = {}

        # Create metric context
        context = MetricContext(
            annotation=annotation,
            rag_response=rag_response,
        )

        registry = get_registry()

        for metric_key in selected_metrics:
            try:
                metric_class = registry.get(metric_key)
                if metric_class:
                    metric = metric_class()
                    result = await metric.calculate(context)
                    results[metric_key] = {
                        "name": result.metric_name,
                        "score": result.score,
                        "passed": result.passed,
                        "details": result.details,
                    }
            except Exception as e:
                logger.warning(f"Failed to calculate {metric_key}: {e}")
                results[metric_key] = {"error": str(e)}

        return results

    async def run_single_streaming(
        adapter,
        query: str,
        history: list,
        thinking: bool,
    ):
        """Run single interface streaming query."""
        output = "### 🔄 RAG 处理过程\n\n"
        rag_response = None

        # Convert history from dataframe format if needed
        if history and isinstance(history, list):
            if isinstance(history[0], list):
                # Convert from dataframe format [[role, content], ...]
                history = [f"{row[0]}: {row[1]}" for row in history if row[0] and row[1]]

        try:
            async for chunk in adapter.stream_query(
                query=query,
                conversation_history=history,
                enable_thinking=thinking,
            ):
                output = format_streaming_chunk(output, chunk)

                if chunk.is_final and chunk.metadata:
                    # Get the final response
                    latency_ms = chunk.metadata.get("latency_ms", 0)

                    # Build a minimal RAGResponse for timing display
                    rag_response = RAGResponse(
                        query=query,
                        final_answer=chunk.content,
                        latency_ms=latency_ms,
                        stage_timing=StageTiming(total_ms=latency_ms),
                    )

                timing_data = []
                if rag_response and rag_response.stage_timing:
                    timing_data = format_timing(rag_response.stage_timing)

                yield (
                    output,
                    rag_response.final_answer if rag_response else "",
                    rag_response.to_dict() if rag_response else {},
                    timing_data,
                )

            # Initialize adapter if needed
            if hasattr(adapter, '_initialized') and not adapter._initialized:
                await adapter.initialize()

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield (
                f"### ❌ 错误\n\n{str(e)}",
                "",
                {"error": str(e)},
                [],
            )

    async def run_dual_streaming(
        adapter_1,
        adapter_2,
        query: str,
        history: list,
        thinking: bool,
    ):
        """Run dual interface streaming query in parallel."""
        output_1 = "### 🔄 接口1 处理过程\n\n"
        output_2 = "### 🔄 接口2 处理过程\n\n"

        response_1 = None
        response_2 = None

        # Convert history from dataframe format if needed
        if history and isinstance(history, list):
            if isinstance(history[0], list):
                history = [f"{row[0]}: {row[1]}" for row in history if row[0] and row[1]]

        # Initialize adapters if needed
        for adapter in [adapter_1, adapter_2]:
            if hasattr(adapter, '_initialized') and not adapter._initialized:
                await adapter.initialize()

        async def stream_adapter_1():
            nonlocal output_1, response_1
            async for chunk in adapter_1.stream_query(
                query=query,
                conversation_history=history,
                enable_thinking=thinking,
            ):
                output_1 = format_streaming_chunk(output_1, chunk)
                if chunk.is_final and chunk.metadata:
                    latency_ms = chunk.metadata.get("latency_ms", 0)
                    response_1 = RAGResponse(
                        query=query,
                        final_answer=chunk.content,
                        latency_ms=latency_ms,
                        stage_timing=StageTiming(total_ms=latency_ms),
                    )

        async def stream_adapter_2():
            nonlocal output_2, response_2
            async for chunk in adapter_2.stream_query(
                query=query,
                conversation_history=history,
                enable_thinking=thinking,
            ):
                output_2 = format_streaming_chunk(output_2, chunk)
                if chunk.is_final and chunk.metadata:
                    latency_ms = chunk.metadata.get("latency_ms", 0)
                    response_2 = RAGResponse(
                        query=query,
                        final_answer=chunk.content,
                        latency_ms=latency_ms,
                        stage_timing=StageTiming(total_ms=latency_ms),
                    )

        # Run both streams concurrently
        gen_1 = stream_adapter_1()
        gen_2 = stream_adapter_2()

        done_1 = False
        done_2 = False

        while not (done_1 and done_2):
            if not done_1:
                try:
                    await gen_1.__anext__()
                except StopAsyncIteration:
                    done_1 = True

            if not done_2:
                try:
                    await gen_2.__anext__()
                except StopAsyncIteration:
                    done_2 = True

            timing_1 = format_timing(response_1.stage_timing) if response_1 else []
            timing_2 = format_timing(response_2.stage_timing) if response_2 else []

            yield (
                output_1,
                output_2,
                response_1.final_answer if response_1 else "",
                response_2.final_answer if response_2 else "",
                response_1.to_dict() if response_1 else {},
                response_2.to_dict() if response_2 else {},
                timing_1,
                timing_2,
            )

            await asyncio.sleep(0.05)

    async def run_single_query(
        input_mode: str,
        query: str,
        conv_history: list,
        thinking: bool,
        annotation_id: Optional[str],
        run_mode: str,
        interface_name: str,
        iface_1: str,
        iface_2: str,
        ret_metrics: list[str],
        gen_metrics: list[str],
    ):
        """Run single query with streaming output."""

        # Prepare query parameters
        annotation = None
        history = []
        enable_think = thinking

        if input_mode == "测试集选择" and annotation_id:
            handler = await get_annotation_handler()
            annotation = await handler.get(annotation_id)
            if annotation:
                query = annotation.query
                history = annotation.conversation_history or []
                enable_think = annotation.enable_thinking
        else:
            # Convert dataframe history to list of strings
            if conv_history and isinstance(conv_history, list):
                if isinstance(conv_history[0], list):
                    history = [f"{row[0]}: {row[1]}" for row in conv_history if row[0] and row[1]]

        if not query or not query.strip():
            yield (
                "❌ 请输入查询内容",
                "",
                {},
                [],
                gr.update(visible=False),
                {},
                "",
            )
            return

        # Combine selected metrics
        selected_metrics = list(ret_metrics) + list(gen_metrics)

        # Run based on mode
        if run_mode == "双接口对比":
            adapter_1 = get_adapter_by_name(iface_1)
            adapter_2 = get_adapter_by_name(iface_2)

            async for result in run_dual_streaming(
                adapter_1, adapter_2, query, history, enable_think
            ):
                (
                    out_1, out_2,
                    ans_1, ans_2,
                    state_1, state_2,
                    timing_1, timing_2,
                ) = result

                yield (
                    out_1, out_2,
                    ans_1, ans_2,
                    state_1, state_2,
                    timing_1, timing_2,
                    gr.update(visible=input_mode == "测试集选择"),
                    {},
                    "",
                )

            # Calculate metrics for both interfaces if test set mode
            metrics_1 = {}
            metrics_2 = {}
            if annotation and selected_metrics:
                if response_1:
                    metrics_1 = await calculate_selected_metrics(
                        annotation, response_1, selected_metrics
                    )
                if response_2:
                    metrics_2 = await calculate_selected_metrics(
                        annotation, response_2, selected_metrics
                    )

            summary_1 = await format_metrics_summary(metrics_1)
            summary_2 = await format_metrics_summary(metrics_2)
            comparison = f"### 接口1指标\n\n{summary_1}\n\n### 接口2指标\n\n{summary_2}"

            yield (
                out_1, out_2,
                ans_1, ans_2,
                state_1, state_2,
                timing_1, timing_2,
                gr.update(visible=input_mode == "测试集选择"),
                metrics_1,
                comparison,
            )
        else:
            # Single interface mode
            adapter = get_adapter_by_name(interface_name)

            async for result in run_single_streaming(
                adapter, query, history, enable_think
            ):
                (
                    output,
                    answer,
                    state,
                    timing,
                ) = result

                yield (
                    output,
                    answer,
                    state,
                    timing,
                    gr.update(visible=input_mode == "测试集选择" and annotation is not None),
                    {},
                    "",
                )

            # Calculate metrics if test set mode
            if annotation and selected_metrics:
                # Re-run query to get full response for metrics
                response = await adapter.query(
                    query=query,
                    conversation_history=history,
                    enable_thinking=enable_think,
                )
                metrics = await calculate_selected_metrics(
                    annotation, response, selected_metrics
                )
                summary = await format_metrics_summary(metrics)

                yield (
                    output,
                    answer,
                    state,
                    timing,
                    gr.update(visible=True),
                    metrics,
                    summary,
                )

    def clear_outputs():
        """Clear all output fields."""
        return (
            "点击'运行'开始处理...",
            "",
            {},
            [],
            gr.update(visible=False),
            {},
            "",
            # Dual mode outputs
            "点击'运行'开始处理...",
            "点击'运行'开始处理...",
            "",
            "",
            {},
            {},
            [],
            [],
            {},
            {},
            "",
        )

    # ===== Connect Events =====

    input_mode.change(
        fn=toggle_input_mode,
        inputs=[input_mode],
        outputs=[
            custom_input_group,
            test_set_group,
            metrics_result_group,
        ],
        show_progress='hidden',
    )

    run_mode.change(
        fn=toggle_run_mode,
        inputs=[run_mode],
        outputs=[
            single_interface_group,
            dual_interface_group,
            single_output_group,
            dual_output_group,
            single_answer_group,
            dual_answer_group,
            single_state_group,
            dual_state_group,
            single_timing_group,
            dual_timing_group,
            single_metrics_group,
            dual_metrics_group,
        ],
        show_progress='hidden',
    )

    refresh_annotations_btn.click(
        fn=load_test_set_annotations,
        outputs=[annotation_selector],
    )

    annotation_selector.change(
        fn=on_annotation_selected,
        inputs=[annotation_selector],
        outputs=[gt_documents_display, standard_answers_display],
    )

    run_btn.click(
        fn=run_single_query,
        inputs=[
            input_mode,
            query_input,
            conversation_history,
            enable_thinking,
            annotation_selector,
            run_mode,
            interface_selector,
            interface_1,
            interface_2,
            metric_retrieval,
            metric_generation,
        ],
        outputs=[
            # Single mode outputs
            streaming_output,
            final_answer,
            rag_state_display,
            timing_display,
            metrics_result_group,
            metrics_display,
            metrics_summary,
            # Dual mode outputs
            streaming_output_1,
            streaming_output_2,
            final_answer_1,
            final_answer_2,
            rag_state_1,
            rag_state_2,
            timing_1,
            timing_2,
            metrics_display_1,
            metrics_display_2,
            metrics_comparison,
        ],
    )

    clear_btn.click(
        fn=clear_outputs,
        outputs=[
            streaming_output,
            final_answer,
            rag_state_display,
            timing_display,
            metrics_result_group,
            metrics_display,
            metrics_summary,
            # Dual mode outputs
            streaming_output_1,
            streaming_output_2,
            final_answer_1,
            final_answer_2,
            rag_state_1,
            rag_state_2,
            timing_1,
            timing_2,
            metrics_display_1,
            metrics_display_2,
            metrics_comparison,
        ],
    )

    # Return components for initialization
    return {
        "annotation_selector": annotation_selector,
        "load_test_set_annotations": load_test_set_annotations,
    }