"""
Results display tab component for Gradio UI.
Enhanced with modern styling and improved data visualization.
"""

import asyncio
from typing import Optional

import gradio as gr

from ...models.evaluation_result import EvaluationResult, EvaluationRun
from ...evaluation.result_manager import get_result_manager
from ...annotation.annotation_handler import get_annotation_handler
from ...rag.base_adapter import RAGAdapter, StreamingChunk
from ...rag.mock_adapter import MockRAGAdapter
from ...rag.langgraph_adapter import LangGraphAdapter
from ...core.logging import logger


def create_results_tab() -> None:
    """Create the results display and analysis tab with enhanced styling."""

    # Header
    gr.Markdown("""
    ### 📊 评测结果查看与分析
    查看评测运行结果、分析指标详情、导出数据报告
    """)

    with gr.Row():
        # Left column: Run selection
        with gr.Column(scale=1):
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**📋 选择评测运行**")

                run_selector = gr.Dropdown(
                    label="评测运行列表",
                    choices=[],
                    interactive=True,
                )
                refresh_runs_btn = gr.Button(
                    "🔄 刷新列表",
                    variant="secondary",
                )

                run_info = gr.JSON(
                    label="运行信息",
                    value={},
                )

        # Right column: Results
        with gr.Column(scale=3):
            with gr.Group(elem_classes=["gr-box"]):
                gr.Markdown("**📊 结果概览**")

                summary_display = gr.Markdown(
                    "👈 请从左侧选择一个评测运行查看结果",
                    elem_classes=["placeholder-text"],
                )

                with gr.Row():
                    interface_filter = gr.Dropdown(
                        label="🔌 接口筛选",
                        choices=["全部"],
                        value="全部",
                        scale=1,
                    )
                    status_filter = gr.Dropdown(
                        label="✅ 状态筛选",
                        choices=["全部", "成功", "失败"],
                        value="全部",
                        scale=1,
                    )
                    tag_filter = gr.Dropdown(
                        label="🏷️ 标签筛选",
                        choices=["全部"],
                        value="全部",
                        scale=1,
                    )

                results_dataframe = gr.Dataframe(
                    headers=[
                        "ID", "查询", "标签", "成功", "平均分", "接口", "耗时 (ms)"
                    ],
                    datatype=["str", "str", "str", "bool", "number", "str", "number"],
                    interactive=False,
                    label="评测结果列表（点击行查看详情）",
                    wrap=True,
                )

    # Detail view
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**🔍 单条详情**")

        with gr.Row():
            with gr.Column(scale=2):
                detail_query = gr.Textbox(
                    label="用户查询",
                    interactive=False,
                    placeholder="选择结果查看详情...",
                )
                detail_answer = gr.Textbox(
                    label="RAG 回答",
                    interactive=False,
                    lines=5,
                    placeholder="回答内容将显示在这里...",
                )

            with gr.Column(scale=1):
                detail_rag_info = gr.JSON(
                    label="RAG 响应信息",
                    value={},
                )
                detail_metrics = gr.JSON(
                    label="评测指标",
                    value={},
                )

    # Intermediate results display
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**🔧 RAG 处理链路详情**")

        with gr.Tabs() as intermediate_tabs:
            # Query Rewrite Tab
            with gr.TabItem("📝 查询改写"):
                query_rewrite_info = gr.Markdown(
                    "选择结果查看查询改写详情...",
                    elem_classes=["placeholder-text"],
                )

            # FAQ Match Tab
            with gr.TabItem("🎯 FAQ 匹配"):
                faq_match_info = gr.Markdown(
                    "选择结果查看 FAQ 匹配详情...",
                    elem_classes=["placeholder-text"],
                )

            # Retrieval Tab
            with gr.TabItem("📚 检索结果"):
                retrieval_table = gr.Dataframe(
                    headers=["排名", "文档ID", "内容摘要", "得分", "来源"],
                    datatype=["number", "str", "str", "number", "str"],
                    interactive=False,
                    label="检索到的文档",
                )

            # Rerank Tab
            with gr.TabItem("🔄 重排序"):
                rerank_table = gr.Dataframe(
                    headers=["排名", "文档ID", "内容摘要", "原得分", "重排得分"],
                    datatype=["number", "str", "str", "number", "number"],
                    interactive=False,
                    label="重排序后的文档",
                )

            # Thinking Tab
            with gr.TabItem("💭 思考过程"):
                thinking_info = gr.Markdown(
                    "选择结果查看思考过程（如果启用）...",
                    elem_classes=["placeholder-text"],
                )

            # Performance Analysis Tab
            with gr.TabItem("⚡ 性能分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        performance_summary = gr.Markdown(
                            "选择结果查看性能分析...",
                            elem_classes=["placeholder-text"],
                        )
                    with gr.Column(scale=1):
                        performance_table = gr.Dataframe(
                            headers=["阶段", "耗时 (ms)", "占比 (%)", "数据来源"],
                            datatype=["str", "number", "number", "str"],
                            interactive=False,
                            label="各阶段耗时详情",
                        )

    # Tag management section
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**🏷️ 标签管理**")

        with gr.Row():
            with gr.Column(scale=2):
                current_tags = gr.Markdown(
                    "**当前标签**: 无",
                    elem_classes=["tag-display"],
                )

            with gr.Column(scale=2):
                with gr.Row():
                    preset_tags = gr.Dropdown(
                        label="快速添加预设标签",
                        choices=[
                            "✅ 优秀案例",
                            "⚠️ 需复审",
                            "❌ 典型错误",
                            "🔍 难度较高",
                            "💡 答案质量好",
                            "📊 数据问题",
                            "🤖 RAG 问题",
                        ],
                        value=None,
                        interactive=True,
                    )

        with gr.Row():
            new_tag_input = gr.Textbox(
                label="自定义标签",
                placeholder="输入新标签...",
                scale=2,
            )
            add_tag_btn = gr.Button(
                "➕ 添加标签",
                variant="primary",
                scale=1,
            )
            remove_tag_btn = gr.Button(
                "➖ 移除标签",
                variant="secondary",
                scale=1,
            )

        tag_status = gr.Markdown("")

        with gr.Row():
            rerun_btn = gr.Button(
                "🔄 重跑此条",
                variant="secondary",
            )
            streaming_output = gr.Markdown(
                "",
                visible=False,
                elem_classes=["streaming-output"],
            )

    # Comparison view (for dual-RAG)
    with gr.Group(
        elem_classes=["gr-box"],
        visible=False,
    ) as comparison_row:
        gr.Markdown("**⚖️ 双接口对比**")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**🔵 接口 1 结果**")
                compare_1 = gr.JSON()
            with gr.Column():
                gr.Markdown("**🟢 接口 2 结果**")
                compare_2 = gr.JSON()

    # Export section
    with gr.Group(elem_classes=["gr-box"]):
        gr.Markdown("**📥 导出结果**")
        with gr.Row():
            export_format = gr.Dropdown(
                label="导出格式",
                choices=["JSON", "CSV"],
                value="JSON",
                scale=1,
            )
            export_btn = gr.Button(
                "📥 导出当前运行结果",
                variant="primary",
                scale=1,
            )
            export_status = gr.Markdown(
                "",
                elem_classes=["export-status"],
            )

    # ===== Event Handlers =====

    async def load_runs():
        """Load available evaluation runs."""
        manager = await get_result_manager()
        runs = await manager.list_runs(limit=50)

        choices = [
            (
                f"{r.name or r.id[:8]} - {r.status} ({r.completed_count}/{r.total_annotations})",
                r.id,
            )
            for r in runs
        ]

        return gr.update(choices=choices)

    async def load_run_details(run_id: str):
        """Load detailed information for a specific run."""
        if not run_id:
            return (
                gr.update(value={}),
                "👈 选择一个评测运行查看结果",
                gr.update(value=[]),
                gr.update(choices=["全部"], value="全部"),
            )

        manager = await get_result_manager()
        run = await manager.get_run(run_id)

        if not run:
            return (
                gr.update(value={}),
                "❌ 运行不存在",
                gr.update(value=[]),
                gr.update(choices=["全部"], value="全部"),
            )

        # Run info
        run_info_data = {
            "名称": run.name,
            "状态": run.status,
            "总数据": run.total_annotations,
            "完成": run.completed_count,
            "失败": run.failed_count,
            "耗时": f"{run.duration_seconds:.1f}s",
            "接口": run.rag_interfaces,
            "运行标签": run.tags if run.tags else [],
        }

        # Summary markdown with improved formatting
        summary_md = f"""
        ### 📊 评测概览

        | 指标 | 数值 |
        |------|------|
        | 状态 | {run.status} |
        | 完成 | {run.completed_count}/{run.total_annotations} |
        | 失败 | {run.failed_count} |
        | 耗时 | {run.duration_seconds:.1f}s |

        ### 🏆 各接口得分
        """

        for iface, stats in run.summary_by_interface.items():
            avg_score = stats.get('average_score', 0)
            success_rate = stats.get('success_rate', 0)

            # Color-coded score display
            if avg_score >= 0.8:
                score_color = "green"
            elif avg_score >= 0.6:
                score_color = "blue"
            elif avg_score >= 0.4:
                score_color = "orange"
            else:
                score_color = "red"

            summary_md += f"""

        #### {iface}
        - 平均分：<span style="color: {score_color}; font-weight: bold;">{avg_score:.2%}</span>
        - 成功率：{success_rate:.1%}
        """

        # Get all tags used in the run
        all_tags = await manager.get_all_tags(run_id)
        tag_choices = ["全部"] + [t["name"] for t in all_tags.get("result_tags", [])]

        # Results dataframe
        results = await manager.get_results_by_run(run_id, limit=100)
        data = []
        for r in results:
            tags_str = ", ".join(r.tags) if r.tags else "-"
            data.append([
                r.id[:8] + "...",
                r.annotation.query[:30] + "..." if r.annotation else "N/A",
                tags_str,
                r.success,
                round(r.metrics.average_score, 3) if r.metrics else 0,
                r.rag_interface,
                round(r.duration_ms, 0),
            ])

        return (
            gr.update(value=run_info_data),
            summary_md,
            gr.update(value=data),
            gr.update(choices=tag_choices, value="全部"),
        )

    async def load_result_detail(evt: gr.SelectData, run_id: str):
        """Load detail for a selected result row."""
        logger.info(f"load_result_detail called: run_id={run_id}, evt.index={evt.index}, type={type(evt.index)}")

        if not run_id:
            logger.warning("load_result_detail: no run_id")
            return [gr.update() for _ in range(13)]

        manager = await get_result_manager()

        # First get the run directly to check if it has results
        run = await manager.get_run(run_id)
        if run:
            logger.info(f"get_run returned run with {len(run.results)} results")
        else:
            logger.warning(f"get_run returned None for run_id={run_id}")

        results = await manager.get_results_by_run(run_id, limit=100)
        logger.info(f"load_result_detail: get_results_by_run returned {len(results)} results")

        # Handle different Gradio versions
        # Gradio 5.x: evt.index is int (row index)
        # Gradio 6.x: evt.index is list [row, col] or (row, col)
        if evt.index is None:
            logger.warning("load_result_detail: evt.index is None")
            return [gr.update() for _ in range(13)]

        # Extract row index from different formats
        if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 1:
            row_index = evt.index[0]
        elif isinstance(evt.index, int):
            row_index = evt.index
        else:
            logger.warning(f"load_result_detail: unexpected evt.index type: {type(evt.index)}, value: {evt.index}")
            return [gr.update() for _ in range(13)]

        logger.info(f"load_result_detail: row_index={row_index}")

        if row_index >= len(results):
            logger.warning(f"load_result_detail: row_index {row_index} >= len(results) {len(results)}")
            return [gr.update() for _ in range(13)]

        result = results[row_index]
        logger.info(f"load_result_detail: got result, annotation={result.annotation is not None}, rag_response={result.rag_response is not None}")

        if result.annotation:
            logger.info(f"  query: {result.annotation.query[:50] if result.annotation.query else 'None'}...")
        if result.rag_response:
            logger.info(f"  final_answer exists: {result.rag_response.final_answer is not None}")

        # Build query rewrite info
        qr_md = "### 📝 查询改写\n\n"
        if result.rag_response and result.rag_response.query_rewrite:
            qr = result.rag_response.query_rewrite
            qr_md += f"""**原查询**: {qr.original_query}

**改写后**: {qr.rewritten_query}

**改写类型**: {qr.rewrite_type or "未指定"}

**置信度**: {qr.confidence:.2%}
"""
        else:
            qr_md += "*未进行查询改写*"

        # Build FAQ match info
        faq_md = "### 🎯 FAQ 匹配\n\n"
        if result.rag_response and result.rag_response.faq_match:
            fm = result.rag_response.faq_match
            if fm.matched:
                faq_md += f"""✅ **匹配成功**

**FAQ ID**: {fm.faq_id}

**问题**: {fm.faq_question}

**答案**: {fm.faq_answer}

**置信度**: {fm.confidence:.2%}

**相似度**: {fm.similarity_score:.2%}
"""
            else:
                faq_md += "❌ **未匹配到 FAQ**"
        else:
            faq_md += "*未进行 FAQ 匹配*"

        # Build retrieval table
        retrieval_data = []
        if result.rag_response and result.rag_response.retrieval_results:
            for r in result.rag_response.retrieval_results:
                retrieval_data.append([
                    r.rank,
                    r.document_id,
                    r.content[:100] + "..." if len(r.content) > 100 else r.content,
                    round(r.score, 4),
                    r.metadata.get("source", "未知"),
                ])

        # Build rerank table
        rerank_data = []
        if result.rag_response and result.rag_response.rerank_results:
            for r in result.rag_response.rerank_results:
                rerank_data.append([
                    r.rank,
                    r.document_id,
                    r.content[:100] + "..." if len(r.content) > 100 else r.content,
                    round(r.original_score, 4),
                    round(r.rerank_score, 4),
                ])

        # Build thinking info
        thinking_md = "### 💭 思考过程\n\n"
        if result.rag_response and result.rag_response.llm_output:
            llm = result.rag_response.llm_output
            if llm.thinking_process:
                thinking_md += f"""```
{llm.thinking_process}
```

**模型**: {llm.model}

**Token 使用**:
- Prompt: {llm.token_usage.get('prompt_tokens', 'N/A')}
- Completion: {llm.token_usage.get('completion_tokens', 'N/A')}
- Total: {llm.token_usage.get('total_tokens', 'N/A')}
"""
            else:
                thinking_md += "*未启用思考模式或无思考过程*"
        else:
            thinking_md += "*无 LLM 输出信息*"

        # Build current tags display
        if result.tags:
            tags_display = "**当前标签**: " + " | ".join([f"`{t}`" for t in result.tags])
        else:
            tags_display = "**当前标签**: 无"

        # Build performance analysis
        perf_md = "### ⚡ 性能分析\n\n"
        perf_table_data = []

        if result.rag_response and result.rag_response.stage_timing:
            timing = result.rag_response.stage_timing
            percentages = timing.get_percentages()
            stage_names = {
                "query_rewrite": "查询改写",
                "faq_match": "FAQ 匹配",
                "retrieval": "检索",
                "rerank": "重排序",
                "generation": "生成",
            }

            perf_md += f"**总耗时**: {timing.total_ms:.2f} ms\n\n"
            perf_md += f"**数据来源**: {timing.source}\n\n"

            for stage, ms in timing.get_stage_timings().items():
                stage_name = stage_names.get(stage, stage)
                pct = percentages.get(stage, 0)
                source = timing.extraction_details.get(stage, "unknown")

                perf_table_data.append([
                    stage_name,
                    round(ms, 2),
                    round(pct, 2),
                    source,
                ])
        else:
            perf_md += "*无 stage_timing 数据*"

        return (
            gr.update(
                value=result.annotation.query if result.annotation else ""
            ),
            gr.update(
                value=result.rag_response.final_answer if result.rag_response else ""
            ),
            gr.update(
                value=result.rag_response.to_dict() if result.rag_response else {}
            ),
            gr.update(
                value=result.metrics.to_dict() if result.metrics else {}
            ),
            gr.update(value=""),  # streaming_output - clear it when viewing detail
            # Intermediate results
            gr.update(value=qr_md),
            gr.update(value=faq_md),
            gr.update(value=retrieval_data),
            gr.update(value=rerank_data),
            gr.update(value=thinking_md),
            # Performance analysis
            gr.update(value=perf_md),
            gr.update(value=perf_table_data),
            # Tags
            gr.update(value=tags_display),
        )

    async def rerun_single(run_id: str, result_id: str):
        """Rerun a single evaluation and stream the output."""
        if not run_id:
            yield "❌ 请先选择一个评测运行"
            return

        manager = await get_result_manager()
        run = await manager.get_run(run_id)

        if not run:
            yield "❌ 运行不存在"
            return

        yield "🔄 正在连接 RAG 服务..."

        # Get annotation
        handler = await get_annotation_handler()

        # Use mock adapter for demo (in production, use real LangGraph adapter)
        adapter = MockRAGAdapter(simulate_latency=True)

        # Get a sample query (in real usage, get from selected result)
        results = await manager.get_results_by_run(run_id, limit=1)
        if results and results[0].annotation:
            query = results[0].annotation.query
        else:
            query = "测试查询"

        # Stream the query execution
        output = "### 🔄 RAG 流式响应\n\n"
        yield output

        current_stage = None
        stage_content = {}

        async for chunk in adapter.stream_query(query=query, enable_thinking=True):
            stage = chunk.stage
            content = chunk.content
            is_final = chunk.is_final
            metadata = chunk.metadata or {}

            # Track stage content
            if stage != current_stage:
                if current_stage:
                    output += "\n\n"
                current_stage = stage

            # Format based on stage
            if stage == "query_rewrite":
                if "query_rewrite" not in stage_content:
                    output += "#### 🔧 查询改写\n"
                    stage_content["query_rewrite"] = True
                output += f"{content}\n"

            elif stage == "faq_match":
                if "faq_match" not in stage_content:
                    output += "\n#### 🎯 FAQ 匹配\n"
                    stage_content["faq_match"] = True
                output += f"{content}\n"

            elif stage == "retrieval":
                if "retrieval" not in stage_content:
                    output += "\n#### 📚 检索结果\n"
                    stage_content["retrieval"] = True
                output += f"{content}\n"

            elif stage == "rerank":
                if "rerank" not in stage_content:
                    output += "\n#### 🔄 重排序\n"
                    stage_content["rerank"] = True
                output += f"{content}\n"

            elif stage == "thinking":
                if "thinking" not in stage_content:
                    output += "\n#### 💭 思考过程\n"
                    stage_content["thinking"] = True
                output += f"```\n{content}\n```\n"

            elif stage == "generation":
                if "generation" not in stage_content:
                    output += "\n#### ✍️ 生成答案\n"
                    stage_content["generation"] = True
                # Show streaming content with cursor effect
                output += f"\r{content}"

            elif stage == "final":
                output += "\n\n---\n"
                output += f"#### ✅ 最终答案\n\n{content}\n\n"

                # Show metadata
                latency = metadata.get("latency_ms", 0)
                output += f"**耗时**: {latency:.0f}ms\n"

                if metadata.get("is_refused"):
                    output += "\n⚠️ 系统拒绝了此查询\n"

                output += "\n✅ 评测完成"

            elif stage == "error":
                output += f"\n\n❌ **错误**: {content}\n"

            yield output
            await asyncio.sleep(0.01)  # Small delay for UI update

    async def export_results(run_id: str, format: str):
        """Export results to file."""
        if not run_id:
            return "❌ 请先选择一个评测运行"

        manager = await get_result_manager()

        try:
            from pathlib import Path
            output_path = await manager.export_run(
                run_id,
                format=format.lower(),
                output_path=Path(f"export_{run_id[:8]}.{format.lower()}"),
            )
            return f"✅ 导出成功：`{output_path}`"
        except Exception as e:
            return f"❌ 导出失败：{str(e)}"

    # State for selected result
    selected_result_id = gr.State(None)
    selected_result_index = gr.State(None)

    async def on_result_select(evt: gr.SelectData):
        """Store selected result index and ID."""
        return evt.index, None  # Will be resolved later with result ID

    async def add_tag_to_result(
        run_id: str,
        result_index: int,
        tag: str,
        preset_tag: Optional[str],
    ):
        """Add a tag to the selected result."""
        if not run_id or result_index is None:
            return "❌ 请先选择一条结果", gr.update()

        # Use preset tag if no custom tag provided
        tag_to_add = tag or preset_tag
        if not tag_to_add:
            return "❌ 请输入或选择一个标签", gr.update()

        manager = await get_result_manager()
        results = await manager.get_results_by_run(run_id, limit=100)

        if result_index >= len(results):
            return "❌ 结果不存在", gr.update()

        result = results[result_index]
        success = await manager.add_result_tag(run_id, result.id, tag_to_add)

        if success:
            # Refresh result detail
            result = results[result_index]  # Get updated result
            if result.tags:
                tags_display = "**当前标签**: " + " | ".join([f"`{t}`" for t in result.tags])
            else:
                tags_display = "**当前标签**: 无"
            return f"✅ 已添加标签: {tag_to_add}", gr.update(value=tags_display)
        else:
            return "❌ 添加失败", gr.update()

    async def remove_tag_from_result(
        run_id: str,
        result_index: int,
        tag: str,
        preset_tag: Optional[str],
    ):
        """Remove a tag from the selected result."""
        if not run_id or result_index is None:
            return "❌ 请先选择一条结果", gr.update()

        # Use preset tag if no custom tag provided
        tag_to_remove = tag or preset_tag
        if not tag_to_remove:
            return "❌ 请输入或选择要移除的标签", gr.update()

        manager = await get_result_manager()
        results = await manager.get_results_by_run(run_id, limit=100)

        if result_index >= len(results):
            return "❌ 结果不存在", gr.update()

        result = results[result_index]
        success = await manager.remove_result_tag(run_id, result.id, tag_to_remove)

        if success:
            # Get updated result from run
            run = await manager.get_run(run_id)
            for r in run.results:
                if r.id == result.id:
                    result = r
                    break
            if result.tags:
                tags_display = "**当前标签**: " + " | ".join([f"`{t}`" for t in result.tags])
            else:
                tags_display = "**当前标签**: 无"
            return f"✅ 已移除标签: {tag_to_remove}", gr.update(value=tags_display)
        else:
            return "❌ 移除失败或标签不存在", gr.update()

    async def filter_by_tag(run_id: str, tag: str):
        """Filter results by tag."""
        if not run_id:
            return gr.update()

        manager = await get_result_manager()
        results = await manager.get_results_by_run(run_id, limit=100)

        data = []
        for r in results:
            # Filter by tag if specified
            if tag and tag != "全部":
                if tag not in r.tags:
                    continue

            tags_str = ", ".join(r.tags) if r.tags else "-"
            data.append([
                r.id[:8] + "...",
                r.annotation.query[:30] + "..." if r.annotation else "N/A",
                tags_str,
                r.success,
                round(r.metrics.average_score, 3) if r.metrics else 0,
                r.rag_interface,
                round(r.duration_ms, 0),
            ])

        return gr.update(value=data)

    # ===== Connect Events =====

    refresh_runs_btn.click(
        fn=load_runs,
        outputs=[run_selector],
    )

    run_selector.change(
        fn=load_run_details,
        inputs=[run_selector],
        outputs=[run_info, summary_display, results_dataframe, tag_filter],
    )

    # Result selection
    def on_result_row_select(evt: gr.SelectData):
        """Handle result row selection."""
        # Handle different Gradio versions
        # Gradio 5.x: evt.index is int (row index)
        # Gradio 6.x: evt.index is list [row, col] or (row, col)
        if evt.index is None:
            return None, None

        if isinstance(evt.index, (list, tuple)) and len(evt.index) >= 1:
            row_index = evt.index[0]
        elif isinstance(evt.index, int):
            row_index = evt.index
        else:
            return None, None

        return row_index, row_index

    results_dataframe.select(
        fn=on_result_row_select,
        outputs=[selected_result_index, selected_result_id],
    )

    results_dataframe.select(
        fn=load_result_detail,
        inputs=[run_selector],
        outputs=[
            detail_query,
            detail_answer,
            detail_rag_info,
            detail_metrics,
            streaming_output,
            # Intermediate results
            query_rewrite_info,
            faq_match_info,
            retrieval_table,
            rerank_table,
            thinking_info,
            # Performance analysis
            performance_summary,
            performance_table,
            # Tags
            current_tags,
        ],
    )

    # Tag filter
    tag_filter.change(
        fn=filter_by_tag,
        inputs=[run_selector, tag_filter],
        outputs=[results_dataframe],
    )

    # Tag management
    add_tag_btn.click(
        fn=add_tag_to_result,
        inputs=[run_selector, selected_result_index, new_tag_input, preset_tags],
        outputs=[tag_status, current_tags],
    )

    remove_tag_btn.click(
        fn=remove_tag_from_result,
        inputs=[run_selector, selected_result_index, new_tag_input, preset_tags],
        outputs=[tag_status, current_tags],
    )

    # Clear preset after selection
    preset_tags.change(
        fn=lambda: gr.update(value=""),
        outputs=[new_tag_input],
    )

    rerun_btn.click(
        fn=rerun_single,
        inputs=[run_selector, selected_result_id],
        outputs=[streaming_output],
    )

    export_btn.click(
        fn=export_results,
        inputs=[run_selector, export_format],
        outputs=[export_status],
    )

    # 返回需要初始化加载的组件和函数
    return {
        "run_selector": run_selector,
        "load_runs": load_runs,
    }