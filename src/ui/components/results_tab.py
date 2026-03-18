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
from ...rag.base_adapter import RAGAdapter
from ...rag.mock_adapter import MockRAGAdapter
from ...utils.async_helpers import run_async
from ...core.logging import logger


def create_results_tab() -> gr.Tab:
    """Create the results display and analysis tab with enhanced styling."""

    with gr.Tab("📈 结果查看") as tab:
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

                    results_dataframe = gr.Dataframe(
                        headers=[
                            "ID", "查询", "成功", "平均分", "接口", "耗时 (ms)"
                        ],
                        datatype=["str", "str", "bool", "number", "str", "number"],
                        interactive=False,
                        label="评测结果列表",
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
                )

            manager = await get_result_manager()
            run = await manager.get_run(run_id)

            if not run:
                return (
                    gr.update(value={}),
                    "❌ 运行不存在",
                    gr.update(value=[]),
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

            # Results dataframe
            results = await manager.get_results_by_run(run_id, limit=100)
            data = []
            for r in results:
                data.append([
                    r.id[:8] + "...",
                    r.annotation.query[:30] + "..." if r.annotation else "N/A",
                    r.success,
                    round(r.metrics.average_score, 3) if r.metrics else 0,
                    r.rag_interface,
                    round(r.duration_ms, 0),
                ])

            return (
                gr.update(value=run_info_data),
                summary_md,
                gr.update(value=data),
            )

        async def load_result_detail(evt: gr.SelectData, run_id: str):
            """Load detail for a selected result row."""
            if not run_id or evt.index is None:
                return [gr.update() for _ in range(5)]

            manager = await get_result_manager()
            results = await manager.get_results_by_run(run_id, limit=100)

            if evt.index >= len(results):
                return [gr.update() for _ in range(5)]

            result = results[evt.index]

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
                gr.update(value=result.to_dict()),
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

            yield "🔄 正在重新评测..."

            # Use mock adapter for demo
            adapter = MockRAGAdapter()

            # Get annotation
            handler = await get_annotation_handler()

            # Simulate streaming output with improved formatting
            output = "### 🔄 RAG 响应\n\n"
            output += "#### 🔧 查询改写\n正在处理...\n\n"

            yield output
            await asyncio.sleep(0.5)

            output += "**改写结果**: 扩展查询\n\n"
            output += "#### 🎯 FAQ 匹配\n正在匹配...\n\n"

            yield output
            await asyncio.sleep(0.5)

            output += "**匹配结果**: 未匹配\n\n"
            output += "#### 📚 检索结果\n正在检索...\n\n"

            yield output
            await asyncio.sleep(0.5)

            output += "**检索到 5 个文档**\n\n"
            output += "#### ✍️ 生成答案\n正在生成...\n\n"

            yield output
            await asyncio.sleep(0.5)

            output += "**答案**: 这是一个模拟的 RAG 响应。\n\n"
            output += "---\n✅ 评测完成"

            yield output

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

        # ===== Connect Events =====
        
        refresh_runs_btn.click(
            fn=lambda: run_async(load_runs()),
            outputs=[run_selector],
        )

        run_selector.change(
            fn=lambda rid: run_async(load_run_details(rid)),
            inputs=[run_selector],
            outputs=[run_info, summary_display, results_dataframe],
        )

        results_dataframe.select(
            fn=lambda evt, rid: run_async(load_result_detail(evt, rid)),
            inputs=[run_selector],
            outputs=[
                detail_query,
                detail_answer,
                detail_rag_info,
                detail_metrics,
                streaming_output,
            ],
        )

        rerun_btn.click(
            fn=lambda rid: run_async(rerun_single(rid, "")),
            inputs=[run_selector],
            outputs=[streaming_output],
        )

        export_btn.click(
            fn=lambda rid, fmt: run_async(export_results(rid, fmt)),
            inputs=[run_selector, export_format],
            outputs=[export_status],
        )

        # Initial load on tab select
        tab.select(
            fn=lambda: run_async(load_runs()),
            outputs=[run_selector],
        )

    return tab
