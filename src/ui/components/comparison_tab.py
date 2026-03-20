"""
Comparison tab component for Gradio UI.
Provides side-by-side comparison of dual RAG evaluation results.
"""

import asyncio
from typing import Optional, Any

import gradio as gr

from ...models.evaluation_result import EvaluationResult, EvaluationRun
from ...evaluation.result_manager import get_result_manager
from ...utils.async_helpers import run_async
from ...core.logging import logger


def create_comparison_tab() -> gr.Tab:
    """Create the dual RAG comparison visualization tab."""

    with gr.Tab("⚖️ 对比分析") as tab:
        # Header
        gr.Markdown("""
        ### ⚖️ 双 RAG 接口对比分析
        并排比较两个 RAG 系统的评测结果，识别性能差异和改进方向
        """)

        with gr.Row():
            # Left column: Run selection
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["gr-box"]):
                    gr.Markdown("**📋 选择对比运行**")

                    run_selector = gr.Dropdown(
                        label="评测运行列表（双接口模式）",
                        choices=[],
                        interactive=True,
                        info="仅显示启用双接口对比的运行",
                    )
                    refresh_btn = gr.Button(
                        "🔄 刷新列表",
                        variant="secondary",
                    )

                    run_info = gr.JSON(
                        label="运行信息",
                        value={},
                    )

        # Summary comparison section
        with gr.Group(elem_classes=["gr-box"]):
            gr.Markdown("**📊 总体对比**")

            with gr.Row():
                # Interface 1 summary
                with gr.Column():
                    gr.Markdown("**🔵 接口 1**")
                    iface1_stats = gr.JSON(
                        label="统计信息",
                        value={},
                    )

                # Comparison metrics
                with gr.Column():
                    gr.Markdown("**📈 差异分析**")
                    comparison_summary = gr.Markdown(
                        "👈 请选择一个双接口评测运行",
                        elem_classes=["placeholder-text"],
                    )

                # Interface 2 summary
                with gr.Column():
                    gr.Markdown("**🟢 接口 2**")
                    iface2_stats = gr.JSON(
                        label="统计信息",
                        value={},
                    )

        # Metric-level comparison
        with gr.Group(elem_classes=["gr-box"]):
            gr.Markdown("**🎯 指标对比详情**")

            with gr.Row():
                category_filter = gr.Dropdown(
                    label="指标类别",
                    choices=["全部", "检索指标", "生成指标", "FAQ指标", "综合指标"],
                    value="全部",
                    scale=1,
                )

            metric_comparison_table = gr.Dataframe(
                headers=["指标名称", "接口 1 得分", "接口 2 得分", "差异", "优势方"],
                datatype=["str", "number", "number", "number", "str"],
                interactive=False,
                label="指标对比表",
                wrap=True,
            )

        # Per-result comparison
        with gr.Group(elem_classes=["gr-box"]):
            gr.Markdown("**📝 单条结果对比**")

            with gr.Row():
                result_selector = gr.Dropdown(
                    label="选择查询",
                    choices=[],
                    interactive=True,
                    scale=2,
                )
                prev_btn = gr.Button("⬅️ 上一个", variant="secondary", scale=1)
                next_btn = gr.Button("➡️ 下一个", variant="secondary", scale=1)

            with gr.Row():
                # Query display
                with gr.Column():
                    query_display = gr.Textbox(
                        label="用户查询",
                        interactive=False,
                        lines=2,
                    )

            with gr.Row():
                # Interface 1 result
                with gr.Column():
                    gr.Markdown("**🔵 接口 1 结果**")
                    iface1_answer = gr.Textbox(
                        label="回答",
                        interactive=False,
                        lines=5,
                    )
                    iface1_metrics = gr.JSON(
                        label="指标详情",
                        value={},
                    )
                    iface1_rag_info = gr.JSON(
                        label="RAG 响应信息",
                        value={},
                    )

                # Interface 2 result
                with gr.Column():
                    gr.Markdown("**🟢 接口 2 结果**")
                    iface2_answer = gr.Textbox(
                        label="回答",
                        interactive=False,
                        lines=5,
                    )
                    iface2_metrics = gr.JSON(
                        label="指标详情",
                        value={},
                    )
                    iface2_rag_info = gr.JSON(
                        label="RAG 响应信息",
                        value={},
                    )

        # Statistical significance
        with gr.Group(elem_classes=["gr-box"]):
            gr.Markdown("**📉 统计显著性检验**")

            significance_info = gr.Markdown(
                "选择运行后显示统计显著性分析结果",
            )

            with gr.Row():
                confidence_level = gr.Slider(
                    minimum=0.90,
                    maximum=0.99,
                    value=0.95,
                    step=0.01,
                    label="置信水平",
                )
                calculate_significance_btn = gr.Button(
                    "🔍 计算显著性",
                    variant="secondary",
                )

        # Export comparison report
        with gr.Group(elem_classes=["gr-box"]):
            gr.Markdown("**📥 导出对比报告**")

            with gr.Row():
                export_format = gr.Dropdown(
                    label="导出格式",
                    choices=["JSON", "CSV", "Markdown"],
                    value="Markdown",
                    scale=1,
                )
                export_btn = gr.Button(
                    "📥 导出对比报告",
                    variant="primary",
                    scale=1,
                )
                export_status = gr.Markdown("")

        # ===== Event Handlers =====

        # State to store current results
        current_run = gr.State(None)
        current_results_1 = gr.State([])
        current_results_2 = gr.State([])
        current_index = gr.State(0)

        async def load_dual_runs():
            """Load evaluation runs with dual RAG interfaces."""
            manager = await get_result_manager()
            runs = await manager.list_runs(limit=50)

            # Filter for dual interface runs
            dual_runs = [
                (f"{r.name or r.id[:8]} - {r.status} (双接口)", r.id)
                for r in runs
                if len(r.rag_interfaces) > 1
            ]

            if not dual_runs:
                dual_runs = [("暂无双接口评测运行", "")]

            return gr.update(choices=dual_runs)

        async def load_comparison_data(run_id: str):
            """Load comparison data for selected run."""
            if not run_id:
                return (
                    gr.update(value={}),
                    gr.update(value={}),
                    gr.update(value={}),
                    "请选择一个双接口评测运行",
                    gr.update(value=[]),
                    gr.update(choices=[]),
                    None,
                    [],
                    [],
                    0,
                )

            manager = await get_result_manager()
            run = await manager.get_run(run_id)

            if not run or len(run.rag_interfaces) < 2:
                return (
                    gr.update(value={}),
                    gr.update(value={}),
                    gr.update(value={}),
                    "❌ 运行不存在或不是双接口模式",
                    gr.update(value=[]),
                    gr.update(choices=[]),
                    None,
                    [],
                    [],
                    0,
                )

            # Run info
            run_info_data = {
                "名称": run.name,
                "状态": run.status,
                "接口": run.rag_interfaces,
                "总数据": run.total_annotations,
                "完成": run.completed_count,
                "耗时": f"{run.duration_seconds:.1f}s",
            }

            # Get interface summaries
            iface1_name = run.rag_interfaces[0]
            iface2_name = run.rag_interfaces[1]

            iface1_stats = run.summary_by_interface.get(iface1_name, {})
            iface2_stats = run.summary_by_interface.get(iface2_name, {})

            # Build comparison summary
            avg1 = iface1_stats.get('average_score', 0)
            avg2 = iface2_stats.get('average_score', 0)
            diff = avg1 - avg2

            if diff > 0.05:
                winner = f"🔵 **{iface1_name}** 领先"
                diff_color = "blue"
            elif diff < -0.05:
                winner = f"🟢 **{iface2_name}** 领先"
                diff_color = "green"
            else:
                winner = "⚖️ 两者相当"
                diff_color = "gray"

            success_rate1 = iface1_stats.get('success_rate', 0)
            success_rate2 = iface2_stats.get('success_rate', 0)

            comparison_md = f"""
            ### 📊 对比摘要

            | 指标 | {iface1_name} | {iface2_name} | 差异 |
            |------|--------------|--------------|------|
            | 平均分 | {avg1:.2%} | {avg2:.2%} | {diff:+.2%} |
            | 成功率 | {success_rate1:.1%} | {success_rate2:.1%} | {(success_rate1-success_rate2):+.1%} |

            **结论**: {winner}

            - 分数差异: <span style="color: {diff_color};">{abs(diff):.2%}</span>
            - 显著性: {"显著" if abs(diff) > 0.1 else "不显著"}
            """

            # Build metric comparison table
            metric_rows = []
            all_results = await manager.get_results_by_run(run_id, limit=1000)

            # Separate results by interface
            results_1 = [r for r in all_results if r.rag_interface == iface1_name]
            results_2 = [r for r in all_results if r.rag_interface == iface2_name]

            # Calculate per-metric averages
            if results_1 and results_2:
                metrics_1 = {}
                metrics_2 = {}

                for r in results_1:
                    if r.success and r.metrics:
                        for m in r.metrics.results:
                            name = m.name
                            if name not in metrics_1:
                                metrics_1[name] = []
                            metrics_1[name].append(m.score)

                for r in results_2:
                    if r.success and r.metrics:
                        for m in r.metrics.results:
                            name = m.name
                            if name not in metrics_2:
                                metrics_2[name] = []
                            metrics_2[name].append(m.score)

                all_metric_names = set(metrics_1.keys()) | set(metrics_2.keys())

                for name in sorted(all_metric_names):
                    avg1 = sum(metrics_1.get(name, [0])) / len(metrics_1.get(name, [1])) if metrics_1.get(name) else 0
                    avg2 = sum(metrics_2.get(name, [0])) / len(metrics_2.get(name, [1])) if metrics_2.get(name) else 0
                    diff = avg1 - avg2

                    if diff > 0.05:
                        winner_icon = "🔵"
                    elif diff < -0.05:
                        winner_icon = "🟢"
                    else:
                        winner_icon = "⚖️"

                    metric_rows.append([name, round(avg1, 4), round(avg2, 4), round(diff, 4), winner_icon])

            # Build result selector choices
            queries = {}
            for r in results_1:
                if r.annotation:
                    queries[r.annotation.query[:50]] = r.annotation_id

            result_choices = [(q, aid) for q, aid in queries.items()]

            return (
                gr.update(value=run_info_data),
                gr.update(value=iface1_stats),
                gr.update(value=iface2_stats),
                comparison_md,
                gr.update(value=metric_rows),
                gr.update(choices=result_choices),
                run,
                results_1,
                results_2,
                0,
            )

        async def load_result_comparison(
            query_id: str,
            results_1: list,
            results_2: list,
            run: Optional[EvaluationRun]
        ):
            """Load comparison for a single result."""
            if not query_id or not results_1 or not results_2:
                return [gr.update() for _ in range(8)]

            # Find results by annotation_id
            result_1 = next((r for r in results_1 if r.annotation_id == query_id), None)
            result_2 = next((r for r in results_2 if r.annotation_id == query_id), None)

            if not result_1 or not result_2:
                return [gr.update() for _ in range(8)]

            query = result_1.annotation.query if result_1.annotation else ""

            answer1 = result_1.rag_response.final_answer if result_1.rag_response else ""
            answer2 = result_2.rag_response.final_answer if result_2.rag_response else ""

            metrics1 = result_1.metrics.to_dict() if result_1.metrics else {}
            metrics2 = result_2.metrics.to_dict() if result_2.metrics else {}

            rag_info1 = result_1.rag_response.to_dict() if result_1.rag_response else {}
            rag_info2 = result_2.rag_response.to_dict() if result_2.rag_response else {}

            return (
                gr.update(value=query),
                gr.update(value=answer1),
                gr.update(value=metrics1),
                gr.update(value=rag_info1),
                gr.update(value=answer2),
                gr.update(value=metrics2),
                gr.update(value=rag_info2),
            )

        async def calculate_statistical_significance(
            results_1: list,
            results_2: list,
            confidence: float
        ):
            """Calculate statistical significance of differences."""
            if not results_1 or not results_2:
                return "❌ 没有足够的数据进行统计检验"

            # Extract scores
            scores_1 = [r.metrics.average_score for r in results_1 if r.success and r.metrics]
            scores_2 = [r.metrics.average_score for r in results_2 if r.success and r.metrics]

            if len(scores_1) < 2 or len(scores_2) < 2:
                return "❌ 样本量不足，需要至少 2 个成功结果"

            import statistics

            mean1 = statistics.mean(scores_1)
            mean2 = statistics.mean(scores_2)
            std1 = statistics.stdev(scores_1) if len(scores_1) > 1 else 0
            std2 = statistics.stdev(scores_2) if len(scores_2) > 1 else 0
            n1 = len(scores_1)
            n2 = len(scores_2)

            # Calculate t-statistic (simplified)
            if std1 == 0 and std2 == 0:
                t_stat = float('inf') if mean1 != mean2 else 0
            else:
                pooled_std = ((std1**2 / n1) + (std2**2 / n2)) ** 0.5
                t_stat = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0

            # Simplified critical value for 95% confidence
            # For proper implementation, use scipy.stats
            critical_values = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            critical = critical_values.get(confidence, 1.96)

            is_significant = t_stat > critical

            result_md = f"""
            ### 📉 统计显著性检验结果

            **检验方法**: 双样本 t 检验

            | 统计量 | 接口 1 | 接口 2 |
            |--------|--------|--------|
            | 样本量 | {n1} | {n2} |
            | 平均分 | {mean1:.4f} | {mean2:.4f} |
            | 标准差 | {std1:.4f} | {std2:.4f} |

            **t 统计量**: {t_stat:.4f}
            **临界值** ({confidence:.0%}): {critical:.4f}

            **结论**: {"✅ 差异具有统计学显著性" if is_significant else "❌ 差异不具有统计学显著性"}

            > 置信水平: {confidence:.0%}
            """

            return result_md

        async def navigate_result(
            direction: int,
            current_idx: int,
            results_1: list,
            results_2: list
        ):
            """Navigate to previous/next result."""
            if not results_1:
                return current_idx, gr.update()

            new_idx = current_idx + direction
            new_idx = max(0, min(new_idx, len(results_1) - 1))

            # Get the annotation_id for the new index
            if results_1[new_idx]:
                new_id = results_1[new_idx].annotation_id
                return new_idx, gr.update(value=new_id)

            return current_idx, gr.update()

        async def export_comparison(
            run: Optional[EvaluationRun],
            results_1: list,
            results_2: list,
            format: str
        ):
            """Export comparison results."""
            if not run:
                return "❌ 请先选择一个评测运行"

            try:
                from pathlib import Path

                iface1 = run.rag_interfaces[0] if run.rag_interfaces else "interface_1"
                iface2 = run.rag_interfaces[1] if len(run.rag_interfaces) > 1 else "interface_2"

                output_path = Path(f"comparison_{run.id[:8]}.{format.lower()}")

                if format.lower() == "json":
                    import json
                    data = {
                        "run_id": run.id,
                        "run_name": run.name,
                        "interfaces": run.rag_interfaces,
                        "summary": {
                            iface1: run.summary_by_interface.get(iface1, {}),
                            iface2: run.summary_by_interface.get(iface2, {}),
                        },
                        "results_count": len(results_1),
                    }
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)

                elif format.lower() == "csv":
                    import csv
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Query", f"{iface1} Score", f"{iface2} Score", "Difference"])
                        for r1, r2 in zip(results_1, results_2):
                            if r1.annotation and r1.metrics and r2.metrics:
                                writer.writerow([
                                    r1.annotation.query,
                                    r1.metrics.average_score,
                                    r2.metrics.average_score,
                                    r1.metrics.average_score - r2.metrics.average_score
                                ])

                elif format.lower() == "markdown":
                    avg1 = run.summary_by_interface.get(iface1, {}).get('average_score', 0)
                    avg2 = run.summary_by_interface.get(iface2, {}).get('average_score', 0)

                    md_content = f"""# RAG 对比报告

## 基本信息
- **运行 ID**: {run.id}
- **运行名称**: {run.name}
- **接口 1**: {iface1}
- **接口 2**: {iface2}

## 总体对比

| 指标 | {iface1} | {iface2} | 差异 |
|------|----------|----------|------|
| 平均分 | {avg1:.2%} | {avg2:.2%} | {(avg1-avg2):+.2%} |
| 成功率 | {run.summary_by_interface.get(iface1, {}).get('success_rate', 0):.1%} | {run.summary_by_interface.get(iface2, {}).get('success_rate', 0):.1%} | - |
| 完成数 | {run.summary_by_interface.get(iface1, {}).get('total', 0)} | {run.summary_by_interface.get(iface2, {}).get('total', 0)} | - |

## 结论
{"接口 1 表现更好" if avg1 > avg2 else "接口 2 表现更好" if avg2 > avg1 else "两者表现相当"}
"""
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(md_content)

                return f"✅ 导出成功：`{output_path}`"

            except Exception as e:
                logger.error(f"Export failed: {e}")
                return f"❌ 导出失败：{str(e)}"

        # ===== Connect Events =====

        refresh_btn.click(
            fn=lambda: run_async(load_dual_runs()),
            outputs=[run_selector],
        )

        run_selector.change(
            fn=lambda rid: run_async(load_comparison_data(rid)),
            inputs=[run_selector],
            outputs=[
                run_info,
                iface1_stats,
                iface2_stats,
                comparison_summary,
                metric_comparison_table,
                result_selector,
                current_run,
                current_results_1,
                current_results_2,
                current_index,
            ],
        )

        result_selector.change(
            fn=lambda qid, r1, r2, run: run_async(load_result_comparison(qid, r1, r2, run)),
            inputs=[result_selector, current_results_1, current_results_2, current_run],
            outputs=[
                query_display,
                iface1_answer,
                iface1_metrics,
                iface1_rag_info,
                iface2_answer,
                iface2_metrics,
                iface2_rag_info,
            ],
        )

        prev_btn.click(
            fn=lambda idx, r1, r2: run_async(navigate_result(-1, idx, r1, r2)),
            inputs=[current_index, current_results_1, current_results_2],
            outputs=[current_index, result_selector],
        )

        next_btn.click(
            fn=lambda idx, r1, r2: run_async(navigate_result(1, idx, r1, r2)),
            inputs=[current_index, current_results_1, current_results_2],
            outputs=[current_index, result_selector],
        )

        calculate_significance_btn.click(
            fn=lambda r1, r2, conf: run_async(calculate_statistical_significance(r1, r2, conf)),
            inputs=[current_results_1, current_results_2, confidence_level],
            outputs=[significance_info],
        )

        export_btn.click(
            fn=lambda run, r1, r2, fmt: run_async(export_comparison(run, r1, r2, fmt)),
            inputs=[current_run, current_results_1, current_results_2, export_format],
            outputs=[export_status],
        )

        # Initial load on tab select
        tab.select(
            fn=lambda: run_async(load_dual_runs()),
            outputs=[run_selector],
        )

    return tab