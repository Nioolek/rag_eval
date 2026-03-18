"""
Statistics tab component for Gradio UI.
"""

import gradio as gr

from ...annotation.statistics import get_statistics
from ...utils.async_helpers import run_async
from ...core.logging import logger


def create_statistics_tab() -> gr.Tab:
    """Create the annotation statistics tab."""

    with gr.Tab("标注统计") as tab:
        gr.Markdown("### 标注数据统计概览")

        with gr.Row():
            refresh_btn = gr.Button("🔄 刷新统计", variant="primary")

        # Basic statistics
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 基础统计")
                total_card = gr.Markdown("加载中...")
            with gr.Column():
                gr.Markdown("#### FAQ & 拒答统计")
                faq_card = gr.Markdown("加载中...")
            with gr.Column():
                gr.Markdown("#### 时间统计")
                time_card = gr.Markdown("加载中...")

        with gr.Row():
            # Language distribution
            with gr.Column():
                gr.Markdown("#### 语言分布")
                language_plot = gr.BarPlot(
                    x="language",
                    y="count",
                    title="语言分布",
                    x_title="语言",
                    y_title="数量",
                )

            # Agent distribution
            with gr.Column():
                gr.Markdown("#### Agent 分布")
                agent_plot = gr.BarPlot(
                    x="agent",
                    y="count",
                    title="Agent 分布",
                    x_title="Agent",
                    y_title="数量",
                )

        with gr.Row():
            # Conversation statistics
            with gr.Column():
                gr.Markdown("#### 对话统计")
                conv_card = gr.Markdown("加载中...")

            # Document statistics
            with gr.Column():
                gr.Markdown("#### 文档统计")
                doc_card = gr.Markdown("加载中...")

        # Custom fields usage
        with gr.Row():
            gr.Markdown("#### 自定义字段使用情况")
            custom_fields_display = gr.JSON(label="自定义字段使用统计")

        async def load_statistics():
            stats_calculator = await get_statistics()
            stats = await stats_calculator.calculate(force_refresh=True)

            # Format basic stats
            total_md = f"""
            **总数:** {stats.total_count}
            - 活跃: {stats.active_count}
            - 已删除: {stats.deleted_count}
            """

            faq_md = f"""
            **FAQ 匹配率:** {stats.faq_matched_rate:.1%}
            - 匹配数: {stats.faq_matched_count}

            **拒答率:** {stats.refusal_rate:.1%}
            - 应拒答数: {stats.should_refuse_count}
            """

            time_md = f"""
            **今日新增:** {stats.created_today}
            **本周新增:** {stats.created_this_week}
            **本月新增:** {stats.created_this_month}
            """

            conv_md = f"""
            **单轮对话:** {stats.single_turn_count}
            **多轮对话:** {stats.multi_turn_count}
            **平均历史长度:** {stats.avg_history_length:.1f}
            """

            doc_md = f"""
            **有 GT 文档:** {stats.with_gt_documents_count}
            **平均 GT 文档数:** {stats.avg_gt_documents:.1f}
            **有标准答案:** {stats.with_standard_answers_count}
            **平均答案数:** {stats.avg_standard_answers:.1f}
            """

            # Prepare plot data
            lang_data = [
                {"language": k, "count": v}
                for k, v in stats.language_distribution.items()
            ]

            agent_data = [
                {"agent": k[:20], "count": v}
                for k, v in stats.agent_distribution.items()
            ]

            return (
                gr.update(value=total_md),
                gr.update(value=faq_md),
                gr.update(value=time_md),
                gr.update(value=conv_md),
                gr.update(value=doc_md),
                gr.update(value=lang_data),
                gr.update(value=agent_data),
                gr.update(value=stats.custom_field_usage),
            )

        # Connect events
        refresh_btn.click(
            fn=lambda: run_async(load_statistics()),
            outputs=[
                total_card, faq_card, time_card, conv_card, doc_card,
                language_plot, agent_plot, custom_fields_display
            ],
        )

        tab.select(
            fn=lambda: run_async(load_statistics()),
            outputs=[
                total_card, faq_card, time_card, conv_card, doc_card,
                language_plot, agent_plot, custom_fields_display
            ],
        )

    return tab