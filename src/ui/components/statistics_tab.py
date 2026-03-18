"""
Statistics tab component for Gradio UI.
Enhanced with modern stat cards and improved visualizations.
"""

import gradio as gr

from ...annotation.statistics import get_statistics
from ...utils.async_helpers import run_async
from ...core.logging import logger


def create_stat_card(value: str, label: str, icon: str = "📊") -> str:
    """
    Create HTML for a statistics card.
    
    Args:
        value: The stat value to display
        label: The stat label
        icon: Emoji icon for the card
        
    Returns:
        HTML string for the stat card
    """
    return f"""
    <div class="stat-card">
        <div class="stat-value">{icon} {value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """


def create_statistics_tab() -> gr.Tab:
    """Create the annotation statistics tab with enhanced styling."""

    with gr.Tab("📊 标注统计") as tab:
        # Header
        gr.Markdown("""
        ### 📈 标注数据统计概览
        实时查看标注数据的分布情况和质量指标
        """)

        # Refresh button
        with gr.Row():
            refresh_btn = gr.Button(
                "🔄 刷新统计",
                variant="primary",
                size="lg",
            )
            gr.Markdown(
                "*数据自动缓存，点击刷新获取最新统计*",
                elem_classes=["markdown-text", "text-muted"],
            )

        # Key Metrics Cards
        gr.Markdown("#### 🔑 核心指标")
        with gr.Row():
            total_card = gr.HTML(value=create_stat_card("-", "标注总数", "📝"))
            active_card = gr.HTML(value=create_stat_card("-", "活跃标注", "✅"))
            faq_card = gr.HTML(value=create_stat_card("-", "FAQ 匹配率", "🎯"))
            refusal_card = gr.HTML(value=create_stat_card("-", "拒答率", "🚫"))

        # Time-based Stats
        gr.Markdown("#### 📅 时间维度统计")
        with gr.Row():
            today_card = gr.HTML(value=create_stat_card("-", "今日新增", "📌"))
            week_card = gr.HTML(value=create_stat_card("-", "本周新增", "📆"))
            month_card = gr.HTML(value=create_stat_card("-", "本月新增", "📋"))

        # Distribution Charts
        gr.Markdown("#### 📊 数据分布")
        with gr.Row():
            # Language distribution
            with gr.Column(scale=1):
                gr.Markdown("**语言分布**")
                language_plot = gr.BarPlot(
                    x="language",
                    y="count",
                    title="",
                    x_title="语言",
                    y_title="数量",
                    color="#3B82F6",
                    height=300,
                )

            # Agent distribution
            with gr.Column(scale=1):
                gr.Markdown("**Agent 分布**")
                agent_plot = gr.BarPlot(
                    x="agent",
                    y="count",
                    title="",
                    x_title="Agent",
                    y_title="数量",
                    color="#10B981",
                    height=300,
                )

        # Detailed Statistics
        gr.Markdown("#### 📋 详细统计")
        with gr.Row():
            with gr.Column():
                gr.Markdown("**对话统计**")
                conv_card = gr.Markdown("""
                - 单轮对话：-
                - 多轮对话：-
                - 平均历史长度：-
                """)
            
            with gr.Column():
                gr.Markdown("**文档统计**")
                doc_card = gr.Markdown("""
                - 有 GT 文档：-
                - 平均 GT 文档数：-
                - 有标准答案：-
                - 平均答案数：-
                """)

        # Custom Fields
        with gr.Row():
            gr.Markdown("**自定义字段使用情况**")
            custom_fields_display = gr.JSON(
                label="自定义字段",
                value={},
            )

        # Async load function
        async def load_statistics():
            """Load and format statistics."""
            stats_calculator = await get_statistics()
            stats = await stats_calculator.calculate(force_refresh=True)

            # Update key metrics cards
            total_html = create_stat_card(
                f"{stats.total_count:,}",
                "标注总数",
                "📝"
            )
            active_html = create_stat_card(
                f"{stats.active_count:,}",
                "活跃标注",
                "✅"
            )
            faq_html = create_stat_card(
                f"{stats.faq_matched_rate:.1%}",
                "FAQ 匹配率",
                "🎯"
            )
            refusal_html = create_stat_card(
                f"{stats.refusal_rate:.1%}",
                "拒答率",
                "🚫"
            )

            # Update time-based cards
            today_html = create_stat_card(
                f"{stats.created_today:,}",
                "今日新增",
                "📌"
            )
            week_html = create_stat_card(
                f"{stats.created_this_week:,}",
                "本周新增",
                "📆"
            )
            month_html = create_stat_card(
                f"{stats.created_this_month:,}",
                "本月新增",
                "📋"
            )

            # Format conversation stats
            conv_md = f"""
            - **单轮对话：** {stats.single_turn_count:,}
            - **多轮对话：** {stats.multi_turn_count:,}
            - **平均历史长度：** {stats.avg_history_length:.1f} 条
            """

            # Format document stats
            doc_md = f"""
            - **有 GT 文档：** {stats.with_gt_documents_count:,}
            - **平均 GT 文档数：** {stats.avg_gt_documents:.1f}
            - **有标准答案：** {stats.with_standard_answers_count:,}
            - **平均答案数：** {stats.avg_standard_answers:.1f}
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
                # Key metrics
                gr.update(value=total_html),
                gr.update(value=active_html),
                gr.update(value=faq_html),
                gr.update(value=refusal_html),
                # Time stats
                gr.update(value=today_html),
                gr.update(value=week_html),
                gr.update(value=month_html),
                # Detailed stats
                gr.update(value=conv_md),
                gr.update(value=doc_md),
                # Plots
                gr.update(value=lang_data),
                gr.update(value=agent_data),
                # Custom fields
                gr.update(value=stats.custom_field_usage),
            )

        # Connect events
        refresh_btn.click(
            fn=lambda: run_async(load_statistics()),
            outputs=[
                # Key metrics
                total_card, active_card, faq_card, refusal_card,
                # Time stats
                today_card, week_card, month_card,
                # Detailed stats
                conv_card, doc_card,
                # Plots
                language_plot, agent_plot,
                # Custom fields
                custom_fields_display,
            ],
        )

        # Load on tab select
        tab.select(
            fn=lambda: run_async(load_statistics()),
            outputs=[
                # Key metrics
                total_card, active_card, faq_card, refusal_card,
                # Time stats
                today_card, week_card, month_card,
                # Detailed stats
                conv_card, doc_card,
                # Plots
                language_plot, agent_plot,
                # Custom fields
                custom_fields_display,
            ],
        )

    return tab
