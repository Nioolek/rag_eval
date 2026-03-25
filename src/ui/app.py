"""
Main application entry point.
RAG Evaluation System - Modern Professional UI
"""

import atexit
from typing import Optional

import gradio as gr

from ..core.config import get_config
from ..core.logging import logger, setup_logging
from ..evaluation.metrics.metric_registry import get_registry
from ..storage.storage_factory import StorageFactory
from .theme import create_modern_theme, CUSTOM_CSS
from .components.annotation_tab import create_annotation_tab
from .components.statistics_tab import create_statistics_tab
from .components.evaluation_tab import create_evaluation_tab
from .components.results_tab import create_results_tab
from .components.comparison_tab import create_comparison_tab
from .components.scheduler_tab import create_scheduler_tab


async def _cleanup_resources():
    """Clean up resources on application shutdown."""
    try:
        logger.info("🧹 Cleaning up resources...")
        await StorageFactory.close_all()
        logger.info("✅ Resources cleaned up successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")


def _sync_cleanup():
    """Synchronous wrapper for cleanup (called by atexit)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule cleanup as a task
            asyncio.create_task(_cleanup_resources())
        else:
            # Run cleanup directly
            loop.run_until_complete(_cleanup_resources())
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


# Register cleanup on module load
atexit.register(_sync_cleanup)


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application with modern professional theme.

    Returns:
        Gradio Blocks application with enhanced UI
    """
    config = get_config()

    # Note: In Gradio 6.x, theme is passed to launch() not Blocks()
    with gr.Blocks(
        title="RAG 评测系统",
        fill_width=True,
    ) as app:
        # ===== Header =====
        gr.HTML(
            """
                <h1>
                    <span style="font-size: 32px;">📊</span>
                    RAG 评测系统
                </h1>
                <p class="subtitle">
                    企业级 RAG 评估框架 · 支持标注管理、并发评测、多维度指标分析
                </p>
            """,
            elem_classes=["app-header"],
        )

        # ===== Main Navigation Tabs =====
        with gr.Tabs(elem_classes=["main-tabs"]) as tabs:
            # Tab 1: 标注管理
            with gr.TabItem("📝 标注管理", id="annotation"):
                annotation_components = create_annotation_tab()

            # Tab 2: 评测执行
            with gr.TabItem("⚡ 评测执行", id="evaluation"):
                evaluation_components = create_evaluation_tab()

            # Tab 3: 结果查看
            with gr.TabItem("📈 结果查看", id="results"):
                results_components = create_results_tab()

            # Tab 4: 对比分析
            with gr.TabItem("⚖️ 对比分析", id="comparison"):
                comparison_components = create_comparison_tab()

            # Tab 5: 标注统计
            with gr.TabItem("📊 标注统计", id="statistics"):
                statistics_components = create_statistics_tab()

            # Tab 6: 定时任务
            with gr.TabItem("⏰ 定时任务", id="scheduler"):
                scheduler_components = create_scheduler_tab()

        # ===== Page Load Events =====
        # 页面加载时自动初始化数据
        async def _init_annotations():
            result = await annotation_components["load_annotations"](1, 20, "")
            return result  # 返回 (dataframe_update, total_count_update, page_num_update)

        async def _init_results():
            """初始化评测结果表格"""
            return await evaluation_components["load_recent_results"]()

        async def _init_run_selector():
            """初始化结果查看页面的运行列表"""
            return await results_components["load_runs"]()

        async def _init_statistics():
            """初始化标注统计页面"""
            return await statistics_components["load_statistics"]()

        async def _init_comparison():
            """初始化对比分析页面的运行列表"""
            return await comparison_components["load_dual_runs"]()

        async def _init_scheduler():
            """初始化定时任务列表"""
            return await scheduler_components["load_scheduled_tasks"]()

        app.load(
            fn=_init_annotations,
            outputs=[
                annotation_components["annotation_list"],
                annotation_components["total_count"],
                annotation_components.get("page_num"),  # 可选：更新页码
            ],
        )

        # 初始化评测结果表格
        app.load(
            fn=_init_results,
            outputs=[evaluation_components["results_table"]],
        )

        # 初始化结果查看页面的运行列表
        app.load(
            fn=_init_run_selector,
            outputs=[results_components["run_selector"]],
        )

        # 初始化标注统计页面
        app.load(
            fn=_init_statistics,
            outputs=[
                statistics_components["total_card"],
                statistics_components["active_card"],
                statistics_components["faq_card"],
                statistics_components["refusal_card"],
                statistics_components["today_card"],
                statistics_components["week_card"],
                statistics_components["month_card"],
                statistics_components["conv_card"],
                statistics_components["doc_card"],
                statistics_components["language_plot"],
                statistics_components["agent_plot"],
                statistics_components["custom_fields_display"],
            ],
        )

        # 初始化对比分析页面的运行列表
        app.load(
            fn=_init_comparison,
            outputs=[comparison_components["run_selector"]],
        )

        # 初始化定时任务列表
        app.load(
            fn=_init_scheduler,
            outputs=[scheduler_components["task_list"]],
        )

        # ===== Footer =====
        gr.HTML(
            """
                <p>
                    RAG Evaluation System v1.0.0
                    <span style="margin: 0 8px;">•</span>
                    © 2024
                    <span style="margin: 0 8px;">•</span>
                    Built with
                    <a href="https://gradio.app" target="_blank">Gradio</a>
                    <span style="margin: 0 8px;">•</span>
                    <a href="https://github.com" target="_blank">GitHub</a>
                </p>
            """,
            elem_classes=["app-footer"],
        )

        # 注意: tabs.select 事件处理器在 Gradio 6.x 中暂时禁用
        # 如果需要恢复tab选择事件，需要更新函数签名

    # 不调用 queue() - 使用 Gradio 6.x 的默认行为
    return app


def run_app(
    server_name: Optional[str] = None,
    server_port: Optional[int] = None,
    share: Optional[bool] = None,
    debug: bool = False,
) -> None:
    """
    Run the Gradio application.

    Args:
        server_name: Server host address
        server_port: Server port
        share: Whether to create a public link
        debug: Enable debug mode
    """
    # Setup logging
    setup_logging(level="DEBUG" if debug else "INFO")
    logger.info("🚀 Starting RAG Evaluation System...")
    logger.info("✨ Modern UI theme enabled")

    # Initialize metrics registry
    registry = get_registry()
    logger.info(f"📊 Loaded {len(registry)} metrics")

    # Get config
    config = get_config()

    # Create theme
    theme = create_modern_theme()

    # Create app with enhanced theme
    app = create_app()

    # Run server
    host = server_name or config.ui.server_name
    port = server_port or config.ui.server_port
    max_threads = config.ui.max_threads
    logger.info(f"Starting server on http://{host}:{port}")
    logger.info(f"Max threads: {max_threads}")

    app.launch(
        server_name=host,
        server_port=port,
        share=share if share is not None else config.ui.share,
        show_error=True,
        quiet=not debug,
        favicon_path=None,
        max_threads=max_threads,
        theme=theme,
        css=CUSTOM_CSS,
    )


def main():
    """Main entry point with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="📊 RAG Evaluation System - Enterprise RAG Assessment Framework"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Server host address (default: from config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: from config)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    run_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
