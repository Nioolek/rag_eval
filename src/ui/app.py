"""
Main application entry point.
RAG Evaluation System - Modern Professional UI
"""

import asyncio
from typing import Optional

import gradio as gr

from ..core.config import get_config
from ..core.logging import logger, setup_logging
from ..storage.storage_factory import init_storage
from ..evaluation.metrics.metric_registry import get_registry
from .theme import create_modern_theme, CUSTOM_CSS
from .components.annotation_tab import create_annotation_tab
from .components.statistics_tab import create_statistics_tab
from .components.evaluation_tab import create_evaluation_tab
from .components.results_tab import create_results_tab
from .components.comparison_tab import create_comparison_tab
from .components.scheduler_tab import create_scheduler_tab


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application with modern professional theme.

    Returns:
        Gradio Blocks application with enhanced UI
    """
    config = get_config()
    theme = create_modern_theme()

    with gr.Blocks(
        title="RAG 评测系统",
        theme=theme,
        css=CUSTOM_CSS,
        fill_width=False,
    ) as app:
        # ===== Header =====
        with gr.HTML(elem_classes=["app-header"]):
            gr.HTML("""
                <h1>
                    <span style="font-size: 32px;">📊</span>
                    RAG 评测系统
                </h1>
                <p class="subtitle">
                    企业级 RAG 评估框架 · 支持标注管理、并发评测、多维度指标分析
                </p>
            """)

        # ===== Main Navigation Tabs =====
        with gr.Tabs(elem_classes=["main-tabs"]) as tabs:
            # Tab 1: 标注管理
            with gr.TabItem("📝 标注管理", id="annotation"):
                create_annotation_tab()

            # Tab 2: 评测执行
            with gr.TabItem("⚡ 评测执行", id="evaluation"):
                create_evaluation_tab()

            # Tab 3: 结果查看
            with gr.TabItem("📈 结果查看", id="results"):
                create_results_tab()

            # Tab 4: 对比分析
            with gr.TabItem("⚖️ 对比分析", id="comparison"):
                create_comparison_tab()

            # Tab 5: 标注统计
            with gr.TabItem("📊 标注统计", id="statistics"):
                create_statistics_tab()

            # Tab 6: 定时任务
            with gr.TabItem("⏰ 定时任务", id="scheduler"):
                create_scheduler_tab()

        # ===== Footer =====
        with gr.HTML(elem_classes=["app-footer"]):
            gr.HTML("""
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
            """)

        # ===== Load on tab select =====
        def on_tab_select(evt: gr.SelectData):
            """Handle tab selection events."""
            logger.info(f"Tab selected: {evt.value}")

        tabs.select(fn=on_tab_select)

    # Enable queue for concurrent request handling
    # max_size limits the number of requests waiting in queue
    app.queue(max_size=20)

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

    # Initialize storage
    asyncio.run(init_storage())

    # Initialize metrics registry
    registry = get_registry()
    logger.info(f"📊 Loaded {len(registry)} metrics")

    # Get config
    config = get_config()

    # Create app with enhanced theme
    app = create_app()

    # Run server
    host = server_name or config.ui.server_name
    port = server_port or config.ui.server_port
    max_threads = config.ui.max_threads
    logger.info(f"🌐 Starting server on http://{host}:{port}")
    logger.info(f"🔧 Max threads: {max_threads}")

    app.launch(
        server_name=host,
        server_port=port,
        share=share if share is not None else config.ui.share,
        show_error=True,
        quiet=not debug,
        favicon_path=None,  # Can add custom favicon later
        max_threads=max_threads,
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
