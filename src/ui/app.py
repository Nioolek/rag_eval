"""
Main application entry point.
"""

import asyncio
from typing import Optional

import gradio as gr

from ..core.config import get_config
from ..core.logging import logger, setup_logging
from ..storage.storage_factory import init_storage
from ..evaluation.metrics.metric_registry import get_registry
from .components.annotation_tab import create_annotation_tab
from .components.statistics_tab import create_statistics_tab
from .components.evaluation_tab import create_evaluation_tab
from .components.results_tab import create_results_tab


def create_app() -> gr.Blocks:
    """
    Create the main Gradio application.

    Returns:
        Gradio Blocks application
    """
    config = get_config()

    with gr.Blocks(
        title="RAG 评测系统",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .markdown-text {
            font-size: 14px;
        }
        """,
    ) as app:
        gr.Markdown(
            """
            # 🎯 RAG 评测系统
            企业级 RAG 系统评测框架 - 支持标注管理、并发评测、多维度指标分析
            """
        )

        with gr.Tabs():
            create_annotation_tab()
            create_statistics_tab()
            create_evaluation_tab()
            create_results_tab()

        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666;">
                RAG Evaluation System v1.0.0 |
                <a href="https://github.com" target="_blank">GitHub</a> |
                Powered by Gradio
            </div>
            """
        )

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
    logger.info("Starting RAG Evaluation System...")

    # Initialize storage
    asyncio.run(init_storage())

    # Initialize metrics registry
    registry = get_registry()
    logger.info(f"Loaded {len(registry)} metrics")

    # Get config
    config = get_config()

    # Create app
    app = create_app()

    # Run
    logger.info(f"Starting server on {server_name or config.ui.server_name}:{server_port or config.ui.server_port}")

    app.launch(
        server_name=server_name or config.ui.server_name,
        server_port=server_port or config.ui.server_port,
        share=share if share is not None else config.ui.share,
        show_error=True,
        quiet=not debug,
    )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Evaluation System")
    parser.add_argument("--host", type=str, help="Server host")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    run_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()