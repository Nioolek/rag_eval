"""
UI module: Gradio frontend components.
"""

from .app import create_app, run_app
from .components.annotation_tab import create_annotation_tab
from .components.statistics_tab import create_statistics_tab
from .components.evaluation_tab import create_evaluation_tab
from .components.results_tab import create_results_tab

__all__ = [
    "create_app",
    "run_app",
    "create_annotation_tab",
    "create_statistics_tab",
    "create_evaluation_tab",
    "create_results_tab",
]