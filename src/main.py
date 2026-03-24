"""
Main module entry point.
Supports both direct execution and module import.
"""

import os
import sys
from pathlib import Path

# Set NO_PROXY to avoid proxy issues with Gradio's internal httpx calls
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

# Allow direct execution by adding project root to path
if __name__ == "__main__" and __package__ is None:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from src.ui.app import main
    main()
else:
    from .ui.app import create_app, run_app, main
    __all__ = ["create_app", "run_app", "main"]