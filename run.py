#!/usr/bin/env python
"""
Entry point for development and debugging.
Run this script directly from IDE or command line.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set NO_PROXY to avoid proxy issues with Gradio's internal httpx calls
import os
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

from src.ui.app import main

if __name__ == "__main__":
    main()