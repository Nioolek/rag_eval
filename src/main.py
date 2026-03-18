"""
Main module entry point.
"""

from .ui.app import create_app, run_app, main

__all__ = ["create_app", "run_app", "main"]


if __name__ == "__main__":
    main()