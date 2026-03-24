"""
Async helper utilities.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar('T')


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """
    Run an async coroutine in a sync context.

    For Gradio 6.x, this function should NOT be used directly.
    Instead, define event handlers as async functions directly.

    This is kept for backwards compatibility but will raise an error
    if called from within Gradio's event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In Gradio's async context, we should not block
        # Return a placeholder or raise an error
        raise RuntimeError(
            "run_async() should not be called from Gradio event handlers. "
            "Define your event handler as an async function directly."
        )
    else:
        return asyncio.run(coro)


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Decorator to convert async function to sync function.

    WARNING: This should not be used for Gradio event handlers.
    Use async functions directly instead.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return run_async(func(*args, **kwargs))
    return wrapper


async def gather_with_concurrency(
    tasks: list[Coroutine[Any, Any, T]],
    concurrency: int = 10,
) -> list[T]:
    """
    Run tasks with limited concurrency.

    Args:
        tasks: List of coroutines to run
        concurrency: Maximum concurrent tasks

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_task(task: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await task

    return await asyncio.gather(*[bounded_task(t) for t in tasks])