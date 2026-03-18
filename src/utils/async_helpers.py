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
    Handles both existing event loop and no event loop cases.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already in an async context, create task
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Decorator to convert async function to sync function.
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