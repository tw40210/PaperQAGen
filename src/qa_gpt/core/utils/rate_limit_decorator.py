import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

T = TypeVar("T")
P = ParamSpec("P")


def handle_openai_errors(
    max_retries: int = 3, wait_time: int = 60
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to handle any errors with retry logic.

    This decorator will retry the function when any error occurs.
    This is particularly useful for handling transient errors like network issues,
    rate limits, API errors, and other temporary issues.

    Args:
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.
        wait_time (int, optional): Time to wait between retries in seconds. Defaults to 60.

    Returns:
        Callable: Decorated function with error handling
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        is_coroutine = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    error_type = type(e).__name__
                    print(
                        f"Error {error_type} hit: {str(e)}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}..."
                    )
                    await asyncio.sleep(wait_time)
            return None  # This line should never be reached due to the raise above

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    error_type = type(e).__name__
                    print(
                        f"Error {error_type} hit: {str(e)}. Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(wait_time)
            return None  # This line should never be reached due to the raise above

        return cast(Callable[P, T], async_wrapper if is_coroutine else sync_wrapper)

    return decorator
