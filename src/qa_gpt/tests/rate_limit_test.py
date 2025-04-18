import time

import pytest

from src.qa_gpt.core.utils.rate_limit_decorator import handle_openai_errors


class CustomError(Exception):
    """Custom error for testing."""

    pass


class NetworkError(Exception):
    """Mock network error."""

    pass


class TimeoutError(Exception):
    """Mock timeout error."""

    pass


@pytest.mark.asyncio
async def test_async_function_success():
    """Test that async function works normally without errors."""

    @handle_openai_errors(max_retries=3, wait_time=0.1)
    async def test_func():
        return "success"

    result = await test_func()
    assert result == "success"


@pytest.mark.asyncio
async def test_async_function_retries():
    """Test that async function retries on errors."""

    attempts = 0

    @handle_openai_errors(max_retries=3, wait_time=0.1)
    async def test_func():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise CustomError("Test error")
        return "success"

    result = await test_func()
    assert result == "success"
    assert attempts == 3


@pytest.mark.asyncio
async def test_async_function_max_retries():
    """Test that async function raises error after max retries."""

    @handle_openai_errors(max_retries=2, wait_time=0.1)
    async def test_func():
        raise CustomError("Test error")

    with pytest.raises(CustomError):
        await test_func()


@pytest.mark.asyncio
async def test_async_function_wait_time():
    """Test that async function waits the correct amount of time between retries."""

    @handle_openai_errors(max_retries=2, wait_time=0.2)
    async def test_func():
        raise CustomError("Test error")

    start_time = time.time()
    with pytest.raises(CustomError):
        await test_func()
    end_time = time.time()

    # Should wait 0.2 seconds between two attempts
    assert end_time - start_time >= 0.2


def test_sync_function_success():
    """Test that sync function works normally without errors."""

    @handle_openai_errors(max_retries=3, wait_time=0.1)
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"


def test_sync_function_retries():
    """Test that sync function retries on errors."""

    attempts = 0

    @handle_openai_errors(max_retries=3, wait_time=0.1)
    def test_func():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise CustomError("Test error")
        return "success"

    result = test_func()
    assert result == "success"
    assert attempts == 3


def test_sync_function_max_retries():
    """Test that sync function raises error after max retries."""

    @handle_openai_errors(max_retries=2, wait_time=0.1)
    def test_func():
        raise CustomError("Test error")

    with pytest.raises(CustomError):
        test_func()


def test_sync_function_wait_time():
    """Test that sync function waits the correct amount of time between retries."""

    @handle_openai_errors(max_retries=2, wait_time=0.2)
    def test_func():
        raise CustomError("Test error")

    start_time = time.time()
    with pytest.raises(CustomError):
        test_func()
    end_time = time.time()

    # Should wait 0.2 seconds between two attempts
    assert end_time - start_time >= 0.2


@pytest.mark.asyncio
async def test_decorator_preserves_function_metadata():
    """Test that the decorator preserves the original function's metadata."""

    @handle_openai_errors()
    async def test_func(a: int, b: str) -> str:
        """Test function docstring."""
        return f"{a}{b}"

    assert test_func.__name__ == "test_func"
    assert test_func.__doc__ == "Test function docstring."
    assert test_func.__annotations__ == {"a": int, "b": str, "return": str}


def test_decorator_with_different_parameters():
    """Test that the decorator works with different max_retries and wait_time values."""

    @handle_openai_errors(max_retries=5, wait_time=0.1)
    def test_func():
        raise CustomError("Test error")

    start_time = time.time()
    with pytest.raises(CustomError):
        test_func()
    end_time = time.time()

    # Should wait 0.1 seconds between 5 attempts
    assert end_time - start_time >= 0.4  # 4 waits between 5 attempts


@pytest.mark.asyncio
async def test_async_function_with_kwargs():
    """Test that async function properly handles keyword arguments."""

    @handle_openai_errors(max_retries=2, wait_time=0.1)
    async def test_func(*, res_obj=None, messages=None):
        if res_obj is None or messages is None:
            raise ValueError("Missing required keyword arguments")
        return f"success: {res_obj}, {messages}"

    result = await test_func(res_obj="test_obj", messages=["test"])
    assert result == "success: test_obj, ['test']"


def test_sync_function_with_kwargs():
    """Test that sync function properly handles keyword arguments."""

    @handle_openai_errors(max_retries=2, wait_time=0.1)
    def test_func(*, res_obj=None, messages=None):
        if res_obj is None or messages is None:
            raise ValueError("Missing required keyword arguments")
        return f"success: {res_obj}, {messages}"

    result = test_func(res_obj="test_obj", messages=["test"])
    assert result == "success: test_obj, ['test']"


@pytest.mark.asyncio
async def test_all_error_types():
    """Test that the decorator handles all error types."""

    error_types = [
        CustomError("Custom error"),
        NetworkError("Network error"),
        TimeoutError("Timeout error"),
        ValueError("Value error"),
        RuntimeError("Runtime error"),
    ]

    @handle_openai_errors(max_retries=3, wait_time=0.1)
    async def test_func(error_obj):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise type(error_obj)(str(error_obj))
        return "success"

    for error in error_types:
        attempts = 0
        result = await test_func(error)
        assert result == "success"
        assert attempts == 3
