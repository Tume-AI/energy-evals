import asyncio
import inspect
from collections.abc import Awaitable, Callable


async def retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Call *fn* up to ``1 + max_retries`` times with exponential backoff.

    Args:
        fn: Async callable to attempt.
        max_retries: Number of retries after the first attempt.
        base_delay: Base delay in seconds; each retry waits ``base_delay * 2**attempt``.
        on_retry: Optional callback invoked before sleeping on each failure.
            Receives ``(attempt, exception, delay_seconds)``.
            May be a plain function or an async coroutine function.
        retryable: Optional predicate; if it returns False for a raised exception,
            that exception is re-raised immediately without retrying (for
            deterministic errors that retrying cannot fix).

    Returns:
        The return value of *fn* on the first successful call.

    Raises:
        The last exception raised by *fn* after all retries are exhausted.
    """
    last_error: BaseException | None = None

    for attempt in range(1 + max_retries):
        try:
            return await fn()
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                raise
            if retryable is not None and not retryable(exc):
                raise
            last_error = exc
            if attempt >= max_retries:
                break

            delay = base_delay * (2 ** attempt)

            if on_retry is not None:
                result = on_retry(attempt, exc, delay)
                if inspect.isawaitable(result):
                    await result
            else:
                await asyncio.sleep(delay)

    raise last_error  # type: ignore[misc]
