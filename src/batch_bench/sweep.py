"""Batch-size search: double until out of memory, then bisect toward the limit."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from batch_bench.runner import TrialResult, TrialStatus

TrialFn = Callable[[int], TrialResult]


def doubling_batch_sizes(minimum: int, maximum: int) -> list[int]:
    """``minimum, 2*minimum, 4*minimum, ...`` up to and including ``maximum``."""
    if minimum < 1:
        raise ValueError("minimum batch size must be >= 1")
    if maximum < minimum:
        raise ValueError("maximum batch size must be >= minimum")
    sizes: list[int] = []
    size = minimum
    while size <= maximum:
        sizes.append(size)
        size *= 2
    return sizes


def parse_int_list(text: str) -> list[int]:
    """Parse ``"1,2,4"`` into ``[1, 2, 4]``; every value must be a non-negative integer."""
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"expected comma-separated non-negative integers, got {text!r}")
        values.append(int(part))
    return values


def midpoint(low: int, high: int) -> int | None:
    """Bisection point between a fitting ``low`` and a failing ``high``.

    From 16 upward the midpoint snaps down to a multiple of 8, the alignment
    tensor cores and matrix engines prefer. Returns None once no new size lies
    strictly between the bounds.
    """
    mid = (low + high) // 2
    if mid >= 16:
        mid -= mid % 8
    return mid if low < mid < high else None


def sweep_batch_sizes(
    trial: TrialFn,
    batch_sizes: Sequence[int],
    *,
    refine_steps: int = 0,
    on_result: Callable[[TrialResult], None] | None = None,
) -> list[TrialResult]:
    """Run ``trial`` over ``batch_sizes`` in ascending order.

    The sweep stops at the first out-of-memory batch size, since larger ones
    cannot fit either. Up to ``refine_steps`` bisection trials then narrow the
    gap between the largest batch that fit and the smallest that did not.
    Non-OOM errors are recorded and the sweep continues.

    Args:
        trial: Runs one batch size and returns its result.
        batch_sizes: Candidate sizes; duplicates are ignored.
        refine_steps: Maximum number of bisection trials after the first OOM.
        on_result: Called with each result as soon as it is available.

    Returns:
        Every result, in the order the trials ran.
    """
    if refine_steps < 0:
        raise ValueError("refine_steps must be >= 0")
    results: list[TrialResult] = []

    def run(size: int) -> TrialResult:
        result = trial(size)
        results.append(result)
        if on_result is not None:
            on_result(result)
        return result

    largest_fit: int | None = None
    first_oom: int | None = None
    for size in sorted(set(batch_sizes)):
        result = run(size)
        if result.status is TrialStatus.OOM:
            first_oom = size
            break
        if result.ok:
            largest_fit = size

    if largest_fit is None or first_oom is None:
        return results
    low, high = largest_fit, first_oom
    for _ in range(refine_steps):
        mid = midpoint(low, high)
        if mid is None:
            break
        result = run(mid)
        if result.ok:
            low = mid
        elif result.status is TrialStatus.OOM:
            high = mid
        else:
            break
    return results
