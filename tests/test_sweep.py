from __future__ import annotations

from collections.abc import Callable

import pytest

from batch_bench.runner import TrialResult, TrialStatus
from batch_bench.sweep import doubling_batch_sizes, midpoint, parse_int_list, sweep_batch_sizes


def fake_trial(fits_up_to: int, errors: frozenset[int] = frozenset()) -> Callable[[int], TrialResult]:
    """Pretend the device holds batches up to ``fits_up_to`` images."""

    def trial(batch_size: int) -> TrialResult:
        if batch_size in errors:
            return TrialResult(batch_size, 0, TrialStatus.ERROR, error="boom")
        if batch_size > fits_up_to:
            return TrialResult(batch_size, 0, TrialStatus.OOM)
        return TrialResult(batch_size, 0, TrialStatus.OK, samples_per_second=float(batch_size))

    return trial


def sizes(results: list[TrialResult]) -> list[int]:
    return [result.batch_size for result in results]


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((1, 8), [1, 2, 4, 8]),
        ((3, 20), [3, 6, 12]),
        ((5, 5), [5]),
        ((1, 1000), [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
    ],
)
def test_doubling_batch_sizes(bounds: tuple[int, int], expected: list[int]):
    assert doubling_batch_sizes(*bounds) == expected


@pytest.mark.parametrize(("bounds", "message"), [((0, 8), "minimum"), ((8, 4), "maximum")])
def test_doubling_batch_sizes_validates(bounds: tuple[int, int], message: str):
    with pytest.raises(ValueError, match=message):
        doubling_batch_sizes(*bounds)


def test_parse_int_list():
    assert parse_int_list("1, 2,4") == [1, 2, 4]
    assert parse_int_list("0") == [0]
    for bad in ("", "1,,2", "a", "-1", "1.5"):
        with pytest.raises(ValueError, match="comma-separated"):
            parse_int_list(bad)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(256, 512, 384), (384, 512, 448), (8, 16, 12), (16, 32, 24), (24, 32, None), (1, 2, None), (2, 4, 3)],
)
def test_midpoint_snaps_to_multiples_of_eight(low: int, high: int, expected: int | None):
    assert midpoint(low, high) == expected


def test_sweep_stops_at_the_first_oom_then_bisects():
    seen: list[TrialResult] = []
    results = sweep_batch_sizes(
        fake_trial(fits_up_to=300), doubling_batch_sizes(1, 1024), refine_steps=3, on_result=seen.append
    )
    # 1..256 fit, 512 OOMs; bisection tries 384 (OOM), 320 (OOM), 288 (fits).
    assert sizes(results) == [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 384, 320, 288]
    assert seen == results


def test_bisection_stops_when_no_size_is_left_between_the_bounds():
    results = sweep_batch_sizes(fake_trial(fits_up_to=2), [1, 2, 4], refine_steps=10)
    assert sizes(results) == [1, 2, 4, 3]


def test_no_oom_means_no_bisection():
    results = sweep_batch_sizes(fake_trial(fits_up_to=10_000), [1, 2, 4], refine_steps=3)
    assert sizes(results) == [1, 2, 4]


def test_oom_at_the_smallest_size_means_nothing_to_bisect():
    results = sweep_batch_sizes(fake_trial(fits_up_to=0), [1, 2, 4], refine_steps=3)
    assert sizes(results) == [1]
    assert results[0].status is TrialStatus.OOM


def test_errors_do_not_stop_the_sweep():
    results = sweep_batch_sizes(fake_trial(fits_up_to=100, errors=frozenset({2})), [4, 1, 2, 2], refine_steps=0)
    assert sizes(results) == [1, 2, 4]
    assert [result.status for result in results] == [TrialStatus.OK, TrialStatus.ERROR, TrialStatus.OK]


def test_bisection_stops_on_an_error():
    results = sweep_batch_sizes(fake_trial(fits_up_to=300, errors=frozenset({384})), [256, 512], refine_steps=3)
    assert sizes(results) == [256, 512, 384]


def test_refine_steps_must_not_be_negative():
    with pytest.raises(ValueError, match="refine_steps"):
        sweep_batch_sizes(fake_trial(fits_up_to=1), [1], refine_steps=-1)
