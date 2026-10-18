from __future__ import annotations

import pytest

from batch_bench.analysis import recommend
from batch_bench.runner import TrialResult, TrialStatus

GIB = 1024**3


def ok(
    batch_size: int,
    throughput: float,
    *,
    workers: int = 4,
    reserved: int | None = None,
    wait: float = 0.0,
    compute: float | None = None,
) -> TrialResult:
    return TrialResult(
        batch_size=batch_size,
        num_workers=workers,
        status=TrialStatus.OK,
        samples_per_second=throughput,
        compute_samples_per_second=compute if compute is not None else throughput,
        data_wait_fraction=wait,
        peak_reserved_bytes=reserved,
    )


def oom(batch_size: int) -> TrialResult:
    return TrialResult(batch_size=batch_size, num_workers=4, status=TrialStatus.OOM)


def test_nothing_to_recommend_without_a_completed_trial():
    assert recommend([]) is None
    assert recommend([oom(1), TrialResult(2, 4, TrialStatus.ERROR, error="x")]) is None


def test_smallest_batch_within_tolerance_wins():
    results = [ok(8, 500), ok(16, 960), ok(32, 1000), ok(64, 1010), oom(128)]
    rec = recommend(results, tolerance=0.05)
    assert rec is not None
    assert rec.fastest.batch_size == 64
    assert rec.recommended.batch_size == 16  # 960 >= 0.95 * 1010
    assert rec.largest_fit == 64
    assert rec.first_oom == 128
    assert rec.notes == ()


def test_zero_tolerance_picks_the_fastest():
    rec = recommend([ok(16, 960), ok(32, 1000)], tolerance=0.0)
    assert rec is not None
    assert rec.recommended.batch_size == 32


def test_ties_prefer_fewer_workers():
    rec = recommend([ok(32, 1000, workers=8), ok(32, 1000, workers=4), oom(64)])
    assert rec is not None
    assert rec.recommended.num_workers == 4


def test_memory_headroom_excludes_nearly_full_runs():
    results = [ok(32, 900, reserved=20 * GIB), ok(64, 1000, reserved=31 * GIB), oom(128)]
    rec = recommend(results, tolerance=0.0, memory_headroom=0.10, total_memory_bytes=32 * GIB)
    assert rec is not None
    assert rec.fastest.batch_size == 64
    assert rec.recommended.batch_size == 32
    assert any("more than 90% of usable memory" in note for note in rec.notes)


def test_unknown_peak_memory_passes_the_headroom_check():
    rec = recommend([ok(32, 1000, reserved=None), oom(64)], total_memory_bytes=GIB)
    assert rec is not None
    assert rec.recommended.batch_size == 32


def test_when_every_run_is_too_full_still_recommend():
    rec = recommend([ok(8, 100, reserved=GIB), oom(16)], total_memory_bytes=GIB)
    assert rec is not None
    assert rec.recommended.batch_size == 8
    assert any("Every run" in note for note in rec.notes)


def test_input_bound_pipelines_are_called_out():
    rec = recommend([ok(32, 400, wait=0.6, compute=1500), oom(64)])
    assert rec is not None
    (note,) = rec.notes
    assert "Input-bound" in note
    assert "60%" in note
    assert "1,500 img/s" in note


def test_missing_oom_suggests_a_larger_sweep():
    rec = recommend([ok(1, 10), ok(2, 30)], max_tested=2)
    assert rec is not None
    assert rec.first_oom is None
    assert rec.notes == (
        "No out-of-memory up to batch 2; raise --max-batch-size (or list larger --batch-sizes) "
        "to find the memory ceiling.",
        "Throughput was still highest at the largest batch tested.",
    )


def test_missing_oom_without_max_tested():
    rec = recommend([ok(2, 30), ok(4, 10)])
    assert rec is not None
    assert rec.notes == (
        "No out-of-memory; raise --max-batch-size (or list larger --batch-sizes) to find the memory ceiling.",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"tolerance": 1.0}, "tolerance"), ({"memory_headroom": -0.1}, "headroom")]
)
def test_arguments_are_validated(kwargs: dict[str, float], message: str):
    with pytest.raises(ValueError, match=message):
        recommend([ok(1, 1)], **kwargs)  # ty: ignore[invalid-argument-type]


def test_to_dict():
    rec = recommend([ok(1, 10), oom(2)])
    assert rec is not None
    data = rec.to_dict()
    assert data["recommended"]["batch_size"] == 1
    assert data["first_oom"] == 2
    assert data["notes"] == []
