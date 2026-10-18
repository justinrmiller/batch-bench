"""End-to-end checks that the runner measures what theory predicts.

Each pipeline's image loads and model steps sleep, so its throughput follows
from how long those calls take and how many workers run them. The fakes
record how long each call *actually* took, and the expected throughput is
computed from the calls the measured batches needed. Sleeps overshoot by
several times on some machines (e.g. virtualized macOS CI runners), and by
more while worker processes are still starting up, so the runner must be
judged against the work it timed, not the nominal delay or the warmup's.

These guard the timing logic itself. Without whole rounds of batches, a 2-
or 4-worker input-bound loader reads 20-25% too fast. Without draining the
prefetch queue after a slow first step, it reads 45-90% too fast.
"""

from __future__ import annotations

import multiprocessing
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator, MutableSequence, Sequence
from typing import Any

import pytest
import torch
from PIL import Image

from batch_bench.devices import Accelerator, Backend
from batch_bench.models import LoadedModel, ModelSpec, Task
from batch_bench.runner import TrialRunner, TrialSettings, effective_batch_counts

# Real timing bugs here are 20%+; this leaves room for scheduler noise.
TOLERANCE = 0.15
# Input-bound rounds end when their slowest worker does, so jitter can make
# a loader much slower than its average worker, but never faster than its
# fastest one. That bound is physical and only needs room for clock noise.
SLOWER_SLACK = 0.30
FASTER_SLACK = 0.10

SETTINGS = TrialSettings(warmup_batches=1, measure_batches=5)

# (sample index, DataLoader worker id or None for the main process, seconds)
Load = tuple[int, int | None, float]


def timed_sleep(delay: float) -> float:
    start = time.perf_counter()
    time.sleep(delay)
    return time.perf_counter() - start


def worker_id() -> int | None:
    info = torch.utils.data.get_worker_info()
    return None if info is None else info.id


class SlowSource:
    """Each image costs ``delay`` seconds to "decode"; real durations go to ``log``."""

    def __init__(self, delay: float, log: MutableSequence[Load]) -> None:
        self.delay = delay
        self.log = log

    def __len__(self) -> int:
        return 1

    def load(self, index: int) -> Image.Image:
        if self.delay:
            self.log.append((index, worker_id(), timed_sleep(self.delay)))
        return Image.new("RGB", (8, 8))


class ZerosProcessor:
    def __call__(self, images: Any, return_tensors: str) -> dict[str, torch.Tensor]:
        batch = images if isinstance(images, list) else [images]
        return {"pixel_values": torch.zeros(len(batch), 3, 8, 8)}


class SlowModel(torch.nn.Module):
    """Each step takes ``delay`` seconds, whatever the batch size; real durations go to ``log``.

    The first step takes ``first_delay`` more, like a GPU's autotuning or compile step.
    """

    def __init__(self, delay: float, first_delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay
        self.first_delay = first_delay
        self.log: list[float] = []
        self.bias = torch.nn.Parameter(torch.zeros(1))

    # torch types Module.forward as a mutable Callable attribute, so any override trips ty.
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        if self.first_delay:
            time.sleep(self.first_delay)
            self.first_delay = 0.0
        if self.delay:
            self.log.append(timed_sleep(self.delay))
        return pixel_values.flatten(1)[:, :4] + self.bias


@pytest.fixture(scope="module")
def shared_log() -> Iterator[Any]:
    """A list that DataLoader worker processes can append to, under fork or spawn."""
    with multiprocessing.Manager() as manager:
        yield manager.list


def measure(source: SlowSource, model: SlowModel, batch_size: int, workers: int) -> tuple[float, float]:
    loaded = LoadedModel(ModelSpec("fake", "fake", Task.EMBEDDING), model, ZerosProcessor())
    result = TrialRunner(loaded, Accelerator(Backend.CPU), source, SETTINGS)(batch_size=batch_size, num_workers=workers)
    assert result.ok, result.error
    assert result.samples_per_second is not None
    assert result.data_wait_fraction is not None
    return result.samples_per_second, result.data_wait_fraction


def measured_batches(workers: int) -> range:
    """Indices of the batches the runner timed, after its warmup."""
    warmup, batches = effective_batch_counts(
        SETTINGS.warmup_batches, SETTINGS.measure_batches, workers, SETTINGS.prefetch_factor
    )
    return range(warmup, warmup + batches)


def measured_loads(log: Sequence[Load], batch_size: int, workers: int) -> list[Load]:
    """The loads that built the timed batches; warmup loads run slower while workers start."""
    batches = measured_batches(workers)
    return [load for load in log if load[0] // batch_size in batches]


@pytest.mark.parametrize("workers", [2, 4])
# A slow first step lets the workers fill the prefetch queue; timing those
# pre-built batches would overstate the loader.
@pytest.mark.parametrize("first_step", [0.0, 1.0], ids=["steady", "slow-first-step"])
def test_input_bound_throughput_matches_the_workers(workers: int, first_step: float, shared_log: Any):
    log = shared_log()
    model = SlowModel(0.0, first_delay=first_step)
    throughput, wait = measure(SlowSource(0.02, log), model, batch_size=4, workers=workers)
    by_worker: defaultdict[int | None, list[float]] = defaultdict(list)
    for _, worker, seconds in measured_loads(list(log), batch_size=4, workers=workers):
        by_worker[worker].append(seconds)
    assert set(by_worker) == set(range(workers))
    # Each worker loads its batch's images one after another, and `workers`
    # of them run in parallel, so together they deliver workers / load_time img/s.
    rates = [workers / statistics.mean(seconds) for seconds in by_worker.values()]
    assert min(rates) * (1 - SLOWER_SLACK) <= throughput <= max(rates) * (1 + FASTER_SLACK)
    assert wait > 0.8


def test_compute_bound_throughput_matches_the_step_time():
    model = SlowModel(0.05)
    throughput, wait = measure(SlowSource(0.0, []), model, batch_size=8, workers=2)
    steps = model.log[-len(measured_batches(workers=2)) :]
    assert throughput == pytest.approx(8 / statistics.mean(steps), rel=TOLERANCE)
    assert wait < 0.2


def test_without_workers_loading_and_compute_add_up():
    source, model = SlowSource(0.01, []), SlowModel(0.04)
    throughput, wait = measure(source, model, batch_size=4, workers=0)
    loads = [seconds for _, _, seconds in measured_loads(source.log, batch_size=4, workers=0)]
    steps = model.log[-len(measured_batches(workers=0)) :]
    load, step = 4 * statistics.mean(loads), statistics.mean(steps)
    assert throughput == pytest.approx(4 / (load + step), rel=TOLERANCE)
    assert wait == pytest.approx(load / (load + step), abs=0.1)
