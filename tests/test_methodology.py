"""End-to-end checks that the runner measures what theory predicts.

Each pipeline's image loads and model steps sleep, so its throughput follows
from how long those calls take and how many workers run them. The fakes
record how long each call *actually* took, and the expected throughput is
computed from those recordings. Sleeps overshoot by several times on some
machines (e.g. virtualized macOS CI runners), and the runner must be judged
against the real work, not the nominal delay.

These guard the timing logic itself, including the worker-burst rounding:
without it, a 4-worker input-bound loader reads about 20% too fast.
"""

from __future__ import annotations

import multiprocessing
import statistics
import time
from collections.abc import Iterator, MutableSequence
from typing import Any

import pytest
import torch
from PIL import Image

from batch_bench.devices import Accelerator, Backend
from batch_bench.models import LoadedModel, ModelSpec, Task
from batch_bench.runner import TrialRunner, TrialSettings

# Real timing bugs here are 20%+; this leaves room for scheduler noise.
TOLERANCE = 0.15


def timed_sleep(delay: float, log: MutableSequence[float]) -> None:
    start = time.perf_counter()
    time.sleep(delay)
    log.append(time.perf_counter() - start)


class SlowSource:
    """Each image costs ``delay`` seconds to "decode"; real durations go to ``log``."""

    def __init__(self, delay: float, log: MutableSequence[float]) -> None:
        self.delay = delay
        self.log = log

    def __len__(self) -> int:
        return 1

    def load(self, index: int) -> Image.Image:
        if self.delay:
            timed_sleep(self.delay, self.log)
        return Image.new("RGB", (8, 8))


class ZerosProcessor:
    def __call__(self, images: Any, return_tensors: str) -> dict[str, torch.Tensor]:
        batch = images if isinstance(images, list) else [images]
        return {"pixel_values": torch.zeros(len(batch), 3, 8, 8)}


class SlowModel(torch.nn.Module):
    """Each step takes ``delay`` seconds, whatever the batch size; real durations go to ``log``."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.log: list[float] = []
        self.bias = torch.nn.Parameter(torch.zeros(1))

    # torch types Module.forward as a mutable Callable attribute, so any override trips ty.
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        if self.delay:
            timed_sleep(self.delay, self.log)
        return pixel_values.flatten(1)[:, :4] + self.bias


@pytest.fixture(scope="module")
def shared_log() -> Iterator[Any]:
    """A list that DataLoader worker processes can append to, under fork or spawn."""
    with multiprocessing.Manager() as manager:
        yield manager.list


def measure(source: SlowSource, model: SlowModel, batch_size: int, workers: int) -> tuple[float, float]:
    loaded = LoadedModel(ModelSpec("fake", "fake", Task.EMBEDDING), model, ZerosProcessor())
    settings = TrialSettings(warmup_batches=1, measure_batches=5)
    result = TrialRunner(loaded, Accelerator(Backend.CPU), source, settings)(batch_size=batch_size, num_workers=workers)
    assert result.ok, result.error
    assert result.samples_per_second is not None
    assert result.data_wait_fraction is not None
    return result.samples_per_second, result.data_wait_fraction


@pytest.mark.parametrize("workers", [2, 4])
def test_input_bound_throughput_matches_the_workers(workers: int, shared_log: Any):
    log = shared_log()
    throughput, wait = measure(SlowSource(0.02, log), SlowModel(0.0), batch_size=4, workers=workers)
    # Each worker loads its batch's images one after another, and `workers`
    # of them run in parallel, so together they deliver workers / load_time img/s.
    expected = workers / statistics.mean(log)
    assert throughput == pytest.approx(expected, rel=TOLERANCE)
    assert wait > 0.8


def test_compute_bound_throughput_matches_the_step_time():
    model = SlowModel(0.05)
    throughput, wait = measure(SlowSource(0.0, []), model, batch_size=8, workers=2)
    assert throughput == pytest.approx(8 / statistics.mean(model.log), rel=TOLERANCE)
    assert wait < 0.2


def test_without_workers_loading_and_compute_add_up():
    source, model = SlowSource(0.01, []), SlowModel(0.04)
    throughput, wait = measure(source, model, batch_size=4, workers=0)
    load, step = 4 * statistics.mean(source.log), statistics.mean(model.log)
    assert throughput == pytest.approx(4 / (load + step), rel=TOLERANCE)
    assert wait == pytest.approx(load / (load + step), abs=0.1)
