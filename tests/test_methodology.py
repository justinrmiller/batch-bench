"""End-to-end checks that the runner measures what theory predicts.

Each pipeline has a known throughput: image loads and model steps sleep for
fixed times, so the expected images/s follows from the worker count. These
guard the timing logic itself, including the worker-burst rounding: without
it, a 4-worker input-bound loader reads about 20% too fast.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import torch
from PIL import Image

from batch_bench.devices import Accelerator, Backend
from batch_bench.models import LoadedModel, ModelSpec, Task
from batch_bench.runner import TrialRunner, TrialSettings

# Sleep-based timings are stable, but CI machines are noisy; real bugs here are 20%+.
TOLERANCE = 0.15


class SlowSource:
    """Each image costs ``delay`` seconds to "decode"."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def __len__(self) -> int:
        return 1

    def load(self, index: int) -> Image.Image:
        time.sleep(self.delay)
        return Image.new("RGB", (8, 8))


class ZerosProcessor:
    def __call__(self, images: Any, return_tensors: str) -> dict[str, torch.Tensor]:
        batch = images if isinstance(images, list) else [images]
        return {"pixel_values": torch.zeros(len(batch), 3, 8, 8)}


class SlowModel(torch.nn.Module):
    """Each step takes ``delay`` seconds, whatever the batch size."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.bias = torch.nn.Parameter(torch.zeros(1))

    # torch types Module.forward as a mutable Callable attribute, so any override trips ty.
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        time.sleep(self.delay)
        return pixel_values.flatten(1)[:, :4] + self.bias


def measure(load_delay: float, step_delay: float, batch_size: int, workers: int) -> tuple[float, float]:
    loaded = LoadedModel(ModelSpec("fake", "fake", Task.EMBEDDING), SlowModel(step_delay), ZerosProcessor())
    settings = TrialSettings(warmup_batches=1, measure_batches=5)
    runner = TrialRunner(loaded, Accelerator(Backend.CPU), SlowSource(load_delay), settings)
    result = runner(batch_size=batch_size, num_workers=workers)
    assert result.ok, result.error
    assert result.samples_per_second is not None
    assert result.data_wait_fraction is not None
    return result.samples_per_second, result.data_wait_fraction


@pytest.mark.parametrize("workers", [2, 4])
def test_input_bound_throughput_matches_the_workers(workers: int):
    # Each worker builds a 4-image batch in 4 x 20 ms; `workers` of them run in parallel.
    expected = workers / (4 * 0.02) * 4
    throughput, wait = measure(load_delay=0.02, step_delay=0.0, batch_size=4, workers=workers)
    assert throughput == pytest.approx(expected, rel=TOLERANCE)
    assert wait > 0.8


def test_compute_bound_throughput_matches_the_step_time():
    throughput, wait = measure(load_delay=0.0, step_delay=0.05, batch_size=8, workers=2)
    assert throughput == pytest.approx(8 / 0.05, rel=TOLERANCE)
    assert wait < 0.2


def test_without_workers_loading_and_compute_add_up():
    throughput, wait = measure(load_delay=0.01, step_delay=0.04, batch_size=4, workers=0)
    assert throughput == pytest.approx(4 / (4 * 0.01 + 0.04), rel=TOLERANCE)
    assert wait == pytest.approx(0.5, abs=0.15)
