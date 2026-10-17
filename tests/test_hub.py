"""Integration tests against real Hub checkpoints.

Deselected by default because they download several GB. Run them with::

    uv run pytest -m hub --no-cov
"""

from __future__ import annotations

import pytest

from batch_bench.data import SyntheticImageSource
from batch_bench.devices import Accelerator, Backend
from batch_bench.models import DEFAULT_PRESETS, PRESETS, Task, load_model
from batch_bench.runner import Mode, TrialRunner, TrialSettings

pytestmark = pytest.mark.hub

SETTINGS = TrialSettings(warmup_batches=0, measure_batches=1)


@pytest.fixture(scope="module")
def source() -> SyntheticImageSource:
    return SyntheticImageSource(pool_size=2)


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_runs_inference(name: str, source: SyntheticImageSource):
    runner = TrialRunner(load_model(PRESETS[name]), Accelerator(Backend.CPU), source, SETTINGS)
    result = runner(batch_size=2, num_workers=0)
    assert result.ok, result.error
    assert runner.output_shape


@pytest.mark.parametrize("task", list(Task))
def test_default_presets_run_training(task: Task, source: SyntheticImageSource):
    spec = PRESETS[DEFAULT_PRESETS[task]]
    settings = TrialSettings(warmup_batches=0, measure_batches=1, mode=Mode.TRAINING)
    runner = TrialRunner(load_model(spec), Accelerator(Backend.CPU), source, settings)
    result = runner(batch_size=2, num_workers=0)
    assert result.ok, result.error
