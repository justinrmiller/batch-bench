from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from batch_bench import runner as runner_module
from batch_bench.data import PreprocessMode, SyntheticImageSource
from batch_bench.devices import Accelerator, Backend
from batch_bench.models import LoadedModel, Task, load_model, resolve_model
from batch_bench.runner import (
    Mode,
    Precision,
    TrialResult,
    TrialRunner,
    TrialSettings,
    TrialStatus,
    effective_batch_counts,
    kill_workers,
    resolve_precision,
)

FAST = TrialSettings(warmup_batches=1, measure_batches=2)
GIB = 1024**3


@pytest.fixture
def segmenter(tiny_segmenter_dir: Path) -> LoadedModel:
    return load_model(resolve_model(str(tiny_segmenter_dir)))


def test_precision_dtypes():
    assert Precision.FP32.dtype is None
    assert Precision.FP16.dtype is torch.float16
    assert Precision.BF16.dtype is torch.bfloat16


class TestResolvePrecision:
    def test_explicit_choice_is_kept(self, cpu: Accelerator):
        assert resolve_precision(Precision.FP16, cpu) is Precision.FP16

    def test_auto_on_cpu_and_mps(self, cpu: Accelerator, make_mps_module):
        assert resolve_precision(Precision.AUTO, cpu) is Precision.FP32
        mps = Accelerator(Backend.MPS, module=make_mps_module([]))
        assert resolve_precision(Precision.AUTO, mps) is Precision.FP16

    def test_auto_on_gpus_follows_bf16_support(self, make_device_module):
        modern = Accelerator(Backend.ROCM, module=make_device_module(bf16=True))
        older = Accelerator(Backend.CUDA, module=make_device_module(bf16=False))
        assert resolve_precision(Precision.AUTO, modern) is Precision.BF16
        assert resolve_precision(Precision.AUTO, older) is Precision.FP16


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision": Precision.AUTO}, "resolve_precision"),
        ({"warmup_batches": -1}, "warmup_batches"),
        ({"measure_batches": 0}, "measure_batches"),
        ({"prefetch_factor": 0}, "prefetch_factor"),
        ({"host_memory_reserve": 1.0}, "host_memory_reserve"),
        ({"host_memory_reserve": -0.1}, "host_memory_reserve"),
    ],
)
def test_settings_are_validated(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        TrialSettings(**kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((3, 16, 0, 2), (3, 16)),  # no workers: nothing is prefetched
        ((0, 5, 0, 2), (0, 5)),
        ((3, 16, 1, 2), (7, 16)),  # 3 + (2 + 1) * 1 + 1
        ((0, 5, 1, 2), (4, 5)),
        ((3, 16, 8, 2), (28, 16)),  # 3 + (2 + 1) * 8 + 1
        ((3, 10, 4, 2), (16, 12)),  # measured batches round up to whole rounds of 4
        ((0, 1, 4, 1), (9, 4)),
        ((9, 17, 8, 3), (42, 24)),
    ],
)
def test_effective_batch_counts_drain_the_prefetch_queue_and_time_whole_rounds(
    args: tuple[int, int, int, int], expected: tuple[int, int]
):
    assert effective_batch_counts(*args) == expected


def test_trial_result_to_dict():
    data = TrialResult(batch_size=4, num_workers=2, status=TrialStatus.OOM, error="boom").to_dict()
    assert data["status"] == "oom"
    assert data["batch_size"] == 4
    assert data["error"] == "boom"
    assert data["peak_host_bytes"] is None


def test_device_and_host_oom_both_mean_out_of_memory():
    assert [status for status in TrialStatus if status.out_of_memory] == [TrialStatus.OOM, TrialStatus.HOST_OOM]


class TestInference:
    def test_successful_trial_reports_metrics(self, segmenter, cpu, tiny_source):
        runner = TrialRunner(segmenter, cpu, tiny_source, FAST)
        result = runner(batch_size=2, num_workers=0)
        assert result.status is TrialStatus.OK
        assert result.ok
        assert result.error is None
        assert result.measured_batches == 2
        assert result.samples_per_second is not None
        assert result.samples_per_second > 0
        assert result.compute_samples_per_second is not None
        assert result.compute_samples_per_second >= result.samples_per_second * 0.5
        assert result.latency_p50_ms is not None
        assert result.latency_p95_ms is not None
        assert result.latency_p95_ms >= result.latency_p50_ms
        assert result.data_wait_fraction is not None
        assert 0.0 <= result.data_wait_fraction < 1.0
        assert result.peak_reserved_bytes is None  # the CPU backend has no allocator stats
        assert runner.output_shape == (4, 16, 16)  # num_labels x (64 / 4) x (64 / 4)
        assert not runner.model.training

    def test_worker_processes_and_sample_preprocessing(self, segmenter, cpu, tiny_source):
        settings = replace(FAST, preprocess=PreprocessMode.SAMPLE)
        result = TrialRunner(segmenter, cpu, tiny_source, settings)(batch_size=2, num_workers=2)
        assert result.ok
        assert result.measured_batches == 2  # already a whole round of 2 workers

    def test_spawned_workers_can_unpickle_the_pipeline(self, segmenter, cpu, tiny_source):
        settings = replace(FAST, mp_context="spawn")
        result = TrialRunner(segmenter, cpu, tiny_source, settings)(batch_size=1, num_workers=1)
        assert result.ok, result.error

    def test_bf16_autocast(self, segmenter, cpu, tiny_source):
        settings = replace(FAST, precision=Precision.BF16)
        assert TrialRunner(segmenter, cpu, tiny_source, settings)(batch_size=1, num_workers=0).ok

    def test_batch_size_must_be_positive(self, segmenter, cpu, tiny_source):
        with pytest.raises(ValueError, match="batch_size"):
            TrialRunner(segmenter, cpu, tiny_source, FAST)(batch_size=0, num_workers=0)


class TestTraining:
    def test_training_leaves_weights_unchanged_and_frees_gradients(self, segmenter, cpu, tiny_source):
        before = {name: p.detach().clone() for name, p in segmenter.model.named_parameters()}
        runner = TrialRunner(segmenter, cpu, tiny_source, replace(FAST, mode=Mode.TRAINING))
        result = runner(batch_size=2, num_workers=0)
        assert result.ok, result.error
        assert runner.model.training
        for name, param in runner.model.named_parameters():
            assert torch.equal(param, before[name]), name
            assert param.grad is None, name


def _replace_forward(monkeypatch: pytest.MonkeyPatch, forward: Callable[[Mapping[str, torch.Tensor]], object]) -> None:
    monkeypatch.setattr(runner_module, "make_forward", lambda task, model, compile_model=False: forward)


class TestFailures:
    def test_out_of_memory_is_reported_not_raised(self, segmenter, cpu, tiny_source, monkeypatch):
        def oom(batch: Mapping[str, torch.Tensor]) -> object:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 GiB")

        _replace_forward(monkeypatch, oom)
        result = TrialRunner(segmenter, cpu, tiny_source, FAST)(batch_size=4, num_workers=0)
        assert result.status is TrialStatus.OOM
        assert result.error is not None
        assert result.error.startswith("OutOfMemoryError: CUDA out of memory")
        assert result.samples_per_second is None
        assert result.measured_batches == 0

    def test_other_errors_are_reported_as_errors(self, segmenter, cpu, tiny_source, monkeypatch):
        def broken(batch: Mapping[str, torch.Tensor]) -> object:
            raise RuntimeError("")

        _replace_forward(monkeypatch, broken)
        result = TrialRunner(segmenter, cpu, tiny_source, FAST)(batch_size=1, num_workers=0)
        assert result.status is TrialStatus.ERROR
        assert result.error == "RuntimeError: "

    def test_wrong_batch_dimension_is_an_error(self, segmenter, cpu, tiny_source, monkeypatch):
        _replace_forward(monkeypatch, lambda batch: torch.zeros(1, 4))
        result = TrialRunner(segmenter, cpu, tiny_source, FAST)(batch_size=2, num_workers=0)
        assert result.status is TrialStatus.ERROR
        assert result.error is not None
        assert "batch dimension 1, expected 2" in result.error


def test_compile_resets_dynamo_for_each_trial(segmenter, cpu, tiny_source, monkeypatch):
    resets: list[None] = []
    monkeypatch.setattr(torch.compiler, "reset", lambda: resets.append(None))
    monkeypatch.setattr(torch, "compile", lambda target: target)
    runner = TrialRunner(segmenter, cpu, tiny_source, replace(FAST, compile_model=True))
    assert runner(batch_size=1, num_workers=0).ok
    assert runner(batch_size=2, num_workers=0).ok
    assert len(resets) == 2


def test_gpu_runner_enables_cudnn_benchmark(segmenter, make_device_module, tiny_source, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    # Constructing the runner moves the model to the device; keep it on the CPU.
    monkeypatch.setattr(Accelerator, "device", property(lambda self: torch.device("cpu")))
    TrialRunner(segmenter, Accelerator(Backend.ROCM, module=make_device_module()), tiny_source, TrialSettings())
    assert torch.backends.cudnn.benchmark is True


@pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true:UserWarning")
def test_peak_memory_comes_from_the_accelerator(segmenter, make_device_module, tiny_source, monkeypatch):
    module = make_device_module()
    monkeypatch.setattr(Accelerator, "device", property(lambda self: torch.device("cpu")))
    accel = Accelerator(Backend.CUDA, module=module)
    result = TrialRunner(segmenter, accel, tiny_source, FAST)(batch_size=1, num_workers=0)
    assert result.ok, result.error
    assert (result.peak_allocated_bytes, result.peak_reserved_bytes) == (100, 200)
    names = [name for name, _ in module.calls]
    assert "reset_peak_memory_stats" in names
    # One per warmup + measured batch, and one per cache release before and after the trial.
    assert names.count("synchronize") == 5


class TestHostMemory:
    """Host RAM is measured and guarded; the fake host has 64 GiB, so the default reserve is 6.4 GiB."""

    def test_trial_reports_how_far_free_host_ram_fell(self, segmenter, cpu, tiny_source, make_host_memory):
        host = make_host_memory([40 * GIB, 37 * GIB])
        result = TrialRunner(segmenter, cpu, tiny_source, FAST, host=host)(batch_size=2, num_workers=0)
        assert result.ok, result.error
        assert result.peak_host_bytes == 3 * GIB

    def test_trial_is_stopped_when_free_ram_falls_into_the_reserve(self, segmenter, cpu, tiny_source, make_host_memory):
        host = make_host_memory([40 * GIB, 40 * GIB, 2 * GIB])
        result = TrialRunner(segmenter, cpu, tiny_source, FAST, host=host)(batch_size=2, num_workers=0)
        assert result.status is TrialStatus.HOST_OOM
        assert result.error == "stopped: free host RAM fell below the 6.4 GiB reserve"
        assert result.peak_host_bytes == 38 * GIB
        assert result.samples_per_second is None

    def test_monitor_thread_kills_the_workers_mid_step(
        self, segmenter, cpu, tiny_source, make_host_memory, monkeypatch
    ):
        host = make_host_memory([40 * GIB])
        killed = threading.Event()
        workers: list[Any] = []

        def spy_kill(batches: Any) -> None:
            workers.extend(batches._workers)
            kill_workers(batches)
            killed.set()

        original = runner_module.make_forward

        def make_slow_forward(task, model, compile_model=False):
            forward = original(task, model, compile_model=compile_model)

            def slow_forward(batch):
                host.script([2 * GIB])  # memory runs low while this step is on the device
                assert killed.wait(timeout=5)
                return forward(batch)

            return slow_forward

        monkeypatch.setattr(runner_module, "kill_workers", spy_kill)
        monkeypatch.setattr(runner_module, "make_forward", make_slow_forward)
        result = TrialRunner(segmenter, cpu, tiny_source, FAST, host=host)(batch_size=1, num_workers=2)
        assert result.status is TrialStatus.HOST_OOM, result.error
        assert [worker.exitcode for worker in workers] == [-signal.SIGKILL] * 2

    def test_batches_projected_to_exhaust_host_ram_are_skipped(self, segmenter, cpu, tiny_source, make_host_memory):
        host = make_host_memory([40 * GIB, 30 * GIB])  # batch 2 takes 10 GiB
        runner = TrialRunner(segmenter, cpu, tiny_source, FAST, host=host)
        assert runner(batch_size=2, num_workers=0).ok
        host.script([40 * GIB])
        # Batch 8 would need 4 x 10 GiB, more than the 33.6 GiB above the reserve.
        skipped = runner(batch_size=8, num_workers=0)
        assert skipped.status is TrialStatus.HOST_OOM
        assert skipped.error == (
            "skipped: needs about 40.0 GiB of host RAM (scaled from batch 2), "
            "33.6 GiB is free above the 6.4 GiB reserve"
        )
        assert skipped.peak_host_bytes is None
        # Projections are per worker count, and a smaller multiple still fits.
        assert runner(batch_size=8, num_workers=1).ok
        assert runner(batch_size=4, num_workers=0).ok
        # Projections scale from the largest trial, so a smaller one doesn't replace it.
        assert runner(batch_size=1, num_workers=0).ok
        assert runner._host_reference[0][0] == 4

    def test_zero_reserve_never_stops_a_trial(self, segmenter, cpu, tiny_source, make_host_memory):
        settings = replace(FAST, host_memory_reserve=0.0)
        host = make_host_memory([40 * GIB, 1])
        assert TrialRunner(segmenter, cpu, tiny_source, settings, host=host)(batch_size=1, num_workers=0).ok


def test_kill_workers_leaves_no_signal_to_raise_later():
    batches: Any = iter(DataLoader(TensorDataset(torch.arange(64)), batch_size=2, num_workers=2))
    next(batches)
    workers = list(batches._workers)
    kill_workers(batches)
    # DataLoader's SIGCHLD handler would raise in here if the workers were still registered.
    for _ in range(20):
        time.sleep(0.01)
    for worker in workers:
        worker.join(timeout=5)
    assert [worker.exitcode for worker in workers] == [-signal.SIGKILL] * 2
    del batches  # shutting the iterator down must not raise either
    kill_workers(iter(DataLoader(TensorDataset(torch.arange(2)))))  # single-process loaders have no workers


def test_close_moves_the_model_back_to_cpu(segmenter, cpu, tiny_source: SyntheticImageSource):
    runner = TrialRunner(segmenter, cpu, tiny_source, FAST)
    runner.close()
    assert next(runner.model.parameters()).device.type == "cpu"


def test_embedding_trial(tiny_embedder_dir: Path, cpu: Accelerator, tiny_source: SyntheticImageSource):
    loaded = load_model(resolve_model(str(tiny_embedder_dir)))
    assert loaded.spec.task is Task.EMBEDDING
    runner = TrialRunner(loaded, cpu, tiny_source, FAST)
    assert runner(batch_size=3, num_workers=0).ok
    assert runner.output_shape == (16,)


def test_detection_training_trial(tiny_detector_dir: Path, cpu: Accelerator, tiny_source: SyntheticImageSource):
    loaded = load_model(resolve_model(str(tiny_detector_dir)))
    runner = TrialRunner(loaded, cpu, tiny_source, replace(FAST, mode=Mode.TRAINING))
    result = runner(batch_size=2, num_workers=0)
    assert result.ok, result.error
    assert runner.output_shape == (5, 4)  # detection tokens x (num_labels + no-object)
