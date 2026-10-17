from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch

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
    resolve_precision,
)

FAST = TrialSettings(warmup_batches=1, measure_batches=2)


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
    ],
)
def test_settings_are_validated(kwargs: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        TrialSettings(**kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((3, 16, 0), (3, 16)),
        ((3, 16, 1), (3, 16)),
        ((3, 16, 8), (8, 16)),
        ((3, 10, 4), (4, 12)),
        ((0, 1, 4), (4, 4)),
        ((9, 16, 8), (16, 16)),
    ],
)
def test_effective_batch_counts_round_to_whole_worker_bursts(args: tuple[int, int, int], expected: tuple[int, int]):
    assert effective_batch_counts(*args) == expected


def test_trial_result_to_dict():
    data = TrialResult(batch_size=4, num_workers=2, status=TrialStatus.OOM, error="boom").to_dict()
    assert data["status"] == "oom"
    assert data["batch_size"] == 4
    assert data["error"] == "boom"


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
        assert result.measured_batches == 2  # rounded up to whole bursts of 2 workers

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
    assert names.count("synchronize") == 3  # one per warmup + measured batch


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
