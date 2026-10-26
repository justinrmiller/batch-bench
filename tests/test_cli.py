from __future__ import annotations

import csv
import json
import os
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from batch_bench import cli
from batch_bench.cli import build_parser, build_source, default_num_workers, main, select_models
from batch_bench.data import FolderImageSource, SyntheticImageSource
from batch_bench.devices import Accelerator, Backend
from batch_bench.models import PRESETS, Task

GIB = 1024**3
QUICK = ["--num-workers", "0", "--warmup-batches", "1", "--measure-batches", "2", "--image-size", "64x48"]


def unwrapped(text: str) -> str:
    """Undo Rich's line wrapping, which leaves a trailing space before each break."""
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no accelerator is present so runs are deterministic on any machine."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


class TestParser:
    def test_run_defaults(self):
        args = build_parser().parse_args(["run"])
        assert args.device == "auto"
        assert args.precision == "auto"
        assert args.mode == "inference"
        assert args.batch_sizes is None
        assert (args.min_batch_size, args.max_batch_size) == (1, 1024)
        assert args.num_workers == [default_num_workers()]
        assert args.output_dir == Path("results")
        assert args.host_memory_reserve == pytest.approx(0.1)

    def test_lists_are_parsed(self):
        args = build_parser().parse_args(
            ["run", "--batch-sizes", "8,16", "--num-workers", "0,4", "--image-size", "320x240"]
        )
        assert args.batch_sizes == [8, 16]
        assert build_parser().parse_args(["run", "--num-workers", "4,8,4"]).num_workers == [4, 8]
        assert build_parser().parse_args(["run", "--tolerance", "0.1"]).tolerance == pytest.approx(0.1)
        assert args.num_workers == [0, 4]
        assert args.image_size == (320, 240)

    @pytest.mark.parametrize(
        "argv",
        [
            ["--batch-sizes", "0,2"],
            ["--batch-sizes", "a"],
            ["--num-workers", "x"],
            ["--min-batch-size", "0"],
            ["--min-batch-size", "two"],
            ["--refine-steps", "-1"],
            ["--tolerance", "1.0"],
            ["--tolerance", "lots"],
            ["--memory-fraction", "0"],
            ["--memory-fraction", "all"],
            ["--host-memory-reserve", "1.0"],
            ["--image-size", "big"],
            ["--device", "tpu"],
            ["--task", "classification"],
        ],
    )
    def test_invalid_values_exit(self, argv: list[str], capsys: pytest.CaptureFixture[str]):
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["run", *argv])
        assert excinfo.value.code == 2
        assert "error" in capsys.readouterr().err


def test_default_num_workers_uses_cpu_affinity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert default_num_workers() == 2


def test_default_num_workers_without_affinity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    assert default_num_workers() == 63  # no upper cap
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert default_num_workers() == 1  # never zero: 0 would preprocess in the main process
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert default_num_workers() == 1


class TestMemoryBudget:
    def test_no_device_memory(self):
        assert cli.memory_budget(None, None, None) is None

    def test_whole_device_when_uncapped_and_idle(self):
        assert cli.memory_budget(32 * GIB, None, None) == 32 * GIB

    def test_fraction_caps_the_budget(self):
        assert cli.memory_budget(32 * GIB, 0.5, 30 * GIB) == 16 * GIB

    def test_memory_held_by_other_processes_is_excluded(self):
        assert cli.memory_budget(32 * GIB, None, 20 * GIB) == 20 * GIB
        assert cli.memory_budget(32 * GIB, 0.9, 20 * GIB) == 20 * GIB


class TestDefaultMemoryFraction:
    def test_mps_is_capped_at_the_recommended_working_set(self):
        assert cli.default_memory_fraction(Backend.MPS, None, {}) == 1.0

    def test_explicit_choices_win(self):
        assert cli.default_memory_fraction(Backend.MPS, 0.5, {}) == 0.5
        assert cli.default_memory_fraction(Backend.MPS, None, {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.4"}) is None

    @pytest.mark.parametrize("backend", [Backend.CUDA, Backend.ROCM, Backend.XPU])
    def test_other_gpus_leave_room_for_a_desktop(self, backend: Backend):
        assert cli.default_memory_fraction(backend, None, {}) == cli.DEFAULT_MEMORY_FRACTION == 0.75
        assert cli.default_memory_fraction(backend, 1.0, {}) == 1.0

    def test_the_cpu_is_not_capped(self):
        assert cli.default_memory_fraction(Backend.CPU, None, {}) is None


class TestSelectModels:
    def test_no_arguments_runs_every_task_default(self):
        assert [spec.name for spec in select_models([], [])] == ["detr-resnet-50", "segformer-b0", "clip-vit-b32"]

    def test_task_selects_its_default(self):
        assert [spec.name for spec in select_models([], ["embedding", "embedding"])] == ["clip-vit-b32"]
        assert len(select_models([], ["all"])) == 3

    def test_task_applies_to_custom_checkpoints(self, tiny_embedder_dir: Path):
        (spec,) = select_models([str(tiny_embedder_dir)], ["image-segmentation"])
        assert spec.task is Task.IMAGE_SEGMENTATION

    def test_presets_keep_their_task_and_duplicates_collapse(self):
        specs = select_models(["segformer-b0", "segformer-b0", "dinov2-small"], ["object-detection", "embedding"])
        assert specs == [PRESETS["segformer-b0"], PRESETS["dinov2-small"]]


def test_build_source(tmp_path: Path):
    synthetic = build_source(None, None, seed=1)
    assert isinstance(synthetic, SyntheticImageSource)
    assert synthetic.size == (640, 480)
    from PIL import Image

    Image.new("RGB", (8, 8)).save(tmp_path / "x.png")
    folder = build_source(tmp_path, (4, 4), seed=0)
    assert isinstance(folder, FolderImageSource)
    assert folder.resize == (4, 4)


def test_models_command_lists_presets(capsys: pytest.CaptureFixture[str]):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    for name in PRESETS:
        assert name in out


def test_devices_command(capsys: pytest.CaptureFixture[str]):
    assert main(["devices"]) == 0
    out = capsys.readouterr().out
    assert f"torch {torch.__version__}" in out
    assert "backend=cpu" in out


def test_run_end_to_end(tiny_segmenter_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    code = main(["run", "-m", str(tiny_segmenter_dir), "--batch-sizes", "1,2", *QUICK, "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "Recommended: batch_size=" in out
    assert "no GPU is visible" in unwrapped(out)

    (json_path,) = tmp_path.glob("*-cpu0.json")
    data = json.loads(json_path.read_text())
    assert data["settings"]["batch_sizes"] == [1, 2]
    assert data["settings"]["image_size"] == [64, 48]
    assert data["settings"]["precision"] == "fp32"
    assert data["settings"]["host_memory_reserve"] == pytest.approx(0.1)
    (model,) = data["models"]
    assert model["task"] == "image-segmentation"
    assert model["parameters"] > 0
    assert [r["status"] for r in model["results"]] == ["ok", "ok"]
    assert all(r["peak_host_bytes"] is not None for r in model["results"])
    assert model["recommendation"]["recommended"]["batch_size"] in (1, 2)

    (csv_path,) = tmp_path.glob("*-cpu0.csv")
    with csv_path.open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_run_training_with_folder_images(tiny_detector_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    for index, size in enumerate([(80, 60), (60, 80)]):
        Image.new("RGB", size, (index * 100, 0, 0)).save(images / f"{index}.jpg")
    argv = ["run", "-m", str(tiny_detector_dir), "--mode", "training", "--min-batch-size", "1", "--max-batch-size", "2"]
    argv += ["--num-workers", "0", "--warmup-batches", "0", "--measure-batches", "1", "--image-dir", str(images)]
    argv += ["--preprocess", "sample", "--no-save"]
    assert main(argv) == 0
    assert "Recommended:" in capsys.readouterr().out


def test_run_reports_load_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    def unreachable(spec: object, revision: str | None = None) -> object:
        raise OSError("We couldn't connect to 'https://huggingface.co'")

    monkeypatch.setattr(cli, "load_model", unreachable)
    code = main(["run", "-m", "segformer-b0", *QUICK, "--batch-sizes", "1", "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "failed to load: OSError" in out
    (json_path,) = tmp_path.glob("*.json")
    assert json.loads(json_path.read_text())["models"][0]["error"].startswith("failed to load")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--device", "rocm"], "rocm is not available"),
        (["--device-index", "3"], "cpu device 3 does not exist"),
        (["--min-batch-size", "8", "--max-batch-size", "4"], "--max-batch-size must be >="),
        (["--memory-fraction", "0.5"], "CPU backend has no device memory"),
        (["--image-dir", "does-not-exist-anywhere"], "no images"),
    ],
)
def test_run_usage_errors(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]):
    assert main(["run", "-m", "segformer-b0", "--no-save", *argv]) == 2
    assert message in unwrapped(capsys.readouterr().out)


def test_interrupt_keeps_finished_trials(
    tiny_segmenter_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    real_sweep = cli.sweep_batch_sizes

    def interrupted_sweep(trial, batch_sizes, *, refine_steps, on_result):
        real_sweep(trial, batch_sizes[:1], refine_steps=0, on_result=on_result)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "sweep_batch_sizes", interrupted_sweep)
    argv = ["run", "-m", str(tiny_segmenter_dir), "-m", "segformer-b0", "--batch-sizes", "1,2", *QUICK]
    code = main([*argv, "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 130
    assert "Interrupted: keeping the trials that finished" in out
    assert "Recommended: batch_size=1" in out
    (json_path,) = tmp_path.glob("*.json")
    data = json.loads(json_path.read_text())
    assert data["interrupted"] is True
    (model,) = data["models"]  # the second model never started
    assert [r["batch_size"] for r in model["results"]] == [1]
    assert model["recommendation"]["recommended"]["batch_size"] == 1


def test_busy_gpu_shrinks_the_memory_budget(
    tiny_segmenter_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(Accelerator, "total_memory", lambda self: 32 * GIB)
    monkeypatch.setattr(Accelerator, "free_memory", lambda self: 20 * GIB)
    budgets: list[int | None] = []
    make_runner = cli.TrialRunner

    def spy(*args: Any, **kwargs: Any) -> Any:
        budgets.append(kwargs.get("device_budget"))
        return make_runner(*args, **kwargs)

    monkeypatch.setattr(cli, "TrialRunner", spy)
    code = main(["run", "-m", str(tiny_segmenter_dir), "--batch-sizes", "1", *QUICK, "--output-dir", str(tmp_path)])
    out = unwrapped(capsys.readouterr().out)
    assert code == 0
    assert "12.0 GiB of 32.0 GiB is already in use" in out
    (json_path,) = tmp_path.glob("*.json")
    assert json.loads(json_path.read_text())["settings"]["memory_budget_bytes"] == 20 * GIB
    assert budgets == [20 * GIB]  # the same budget guards the sweep


def test_mixed_size_folders_turn_autotuning_off(
    tiny_segmenter_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (40, 30)).save(images / "wide.png")
    Image.new("RGB", (30, 40)).save(images / "tall.png")
    monkeypatch.setattr(Accelerator, "supports_cudnn_benchmark", property(lambda self: True))
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    common = ["run", "-m", str(tiny_segmenter_dir), "--batch-sizes", "1", "--num-workers", "0"]
    common += ["--warmup-batches", "0", "--measure-batches", "1", "--image-dir", str(images)]

    assert main([*common, "--output-dir", str(tmp_path / "mixed")]) == 0
    assert "autotuning is off" in unwrapped(capsys.readouterr().out)
    (mixed,) = (tmp_path / "mixed").glob("*.json")
    assert json.loads(mixed.read_text())["settings"]["cudnn_benchmark"] is False

    assert main([*common, "--image-size", "32x32", "--output-dir", str(tmp_path / "fixed")]) == 0
    assert "autotuning is off" not in unwrapped(capsys.readouterr().out)
    (fixed,) = (tmp_path / "fixed").glob("*.json")
    assert json.loads(fixed.read_text())["settings"]["cudnn_benchmark"] is True


def test_cpu_fallback_explains_an_invisible_gpu(
    tiny_segmenter_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(cli, "build_backend", lambda: Backend.ROCM)
    assert main(["run", "-m", str(tiny_segmenter_dir), "--batch-sizes", "1", *QUICK, "--no-save"]) == 0
    assert "no AMD GPU is visible" in unwrapped(capsys.readouterr().out)


def test_devices_explains_an_invisible_gpu(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(cli, "build_backend", lambda: Backend.CUDA)
    assert main(["devices"]) == 0
    assert "no NVIDIA GPU is visible" in unwrapped(capsys.readouterr().out)


def test_errors_with_brackets_are_printed_verbatim(capsys: pytest.CaptureFixture[str]):
    assert main(["run", "-m", "segformer-b0", "--no-save", "--image-dir", "missing-[/dir]"]) == 2
    assert "missing-[/dir]" in capsys.readouterr().out


def test_keyboard_interrupt_exits_130(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    def interrupted(args: object, console: object) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_command", interrupted)
    assert main(["run"]) == 130
    assert "interrupted" in capsys.readouterr().out


def test_python_dash_m_entry_point(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(sys, "argv", ["batch-bench", "models"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("batch_bench", run_name="__main__")
    assert excinfo.value.code == 0
    assert "segformer-b0" in capsys.readouterr().out
