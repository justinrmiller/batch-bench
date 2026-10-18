from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console

from batch_bench.analysis import recommend
from batch_bench.devices import Backend, DeviceInfo
from batch_bench.models import PRESETS
from batch_bench.report import (
    CSV_FIELDS,
    ModelReport,
    RunReport,
    environment_info,
    format_bytes,
    render_device,
    render_model,
    render_trial,
    write_reports,
)
from batch_bench.runner import TrialResult, TrialStatus

GIB = 1024**3
DEVICE = DeviceInfo(
    backend=Backend.ROCM,
    index=0,
    name="AMD Radeon AI PRO R9700",
    total_memory_bytes=32 * GIB,
    arch="gfx1201",
    runtime="HIP 7.2",
)


def ok(batch_size: int, throughput: float, reserved: int = GIB) -> TrialResult:
    return TrialResult(
        batch_size=batch_size,
        num_workers=8,
        status=TrialStatus.OK,
        samples_per_second=throughput,
        compute_samples_per_second=throughput * 1.2,
        latency_p50_ms=12.5,
        latency_p95_ms=14.0,
        data_wait_fraction=0.02,
        peak_allocated_bytes=reserved // 2,
        peak_reserved_bytes=reserved,
        measured_batches=16,
    )


@pytest.fixture
def model_report() -> ModelReport:
    results = [
        ok(16, 900.0),
        ok(32, 1000.0, reserved=2 * GIB),
        TrialResult(64, 8, TrialStatus.OOM, error="OutOfMemoryError: HIP out of memory"),
    ]
    return ModelReport(
        spec=PRESETS["segformer-b0"],
        parameters=3_752_694,
        output_shape=(150, 128, 128),
        results=results,
        recommendation=recommend(results, total_memory_bytes=32 * GIB),
    )


def recording_console() -> Console:
    return Console(record=True, width=200, color_system=None)


@pytest.mark.parametrize(
    ("value", "expected"), [(None, "-"), (0, "0 MiB"), (512 * 1024**2, "512 MiB"), (3 * GIB // 2, "1.5 GiB")]
)
def test_format_bytes(value: int | None, expected: str):
    assert format_bytes(value) == expected


def test_render_device_includes_arch_and_warnings():
    console = recording_console()
    render_device(console, replace(DEVICE, warnings=("no gfx1201 kernels",)))
    text = console.export_text()
    assert "AMD Radeon AI PRO R9700" in text
    assert "arch=gfx1201" in text
    assert "memory=32.0 GiB" in text
    assert "warning: no gfx1201 kernels" in text


def test_render_device_for_cpu_omits_gpu_fields():
    console = recording_console()
    render_device(console, DeviceInfo(Backend.CPU, 0, "x86_64", None))
    text = console.export_text()
    assert "backend=cpu" in text
    assert "index=" not in text
    assert "memory=" not in text


def test_render_trial_lines():
    console = recording_console()
    render_trial(console, ok(32, 1234.5))
    render_trial(console, TrialResult(64, 8, TrialStatus.OOM, error="HIP out of memory"))
    render_trial(console, TrialResult(128, 8, TrialStatus.OOM))
    lines = console.export_text().splitlines()
    assert "1,234.5 img/s" in lines[0]
    assert "peak 1.0 GiB" in lines[0]
    assert "oom" in lines[1]
    assert "HIP out of memory" in lines[1]
    assert lines[2].rstrip().endswith("oom")


def test_render_model_marks_the_recommendation(model_report: ModelReport):
    console = recording_console()
    render_model(console, model_report, 32 * GIB)
    text = console.export_text()
    assert "segformer-b0" in text
    assert "recommended" in text
    assert "fastest" in text
    assert "(6%)" in text  # 2 GiB of 32 GiB
    assert "Recommended: batch_size=32, num_workers=8" in text
    assert "Largest batch that fit: 32, first OOM at 64" in text


def test_render_model_without_results():
    console = recording_console()
    render_model(console, ModelReport(spec=PRESETS["clip-vit-b32"]), None)
    assert "No batch size completed" in console.export_text()


def test_render_model_with_a_load_error():
    console = recording_console()
    render_model(console, ModelReport(spec=PRESETS["clip-vit-b32"], error="failed to load: OSError"), None)
    text = console.export_text()
    assert "error: failed to load: OSError" in text
    assert "No batch size completed" not in text


def test_bracketed_text_is_printed_verbatim():
    # Rich would drop "[see docs]" and raise MarkupError on "[/x]" if this text were parsed as markup.
    console = recording_console()
    render_trial(console, TrialResult(8, 4, TrialStatus.ERROR, error="RuntimeError: boom [see docs] [/x]"))
    results = [ok(1, 10.0)]
    report = ModelReport(
        spec=PRESETS["dinov2-small"],
        results=results,
        recommendation=recommend(results),
        error="failed to load: OSError: [Errno 2] /data/[raw]/model",
    )
    render_model(console, report, None)
    render_device(console, replace(DEVICE, name="GPU [0]", warnings=("kernels for [/gfx1201] missing",)))
    text = console.export_text()
    assert "boom [see docs] [/x]" in text
    assert "[Errno 2] /data/[raw]/model" in text
    assert "GPU [0]" in text
    assert "kernels for [/gfx1201] missing" in text


def test_render_model_prints_notes():
    results = [ok(1, 10.0)]
    report = ModelReport(spec=PRESETS["dinov2-small"], results=results, recommendation=recommend(results))
    console = recording_console()
    render_model(console, report, None)
    assert "note: No out-of-memory" in console.export_text()


def test_write_reports(tmp_path: Path, model_report: ModelReport):
    created = datetime(2026, 10, 17, 12, 30, 5, tzinfo=UTC)
    report = RunReport(device=DEVICE, settings={"mode": "inference"}, models=[model_report], created_at=created)
    json_path, csv_path = write_reports(report, tmp_path / "out")
    assert json_path.name == "20261017-123005-rocm0.json"
    assert csv_path.name == "20261017-123005-rocm0.csv"

    data = json.loads(json_path.read_text())
    assert data["interrupted"] is False
    assert data["device"]["arch"] == "gfx1201"
    assert data["settings"] == {"mode": "inference"}
    assert data["environment"] == environment_info()
    (model,) = data["models"]
    assert model["name"] == "segformer-b0"
    assert model["output_shape"] == [150, 128, 128]
    assert model["recommendation"]["recommended"]["batch_size"] == 32
    assert [r["status"] for r in model["results"]] == ["ok", "ok", "oom"]

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == CSV_FIELDS
    assert [row["recommended"] for row in rows] == ["False", "True", "False"]
    assert rows[1]["peak_reserved_mib"] == "2048.0"
    assert rows[2]["error"] == "OutOfMemoryError: HIP out of memory"


def test_reports_started_in_the_same_second_do_not_overwrite(tmp_path: Path):
    created = datetime(2026, 10, 17, 12, 30, 5, tzinfo=UTC)
    first = write_reports(RunReport(device=DEVICE, settings={"run": 1}, created_at=created), tmp_path)
    second = write_reports(RunReport(device=DEVICE, settings={"run": 2}, created_at=created), tmp_path)
    assert second[0].name == "20261017-123005-rocm0-2.json"
    assert json.loads(first[0].read_text())["settings"] == {"run": 1}
    assert json.loads(second[0].read_text())["settings"] == {"run": 2}


def test_model_report_without_recommendation_serializes():
    data = ModelReport(spec=PRESETS["yolos-small"], error="boom").to_dict()
    assert data["recommendation"] is None
    assert data["output_shape"] is None
    assert data["error"] == "boom"
