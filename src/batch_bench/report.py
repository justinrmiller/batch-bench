"""Console tables and JSON/CSV reports.

Text that comes from outside this package (error messages, model ids, device
names, paths) goes through :func:`rich.markup.escape` before printing. Rich
would otherwise read ``[...]`` in it as markup, dropping the text or raising
``MarkupError`` mid-run.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from batch_bench import __version__
from batch_bench.analysis import Recommendation
from batch_bench.devices import DeviceInfo
from batch_bench.models import ModelSpec
from batch_bench.runner import TrialResult, TrialStatus

_MIB = 1024**2
_GIB = 1024**3

CSV_FIELDS: tuple[str, ...] = (
    "model",
    "model_id",
    "task",
    "batch_size",
    "num_workers",
    "status",
    "samples_per_second",
    "compute_samples_per_second",
    "latency_p50_ms",
    "latency_p95_ms",
    "data_wait_fraction",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "recommended",
    "error",
)

_STATUS_STYLE: dict[TrialStatus, str] = {
    TrialStatus.OK: "green",
    TrialStatus.OOM: "red",
    TrialStatus.ERROR: "yellow",
}


def format_bytes(value: int | None) -> str:
    """Human-readable size, ``-`` when unknown."""
    if value is None:
        return "-"
    if value >= _GIB:
        return f"{value / _GIB:.1f} GiB"
    return f"{value / _MIB:.0f} MiB"


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _mib(value: int | None) -> float | None:
    return None if value is None else round(value / _MIB, 1)


@dataclass
class ModelReport:
    """Everything measured for one model."""

    spec: ModelSpec
    parameters: int | None = None
    output_shape: tuple[int, ...] | None = None
    results: list[TrialResult] = field(default_factory=list)
    recommendation: Recommendation | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "name": self.spec.name,
            "model_id": self.spec.model_id,
            "task": str(self.spec.task),
            "parameters": self.parameters,
            "output_shape": list(self.output_shape) if self.output_shape is not None else None,
            "error": self.error,
            "results": [result.to_dict() for result in self.results],
            "recommendation": self.recommendation.to_dict() if self.recommendation else None,
        }


@dataclass
class RunReport:
    """A full benchmark run: device, settings, and per-model results."""

    device: DeviceInfo
    settings: dict[str, Any]
    models: list[ModelReport] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    # True when Ctrl-C stopped the run; the report then holds the trials that finished.
    interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "created_at": self.created_at.isoformat(),
            "interrupted": self.interrupted,
            "batch_bench_version": __version__,
            "environment": environment_info(),
            "device": self.device.to_dict(),
            "settings": self.settings,
            "models": [model.to_dict() for model in self.models],
        }


def environment_info() -> dict[str, str]:
    """Versions that affect results."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def render_device(console: Console, info: DeviceInfo) -> None:
    """Print a short description of the device."""
    details = [f"[bold]{escape(info.name)}[/bold]", f"backend={info.backend}"]
    if info.backend.device_type != "cpu":
        details.append(f"index={info.index}")
    if info.total_memory_bytes is not None:
        details.append(f"memory={format_bytes(info.total_memory_bytes)}")
    if info.arch:
        details.append(f"arch={info.arch}")
    if info.runtime:
        details.append(escape(info.runtime))
    console.print("Device: " + "  ".join(details))
    for warning in info.warnings:
        console.print(f"[yellow]warning:[/yellow] {escape(warning)}")


def render_trial(console: Console, result: TrialResult) -> None:
    """One progress line per finished trial."""
    style = _STATUS_STYLE[result.status]
    line = f"  batch {result.batch_size:>5}  workers {result.num_workers:>2}  [{style}]{result.status:<5}[/{style}]"
    if result.ok:
        line += (
            f"  {_fmt(result.samples_per_second, ',.1f')} img/s"
            f"  step p50 {_fmt(result.latency_p50_ms, '.1f')} ms"
            f"  data wait {_fmt(result.data_wait_fraction, '.0%')}"
            f"  peak {format_bytes(result.peak_reserved_bytes)}"
        )
    elif result.error:
        line += f"  {escape(result.error)}"
    console.print(line, highlight=False)


def render_model(console: Console, report: ModelReport, memory_budget_bytes: int | None) -> None:
    """Print the results table and recommendation for one model.

    ``Peak mem`` is shown as a share of ``memory_budget_bytes``: the memory
    this process could actually use, the same figure the headroom rule uses.
    """
    rec = report.recommendation
    title = escape(f"{report.spec.name} ({report.spec.task}, {report.spec.model_id})")
    table = Table(title=title, title_justify="left", header_style="bold")
    for column in ("Batch", "Workers", "Status", "img/s", "Dev img/s", "p50 ms", "p95 ms", "Wait"):
        table.add_column(column, justify="right", no_wrap=True)
    table.add_column("Peak mem", justify="right")
    table.add_column("", justify="left")
    for result in sorted(report.results, key=lambda r: (r.num_workers, r.batch_size)):
        style = _STATUS_STYLE[result.status]
        peak = format_bytes(result.peak_reserved_bytes)
        if memory_budget_bytes and result.peak_reserved_bytes is not None:
            peak += f" ({result.peak_reserved_bytes / memory_budget_bytes:.0%})"
        marks = []
        if rec is not None and result is rec.recommended:
            marks.append("[bold green]recommended[/bold green]")
        if rec is not None and result is rec.fastest:
            marks.append("fastest")
        table.add_row(
            str(result.batch_size),
            str(result.num_workers),
            f"[{style}]{result.status}[/{style}]",
            _fmt(result.samples_per_second, ",.1f"),
            _fmt(result.compute_samples_per_second, ",.1f"),
            _fmt(result.latency_p50_ms, ".1f"),
            _fmt(result.latency_p95_ms, ".1f"),
            _fmt(result.data_wait_fraction, ".0%"),
            peak,
            ", ".join(marks),
        )
    console.print(table)
    if report.error:
        console.print(f"[red]error:[/red] {escape(report.error)}")
    if rec is None:
        if not report.error:
            console.print("[red]No batch size completed; nothing to recommend.[/red]")
        return
    best = rec.recommended
    console.print(
        f"[bold]Recommended:[/bold] batch_size={best.batch_size}, num_workers={best.num_workers} "
        f"({_fmt(best.samples_per_second, ',.1f')} img/s, peak {format_bytes(best.peak_reserved_bytes)})"
    )
    oom = f", first OOM at {rec.first_oom}" if rec.first_oom is not None else ""
    console.print(f"Largest batch that fit: {rec.largest_fit}{oom}")
    for note in rec.notes:
        console.print(f"[yellow]note:[/yellow] {escape(note)}")


def csv_rows(report: RunReport) -> list[dict[str, Any]]:
    """Flatten every trial of every model into CSV rows."""
    return [_csv_row(model, result) for model in report.models for result in model.results]


def _csv_row(model: ModelReport, result: TrialResult) -> dict[str, Any]:
    recommended = model.recommendation.recommended if model.recommendation else None
    return {
        "model": model.spec.name,
        "model_id": model.spec.model_id,
        "task": str(model.spec.task),
        "batch_size": result.batch_size,
        "num_workers": result.num_workers,
        "status": str(result.status),
        "samples_per_second": result.samples_per_second,
        "compute_samples_per_second": result.compute_samples_per_second,
        "latency_p50_ms": result.latency_p50_ms,
        "latency_p95_ms": result.latency_p95_ms,
        "data_wait_fraction": result.data_wait_fraction,
        "peak_allocated_mib": _mib(result.peak_allocated_bytes),
        "peak_reserved_mib": _mib(result.peak_reserved_bytes),
        "recommended": result is recommended,
        "error": result.error,
    }


def write_reports(report: RunReport, output_dir: Path) -> tuple[Path, Path]:
    """Write ``<timestamp>-<backend>.json`` and ``.csv`` into ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.created_at:%Y%m%d-%H%M%S}-{report.device.backend}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows(report))
    return json_path, csv_path
