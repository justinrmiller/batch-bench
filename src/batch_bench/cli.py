"""Command-line interface: ``batch-bench run | devices | models``."""

from __future__ import annotations

import argparse
import functools
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from batch_bench import __version__
from batch_bench.analysis import recommend
from batch_bench.data import (
    DEFAULT_IMAGE_SIZE,
    FolderImageSource,
    ImageSize,
    ImageSource,
    PreprocessMode,
    SyntheticImageSource,
    parse_image_size,
)
from batch_bench.devices import (
    VISIBILITY_HINTS,
    Accelerator,
    Backend,
    BackendUnavailableError,
    available_backends,
    build_backend,
    device_count,
    resolve_backend,
)
from batch_bench.models import DEFAULT_PRESETS, PRESETS, ModelSpec, Task, load_model, resolve_model
from batch_bench.report import (
    ModelReport,
    RunReport,
    format_bytes,
    render_device,
    render_model,
    render_trial,
    write_reports,
)
from batch_bench.runner import Mode, Precision, TrialResult, TrialRunner, TrialSettings, resolve_precision
from batch_bench.sweep import doubling_batch_sizes, parse_int_list, sweep_batch_sizes

ALL_TASKS = "all"
# Share of device memory a GPU run may use when --memory-fraction is unset (MPS uses 1.0).
DEFAULT_MEMORY_FRACTION = 0.75

RUN_EXAMPLES = """\
examples:
  batch-bench run                                    default model for each task
  batch-bench run -t embedding --max-batch-size 4096
  batch-bench run -m detr-resnet-50 --mode training
  batch-bench run -m facebook/dinov2-large --image-dir ~/data/val --num-workers 4,8,12

Ctrl-C stops early and still prints and saves the trials that finished.
Exit status: 0 every model got a recommendation, 1 some did not, 2 usage
error, 130 interrupted. See QUICKSTART.md and README.md for more.
"""


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults and keep the examples epilog's line breaks."""


def default_num_workers() -> int:
    """One DataLoader worker per usable CPU, minus one left for the main process (at least 1)."""
    # sched_getaffinity honours CPU pinning (taskset, containers) but is Linux-only.
    affinity = getattr(os, "sched_getaffinity", None)
    cpus = len(affinity(0)) if affinity is not None else os.cpu_count() or 1
    return max(1, cpus - 1)


def default_memory_fraction(backend: Backend, requested: float | None, environ: Mapping[str, str]) -> float | None:
    """The per-process memory cap to apply, if any.

    Other GPUs are capped at :data:`DEFAULT_MEMORY_FRACTION`. A sweep pushes
    the device to out-of-memory on purpose, and when the same GPU drives the
    display, the desktop needs some left: on an R9700 a sweep that filled its
    32 GB crashed GNOME Shell, and every app with it. The cap only covers
    PyTorch's allocator. The ROCm runtime used another 1-4.6 GiB outside it,
    more with DataLoader workers and pinned memory, so at 0.85 detectors came
    within 150 MB of filling the card again.

    MPS lets a process allocate 1.7x Metal's recommended working set by
    default, which is past physical RAM on most Macs. An oversized batch then
    swaps the whole system instead of raising OOM, so MPS is capped at 1.0
    unless the user chose a fraction or set PYTORCH_MPS_HIGH_WATERMARK_RATIO.
    The recommended working set already leaves the system its share.
    """
    if requested is not None:
        return requested
    if backend is Backend.CPU:
        return None
    if backend is Backend.MPS:
        return None if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in environ else 1.0
    return DEFAULT_MEMORY_FRACTION


def memory_budget(total: int | None, fraction: float | None, free: int | None) -> int | None:
    """Bytes this process can actually use; the headroom rule is applied to this.

    That is the device total, scaled by any per-process cap
    (``--memory-fraction``), and no more than was free before the first model
    loaded. Memory held by other processes, such as a desktop session on the
    same GPU, is not available to the benchmark. None when the device has no
    memory figure (CPU).
    """
    if total is None:
        return None
    budget = int(total * fraction) if fraction is not None else total
    return min(budget, free) if free is not None else budget


def _positive_int(text: str) -> int:
    value = _non_negative_int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {text}")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1), got {text}")
    return value


def _memory_fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 < value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be in (0, 1], got {text}")
    return value


def _batch_sizes(text: str) -> list[int]:
    try:
        values = parse_int_list(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("batch sizes must be >= 1")
    return values


def _worker_counts(text: str) -> list[int]:
    try:
        values = parse_int_list(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return list(dict.fromkeys(values))  # drop repeats, keep order


def _image_size(text: str) -> ImageSize:
    try:
        return parse_image_size(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    """The ``batch-bench`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="batch-bench",
        description="Find the optimal DataLoader batch size for HuggingFace vision models on your GPU.",
        epilog="Start with `batch-bench devices`, then `batch-bench run`. See QUICKSTART.md.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices", help="list the accelerators this torch build can use")
    commands.add_parser("models", help="list the model presets")

    run = commands.add_parser(
        "run",
        help="sweep batch sizes and recommend one",
        description="Sweep DataLoader batch sizes for one or more models and recommend one per model.",
        epilog=RUN_EXAMPLES,
        formatter_class=_HelpFormatter,
    )
    models = run.add_argument_group("models")
    models.add_argument(
        "-m",
        "--model",
        action="append",
        default=[],
        help="preset name (see `batch-bench models`), Hub id, or local directory; repeatable",
    )
    models.add_argument(
        "-t",
        "--task",
        action="append",
        default=[],
        choices=[*Task, ALL_TASKS],
        help="without --model: benchmark this task's default preset; with --model: the task "
        "of non-preset checkpoints whose config does not reveal it; repeatable",
    )
    models.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")

    device = run.add_argument_group("device")
    device.add_argument(
        "--device",
        default="auto",
        choices=["auto", *Backend],
        help="backend to use; auto picks the first visible GPU, else the CPU",
    )
    device.add_argument("--device-index", type=_non_negative_int, default=0, help="GPU ordinal")
    device.add_argument(
        "--precision",
        default=Precision.AUTO,
        choices=list(Precision),
        help="autocast precision; auto = bf16 if natively supported, else fp16 (fp32 on CPU)",
    )
    device.add_argument("--mode", default=Mode.INFERENCE, choices=list(Mode), help="what each step does")
    device.add_argument(
        "--memory-fraction",
        type=_memory_fraction,
        default=None,
        help=f"cap the process at this fraction of device memory; unset: {DEFAULT_MEMORY_FRACTION} "
        "(1.0 on MPS), leaving room for a desktop on the same GPU; 1.0 uses all of it",
    )
    device.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    device.add_argument(
        "--no-cudnn-benchmark",
        action="store_true",
        help="disable cuDNN/MIOpen autotuning (shortens first-run warmup on ROCm); "
        "always off for --image-dir without --image-size",
    )

    sweep = run.add_argument_group("sweep")
    sweep.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=None,
        help="explicit comma-separated batch sizes; overrides --min/--max-batch-size",
    )
    sweep.add_argument("--min-batch-size", type=_positive_int, default=1, help="first size of the doubling sweep")
    sweep.add_argument("--max-batch-size", type=_positive_int, default=1024, help="last size of the doubling sweep")
    sweep.add_argument(
        "--refine-steps",
        type=_non_negative_int,
        default=3,
        help="bisection trials between the largest fitting batch and the first OOM",
    )
    sweep.add_argument(
        "--num-workers",
        type=_worker_counts,
        default=[default_num_workers()],
        help="DataLoader worker counts to try, comma-separated (e.g. 4,8,12)",
    )
    sweep.add_argument(
        "--warmup-batches",
        type=_non_negative_int,
        default=3,
        help="untimed batches per trial; with workers, (prefetch factor + 1) x workers + 1 more drain the queue",
    )
    sweep.add_argument(
        "--measure-batches",
        type=_positive_int,
        default=16,
        help="timed batches per trial; with workers, rounded up to a multiple of the worker count",
    )
    sweep.add_argument("--prefetch-factor", type=_positive_int, default=2, help="batches prefetched per worker")
    sweep.add_argument(
        "--tolerance",
        type=_fraction,
        default=0.05,
        help="recommend the smallest batch within this fraction of peak throughput",
    )
    sweep.add_argument(
        "--memory-headroom",
        type=_fraction,
        default=0.10,
        help="fraction of the usable memory (see --memory-fraction) the recommendation leaves free",
    )
    sweep.add_argument(
        "--host-memory-reserve",
        type=_fraction,
        default=0.10,
        help="fraction of host RAM to keep free: batch sizes projected to need more are skipped "
        "and a trial that cuts into it is stopped",
    )

    data = run.add_argument_group("data")
    data.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="benchmark on images from this directory instead of synthetic JPEGs",
    )
    data.add_argument(
        "--image-size",
        type=_image_size,
        default=None,
        help=f"WIDTHxHEIGHT of the synthetic images, {DEFAULT_IMAGE_SIZE[0]}x{DEFAULT_IMAGE_SIZE[1]} if unset; "
        "with --image-dir, resize your images to this size",
    )
    data.add_argument(
        "--preprocess",
        default=PreprocessMode.BATCH,
        choices=list(PreprocessMode),
        help="run the image processor per batch (collate_fn) or per sample (__getitem__)",
    )
    data.add_argument("--no-pin-memory", action="store_true", help="disable pinned host memory")
    data.add_argument(
        "--mp-context",
        default=None,
        choices=["fork", "spawn", "forkserver"],
        help="how DataLoader workers start; unset uses Python's platform default",
    )
    data.add_argument("--seed", type=int, default=0, help="seed for synthetic images and the --image-dir order")

    output = run.add_argument_group("output")
    output.add_argument("--output-dir", type=Path, default=Path("results"), help="where to write JSON/CSV")
    output.add_argument("--no-save", action="store_true", help="print results without writing files")
    return parser


def _expand_tasks(tasks: Sequence[str]) -> list[Task]:
    expanded: list[Task] = []
    for task in tasks:
        for item in list(Task) if task == ALL_TASKS else [Task(task)]:
            if item not in expanded:
                expanded.append(item)
    return expanded


def select_models(names: Sequence[str], tasks: Sequence[str], revision: str | None = None) -> list[ModelSpec]:
    """Resolve ``--model``/``--task`` into the specs to benchmark.

    With no ``--model``, each task's default preset runs (all tasks if none
    is given). With ``--model``, a single ``--task`` applies to non-preset
    checkpoints; otherwise their task is inferred from the config.
    """
    expanded = _expand_tasks(tasks)
    if not names:
        return [PRESETS[DEFAULT_PRESETS[task]] for task in expanded or list(Task)]
    hint = expanded[0] if len(expanded) == 1 else None
    specs: list[ModelSpec] = []
    for name in names:
        spec = resolve_model(name, task=hint, revision=revision)
        if spec not in specs:
            specs.append(spec)
    return specs


def build_source(image_dir: Path | None, image_size: ImageSize | None, seed: int) -> ImageSource:
    """Images to feed the DataLoader: a folder, or synthetic JPEGs."""
    if image_dir is not None:
        return FolderImageSource(image_dir, resize=image_size, seed=seed)
    return SyntheticImageSource(size=image_size or DEFAULT_IMAGE_SIZE, seed=seed)


@dataclass(frozen=True)
class SweepPlan:
    """What to sweep and how to choose, shared by every model in a run."""

    batch_sizes: list[int]
    worker_counts: list[int]
    refine_steps: int
    tolerance: float
    memory_headroom: float
    memory_budget_bytes: int | None = None


def benchmark_model(
    report: ModelReport,
    accelerator: Accelerator,
    source: ImageSource,
    settings: TrialSettings,
    plan: SweepPlan,
    console: Console,
    revision: str | None = None,
) -> None:
    """Load ``report.spec``, sweep every worker count, and fill ``report`` in place.

    Each trial is recorded as soon as it finishes, and the recommendation is
    computed even if the sweep is interrupted, so a Ctrl-C keeps everything
    measured up to that point.
    """
    spec = report.spec
    console.print(f"\n[bold]{escape(spec.name)}[/bold]: loading {escape(spec.model_id)}")
    try:
        loaded = load_model(spec, revision=revision)
        runner = TrialRunner(loaded, accelerator, source, settings)
    except Exception as exc:  # network, gated repo, unsupported architecture, OOM on load
        report.error = f"failed to load: {type(exc).__name__}: {exc}"
        console.print(f"[red]{escape(report.error)}[/red]")
        return
    report.parameters = loaded.num_parameters

    def record(result: TrialResult) -> None:
        report.results.append(result)
        render_trial(console, result)

    try:
        for workers in plan.worker_counts:
            sweep_batch_sizes(
                functools.partial(runner, num_workers=workers),
                plan.batch_sizes,
                refine_steps=plan.refine_steps,
                on_result=record,
            )
    finally:
        runner.close()
        report.output_shape = runner.output_shape
        report.recommendation = recommend(
            report.results,
            tolerance=plan.tolerance,
            memory_headroom=plan.memory_headroom,
            total_memory_bytes=plan.memory_budget_bytes,
            max_tested=max(plan.batch_sizes),
        )


def run_command(args: argparse.Namespace, console: Console) -> int:
    """``batch-bench run``: returns 0 when every model produced a recommendation."""
    backend = resolve_backend(args.device)
    if args.device_index >= device_count(backend):
        raise BackendUnavailableError(f"{backend} device {args.device_index} does not exist")
    accelerator = Accelerator(backend, args.device_index)
    accelerator.activate()
    memory_fraction = default_memory_fraction(backend, args.memory_fraction, os.environ)
    if memory_fraction is not None:
        accelerator.set_memory_fraction(memory_fraction)
    info = accelerator.info()
    render_device(console, info)
    if backend is Backend.CPU and args.device == "auto":
        built_for = build_backend()
        reason = (
            VISIBILITY_HINTS[built_for]
            if built_for is not Backend.CPU
            else "If you have a GPU, install its build: uv sync --extra rocm | cuda | xpu | mps"
        )
        console.print(
            f"[yellow]warning:[/yellow] no GPU is visible to this torch build ({torch.__version__}), "
            f"so the benchmark runs on the CPU. {escape(reason)}"
        )
    total, free = accelerator.total_memory(), accelerator.free_memory()
    budget = memory_budget(total, memory_fraction, free)
    if total and free is not None and free < 0.9 * total:
        console.print(
            f"[yellow]warning:[/yellow] {format_bytes(total - free)} of {format_bytes(total)} is already in use "
            f"(other processes or the display). Recommendations keep their headroom within the "
            f"{format_bytes(budget)} that is free."
        )

    # Folder images of mixed sizes give batches of varying shape for models
    # that keep aspect ratio (e.g. DETR, YOLOS). cuDNN/MIOpen autotune
    # every new shape, so tuning would land inside the timed batches.
    autotune = not args.no_cudnn_benchmark
    if autotune and args.image_dir is not None and args.image_size is None:
        autotune = False
        if accelerator.supports_cudnn_benchmark:
            console.print(
                "[yellow]note:[/yellow] --image-dir without --image-size can give batches of varying shape, "
                "so cuDNN/MIOpen autotuning is off to keep re-tuning out of the timings. "
                "Pass --image-size to fix the input size and keep autotuning on."
            )
    settings = TrialSettings(
        mode=Mode(args.mode),
        precision=resolve_precision(Precision(args.precision), accelerator),
        warmup_batches=args.warmup_batches,
        measure_batches=args.measure_batches,
        preprocess=PreprocessMode(args.preprocess),
        pin_memory=not args.no_pin_memory,
        prefetch_factor=args.prefetch_factor,
        compile_model=args.compile,
        cudnn_benchmark=autotune,
        mp_context=args.mp_context,
        host_memory_reserve=args.host_memory_reserve,
    )
    if args.batch_sizes is None and args.max_batch_size < args.min_batch_size:
        raise ValueError("--max-batch-size must be >= --min-batch-size")
    plan = SweepPlan(
        batch_sizes=args.batch_sizes or doubling_batch_sizes(args.min_batch_size, args.max_batch_size),
        worker_counts=args.num_workers,
        refine_steps=args.refine_steps,
        tolerance=args.tolerance,
        memory_headroom=args.memory_headroom,
        memory_budget_bytes=budget,
    )
    specs = select_models(args.model, args.task, args.revision)
    source = build_source(args.image_dir, args.image_size, args.seed)
    console.print(
        f"Mode: {settings.mode}, precision: {settings.precision}, batch sizes: {plan.batch_sizes}, "
        f"workers: {plan.worker_counts}, preprocess: {settings.preprocess}"
    )

    recorded: dict[str, Any] = {
        **asdict(settings),
        **asdict(plan),
        "image_source": str(args.image_dir) if args.image_dir else "synthetic",
        "image_size": list(args.image_size or DEFAULT_IMAGE_SIZE) if args.image_dir is None else args.image_size,
        "memory_fraction": memory_fraction,
    }
    run_report = RunReport(device=info, settings=recorded)
    exit_code = 0
    for spec in specs:
        model_report = ModelReport(spec=spec)
        run_report.models.append(model_report)
        try:
            benchmark_model(model_report, accelerator, source, settings, plan, console, args.revision)
        except KeyboardInterrupt:
            run_report.interrupted = True
        render_model(console, model_report, budget)
        if model_report.recommendation is None:
            exit_code = 1
        if run_report.interrupted:
            console.print("[yellow]Interrupted: keeping the trials that finished.[/yellow]")
            break

    if not args.no_save:
        json_path, csv_path = write_reports(run_report, args.output_dir)
        console.print(f"\nWrote {escape(str(json_path))} and {escape(str(csv_path))}")
    return 130 if run_report.interrupted else exit_code


def devices_command(console: Console) -> int:
    """``batch-bench devices``."""
    builds = {"cuda": torch.version.cuda, "hip": torch.version.hip, "xpu": getattr(torch.version, "xpu", None)}
    flavours = ", ".join(f"{name} {version}" for name, version in builds.items() if version)
    console.print(f"torch {torch.__version__} ({flavours or 'CPU-only build'})")
    available = available_backends()
    for backend in available:
        for index in range(device_count(backend)):
            render_device(console, Accelerator(backend, index).info())
    built_for = build_backend()
    if built_for not in available:
        console.print(f"[yellow]warning:[/yellow] {escape(VISIBILITY_HINTS[built_for])}")
    return 0


def models_command(console: Console) -> int:
    """``batch-bench models``."""
    defaults = set(DEFAULT_PRESETS.values())
    table = Table(header_style="bold")
    # Names are meant to be copied, so they never wrap; long Hub ids fold instead.
    table.add_column("Preset", no_wrap=True, min_width=max(len(name) for name in PRESETS) + 2)
    table.add_column("Task", no_wrap=True)
    table.add_column("Hub id", overflow="fold")
    table.add_column("Description")
    for spec in PRESETS.values():
        name = f"{spec.name} *" if spec.name in defaults else spec.name
        table.add_row(name, str(spec.task), spec.model_id, spec.description)
    console.print(table)
    console.print("* default for its task. Any Hub id or local directory also works with --model.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``batch-bench`` script."""
    args = build_parser().parse_args(argv)
    console = Console()
    # Loading vision towers from full CLIP/SigLIP checkpoints lists every
    # unused text weight; that noise hides the benchmark output.
    transformers.logging.set_verbosity_error()
    transformers.logging.disable_progress_bar()
    try:
        if args.command == "devices":
            return devices_command(console)
        if args.command == "models":
            return models_command(console)
        return run_command(args, console)
    except (BackendUnavailableError, ValueError, OSError) as exc:
        console.print(f"[red]error:[/red] {escape(str(exc))}")
        return 2
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
        return 130
