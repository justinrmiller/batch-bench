"""Measure throughput, latency and peak memory for one (batch size, workers) configuration."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.signal_handling import _remove_worker_pids
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

from batch_bench.data import ImageDataset, ImageSource, PreprocessMode, build_loader
from batch_bench.devices import Accelerator, Backend, is_oom_error
from batch_bench.hostmem import HostMemory, HostMemoryMonitor
from batch_bench.models import LoadedModel, make_forward, primary_output, surrogate_loss


class Mode(StrEnum):
    """What each step does."""

    INFERENCE = "inference"  # forward pass under torch.inference_mode
    TRAINING = "training"  # forward + backward + AdamW step


class Precision(StrEnum):
    """Autocast precision; AUTO picks the fastest safe choice for the backend."""

    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    @property
    def dtype(self) -> torch.dtype | None:
        """The autocast dtype, or None for full float32."""
        return {Precision.FP16: torch.float16, Precision.BF16: torch.bfloat16}.get(self)


def resolve_precision(requested: Precision, accelerator: Accelerator) -> Precision:
    """Replace AUTO with bf16 where supported, fp16 on MPS or older GPUs, and fp32 on CPU."""
    if requested is not Precision.AUTO:
        return requested
    if accelerator.backend is Backend.CPU:
        return Precision.FP32
    if accelerator.backend is Backend.MPS:
        return Precision.FP16
    return Precision.BF16 if accelerator.supports_bf16() else Precision.FP16


class TrialStatus(StrEnum):
    """Outcome of one trial."""

    OK = "ok"
    OOM = "oom"  # device memory ran out
    HOST_OOM = "host-oom"  # host RAM would run out (skipped) or ran low (stopped)
    ERROR = "error"

    @property
    def out_of_memory(self) -> bool:
        """Whether the batch did not fit, in device memory or in host RAM."""
        return self in (TrialStatus.OOM, TrialStatus.HOST_OOM)


@dataclass(frozen=True)
class TrialSettings:
    """Knobs shared by every trial of a sweep."""

    mode: Mode = Mode.INFERENCE
    precision: Precision = Precision.FP32
    warmup_batches: int = 3
    measure_batches: int = 16
    preprocess: PreprocessMode = PreprocessMode.BATCH
    pin_memory: bool = True
    prefetch_factor: int = 2
    compile_model: bool = False
    cudnn_benchmark: bool = True
    mp_context: str | None = None
    # Share of host RAM to keep free; see TrialRunner.
    host_memory_reserve: float = 0.10

    def __post_init__(self) -> None:
        if self.precision is Precision.AUTO:
            raise ValueError("resolve Precision.AUTO with resolve_precision() first")
        if self.warmup_batches < 0:
            raise ValueError("warmup_batches must be >= 0")
        if self.measure_batches < 1:
            raise ValueError("measure_batches must be >= 1")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be >= 1")
        if not 0.0 <= self.host_memory_reserve < 1.0:
            raise ValueError("host_memory_reserve must be in [0, 1)")


@dataclass(frozen=True)
class TrialResult:
    """Measurements for one (batch size, workers) configuration.

    Attributes:
        samples_per_second: End-to-end images/s, including time spent waiting
            on the DataLoader. This is what a real job achieves.
        compute_samples_per_second: Images/s from the median device step alone;
            the ceiling if data loading were free.
        latency_p50_ms / latency_p95_ms: Device step time per batch (copy to
            device, forward, and backward/optimizer in training mode).
        data_wait_fraction: Share of wall time spent blocked on the DataLoader.
            High values mean the job is input-bound.
        peak_allocated_bytes / peak_reserved_bytes: Peak tensor memory and peak
            memory held by the caching allocator. Reserved is what can OOM.
        peak_host_bytes: How far free host RAM fell during the trial: worker
            processes, queued and pinned batches, plus whatever else the system
            allocated meanwhile. On Apple Silicon this includes GPU memory,
            which is system memory there.
    """

    batch_size: int
    num_workers: int
    status: TrialStatus
    samples_per_second: float | None = None
    compute_samples_per_second: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    data_wait_fraction: float | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    peak_host_bytes: int | None = None
    measured_batches: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the trial completed."""
        return self.status is TrialStatus.OK

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        data = asdict(self)
        data["status"] = str(self.status)
        return data


def effective_batch_counts(warmup: int, measure: int, num_workers: int, prefetch_factor: int) -> tuple[int, int]:
    """Untimed and timed batch counts that keep worker effects out of the timings.

    Warmup gets ``(prefetch_factor + 1) * num_workers + 1`` extra batches. A
    slow warmup step (cuDNN/MIOpen autotuning, ``torch.compile``, lazy
    initialization) lets the workers fill the prefetch queue, up to
    ``prefetch_factor * num_workers`` batches. Timing those pre-built batches
    would overstate an input-bound loader, by about 2x with 8 workers at the
    defaults. The extras empty that queue and run one more round, so timing
    starts from a steady flow wherever in the first ``warmup`` batches the
    slow step fell. They also keep worker start-up out of the timings.

    The measured count rounds up to a multiple of ``num_workers``. Each worker
    builds a whole batch, so an input-bound loader delivers batches in rounds
    of ``num_workers``, and only whole rounds time its true rate.

    With ``num_workers=0`` the main process loads data, nothing is prefetched,
    and the counts are used as given.
    """
    if num_workers == 0:
        return warmup, measure
    drain = (prefetch_factor + 1) * num_workers + 1
    rounds = -(-measure // num_workers)
    return warmup + drain, rounds * num_workers


def kill_workers(batches: object) -> None:
    """SIGKILL the worker processes of a DataLoader iterator, freeing their memory at once.

    The workers are unregistered from DataLoader's SIGCHLD handler first. That
    handler raises in the main thread, at whatever line it is running, for
    every SIGCHLD while a dead worker is registered, so with several workers
    some raises would land outside the trial's error handling. Unregistered,
    the iterator still notices the dead workers itself and raises from
    ``next()``. These are private DataLoader internals, but the only handle on
    the processes.
    """
    if not isinstance(batches, _MultiProcessingDataLoaderIter):
        return  # no workers: the main process loads the data itself
    if batches._worker_pids_set:
        _remove_worker_pids(id(batches))
        batches._worker_pids_set = False
    for worker in batches._workers:
        worker.kill()


def _gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def _describe_error(exc: BaseException, limit: int = 300) -> str:
    lines = str(exc).strip().splitlines()
    first = lines[0] if lines else ""
    return f"{type(exc).__name__}: {first}"[:limit]


Batch = Mapping[str, torch.Tensor]
Step = Callable[[Batch], None]


class TrialRunner:
    """Runs trials for one model on one device.

    Calling the runner with a batch size and worker count builds a fresh
    DataLoader, runs ``warmup + measure`` steps, and returns a
    :class:`TrialResult`. Out-of-memory errors are caught and reported, so a
    sweep can probe until the device is full.

    Host RAM is guarded too, since it can run out before device memory does.
    A batch size projected to need more than is free above
    ``settings.host_memory_reserve`` is skipped, and a trial that cuts into the
    reserve is stopped by killing its DataLoader workers. Both report
    :attr:`TrialStatus.HOST_OOM`, which the sweep treats like a device OOM.
    ``host`` reads system RAM; tests pass a fake.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        accelerator: Accelerator,
        source: ImageSource,
        settings: TrialSettings,
        host: HostMemory | None = None,
    ) -> None:
        self.spec = loaded.spec
        self.processor = loaded.processor
        self.accelerator = accelerator
        self.source = source
        self.settings = settings
        self.host = host or HostMemory()
        # Trials stop once free host RAM falls below this.
        self.host_floor = int(self.host.total() * settings.host_memory_reserve)
        # Largest successful (batch size, peak host bytes) per worker count, for projections.
        self._host_reference: dict[int, tuple[int, int]] = {}
        # Per-sample output shape (e.g. (512,) for an embedding), set by the first step.
        self.output_shape: tuple[int, ...] | None = None
        if accelerator.supports_cudnn_benchmark:
            # Inputs have a fixed shape per trial, so autotuning pays off. On ROCm
            # this drives MIOpen's kernel search, which the warmup batches absorb.
            torch.backends.cudnn.benchmark = settings.cudnn_benchmark
        self.model = loaded.model.to(accelerator.device)

    def __call__(self, batch_size: int, num_workers: int) -> TrialResult:
        """Measure one configuration."""
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        shortfall = self._host_shortfall(batch_size, num_workers)
        if shortfall is not None:
            return TrialResult(
                batch_size=batch_size, num_workers=num_workers, status=TrialStatus.HOST_OOM, error=shortfall
            )
        warmup, measure = effective_batch_counts(
            self.settings.warmup_batches, self.settings.measure_batches, num_workers, self.settings.prefetch_factor
        )
        accelerator = self.accelerator
        pin_memory = self.settings.pin_memory and accelerator.supports_pin_memory
        loader = build_loader(
            ImageDataset(self.source, batch_size * (warmup + measure)),
            self.processor,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=self.settings.preprocess,
            pin_memory=pin_memory,
            prefetch_factor=self.settings.prefetch_factor,
            mp_context=self.settings.mp_context,
        )
        self._release()
        accelerator.reset_peak_memory()

        waits: list[float] = []
        computes: list[float] = []
        started = finished = 0.0
        status, error = TrialStatus.OK, None
        batches: Any = None
        step: Step | None = None

        def stop_workers() -> None:
            # Runs on the monitor thread; the DataLoader then raises on this one.
            kill_workers(batches)

        with HostMemoryMonitor(self.host, self.host_floor, stop_workers) as monitor:
            try:
                step = self._make_step(batch_size)
                batches = iter(loader)
                for index in range(warmup + measure):
                    fetch_start = time.perf_counter()
                    batch = next(batches)
                    fetched = time.perf_counter()
                    step(self._to_device(batch, non_blocking=pin_memory))
                    accelerator.synchronize()
                    accelerator.sample_memory()
                    finished = time.perf_counter()
                    if index == warmup:
                        started = fetch_start
                    if index >= warmup:
                        waits.append(fetched - fetch_start)
                        computes.append(finished - fetched)
                    monitor.sample()
                    if monitor.tripped:
                        break
            except Exception as exc:  # classified, not swallowed: OOM is an expected outcome
                status = TrialStatus.OOM if is_oom_error(exc) else TrialStatus.ERROR
                error = _describe_error(exc)
        if monitor.tripped:  # whatever the killed workers raised, host RAM was the cause
            status = TrialStatus.HOST_OOM
            error = f"stopped: free host RAM fell below the {_gib(self.host_floor)} reserve"
        peak_host = monitor.peak_bytes
        memory = accelerator.peak_memory()
        # Shut DataLoader workers down and free activations and optimizer state.
        batches = None  # rebound, not deleted: stop_workers refers to it
        del loader, step
        self._release()

        if status is not TrialStatus.OK:
            return TrialResult(
                batch_size=batch_size,
                num_workers=num_workers,
                status=status,
                peak_allocated_bytes=memory.allocated,
                peak_reserved_bytes=memory.reserved,
                peak_host_bytes=peak_host,
                error=error,
            )
        reference = self._host_reference.get(num_workers)
        if reference is None or batch_size >= reference[0]:
            self._host_reference[num_workers] = (batch_size, peak_host)
        wall = finished - started
        median_compute = float(np.median(computes))
        return TrialResult(
            batch_size=batch_size,
            num_workers=num_workers,
            status=status,
            samples_per_second=measure * batch_size / wall,
            compute_samples_per_second=batch_size / median_compute,
            latency_p50_ms=median_compute * 1000,
            latency_p95_ms=float(np.percentile(computes, 95)) * 1000,
            data_wait_fraction=sum(waits) / wall,
            peak_allocated_bytes=memory.allocated,
            peak_reserved_bytes=memory.reserved,
            peak_host_bytes=peak_host,
            measured_batches=measure,
        )

    def _host_shortfall(self, batch_size: int, num_workers: int) -> str | None:
        """Why ``batch_size`` would exhaust host RAM, or None if it should fit.

        Queued batches and the workers' decoded images grow linearly with the
        batch size, so the projection scales the largest successful trial with
        the same worker count. Until one exists nothing is skipped, but the
        monitor still stops a trial that runs low.
        """
        reference = self._host_reference.get(num_workers)
        if reference is None or batch_size <= reference[0]:
            return None
        reference_batch, reference_peak = reference
        needed = reference_peak * batch_size // reference_batch
        room = max(0, self.host.available() - self.host_floor)
        if needed <= room:
            return None
        return (
            f"skipped: needs about {_gib(needed)} of host RAM (scaled from batch {reference_batch}), "
            f"{_gib(room)} is free above the {_gib(self.host_floor)} reserve"
        )

    def close(self) -> None:
        """Move the model off the device and free its memory."""
        self.model.to("cpu")
        self._release()

    def _release(self) -> None:
        self.model.zero_grad(set_to_none=True)
        self.accelerator.empty_cache()

    def _to_device(self, batch: Batch, *, non_blocking: bool) -> dict[str, torch.Tensor]:
        device = self.accelerator.device
        return {key: value.to(device, non_blocking=non_blocking) for key, value in batch.items()}

    def _autocast(self) -> AbstractContextManager[Any]:
        dtype = self.settings.precision.dtype
        if dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.accelerator.backend.device_type, dtype=dtype)

    def _check_output(self, output: object, batch_size: int) -> None:
        primary = primary_output(self.spec.task, output)
        if primary.shape[0] != batch_size:
            raise RuntimeError(f"model returned batch dimension {primary.shape[0]}, expected {batch_size}")
        self.output_shape = tuple(primary.shape[1:])

    def _make_step(self, batch_size: int) -> Step:
        settings = self.settings
        if settings.compile_model:
            # Each batch size is a new input shape; resetting avoids hitting the
            # recompile limit, after which torch.compile silently runs eager.
            torch.compiler.reset()
        forward = make_forward(self.spec.task, self.model, compile_model=settings.compile_model)

        if settings.mode is Mode.INFERENCE:
            self.model.eval()

            def inference_step(batch: Batch) -> None:
                with torch.inference_mode(), self._autocast():
                    output = forward(batch)
                self._check_output(output, batch_size)

            return inference_step

        self.model.train()
        # lr=0 keeps the weights identical across trials while still paying the
        # full optimizer cost (AdamW state is 2x the parameter memory).
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0)

        def training_step(batch: Batch) -> None:
            optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                output = forward(batch)
                loss = surrogate_loss(output)
            self._check_output(output, batch_size)
            del output
            loss.backward()
            optimizer.step()

        return training_step
