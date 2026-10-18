"""Accelerator detection and a uniform wrapper over the per-backend torch APIs.

PyTorch exposes each GPU vendor through a different namespace:

* NVIDIA CUDA and AMD ROCm both use ``torch.cuda``; ROCm builds set ``torch.version.hip``.
* Intel GPUs use ``torch.xpu``.
* Apple Silicon uses ``torch.mps``, which has no peak-memory counters.

:class:`Accelerator` hides those differences behind one small interface so the
benchmark loop never branches on the backend.
"""

from __future__ import annotations

import gc
import platform
import re
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import torch


class Backend(StrEnum):
    """Hardware backends the benchmark can drive."""

    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    MPS = "mps"
    CPU = "cpu"

    @property
    def device_type(self) -> str:
        """The torch device type; ROCm reuses ``cuda`` through HIP."""
        return "cuda" if self is Backend.ROCM else self.value


class BackendUnavailableError(RuntimeError):
    """Raised when the requested backend is not usable with the installed torch build."""


# How to get a working torch build for each backend, shown when it is missing.
INSTALL_HINTS: dict[Backend, str] = {
    Backend.CUDA: "install the NVIDIA build with `uv sync --extra cuda`",
    Backend.ROCM: (
        "install the ROCm build with `uv sync --extra rocm` (Linux only) and make sure "
        "your user is in the `render` and `video` groups"
    ),
    Backend.XPU: ("install the Intel build with `uv sync --extra xpu` and Intel's GPU driver (Level Zero runtime)"),
    Backend.MPS: "use an Apple Silicon Mac with macOS 14+ and `uv sync --extra mps`",
    Backend.CPU: "the CPU backend is always available",
}

# Why a GPU build might not see its GPU, shown when it falls back to the CPU.
VISIBILITY_HINTS: dict[Backend, str] = {
    Backend.ROCM: (
        "The ROCm build is installed but no AMD GPU is visible. Check that your user can open /dev/kfd "
        "and /dev/dri (the `render` and `video` groups), that the amdgpu driver supports the GPU, and "
        "that HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES are not hiding it."
    ),
    Backend.CUDA: (
        "The CUDA build is installed but no NVIDIA GPU is visible. Check `nvidia-smi`, that the driver "
        "is new enough for CUDA 13 (R580+), and that CUDA_VISIBLE_DEVICES is not hiding the GPU."
    ),
    Backend.XPU: (
        "The XPU build is installed but no Intel GPU is visible. Check Intel's GPU driver (Level Zero "
        "runtime) and that ZE_AFFINITY_MASK is not hiding the GPU."
    ),
    Backend.MPS: "MPS is unavailable: it needs an Apple Silicon Mac running macOS 14 or newer.",
}

# Lower-cased substrings that identify an out-of-memory failure. Only some
# backends raise ``torch.OutOfMemoryError``; the rest raise a RuntimeError
# whose message is the only signal.
_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",  # CUDA, HIP, MPS ("MPS backend out of memory"), XPU
    "out_of_device_memory",  # Intel Level Zero / Unified Runtime
    "alloc_failed",  # CUDNN_STATUS_ALLOC_FAILED, CUBLAS_STATUS_ALLOC_FAILED
    "allocfailed",  # miopenStatusAllocFailed (ROCm)
    "can't allocate memory",  # CPU DefaultCPUAllocator
    "larger than 4gb",  # Intel GPUs that cap a single allocation at 4 GiB
)


def is_oom_error(exc: BaseException) -> bool:
    """Return True when ``exc`` means the batch did not fit in device memory."""
    if isinstance(exc, torch.OutOfMemoryError | MemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MARKERS)


def _cuda_like_backend() -> Backend | None:
    """CUDA or ROCm, whichever this torch build drives through ``torch.cuda``."""
    if not torch.cuda.is_available():
        return None
    return Backend.ROCM if torch.version.hip else Backend.CUDA


def _xpu_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


def _mps_available() -> bool:
    return bool(torch.backends.mps.is_available())


def available_backends() -> list[Backend]:
    """Backends usable in this process, most preferred first; CPU is always last."""
    found: list[Backend] = []
    if (cuda_like := _cuda_like_backend()) is not None:
        found.append(cuda_like)
    if _xpu_available():
        found.append(Backend.XPU)
    if _mps_available():
        found.append(Backend.MPS)
    found.append(Backend.CPU)
    return found


def build_backend() -> Backend:
    """The backend the installed torch build was made for, whether or not a device is visible."""
    if torch.version.hip:
        return Backend.ROCM
    if torch.version.cuda:
        return Backend.CUDA
    if getattr(torch.version, "xpu", None):
        return Backend.XPU
    if sys.platform == "darwin":
        return Backend.MPS
    return Backend.CPU


def resolve_backend(requested: str = "auto") -> Backend:
    """Turn a ``--device`` value into a usable :class:`Backend`.

    ``auto`` picks the first available accelerator and falls back to CPU.
    """
    available = available_backends()
    if requested == "auto":
        return available[0]
    try:
        backend = Backend(requested)
    except ValueError:
        choices = ", ".join(["auto", *Backend])
        raise BackendUnavailableError(f"unknown device {requested!r}; choose one of {choices}") from None
    if backend not in available:
        names = ", ".join(available)
        raise BackendUnavailableError(
            f"{backend} is not available with this torch build ({torch.__version__}); "
            f"available: {names}. To fix it, {INSTALL_HINTS[backend]}."
        )
    return backend


def _backend_module(backend: Backend) -> Any:  # noqa: ANN401 - torch.cuda/xpu/mps share no type
    """The torch namespace implementing ``backend``, or None for CPU."""
    if backend is Backend.CPU:
        return None
    return getattr(torch, backend.device_type)


def _parse_compute_capability(arch: str) -> tuple[int, int] | None:
    """``sm_86`` / ``compute_120`` -> ``(8, 6)`` / ``(12, 0)``."""
    match = re.fullmatch(r"(?:sm|compute)_(\d+)[a-z]?", arch)
    if match is None:
        return None
    number = int(match.group(1))
    return number // 10, number % 10


def arch_warnings(backend: Backend, arch: str | None, compiled: tuple[str, ...]) -> tuple[str, ...]:
    """Warn when the installed torch build ships no kernels for the device.

    CUDA cubins run on any GPU with the same major and an equal or newer minor
    version, and PTX (``compute_XY``) can be JIT-compiled for newer GPUs. ROCm
    code objects must match the ``gfx`` target exactly.
    """
    if not arch or not compiled:
        return ()
    if backend is Backend.ROCM:
        if arch in compiled:
            return ()
        return (
            f"This torch build has no kernels for {arch} (built for {', '.join(compiled)}). "
            "Install a ROCm build that targets it, e.g. AMD's wheels from repo.radeon.com.",
        )
    if backend is Backend.CUDA:
        device = _parse_compute_capability(arch)
        if device is None:
            return ()
        for entry in compiled:
            built = _parse_compute_capability(entry)
            if built is None:
                continue
            if entry.startswith("sm_") and built[0] == device[0] and built[1] <= device[1]:
                return ()
            if entry.startswith("compute_") and built <= device:
                return ()
        return (
            f"This torch build has no kernels for {arch} (built for {', '.join(compiled)}). "
            "Install a CUDA build that supports this GPU.",
        )
    return ()


@dataclass(frozen=True)
class DeviceInfo:
    """Static description of the device a benchmark ran on."""

    backend: Backend
    index: int
    name: str
    total_memory_bytes: int | None
    arch: str | None = None
    runtime: str | None = None
    compiled_archs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        data = asdict(self)
        data["backend"] = str(self.backend)
        data["compiled_archs"] = list(self.compiled_archs)
        data["warnings"] = list(self.warnings)
        return data


@dataclass(frozen=True)
class MemoryStats:
    """Peak memory observed during a trial, in bytes (None when unknown)."""

    allocated: int | None
    reserved: int | None


class Accelerator:
    """One device behind a backend-agnostic interface.

    Args:
        backend: Which backend drives the device.
        index: Device ordinal for multi-GPU machines (ignored for MPS and CPU).
        module: Override for the torch namespace (``torch.cuda`` etc.); tests
            inject fakes here.
    """

    def __init__(self, backend: Backend, index: int = 0, module: Any | None = None) -> None:  # noqa: ANN401
        self.backend = backend
        self.index = index
        self._module = module if module is not None else _backend_module(backend)
        # MPS exposes no peak counters, so peaks are tracked by polling.
        self._sampled_reserved = 0

    @property
    def device(self) -> torch.device:
        """The torch device to place tensors on."""
        if self.backend in (Backend.CPU, Backend.MPS):
            return torch.device(self.backend.device_type)
        return torch.device(self.backend.device_type, self.index)

    @property
    def has_peak_counters(self) -> bool:
        """Whether the allocator tracks peak usage itself (CUDA, ROCm, XPU)."""
        return self.backend in (Backend.CUDA, Backend.ROCM, Backend.XPU)

    @property
    def supports_pin_memory(self) -> bool:
        """Pinned host memory speeds up host-to-device copies; MPS and CPU lack it."""
        return self.backend in (Backend.CUDA, Backend.ROCM, Backend.XPU)

    @property
    def supports_cudnn_benchmark(self) -> bool:
        """Whether cuDNN (NVIDIA) or MIOpen (ROCm) can autotune convolution kernels."""
        return self.backend in (Backend.CUDA, Backend.ROCM)

    def activate(self) -> None:
        """Make this device current so allocations without an index land on it."""
        if self.has_peak_counters:
            self._module.set_device(self.index)

    def synchronize(self) -> None:
        """Block until all queued device work has finished."""
        if self.backend is Backend.MPS:
            self._module.synchronize()
        elif self._module is not None:
            self._module.synchronize(self.index)

    def empty_cache(self) -> None:
        """Collect garbage and return cached allocator blocks to the driver."""
        gc.collect()
        if self._module is not None:
            self._module.empty_cache()

    def reset_peak_memory(self) -> None:
        """Start a fresh peak-memory window."""
        self._sampled_reserved = 0
        if self.has_peak_counters:
            self._module.reset_peak_memory_stats(self.index)

    def sample_memory(self) -> None:
        """Poll memory use; only needed on MPS, which has no peak counters.

        ``driver_allocated_memory`` includes the MPS caching allocator, which
        keeps freed blocks, so sampling it after each step approximates the
        peak reserved memory during that step.
        """
        if self.backend is Backend.MPS:
            self._sampled_reserved = max(self._sampled_reserved, int(self._module.driver_allocated_memory()))

    def peak_memory(self) -> MemoryStats:
        """Peak memory since the last :meth:`reset_peak_memory`."""
        if self.has_peak_counters:
            return MemoryStats(
                allocated=int(self._module.max_memory_allocated(self.index)),
                reserved=int(self._module.max_memory_reserved(self.index)),
            )
        if self.backend is Backend.MPS:
            return MemoryStats(allocated=None, reserved=self._sampled_reserved or None)
        return MemoryStats(allocated=None, reserved=None)

    def free_memory(self) -> int | None:
        """Device memory free right now, as the driver reports it; None on MPS and CPU.

        Unlike :meth:`total_memory`, this excludes memory held by other
        processes, such as a desktop session on the same GPU.
        """
        if self.has_peak_counters:
            return int(self._module.mem_get_info(self.index)[0])
        return None

    def total_memory(self) -> int | None:
        """Memory budget of the device in bytes (MPS: Metal's recommended working set)."""
        if self.has_peak_counters:
            return int(self._module.get_device_properties(self.index).total_memory)
        if self.backend is Backend.MPS:
            return int(self._module.recommended_max_memory())
        return None

    def set_memory_fraction(self, fraction: float) -> None:
        """Cap this process at ``fraction`` of device memory, so OOM happens early and cleanly."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"memory fraction must be in (0, 1], got {fraction}")
        if self.backend is Backend.CPU:
            raise ValueError("the CPU backend has no device memory to cap")
        if self.backend is Backend.MPS:
            self._module.set_per_process_memory_fraction(fraction)
        else:
            self._module.set_per_process_memory_fraction(fraction, self.index)

    def supports_bf16(self) -> bool:
        """Whether bfloat16 math is supported natively."""
        if self.backend in (Backend.CUDA, Backend.ROCM, Backend.XPU):
            return bool(self._module.is_bf16_supported())
        # MPS supports bf16 on macOS 14+, which torch>=2.14 already requires.
        return True

    def info(self) -> DeviceInfo:
        """Describe the device, including build/arch compatibility warnings."""
        if self.backend is Backend.CPU:
            name = platform.processor() or platform.machine() or "CPU"
            return DeviceInfo(self.backend, 0, name, None)
        if self.backend is Backend.MPS:
            return DeviceInfo(
                backend=self.backend,
                index=0,
                name=f"Apple {platform.machine()} GPU",
                total_memory_bytes=self.total_memory(),
                runtime=f"macOS {platform.mac_ver()[0] or 'unknown'}",
            )

        props = self._module.get_device_properties(self.index)
        compiled = tuple(self._module.get_arch_list()) if hasattr(self._module, "get_arch_list") else ()
        arch: str | None
        if self.backend is Backend.ROCM:
            # e.g. "gfx1201" for the Radeon AI PRO R9700; strip feature flags like ":xnack-".
            arch = str(getattr(props, "gcnArchName", "")).split(":")[0] or None
            runtime = f"HIP {torch.version.hip}"
        elif self.backend is Backend.CUDA:
            arch = f"sm_{props.major}{props.minor}"
            runtime = f"CUDA {torch.version.cuda}"
        else:
            arch = None
            driver = getattr(props, "driver_version", None)
            runtime = f"XPU driver {driver}" if driver else "XPU"
        return DeviceInfo(
            backend=self.backend,
            index=self.index,
            name=str(props.name),
            total_memory_bytes=int(props.total_memory),
            arch=arch,
            runtime=runtime,
            compiled_archs=compiled,
            warnings=arch_warnings(self.backend, arch, compiled),
        )


def device_count(backend: Backend) -> int:
    """Number of devices visible for ``backend`` (MPS and CPU expose one)."""
    if backend in (Backend.CPU, Backend.MPS):
        return 1
    return int(_backend_module(backend).device_count())
