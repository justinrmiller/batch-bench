"""Host RAM: how much is free, a per-trial monitor, and projections for larger batches.

Host memory grows with the batch size times the worker count. Every DataLoader
worker builds a whole batch, and up to ``prefetch_factor * num_workers``
finished batches wait in the main process, pinned and rounded up to a power of
two. A sweep that doubles the batch until the GPU runs out of memory can
exhaust host RAM first. The kernel's OOM killer then picks a victim, which may
be the desktop session rather than the benchmark.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil

# How often the background thread reads free memory. One reading costs ~30 us.
POLL_SECONDS = 0.05


def _read_int(path: Path) -> int | None:
    """An integer control file, or None when it is missing or reads ``max`` (no limit)."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _cgroup_limits(proc_cgroup: Path, root: Path) -> list[tuple[Path, int]]:
    """The cgroup v2 directories at and above this process that cap memory, with their caps.

    A container's memory limit, or ``systemd-run -p MemoryMax=``, lives here
    and is invisible to system-wide figures. Empty without cgroup v2, e.g. on
    macOS, Windows or a cgroup v1 host.
    """
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return []
    relative = next((line[3:] for line in lines if line.startswith("0::")), None)
    if relative is None:
        return []
    directory = root / relative.strip().lstrip("/")
    limits = []
    for candidate in (directory, *directory.parents):
        if not candidate.is_relative_to(root):
            break
        limit = _read_int(candidate / "memory.max")
        if limit is not None:
            limits.append((candidate, limit))
    return limits


def _cgroup_room(directory: Path, limit: int) -> int:
    """Bytes left under a cgroup's cap. Inactive page cache counts as free: the kernel reclaims it first."""
    usage = _read_int(directory / "memory.current") or 0
    try:
        stat = (directory / "memory.stat").read_text().split()
    except OSError:
        stat = []
    pairs = dict(zip(stat[::2], stat[1::2], strict=False))
    return limit - usage + int(pairs.get("inactive_file", 0))


class HostMemory:
    """RAM figures for this process: the system's, narrowed by any cgroup memory cap.

    Tests substitute a fake, or point ``proc_cgroup`` and ``cgroup_root`` at a
    fake cgroup tree.
    """

    def __init__(
        self, proc_cgroup: Path = Path("/proc/self/cgroup"), cgroup_root: Path = Path("/sys/fs/cgroup")
    ) -> None:
        self._cgroup_limits = _cgroup_limits(proc_cgroup, cgroup_root)

    def available(self) -> int:
        """Bytes that can be allocated without swapping or hitting a cgroup cap."""
        available = int(psutil.virtual_memory().available)
        for directory, limit in self._cgroup_limits:
            available = min(available, _cgroup_room(directory, limit))
        return max(0, available)

    def total(self) -> int:
        """Physical RAM, or the tightest cgroup cap if that is smaller."""
        return min([int(psutil.virtual_memory().total), *(limit for _, limit in self._cgroup_limits)])


def round_up_to_power_of_two(value: int) -> int:
    """``value`` rounded up as PyTorch's caching host allocator rounds a pinned buffer."""
    return 1 << (value - 1).bit_length() if value > 0 else 0


@dataclass(frozen=True)
class HostFootprint:
    """How much host RAM a successful trial took, to project larger batches from.

    Attributes:
        batch_size: The trial's batch size.
        peak_bytes: How far free host RAM fell during it.
        pinned_bytes: The part of that held as pinned memory at most, in whole
            allocator blocks; 0 when nothing was pinned or the backend can't tell.
        tensor_bytes: Size of each tensor in one batch; each is pinned separately.
    """

    batch_size: int
    peak_bytes: int
    pinned_bytes: int = 0
    tensor_bytes: tuple[int, ...] = ()

    def project(self, batch_size: int) -> int:
        """Host RAM a trial at ``batch_size`` should take.

        Most of the footprint grows linearly with the batch: the batches
        workers are building, and batches in shared memory. Pinned
        batches grow in steps instead, because the allocator rounds each
        tensor's buffer up to a power of two. A batch 1.75x the size can need
        twice the pinned memory, and one 1.25x the size may need no more.
        """
        scale = batch_size / self.batch_size
        per_batch = sum(round_up_to_power_of_two(size) for size in self.tensor_bytes)
        if not (self.pinned_bytes and per_batch):
            return int(self.peak_bytes * scale)
        batches = self.pinned_bytes / per_batch  # pinned at once, e.g. a full prefetch queue
        pinned = batches * sum(
            round_up_to_power_of_two(size * batch_size // self.batch_size) for size in self.tensor_bytes
        )
        return int(max(0, self.peak_bytes - self.pinned_bytes) * scale + pinned)


class HostMemoryMonitor:
    """Tracks how far free host RAM drops during a trial, and trips below a floor.

    Use it as a context manager around the trial. A background thread reads
    free memory every :data:`POLL_SECONDS`, which catches DataLoader workers
    filling memory while the main thread is blocked; :meth:`sample` takes an
    extra reading on demand.

    The peak is the drop in system-wide available memory, not a sum of process
    sizes. Forked workers share most of their pages with the main process, so
    summing RSS would count those once per worker. Memory other processes
    allocate meanwhile counts too, which is what decides whether the machine
    runs out.

    Args:
        host: Where readings come from.
        floor_bytes: Trip when available memory falls below this.
        on_trip: Called once, from whichever thread sees the trip first.
    """

    def __init__(self, host: HostMemory, floor_bytes: int, on_trip: Callable[[], None]) -> None:
        self.host = host
        self.floor_bytes = floor_bytes
        self.on_trip = on_trip
        self.start_available = host.available()
        self.lowest_available = self.start_available
        self.tripped = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, name="host-memory-monitor", daemon=True)

    @property
    def peak_bytes(self) -> int:
        """Largest drop in available memory since the monitor was created."""
        return max(0, self.start_available - self.lowest_available)

    def sample(self) -> None:
        """Take one reading, tripping the monitor if available memory is below the floor."""
        available = self.host.available()
        with self._lock:
            self.lowest_available = min(self.lowest_available, available)
            if self.tripped or available >= self.floor_bytes:
                return
            self.tripped = True
        self.on_trip()

    def _poll(self) -> None:
        while not self._stop.wait(POLL_SECONDS):
            self.sample()

    def __enter__(self) -> HostMemoryMonitor:
        self.sample()
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()
