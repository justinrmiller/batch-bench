"""Host RAM: how much is free, and a per-trial monitor that measures and guards it.

Host memory grows with the batch size times the worker count. Every DataLoader
worker decodes and preprocesses a whole batch at once, and up to
``prefetch_factor * num_workers`` finished batches wait in the main process,
pinned and rounded up to a power of two. A sweep that doubles the batch until
the GPU runs out of memory can exhaust host RAM first. The kernel's OOM killer
then picks a victim, which may be the desktop session rather than the
benchmark.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
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
