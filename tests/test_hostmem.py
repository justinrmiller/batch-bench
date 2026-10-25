from __future__ import annotations

import threading
from pathlib import Path

import psutil

from batch_bench.hostmem import HostMemory, HostMemoryMonitor

GIB = 1024**3


def test_real_host_memory_figures():
    host = HostMemory()
    assert 0 < host.available() <= host.total()


def test_peak_is_the_largest_drop_below_the_starting_level(make_host_memory):
    host = make_host_memory([40 * GIB, 37 * GIB, 34 * GIB, 39 * GIB])
    monitor = HostMemoryMonitor(host, floor_bytes=0, on_trip=lambda: None)
    for _ in range(3):
        monitor.sample()
    assert monitor.peak_bytes == 6 * GIB
    assert not monitor.tripped


def test_memory_freed_by_others_does_not_make_the_peak_negative(make_host_memory):
    monitor = HostMemoryMonitor(make_host_memory([10 * GIB, 12 * GIB]), floor_bytes=0, on_trip=lambda: None)
    monitor.sample()
    assert monitor.peak_bytes == 0


def test_trips_once_below_the_floor(make_host_memory):
    trips: list[None] = []
    host = make_host_memory([8 * GIB, 1 * GIB])
    monitor = HostMemoryMonitor(host, floor_bytes=2 * GIB, on_trip=lambda: trips.append(None))
    monitor.sample()
    monitor.sample()
    assert monitor.tripped
    assert trips == [None]


def test_entering_takes_a_reading(make_host_memory):
    monitor = HostMemoryMonitor(make_host_memory([8 * GIB, 1 * GIB]), floor_bytes=2 * GIB, on_trip=lambda: None)
    with monitor:
        assert monitor.tripped


def test_background_thread_polls_until_exit(make_host_memory):
    # Only the first two readings are taken synchronously (creation and entry),
    # so tripping on the third proves the thread polls.
    tripped = threading.Event()
    host = make_host_memory([8 * GIB, 8 * GIB, 1 * GIB])
    with HostMemoryMonitor(host, floor_bytes=2 * GIB, on_trip=tripped.set) as monitor:
        assert tripped.wait(timeout=5)
    assert monitor.tripped
    assert not monitor._thread.is_alive()


def write_cgroup(root: Path, relative: str, **files: str) -> None:
    """Create a cgroup directory; ``memory_max="1"`` writes the ``memory.max`` file."""
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name.replace("_", ".")).write_text(text)


class TestCgroupLimits:
    def test_the_tightest_cap_at_or_above_the_process_wins(self, tmp_path: Path):
        root = tmp_path / "cgroup"
        stat = "anon 1\ninactive_file {}\n"
        write_cgroup(root, "slice", memory_max=str(2 * GIB), memory_current=str(GIB), memory_stat=stat.format(0))
        write_cgroup(
            root,
            "slice/scope",
            memory_max=str(GIB),
            memory_current=str(GIB * 3 // 4),
            memory_stat=stat.format(GIB // 4),
        )
        write_cgroup(root, "slice/scope/leaf", memory_max="max")
        proc = tmp_path / "proc-cgroup"
        proc.write_text("0::/slice/scope/leaf\n")
        host = HostMemory(proc_cgroup=proc, cgroup_root=root)
        assert host.total() == GIB
        # The scope has 1 GiB - 3/4 GiB used + 1/4 GiB of reclaimable cache left; the slice has 1 GiB.
        assert host.available() == GIB // 2

    def test_a_container_cap_at_the_namespace_root(self, tmp_path: Path):
        write_cgroup(tmp_path, "", memory_max=str(GIB))  # no memory.current or memory.stat
        proc = tmp_path / "proc-cgroup"
        proc.write_text("0::/\n")
        host = HostMemory(proc_cgroup=proc, cgroup_root=tmp_path)
        assert host.total() == GIB
        assert host.available() == min(GIB, psutil.virtual_memory().available)

    def test_no_cgroup_v2_means_system_figures(self, tmp_path: Path):
        proc = tmp_path / "proc-cgroup"
        proc.write_text("12:memory:/docker/abc\n")  # cgroup v1 only
        for host in (HostMemory(proc_cgroup=proc), HostMemory(proc_cgroup=tmp_path / "missing")):
            assert host.total() == psutil.virtual_memory().total
