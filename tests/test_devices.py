from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from batch_bench import devices
from batch_bench.devices import (
    Accelerator,
    Backend,
    BackendUnavailableError,
    DeviceInfo,
    arch_warnings,
    available_backends,
    device_count,
    is_oom_error,
    resolve_backend,
)


def test_rocm_uses_the_cuda_device_type():
    assert Backend.ROCM.device_type == "cuda"
    assert Backend.XPU.device_type == "xpu"
    assert Backend.CPU.device_type == "cpu"


@pytest.mark.parametrize(
    "exc",
    [
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        MemoryError(),
        RuntimeError("HIP out of memory. Tried to allocate 1.50 GiB"),
        RuntimeError("MPS backend out of memory (MPS allocated: 17.00 GB)"),
        RuntimeError("XPU out of memory. Tried to allocate 3.00 GiB"),
        RuntimeError("level_zero backend failed with error: 39 (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)"),
        RuntimeError("miopenStatusAllocFailed"),
        RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
        RuntimeError("Current platform can NOT allocate memory block with size larger than 4GB!"),
        RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 9 bytes"),
    ],
)
def test_oom_errors_are_recognized(exc: BaseException):
    assert is_oom_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
        ValueError("out of memory"),  # not a RuntimeError: a bug, not an allocation failure
        KeyError("pixel_values"),
    ],
)
def test_other_errors_are_not_oom(exc: BaseException):
    assert not is_oom_error(exc)


@pytest.fixture
def no_accelerators(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    return monkeypatch


def test_cpu_is_the_fallback(no_accelerators: pytest.MonkeyPatch):
    assert available_backends() == [Backend.CPU]
    assert resolve_backend("auto") is Backend.CPU


def test_hip_build_is_reported_as_rocm(no_accelerators: pytest.MonkeyPatch):
    no_accelerators.setattr(torch.cuda, "is_available", lambda: True)
    no_accelerators.setattr(torch.version, "hip", "7.2.26015")
    assert available_backends() == [Backend.ROCM, Backend.CPU]
    assert resolve_backend("auto") is Backend.ROCM


def test_cuda_build_is_reported_as_cuda(no_accelerators: pytest.MonkeyPatch):
    no_accelerators.setattr(torch.cuda, "is_available", lambda: True)
    no_accelerators.setattr(torch.version, "hip", None)
    assert resolve_backend("auto") is Backend.CUDA


def test_xpu_and_mps_detection_order(no_accelerators: pytest.MonkeyPatch):
    no_accelerators.setattr(torch.xpu, "is_available", lambda: True)
    no_accelerators.setattr(torch.backends.mps, "is_available", lambda: True)
    assert available_backends() == [Backend.XPU, Backend.MPS, Backend.CPU]


def test_missing_xpu_namespace_is_tolerated(no_accelerators: pytest.MonkeyPatch):
    no_accelerators.delattr(torch, "xpu")
    assert available_backends() == [Backend.CPU]


def test_explicit_backend_is_validated(no_accelerators: pytest.MonkeyPatch):
    assert resolve_backend("cpu") is Backend.CPU
    with pytest.raises(BackendUnavailableError, match="uv sync --extra rocm"):
        resolve_backend("rocm")
    with pytest.raises(BackendUnavailableError, match="unknown device 'tpu'"):
        resolve_backend("tpu")


def test_device_count(monkeypatch: pytest.MonkeyPatch):
    assert device_count(Backend.CPU) == 1
    assert device_count(Backend.MPS) == 1
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert device_count(Backend.ROCM) == 2


class TestArchWarnings:
    def test_rocm_requires_an_exact_gfx_match(self):
        assert arch_warnings(Backend.ROCM, "gfx1201", ("gfx1100", "gfx1201")) == ()
        # Build lists can carry feature flags; the base target is what matters.
        assert arch_warnings(Backend.ROCM, "gfx90a", ("gfx90a:sramecc+:xnack-",)) == ()
        (warning,) = arch_warnings(Backend.ROCM, "gfx1201", ("gfx90a", "gfx1100"))
        assert "gfx1201" in warning
        assert "repo.radeon.com" in warning

    def test_cuda_cubins_run_on_newer_minor_versions(self):
        assert arch_warnings(Backend.CUDA, "sm_89", ("sm_80", "sm_86", "sm_90")) == ()

    def test_cuda_ptx_can_be_jit_compiled_for_newer_gpus(self):
        assert arch_warnings(Backend.CUDA, "sm_120", ("sm_80", "sm_90", "compute_90")) == ()

    def test_cuda_without_compatible_code_warns(self):
        (warning,) = arch_warnings(Backend.CUDA, "sm_120", ("sm_80", "sm_90", "sm_90a"))
        assert "sm_120" in warning

    def test_cuda_ignores_unparseable_entries(self):
        assert arch_warnings(Backend.CUDA, "sm_89", ("weird", "sm_86")) == ()
        assert arch_warnings(Backend.CUDA, "unknown", ("sm_86",)) == ()

    def test_nothing_to_check(self):
        assert arch_warnings(Backend.CUDA, None, ("sm_86",)) == ()
        assert arch_warnings(Backend.ROCM, "gfx1201", ()) == ()
        assert arch_warnings(Backend.XPU, "bmg", ("pvc",)) == ()


class TestCudaLikeAccelerator:
    def test_memory_and_sync_calls_target_the_device_index(self, make_device_module):
        module = make_device_module()
        accel = Accelerator(Backend.CUDA, index=1, module=module)
        accel.activate()
        accel.synchronize()
        accel.reset_peak_memory()
        accel.empty_cache()
        accel.set_memory_fraction(0.5)
        assert module.calls == [
            ("set_device", (1,)),
            ("synchronize", (1,)),
            ("reset_peak_memory_stats", (1,)),
            ("empty_cache", ()),
            ("set_per_process_memory_fraction", (0.5, 1)),
        ]
        assert accel.device == torch.device("cuda", 1)
        assert accel.has_peak_counters
        assert accel.supports_pin_memory
        assert accel.supports_cudnn_benchmark

    def test_peak_total_and_free_memory(self, make_device_module):
        accel = Accelerator(Backend.CUDA, module=make_device_module(total_memory=1234))
        assert accel.peak_memory() == devices.MemoryStats(allocated=100, reserved=200)
        assert accel.total_memory() == 1234
        assert accel.free_memory() == 617

    def test_bf16_support_comes_from_the_backend(self, make_device_module):
        assert Accelerator(Backend.CUDA, module=make_device_module(bf16=True)).supports_bf16()
        # Emulated bf16 (pre-Ampere CUDA, some Intel GPUs) does not count.
        assert not Accelerator(Backend.CUDA, module=make_device_module(bf16=False)).supports_bf16()
        assert not Accelerator(Backend.XPU, module=make_device_module(bf16=False)).supports_bf16()

    def test_unreadable_free_memory_is_unknown(self, make_device_module):
        module = make_device_module()
        module.mem_get_info = module.failing_mem_get_info
        assert Accelerator(Backend.XPU, module=module).free_memory() is None

    def test_cuda_info(self, make_device_module, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.version, "cuda", "13.0")
        module = make_device_module(name="NVIDIA GeForce RTX 4090", major=8, minor=9, arch_list=["sm_80", "sm_86"])
        info = Accelerator(Backend.CUDA, module=module).info()
        assert info.name == "NVIDIA GeForce RTX 4090"
        assert info.arch == "sm_89"
        assert info.runtime == "CUDA 13.0"
        assert info.warnings == ()

    def test_rocm_info_reports_the_gfx_target(self, make_device_module, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.version, "hip", "7.2.26015")
        module = make_device_module(
            name="AMD Radeon AI PRO R9700",
            gcn_arch="gfx1201:sramecc-:xnack-",
            arch_list=["gfx1100", "gfx1201"],
        )
        info = Accelerator(Backend.ROCM, module=module).info()
        assert info.arch == "gfx1201"
        assert info.runtime == "HIP 7.2.26015"
        assert info.total_memory_bytes == 32 * 1024**3
        assert info.warnings == ()
        assert info.compiled_archs == ("gfx1100", "gfx1201")

    def test_rocm_info_warns_when_the_build_lacks_the_gpu(self, make_device_module):
        module = make_device_module(gcn_arch="gfx1201", arch_list=["gfx90a"])
        (warning,) = Accelerator(Backend.ROCM, module=module).info().warnings
        assert "gfx1201" in warning

    def test_xpu_info_reports_the_driver(self, make_device_module):
        module = make_device_module(name="Intel(R) Arc(TM) B580", driver_version="1.6.33578")
        info = Accelerator(Backend.XPU, module=module).info()
        assert info.arch is None
        assert info.runtime == "XPU driver 1.6.33578"
        no_driver = make_device_module(driver_version=None)
        assert Accelerator(Backend.XPU, module=no_driver).info().runtime == "XPU"

    def test_info_without_arch_list(self, make_device_module):
        props = make_device_module().props
        stripped = SimpleNamespace(get_device_properties=lambda index: props)
        assert Accelerator(Backend.CUDA, module=stripped).info().compiled_archs == ()


class TestMpsAccelerator:
    def test_peak_reserved_is_the_largest_sample(self, make_mps_module):
        module = make_mps_module([300, 900, 600])
        accel = Accelerator(Backend.MPS, module=module)
        accel.reset_peak_memory()
        for _ in range(3):
            accel.sample_memory()
        assert accel.peak_memory() == devices.MemoryStats(allocated=None, reserved=900)
        accel.reset_peak_memory()
        assert accel.peak_memory().reserved is None

    def test_mps_calls_take_no_device_index(self, make_mps_module):
        module = make_mps_module([])
        accel = Accelerator(Backend.MPS, module=module)
        accel.activate()  # no-op on MPS
        accel.synchronize()
        accel.set_memory_fraction(0.8)
        assert module.calls == [("synchronize", ()), ("set_per_process_memory_fraction", (0.8,))]
        assert accel.device == torch.device("mps")
        assert accel.total_memory() == 24 * 1024**3
        assert accel.supports_bf16()
        assert not accel.supports_pin_memory
        assert not accel.has_peak_counters

    def test_mps_info(self, make_mps_module):
        info = Accelerator(Backend.MPS, module=make_mps_module([])).info()
        assert info.backend is Backend.MPS
        assert info.total_memory_bytes == 24 * 1024**3
        assert info.runtime is not None
        assert info.runtime.startswith("macOS")


class TestCpuAccelerator:
    def test_cpu_has_no_device_memory_figures(self, cpu: Accelerator):
        assert cpu.free_memory() is None

    def test_cpu_has_no_device_memory(self, cpu: Accelerator):
        cpu.activate()
        cpu.synchronize()
        cpu.sample_memory()
        cpu.reset_peak_memory()
        cpu.empty_cache()
        assert cpu.peak_memory() == devices.MemoryStats(allocated=None, reserved=None)
        assert cpu.total_memory() is None
        assert cpu.device == torch.device("cpu")
        assert cpu.supports_bf16()
        assert not cpu.supports_cudnn_benchmark

    def test_cpu_memory_fraction_is_rejected(self, cpu: Accelerator):
        with pytest.raises(ValueError, match="CPU backend"):
            cpu.set_memory_fraction(0.5)

    @pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
    def test_memory_fraction_must_be_a_fraction(self, cpu: Accelerator, fraction: float):
        with pytest.raises(ValueError, match="memory fraction"):
            cpu.set_memory_fraction(fraction)

    def test_cpu_info(self, cpu: Accelerator):
        info = cpu.info()
        assert info.backend is Backend.CPU
        assert info.name
        assert info.total_memory_bytes is None


def test_device_info_to_dict_is_json_friendly():
    info = DeviceInfo(
        backend=Backend.ROCM,
        index=0,
        name="AMD Radeon AI PRO R9700",
        total_memory_bytes=32,
        arch="gfx1201",
        compiled_archs=("gfx1201",),
        warnings=("w",),
    )
    data = info.to_dict()
    assert data["backend"] == "rocm"
    assert data["compiled_archs"] == ["gfx1201"]
    assert data["warnings"] == ["w"]


@pytest.mark.parametrize(
    ("hip", "cuda", "xpu", "platform", "expected"),
    [
        ("7.2.26015", None, None, "linux", Backend.ROCM),
        (None, "13.0", None, "linux", Backend.CUDA),
        (None, None, "20260100", "linux", Backend.XPU),
        (None, None, None, "darwin", Backend.MPS),
        (None, None, None, "linux", Backend.CPU),
    ],
)
def test_build_backend(monkeypatch: pytest.MonkeyPatch, hip, cuda, xpu, platform, expected):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "xpu", xpu, raising=False)
    monkeypatch.setattr(devices.sys, "platform", platform)
    assert devices.build_backend() is expected


def test_every_gpu_backend_has_a_visibility_hint():
    assert set(devices.VISIBILITY_HINTS) == set(Backend) - {Backend.CPU}
    assert "render" in devices.VISIBILITY_HINTS[Backend.ROCM]
