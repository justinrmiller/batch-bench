"""Shared fixtures: tiny local checkpoints and fake accelerator namespaces.

Every model here is randomly initialized from a miniature config and saved
with ``save_pretrained``, so tests exercise the real loading code without
network access or a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import transformers
from transformers import (
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    SegformerConfig,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    SwinConfig,
    YolosConfig,
    YolosForObjectDetection,
    YolosImageProcessor,
)

from batch_bench.data import SyntheticImageSource
from batch_bench.devices import Accelerator, Backend

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

GIB = 1024**3


def _post_init(**kwargs: Any) -> dict[str, Any]:
    """Config kwargs consumed by ``__post_init__(**kwargs)``, which type checkers cannot see."""
    return kwargs


def _save(directory: Path, model: transformers.PreTrainedModel, processor: Any) -> Path:
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_detector_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    config = YolosConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=37,
        image_size=[64, 64],
        patch_size=16,
        num_detection_tokens=5,
        **_post_init(num_labels=3),
    )
    return _save(
        tmp_path_factory.mktemp("tiny-yolos"),
        YolosForObjectDetection(config),
        YolosImageProcessor(size={"height": 64, "width": 64}),
    )


@pytest.fixture(scope="session")
def tiny_segmenter_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    config = SegformerConfig(
        num_encoder_blocks=2,
        depths=[1, 1],
        sr_ratios=[2, 1],
        hidden_sizes=[8, 16],
        patch_sizes=[7, 3],
        strides=[4, 2],
        num_attention_heads=[1, 1],
        mlp_ratios=[2, 2],
        decoder_hidden_size=16,
        **_post_init(num_labels=4),
    )
    return _save(
        tmp_path_factory.mktemp("tiny-segformer"),
        SegformerForSemanticSegmentation(config),
        SegformerImageProcessor(size={"height": 64, "width": 64}),
    )


@pytest.fixture(scope="session")
def tiny_universal_segmenter_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    backbone = SwinConfig(
        image_size=64,
        patch_size=4,
        embed_dim=32,
        depths=[1, 1, 1, 1],
        num_heads=[1, 1, 2, 2],
        window_size=2,
        **_post_init(out_features=["stage1", "stage2", "stage3", "stage4"]),
    )
    config = Mask2FormerConfig(
        backbone_config=backbone,
        hidden_dim=32,
        mask_feature_size=32,
        feature_size=32,
        num_queries=4,
        encoder_layers=1,
        decoder_layers=2,
        encoder_feedforward_dim=32,
        dim_feedforward=32,
        num_attention_heads=2,
        feature_strides=[4, 8, 16, 32],
        common_stride=4,
        **_post_init(num_labels=3),
    )
    return _save(
        tmp_path_factory.mktemp("tiny-mask2former"),
        Mask2FormerForUniversalSegmentation(config),
        Mask2FormerImageProcessor(size={"height": 64, "width": 64}),
    )


@pytest.fixture(scope="session")
def tiny_embedder_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
        projection_dim=16,
    )
    return _save(
        tmp_path_factory.mktemp("tiny-clip-vision"),
        CLIPVisionModelWithProjection(config),
        CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}),
    )


@pytest.fixture
def tiny_source() -> SyntheticImageSource:
    return SyntheticImageSource(size=(64, 48), pool_size=4)


@pytest.fixture
def cpu() -> Accelerator:
    return Accelerator(Backend.CPU)


@pytest.fixture(autouse=True)
def host_cache_releases(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Record ``torch.accelerator.empty_host_cache`` calls, so faked GPUs never reach the real one."""
    calls: list[None] = []
    monkeypatch.setattr(torch.accelerator, "empty_host_cache", lambda: calls.append(None))
    return calls


class FakeHostMemory:
    """Stands in for :class:`batch_bench.hostmem.HostMemory`.

    ``available`` returns the scripted readings in order, then repeats the
    last one. The monitor thread reads it too, so tests should only rely on
    which values appear, not on when.
    """

    def __init__(self, readings: list[int], total: int = 64 * GIB) -> None:
        self.readings = list(readings)
        self.total_bytes = total
        self.reads = 0

    def script(self, readings: list[int]) -> None:
        """Start over with new readings."""
        self.readings = list(readings)
        self.reads = 0

    def available(self) -> int:
        value = self.readings[min(self.reads, len(self.readings) - 1)]
        self.reads += 1
        return value

    def total(self) -> int:
        return self.total_bytes


@pytest.fixture
def make_host_memory() -> Callable[..., FakeHostMemory]:
    """Factory for fake host RAM with scripted ``available`` readings."""
    return FakeHostMemory


class FakeDeviceModule:
    """Stands in for ``torch.cuda`` / ``torch.xpu``: records calls, returns canned values."""

    def __init__(self, props: SimpleNamespace, arch_list: list[str], bf16: bool = True) -> None:
        self.props = props
        self.arch_list = arch_list
        self.bf16 = bf16
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.allocated = 100
        self.reserved = 200
        self.free = props.total_memory // 2

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def set_device(self, index: int) -> None:
        self._record("set_device", index)

    def synchronize(self, index: int) -> None:
        self._record("synchronize", index)

    def empty_cache(self) -> None:
        self._record("empty_cache")

    def reset_peak_memory_stats(self, index: int) -> None:
        self._record("reset_peak_memory_stats", index)

    def max_memory_allocated(self, index: int) -> int:
        return self.allocated

    def max_memory_reserved(self, index: int) -> int:
        return self.reserved

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return self.props

    def get_arch_list(self) -> list[str]:
        return self.arch_list

    def set_per_process_memory_fraction(self, fraction: float, index: int) -> None:
        self._record("set_per_process_memory_fraction", fraction, index)

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        # Like torch: emulated bf16 counts unless the caller excludes it.
        return self.bf16 or including_emulation

    def failing_mem_get_info(self, index: int) -> tuple[int, int]:
        raise RuntimeError("The device does not support querying free memory")

    def mem_get_info(self, index: int) -> tuple[int, int]:
        return self.free, self.props.total_memory


class FakeMpsModule:
    """Stands in for ``torch.mps``, whose memory API differs from CUDA's."""

    def __init__(self, driver_readings: list[int]) -> None:
        self.driver_readings = list(driver_readings)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def synchronize(self) -> None:
        self.calls.append(("synchronize", ()))

    def empty_cache(self) -> None:
        self.calls.append(("empty_cache", ()))

    def driver_allocated_memory(self) -> int:
        return self.driver_readings.pop(0)

    def recommended_max_memory(self) -> int:
        return 24 * 1024**3

    def set_per_process_memory_fraction(self, fraction: float) -> None:
        self.calls.append(("set_per_process_memory_fraction", (fraction,)))


@pytest.fixture
def make_device_module() -> Callable[..., FakeDeviceModule]:
    """Factory for fake CUDA/ROCm/XPU namespaces."""

    def factory(
        *,
        name: str = "Fake GPU",
        total_memory: int = 32 * 1024**3,
        major: int = 8,
        minor: int = 9,
        gcn_arch: str = "",
        driver_version: str | None = None,
        arch_list: list[str] | None = None,
        bf16: bool = True,
    ) -> FakeDeviceModule:
        props = SimpleNamespace(
            name=name,
            total_memory=total_memory,
            major=major,
            minor=minor,
            gcnArchName=gcn_arch,
            driver_version=driver_version,
        )
        return FakeDeviceModule(props, arch_list if arch_list is not None else [], bf16=bf16)

    return factory


@pytest.fixture
def make_mps_module() -> Callable[[list[int]], FakeMpsModule]:
    """Factory for fake ``torch.mps`` namespaces with scripted memory readings."""
    return FakeMpsModule
