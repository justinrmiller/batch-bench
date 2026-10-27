"""The wiki generator's reproduce line, which must repeat a run's settings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "wiki.py"


@pytest.fixture(scope="module")
def wiki() -> ModuleType:
    spec = importlib.util.spec_from_file_location("wiki", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["wiki"] = module  # dataclasses look their module up while being created
    spec.loader.exec_module(module)
    return module


def reproduce(wiki: ModuleType, **settings: Any) -> str:
    recorded = {
        "mode": "inference",
        "precision": "bf16",
        "worker_counts": [1, 2, 4],
        "batch_sizes": [1, 2, 4, 8],
        "image_source": "synthetic",
        "image_size": [640, 480],
        "memory_fraction": 0.75,
        "compile_model": False,
        "pin_memory": True,
        "cudnn_benchmark": True,
    } | settings
    data = {"settings": recorded, "device": {"index": 0, "backend": "rocm"}}
    run = wiki.Run(Path("run.json"), data, {"name": "clip-vit-b32"}, wiki.HARDWARE["amd-r9700"])
    return wiki.reproduce(run)


def test_defaults_add_no_flags(wiki: ModuleType):
    assert reproduce(wiki) == (
        "uv run batch-bench run -m clip-vit-b32 --precision bf16 --num-workers 1,2,4 --max-batch-size 8"
    )


def test_seed_and_revision_are_repeated(wiki: ModuleType):
    line = reproduce(wiki, seed=7, revision="abc123")
    assert "--seed 7" in line
    assert "--revision abc123" in line
    assert "--seed" not in reproduce(wiki, seed=0, revision=None)


@pytest.mark.parametrize(
    ("settings", "flag"),
    [
        ({"cudnn_benchmark": False}, True),
        # Folder images without --image-size turn autotuning off by themselves.
        ({"cudnn_benchmark": False, "image_source": "/data", "image_size": None}, False),
        ({"cudnn_benchmark": False, "image_source": "/data", "image_size": [512, 512]}, True),
    ],
)
def test_autotuning_turned_off_by_hand_is_repeated(wiki: ModuleType, settings: dict[str, Any], flag: bool):
    assert ("--no-cudnn-benchmark" in reproduce(wiki, **settings)) is flag
