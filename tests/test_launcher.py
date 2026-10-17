from __future__ import annotations

import importlib.util

import pytest

from batch_bench import launcher


def test_missing_torch_prints_install_instructions(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, package: str | None = None) -> object:
        return None if name == "torch" else real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    assert launcher.main(["models"]) == 2
    err = capsys.readouterr().err
    assert "PyTorch is not installed" in err
    assert "uv sync --extra rocm" in err


def test_delegates_to_the_cli_when_torch_is_present(capsys: pytest.CaptureFixture[str]):
    assert launcher.main(["models"]) == 0
    assert "segformer-b0" in capsys.readouterr().out
