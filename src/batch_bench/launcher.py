"""Console entry point that explains a missing PyTorch install instead of crashing.

PyTorch is not a base dependency: the right build depends on the GPU, so it
comes from a uv extra. Without one, importing the CLI would die with a bare
``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence

MISSING_TORCH = """\
error: PyTorch is not installed.

batch-bench needs the PyTorch build for your GPU, chosen with a uv extra:

    uv sync --extra rocm    # AMD on Linux, e.g. Radeon AI PRO R9700
    uv sync --extra cuda    # NVIDIA
    uv sync --extra xpu     # Intel
    uv sync --extra mps     # Apple Silicon
    uv sync --extra cpu     # no GPU

A plain `uv sync` installs no PyTorch and removes one an extra installed,
so pass the same --extra every time you sync.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI, or print install instructions when PyTorch is missing."""
    if importlib.util.find_spec("torch") is None:
        sys.stderr.write(MISSING_TORCH)
        return 2
    # Deferred: the CLI imports torch at module level.
    from batch_bench.cli import main as cli_main

    return cli_main(argv)
