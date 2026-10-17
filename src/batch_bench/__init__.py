"""Find the optimal DataLoader batch size for HuggingFace vision models on any GPU."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("batch-bench")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
