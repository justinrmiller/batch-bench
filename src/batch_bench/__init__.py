"""Find the optimal DataLoader batch size for HuggingFace vision models on any GPU."""

import logging
from importlib.metadata import PackageNotFoundError, version

# transformers imports torch.distributed.elastic, which logs "NOTE: Redirects
# are currently not supported in MacOs/Windows" while loading. That happens
# once per process, so once per DataLoader worker where workers are spawned
# (macOS, Windows). batch-bench never uses those redirects. This package is
# imported before transformers in every worker, so the filter is in place in time.
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").addFilter(
    lambda record: "Redirects are currently not supported" not in record.getMessage()
)

try:
    __version__ = version("batch-bench")
except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
