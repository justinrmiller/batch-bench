"""Turn sweep results into a batch-size recommendation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from batch_bench.runner import TrialResult, TrialStatus


@dataclass(frozen=True)
class Recommendation:
    """The suggested configuration and how it was chosen.

    Attributes:
        recommended: Smallest batch size (then fewest workers) within
            ``tolerance`` of the best throughput that also leaves the requested
            memory headroom. Smaller batches cut latency and memory, and
            leave room for larger inputs at no real throughput cost.
        fastest: Highest end-to-end throughput observed.
        largest_fit: Largest batch size that ran.
        first_oom: Smallest batch size that ran out of device memory, or was
            skipped as projected to, if any. Host RAM limits are reported in
            ``notes``.
        notes: Human-readable caveats, such as an input-bound pipeline.
    """

    recommended: TrialResult
    fastest: TrialResult
    largest_fit: int
    first_oom: int | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation."""
        return {
            "recommended": self.recommended.to_dict(),
            "fastest": self.fastest.to_dict(),
            "largest_fit": self.largest_fit,
            "first_oom": self.first_oom,
            "notes": list(self.notes),
        }


def _throughput(result: TrialResult) -> float:
    return result.samples_per_second or 0.0


def recommend(
    results: Sequence[TrialResult],
    *,
    tolerance: float = 0.05,
    memory_headroom: float = 0.10,
    total_memory_bytes: int | None = None,
    input_bound_threshold: float = 0.15,
    max_tested: int | None = None,
) -> Recommendation | None:
    """Pick a batch size from ``results``; None when no trial succeeded.

    Args:
        results: Trials for one model, any order, possibly several worker counts.
        tolerance: Accept configurations within this fraction of the best throughput.
        memory_headroom: Fraction of usable memory to keep free; trials whose
            peak reserved memory exceeds the rest are not recommended.
        total_memory_bytes: Usable memory budget; None disables the headroom check.
        input_bound_threshold: Data-wait fraction above which the pipeline is
            reported as input-bound.
        max_tested: Largest batch size the sweep was allowed to try, used to
            tell "memory ceiling not reached" apart from "nothing larger fits".
    """
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must be in [0, 1)")
    if not 0.0 <= memory_headroom < 1.0:
        raise ValueError("memory_headroom must be in [0, 1)")
    completed = [r for r in results if r.ok and r.samples_per_second]
    if not completed:
        return None
    notes: list[str] = []
    fastest = max(completed, key=_throughput)

    candidates = completed
    if total_memory_bytes is not None:
        budget = (1.0 - memory_headroom) * total_memory_bytes
        within = [r for r in completed if r.peak_reserved_bytes is None or r.peak_reserved_bytes <= budget]
        if within:
            candidates = within
            if fastest not in within:
                notes.append(
                    f"The fastest run (batch {fastest.batch_size}) used more than "
                    f"{1 - memory_headroom:.0%} of usable memory, so it is not recommended."
                )
        else:
            notes.append(f"Every run used more than {1 - memory_headroom:.0%} of usable memory.")

    best = max(_throughput(r) for r in candidates)
    acceptable = [r for r in candidates if _throughput(r) >= (1.0 - tolerance) * best]
    recommended = min(acceptable, key=lambda r: (r.batch_size, r.num_workers))

    oom_sizes = [r.batch_size for r in results if r.status is TrialStatus.OOM]
    first_oom = min(oom_sizes) if oom_sizes else None
    host_oom_sizes = [r.batch_size for r in results if r.status is TrialStatus.HOST_OOM]
    largest_fit = max(r.batch_size for r in completed)

    wait = recommended.data_wait_fraction or 0.0
    if wait >= input_bound_threshold:
        ceiling = recommended.compute_samples_per_second or 0.0
        notes.append(
            f"Input-bound: the device waited on the DataLoader {wait:.0%} of the time. "
            f"The device alone could do {ceiling:,.0f} img/s versus "
            f"{_throughput(recommended):,.0f} img/s end-to-end. Add workers (--num-workers), "
            "speed up decoding/preprocessing, or move preprocessing to the GPU before "
            "raising the batch size."
        )
    if host_oom_sizes:
        notes.append(
            f"Host RAM ran short at batch {min(host_oom_sizes)}. "
            "Fewer workers (--num-workers), --prefetch-factor 1 or --no-pin-memory need less host memory."
        )
    elif first_oom is None:
        limit = f" up to batch {max_tested}" if max_tested is not None else ""
        notes.append(
            f"No out-of-memory{limit}; raise --max-batch-size (or list larger --batch-sizes) "
            "to find the memory ceiling."
        )
        if fastest.batch_size == largest_fit:
            notes.append("Throughput was still highest at the largest batch tested.")
    return Recommendation(
        recommended=recommended,
        fastest=fastest,
        largest_fit=largest_fit,
        first_oom=first_oom,
        notes=tuple(notes),
    )
