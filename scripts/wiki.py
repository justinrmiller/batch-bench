"""Regenerate the GitHub wiki's results pages from ``benchmarks/``.

Each folder in ``benchmarks/`` holds one machine's ``batch-bench run`` output,
the ``.json`` and ``.csv`` pairs copied from ``results/``. This writes Home,
one page per task, Hardware and the sidebar into a clone of the wiki::

    git clone https://github.com/justinrmiller/batch-bench.wiki.git ../batch-bench.wiki
    make wiki WIKI=../batch-bench.wiki

Then review, commit and push the wiki clone. What the data can't say (machine
specs, commentary on a result) lives in ``HARDWARE`` and ``OBSERVATIONS``
below, so regenerating keeps it.
"""

# The pages use typographic multiplication, minus and en-dash signs on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from batch_bench.models import DEFAULT_PRESETS, PRESETS, Task
from batch_bench.sweep import doubling_batch_sizes

REPO = "https://github.com/justinrmiller/batch-bench"
MIB, GIB = 2**20, 2**30
BACKENDS = {"cuda": "CUDA", "rocm": "ROCm", "xpu": "XPU", "mps": "MPS", "cpu": "CPU"}


@dataclass(frozen=True)
class Hardware:
    """A test machine: one folder in ``benchmarks/``.

    Attributes:
        key: Folder name in ``benchmarks/``.
        title: Section heading for the machine on every page.
        specs: Rows for the Hardware page, ahead of the ones read from the runs.
        notes: Bullets under the machine's Hardware section.
    """

    key: str
    title: str
    specs: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


# Every page shows these machines, in this order, with or without results.
# A folder in benchmarks/ that isn't listed still gets a section, named after
# the device its runs report.
HARDWARE = {
    hw.key: hw
    for hw in (
        Hardware(
            "apple-m4-pro",
            "Apple M4 Pro (MPS)",
            specs=(
                ("Chip", "Apple M4 Pro: 12-core CPU (8 performance + 4 efficiency), 16-core GPU"),
                ("Memory", "48 GB unified"),
                ("Backend", "PyTorch MPS (`uv sync --extra mps`)"),
                ("batch-bench default workers", "11 (CPUs − 1)"),
            ),
            notes=(
                "The memory budget is Metal's recommended working set, not the full 48 GB. batch-bench caps MPS "
                "at 1.0× of it so an oversized batch fails with out-of-memory instead of swapping.",
                "MPS has no peak-memory counters. batch-bench polls `driver_allocated_memory()` after every step, "
                "so peak memory is approximate and moves in coarse steps.",
                "DataLoader workers start with `spawn` and share unified memory with the GPU.",
            ),
        ),
        Hardware(
            "amd-r9700",
            "AMD Radeon AI PRO R9700 (ROCm)",
            specs=(
                ("GPU", "AMD Radeon AI PRO R9700: RDNA 4 (`gfx1201`), 32 GB"),
                ("Backend", "PyTorch ROCm 7.2 wheels (`uv sync --extra rocm`)"),
                ("Host", "_To be added with the first results._"),
            ),
            notes=(
                "MIOpen searches for kernels the first time it sees each batch shape. Warmup absorbs it, so the "
                "numbers are unaffected, but first runs take longer.",
            ),
        ),
    )
}

# Wiki page name, label and intro for each task.
TASK_PAGES = {
    Task.OBJECT_DETECTION: (
        "Object-Detection",
        "Object detection",
        "Detectors return boxes and class scores. The DETR and YOLOS presets keep each image's aspect ratio "
        "and pad the batch; the RT-DETR, RT-DETRv2 and D-FINE presets resize every image to 640×640.",
    ),
    Task.IMAGE_SEGMENTATION: (
        "Image-Segmentation",
        "Image segmentation",
        "SegFormer and UperNet return per-pixel class logits (semantic segmentation); Mask2Former returns "
        "mask and class predictions per query (instance segmentation).",
    ),
    Task.EMBEDDING: (
        "Embedding",
        "Embedding",
        "Embedding models turn each image into one feature vector. The CLIP and SigLIP 2 presets load the "
        "vision tower only, which is what an embedding job keeps on the GPU.",
    ),
}

# Commentary on the latest run, keyed by (model, machine, mode). Rewrite it
# when a new run changes the numbers it quotes.
OBSERVATIONS: dict[tuple[str, str, str], list[str]] = {
    ("clip-vit-b32", "apple-m4-pro", "inference"): [
        "**Throughput flattens after batch 64.** Batch 1 → 64 gains 4.6×. Batch 128 adds only 2.4% more, "
        "and the peak is 518.1 img/s at batch 512.",
        "**Past 512 it gets slower:** −6.6% at 1024 and −9.4% at 2048. Stay at or below 512.",
        "**Bigger batches mostly add latency.** Device time per image falls from 9.2 ms at batch 1 to about "
        "2.0 ms at batch 64 and barely moves after that. The step time keeps growing: 129 ms at 64, "
        "252 ms at 128, 986 ms at 512.",
        "**The GPU is the bottleneck, even with one DataLoader worker.** Data wait is ≤1.3% at every size "
        "and ≤0.3% from batch 32 up. More workers can win back at most that wait share.",
        "**Memory is not the constraint.** Nothing ran out of memory up to batch 2048, which peaked at "
        "6.9 GiB (19% of the 37.4 GiB budget).",
        "**Steps are steady.** p95 is within 5% of p50 at every size, so there's no sign of throttling.",
        "**64 vs. 128 is within run-to-run noise.** Batch 64 was 4.4% below the fastest result in this run, "
        "inside the 5% tolerance. In the earlier run it was 7.6% below, so that run picked 128. Batch 64 "
        "moved 3.3% between runs; every other size above 1 moved ≤1.2%. Pick **64** for half the step "
        "latency, or **128** for 2.4% more throughput and a choice that holds across runs.",
        "**Measured before the prefetch-drain timing fix** (4c7bfd7), which corrected inflated throughput for "
        "input-bound loaders. These runs waited on data ≤1.3% of the time, so the fix shouldn't change them; "
        "a rerun would confirm.",
    ],
    ("segformer-b0", "apple-m4-pro", "inference"): [
        "**Small batches win.** Throughput peaks at batch 8 (94.5 img/s), and batch 4 gets 94.4, so it's the "
        "pick. Batch 1 already reaches 78.8 img/s: batching adds only 20% before the GPU is saturated.",
        "**Past 8 it gets slower, not just flatter:** −15% at 16, −19% at 32, −24% at 64 and −65% at 128. "
        "Device time per image climbs from 10.5 ms at batch 4 and 8 to 13.8 ms at 64 and 30.2 ms at 128.",
        "**There's no throughput to find above 128.** batch-bench notes that nothing ran out of memory, but "
        "with throughput falling, a larger `--max-batch-size` would only locate the memory ceiling.",
        "**Activations, not weights, take the memory.** Peak memory is 1.2 GiB at the pick, 10.6 GiB at 64 "
        "and 17.7 GiB (47% of the budget) at 128: 13× what CLIP ViT-B/32 used at batch 128 on this machine, "
        "from a model with a 23rd of its parameters (3.8M vs. 87.8M).",
        "**The slowdown at 128 isn't the Mac swapping.** System swap use was the same before and after the "
        "run (1.2 GB).",
        "**The GPU is the bottleneck with one DataLoader worker.** Data wait is ≤1.3% at every size, and "
        "img/s is within 1% of Dev img/s.",
        "**Steps are steady.** p95 is within 2.1% of p50 at every size.",
        "**Measured after the prefetch-drain timing fix** (4c7bfd7), unlike the CLIP runs on this machine.",
    ],
}


# ---------------------------------------------------------------- formatting


def _bytes(value: int | None) -> str:
    """Human-readable size, matching ``batch-bench run``'s tables."""
    if value is None:
        return "-"
    return f"{value / GIB:.1f} GiB" if value >= GIB else f"{value / MIB:.0f} MiB"


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _peak(result: dict, budget: int | None) -> str:
    text = _bytes(result["peak_reserved_bytes"])
    if budget and result["peak_reserved_bytes"] is not None:
        text += f" ({result['peak_reserved_bytes'] / budget:.0%})"
    return text


def _code_flags(text: str) -> str:
    """Put backticks around the --flags in batch-bench's plain-text notes."""
    return re.sub(r"(?<!`)(--[a-z][a-z-]*)", r"`\1`", text)


def _slug(text: str) -> str:
    """The anchor GitHub gives a heading."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    """A Markdown table; ``align`` has one ``l`` or ``r`` per column."""
    rule = ["---:" if a == "r" else "---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(rule) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


# ---------------------------------------------------------------- loading


@dataclass
class Run:
    """One model's results from one results file."""

    path: Path
    file: dict
    model: dict
    hardware: Hardware

    @property
    def settings(self) -> dict:
        """The run's ``settings`` block."""
        return self.file["settings"]

    @property
    def mode(self) -> str:
        """``inference`` or ``training``."""
        return self.settings["mode"]

    @property
    def created(self) -> str:
        """Start time, sortable."""
        return self.file["created_at"]

    @property
    def rec(self) -> dict | None:
        """The model's recommendation, or None if no batch size completed."""
        return self.model["recommendation"]

    def when(self) -> str:
        """Start time for display."""
        return datetime.fromisoformat(self.created).strftime("%Y-%m-%d %H:%M UTC")

    def data_links(self) -> str:
        """Links to this run's files in the repository."""
        base = f"{REPO}/blob/main/benchmarks/{self.hardware.key}/{self.path.stem}"
        return f"[JSON]({base}.json) · [CSV]({base}.csv)"


Index = dict[str, dict[str, list[Run]]]


def load(benchmarks: Path) -> tuple[list[Run], list[Hardware]]:
    """Every model run under ``benchmarks/``, and the machines to show."""
    hardware = dict(HARDWARE)
    runs: list[Run] = []
    for path in sorted(benchmarks.glob("*/*.json")):
        data = json.loads(path.read_text())
        key = path.parent.name
        if key not in hardware:
            device = data["device"]
            hardware[key] = Hardware(key, f"{device['name']} ({BACKENDS.get(device['backend'], device['backend'])})")
        runs.extend(Run(path, data, model, hardware[key]) for model in data["models"])
    return runs, list(hardware.values())


def index_runs(runs: list[Run]) -> Index:
    """Runs by model name, then by machine."""
    index: Index = defaultdict(lambda: defaultdict(list))
    for run in runs:
        index[run.model["name"]][run.hardware.key].append(run)
    return index


def model_names(task: Task, runs: list[Run]) -> list[str]:
    """The task's presets (default first), then any other checkpoints that have results."""
    default = DEFAULT_PRESETS[task]
    names = [default, *(name for name, spec in PRESETS.items() if spec.task is task and name != default)]
    return names + sorted({r.model["name"] for r in runs if r.model["task"] == task.value} - set(names))


# ---------------------------------------------------------------- sections


def reproduce(run: Run) -> str:
    """The ``batch-bench run`` command that repeats this run's settings."""
    s, device = run.settings, run.file["device"]
    parts = ["uv run batch-bench run", f"-m {run.model['name']}"]
    if device["index"]:
        parts.append(f"--device-index {device['index']}")
    if s["mode"] != "inference":
        parts.append(f"--mode {s['mode']}")
    parts.append(f"--precision {s['precision']}")
    parts.append("--num-workers " + ",".join(map(str, s["worker_counts"])))
    sizes = s["batch_sizes"]
    if sizes == doubling_batch_sizes(min(sizes), max(sizes)):
        if min(sizes) != 1:
            parts.append(f"--min-batch-size {min(sizes)}")
        if max(sizes) != 1024:
            parts.append(f"--max-batch-size {max(sizes)}")
    else:
        parts.append("--batch-sizes " + ",".join(map(str, sizes)))
    if s["image_source"] != "synthetic":
        parts.append("--image-dir <your-images>")
    if s["image_size"] and list(s["image_size"]) != [640, 480]:
        parts.append(f"--image-size {s['image_size'][0]}x{s['image_size'][1]}")
    defaults = {
        "warmup_batches": ("--warmup-batches", 3),
        "measure_batches": ("--measure-batches", 16),
        "prefetch_factor": ("--prefetch-factor", 2),
        "refine_steps": ("--refine-steps", 3),
        "tolerance": ("--tolerance", 0.05),
        "memory_headroom": ("--memory-headroom", 0.1),
        "preprocess": ("--preprocess", "batch"),
        "mp_context": ("--mp-context", None),
    }
    parts += [f"{flag} {s[key]}" for key, (flag, default) in defaults.items() if s.get(key) != default]
    # MPS runs are capped at 1.0 unless told otherwise, so that isn't a choice.
    fraction = s.get("memory_fraction")
    if fraction is not None and not (device["backend"] == "mps" and fraction == 1.0):
        parts.append(f"--memory-fraction {fraction}")
    if s["compile_model"]:
        parts.append("--compile")
    if not s["pin_memory"]:
        parts.append("--no-pin-memory")
    return " ".join(parts)


def settings_line(run: Run) -> str:
    """One line of run conditions and software versions."""
    s, env = run.settings, run.file["environment"]
    if s["image_source"] == "synthetic":
        source = f"synthetic {s['image_size'][0]}×{s['image_size'][1]} JPEGs"
    else:
        source = "folder images (`--image-dir`)"
    workers = ", ".join(map(str, s["worker_counts"]))
    items = [
        s["mode"].capitalize(),
        f"{s['precision']} autocast",
        f"{workers} DataLoader worker{'' if workers == '1' else 's'}",
        source,
        f"{s['warmup_batches']} warmup + {s['measure_batches']} measured batches per trial",
    ]
    if s["compile_model"]:
        items.append("`torch.compile`")
    items.append(
        f"batch-bench {run.file['batch_bench_version']}, torch {env['torch']}, "
        f"transformers {env['transformers']}, Python {env['python']}, {run.file['device']['runtime']}"
    )
    items.append(f"run {run.when()}")
    return " · ".join(items)


def headline(run: Run) -> str:
    """The recommendation, fastest batch and memory ceiling in two sentences."""
    rec, budget = run.rec, run.settings["memory_budget_bytes"]
    if rec is None:
        error = run.model["error"]
        return f"**Failed:** {error}" if error else "**No batch size completed; nothing to recommend.**"
    best, fastest = rec["recommended"], rec["fastest"]
    memory = _bytes(best["peak_reserved_bytes"])
    if budget and best["peak_reserved_bytes"] is not None:
        memory += f" ({best['peak_reserved_bytes'] / budget:.0%} of the {_bytes(budget)} budget)"
    oom = f"first out-of-memory at {rec['first_oom']}" if rec["first_oom"] is not None else "no out-of-memory"
    return (
        f"**Recommended: batch {best['batch_size']}, {best['num_workers']} worker"
        f"{'' if best['num_workers'] == 1 else 's'}.** "
        f"{_fmt(best['samples_per_second'], ',.1f')} img/s, step p50 {_fmt(best['latency_p50_ms'], '.1f')} ms, "
        f"peak memory {memory}. "
        f"Fastest: batch {fastest['batch_size']} at {_fmt(fastest['samples_per_second'], ',.1f')} img/s. "
        f"Largest batch that fit: {rec['largest_fit']} ({oom})."
    )


def _is(result: dict, chosen: dict | None) -> bool:
    return chosen is not None and (result["batch_size"], result["num_workers"]) == (
        chosen["batch_size"],
        chosen["num_workers"],
    )


def results_table(run: Run) -> list[str]:
    """Every trial, in the same columns as ``batch-bench run`` prints."""
    rec, budget = run.rec, run.settings["memory_budget_bytes"]
    rows = []
    for r in sorted(run.model["results"], key=lambda r: (r["num_workers"], r["batch_size"])):
        marks = []
        if _is(r, rec and rec["recommended"]):
            marks.append("**recommended**")
        if _is(r, rec and rec["fastest"]):
            marks.append("fastest")
        if r["error"]:
            marks.append(r["error"].replace("|", "\\|")[:120])
        rows.append(
            [
                str(r["batch_size"]),
                str(r["num_workers"]),
                r["status"],
                _fmt(r["samples_per_second"], ",.1f"),
                _fmt(r["compute_samples_per_second"], ",.1f"),
                _fmt(r["latency_p50_ms"], ".1f"),
                _fmt(r["latency_p95_ms"], ".1f"),
                _fmt(r["data_wait_fraction"], ".0%"),
                _peak(r, budget),
                ", ".join(marks),
            ]
        )
    header = ["Batch", "Workers", "Status", "img/s", "Dev img/s", "p50 ms", "p95 ms", "Wait", "Peak mem", ""]
    return _table(header, rows, "rrlrrrrrrl")


def history(runs: list[Run]) -> list[str]:
    """Collapsed list of every run for one model, machine and mode, newest first."""
    rows = []
    for run in runs:
        sizes = [r["batch_size"] for r in run.model["results"]]
        rec = run.rec
        picks = [
            f"{rec[k]['batch_size']} ({rec[k]['samples_per_second']:,.1f} img/s)" if rec else "-"
            for k in ("recommended", "fastest")
        ]
        tested = f"{min(sizes)}–{max(sizes)}" if sizes else "-"
        rows.append([run.when(), tested, *picks, run.data_links()])
    out = [
        "<details>",
        f"<summary>All runs ({len(runs)})</summary>",
        "",
        *_table(["Run", "Batch sizes", "Recommended", "Fastest", "Data"], rows, "lrrrl"),
    ]
    if len(runs) > 1:
        shown = runs[:4]
        by_run = [
            {(r["batch_size"], r["num_workers"]): r["samples_per_second"] for r in run.model["results"]}
            for run in shown
        ]
        keys = sorted(set().union(*by_run), key=lambda k: (k[1], k[0]))
        multi_worker = len({k[1] for k in keys}) > 1
        header = ["Batch", *(["Workers"] if multi_worker else []), *(run.when() for run in shown), "Change"]
        cmp_rows = []
        for key in keys:
            newest, oldest = by_run[0].get(key), by_run[-1].get(key)
            change = f"{newest / oldest - 1:+.1%}" if newest and oldest else "-"
            cmp_rows.append(
                [
                    str(key[0]),
                    *([str(key[1])] if multi_worker else []),
                    *(_fmt(m.get(key), ",.1f") for m in by_run),
                    change,
                ]
            )
        out += [
            "",
            "End-to-end img/s by batch size, newest run first. *Change* is the newest run against the oldest shown.",
            "",
            *_table(header, cmp_rows, "r" * len(header)),
        ]
    return [*out, "", "</details>"]


def hardware_section(model_name: str, hw: Hardware, runs: list[Run]) -> list[str]:
    """One machine's subsection: the latest run per mode, then its history."""
    if not runs:
        return [
            f"### {hw.title}",
            "",
            f"_No results yet._ Run `uv run batch-bench run -m {model_name}` on this machine.",
            "",
        ]
    by_mode: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        by_mode[run.mode].append(run)
    out = []
    for mode in sorted(by_mode, key=lambda m: m != "inference"):
        mode_runs = sorted(by_mode[mode], key=lambda r: r.created, reverse=True)
        latest = mode_runs[0]
        title = hw.title if mode == "inference" else f"{hw.title}, {mode}"
        out += [f"### {title}", "", headline(latest), "", f"<sub>{settings_line(latest)}</sub>", ""]
        if latest.model["results"]:
            out += [*results_table(latest), ""]
        if observations := OBSERVATIONS.get((model_name, hw.key, mode)):
            out += ["**Observations**", "", *(f"- {o}" for o in observations), ""]
        if notes := (latest.rec or {}).get("notes"):
            out += ["**Notes from batch-bench**", "", *(f"- {_code_flags(n)}" for n in notes), ""]
        out += ["**Reproduce**", "", "```bash", reproduce(latest), "```", ""]
        out += [*history(mode_runs), ""]
    return out


def comparison(hardware: list[Hardware], runs_by_hw: dict[str, list[Run]]) -> list[str]:
    """Latest inference recommendation per machine, once two or more have one."""
    rows = []
    for hw in hardware:
        inference = sorted((r for r in runs_by_hw.get(hw.key, []) if r.mode == "inference"), key=lambda r: r.created)
        if not inference:
            continue
        run = inference[-1]
        if run.rec is None:
            rows.append([hw.title, "-", "-", "-", "-", "-"])
            continue
        best, fastest = run.rec["recommended"], run.rec["fastest"]
        rows.append(
            [
                hw.title,
                f"**{best['batch_size']}**",
                _fmt(best["samples_per_second"], ",.1f"),
                _fmt(best["latency_p50_ms"], ".1f"),
                _peak(best, run.settings["memory_budget_bytes"]),
                f"{fastest['batch_size']} ({fastest['samples_per_second']:,.1f})",
            ]
        )
    if len(rows) < 2:
        return []
    header = ["Machine", "Recommended batch", "img/s", "Step p50 ms", "Peak mem", "Fastest batch (img/s)"]
    return [*_table(header, rows, "lrrrrr"), ""]


def cell(runs: list[Run]) -> str:
    """Summary-table cell: the latest inference recommendation."""
    inference = sorted((r for r in runs if r.mode == "inference"), key=lambda r: r.created)
    if not inference:
        return "—"
    rec = inference[-1].rec
    if rec is None:
        return "failed"
    best = rec["recommended"]
    return f"**{best['batch_size']}** · {best['samples_per_second']:,.0f} img/s"


# ---------------------------------------------------------------- pages


def task_page(task: Task, runs: list[Run], hardware: list[Hardware]) -> str:
    """A task's page: summary, then a section per model and a subsection per machine."""
    _, _, intro = TASK_PAGES[task]
    names = model_names(task, runs)
    index = index_runs(runs)
    summary = [[f"[{n}](#{_slug(n)})", *(cell(index[n][hw.key]) for hw in hardware)] for n in names]
    out = [
        intro,
        "",
        "Each model below has one section per test machine ([Hardware](Hardware)). The summary shows the "
        "recommended batch size and the end-to-end throughput it gets in inference; — means not run yet.",
        "",
        *_table(["Model", *(hw.title for hw in hardware)], summary, "l" + "r" * len(hardware)),
        "",
    ]
    for name in names:
        spec = PRESETS.get(name)
        model_runs = [r for r in runs if r.model["name"] == name]
        model_id = spec.model_id if spec else model_runs[0].model["model_id"]
        params = next((r.model["parameters"] for r in model_runs if r.model["parameters"]), None)
        about = [f"[`{model_id}`](https://huggingface.co/{model_id})"]
        if spec and spec.description:
            about.append(spec.description)
        if params:
            about.append(f"{params / 1e6:.1f}M parameters")
        if DEFAULT_PRESETS[task] == name:
            about.append(f"default preset for `-t {task.value}`")
        if spec is None:
            about.append("not a preset")
        out += [f"## {name}", "", " · ".join(about), ""]
        out += comparison(hardware, index[name])
        for hw in hardware:
            out += hardware_section(name, hw, index[name][hw.key])
    return "\n".join(out).rstrip() + "\n"


def home_page(runs: list[Run], hardware: list[Hardware]) -> str:
    """Overview: results matrix, page guide, method and how to read the tables."""
    index = index_runs(runs)
    rows = [
        [label, f"[{name}]({page}#{_slug(name)})", *(cell(index[name][hw.key]) for hw in hardware)]
        for task, (page, label, _) in TASK_PAGES.items()
        for name in model_names(task, runs)
    ]
    columns = [
        ["img/s", "End-to-end images per second, including time waiting on the DataLoader. What a real job gets."],
        ["Dev img/s", "Images per second from the median device step alone: the ceiling if data loading were free."],
        [
            "p50 ms / p95 ms",
            "Device step time per batch: copy and forward (plus backward and optimizer step in training).",
        ],
        ["Wait", "Share of wall time the device sat blocked on the DataLoader."],
        ["Peak mem", "Peak device memory, with its share of the memory batch-bench could use."],
    ]
    lines = [
        "**batch-bench** finds the DataLoader batch size that gets the most images per second out of a GPU "
        "for HuggingFace object detection, image segmentation and embedding models. This wiki collects its "
        "results on real hardware.",
        "",
        f"New to the tool? [QUICKSTART]({REPO}/blob/main/QUICKSTART.md) gets you from install to a "
        f"recommendation; the [README]({REPO}#readme) is the full reference.",
        "",
        "## Results at a glance",
        "",
        "Recommended batch size and the end-to-end throughput it gets, inference, latest run per machine. "
        "— means not run yet. Each model links to its full results.",
        "",
        *_table(["Task", "Model", *(hw.title for hw in hardware)], rows, "ll" + "r" * len(hardware)),
        "",
        "## Pages",
        "",
        "- **[Object Detection](Object-Detection)**: DETR, RT-DETR, RT-DETRv2, D-FINE, YOLOS",
        "- **[Image Segmentation](Image-Segmentation)**: SegFormer B0 and B5, UperNet ConvNeXt-S, Mask2Former Swin-T",
        "- **[Embedding](Embedding)**: CLIP ViT-B/32 and ViT-L/14, SigLIP 2, DINOv2 S and B",
        "- **[Hardware](Hardware)**: the test machines and their software",
        "",
        "Every model page has one section per model and one subsection per machine. A subsection has the "
        "recommendation, the run settings, the full results table, notes, the command to reproduce it, and "
        "the raw JSON and CSV for every run.",
        "",
        "## How the batch size is chosen",
        "",
        "1. **Sweep.** Batch sizes double (1, 2, 4, …) until a trial runs out of memory or reaches "
        "`--max-batch-size`, then a few bisection steps narrow in on the largest batch that fits.",
        "2. **Recommend.** The pick is the *smallest* batch within 5% of the best end-to-end throughput "
        "whose peak memory stays under 90% of usable memory. Past that knee, bigger batches cost memory and "
        "latency but add no throughput.",
        "3. **Diagnose.** If the GPU waited on the DataLoader for 15% or more of the run, the run is "
        "*input-bound*, and more workers or cheaper preprocessing will help where a bigger batch won't.",
        "",
        "## Reading a results table",
        "",
        *_table(["Column", "Meaning"], columns, "ll"),
        "",
        "Patterns worth recognising:",
        "",
        "- **img/s rises, then flattens.** Normal. The recommendation sits where it flattens.",
        "- **Dev img/s is far above img/s and Wait is high.** The GPU is starved: add workers or speed up "
        "decoding rather than raising the batch size.",
        "- **p95 is well above p50.** Uneven steps: other GPU work, throttling, or memory pressure. Treat that "
        "batch size with suspicion.",
        "- **img/s drops at the largest sizes.** Allocator fragmentation or (on MPS) memory pressure. Stay below it.",
        "",
        "## Test conditions",
        "",
        "Unless a section says otherwise:",
        "",
        "- Every trial runs the whole pipeline a real job runs: JPEG decode and the model's HuggingFace image "
        "processor in DataLoader workers, the copy to the device, then the forward pass.",
        "- Images are synthetic 640×480 JPEGs, decoded for every sample, so decode cost is realistic without "
        "a dataset.",
        "- Inference with autocast and float32 weights; 3 warmup and 16 measured batches per trial.",
        "",
        "Results depend on the pipeline as well as the GPU. With your own images (`--image-dir`) decode cost "
        "and input sizes change, so rerun on your data before relying on a number.",
        "",
    ]
    return "\n".join(lines)


def hardware_page(runs: list[Run], hardware: list[Hardware]) -> str:
    """Specs, software versions and backend notes for each machine."""
    out = ["The machines behind the results. Software versions come from the latest run on each machine.", ""]
    for hw in hardware:
        hw_runs = sorted((r for r in runs if r.hardware.key == hw.key), key=lambda r: r.created)
        specs = list(hw.specs)
        if hw_runs:
            latest = hw_runs[-1].file
            env, device = latest["environment"], latest["device"]
            if not specs:
                specs.append(("Device", f"{device['name']} ({BACKENDS.get(device['backend'], device['backend'])})"))
            specs.append(("OS", device["runtime"] if device["backend"] == "mps" else env["platform"]))
            if device.get("arch"):
                specs.append(("Arch", f"`{device['arch']}`"))
            budget, total = latest["settings"]["memory_budget_bytes"], device["total_memory_bytes"]
            of_total = f" of the {_bytes(total)} the device reports" if budget != total else ""
            specs.append(("Memory budget", f"{_bytes(budget)}{of_total}"))
            specs.append(
                (
                    "Software",
                    f"Python {env['python']}, torch {env['torch']}, transformers {env['transformers']}, "
                    f"batch-bench {latest['batch_bench_version']}",
                )
            )
            specs.append(("Models run", ", ".join(f"`{m}`" for m in sorted({r.model["name"] for r in hw_runs}))))
        else:
            specs.append(("Results", "_None yet._"))
        out += [f"## {hw.title}", "", *_table(["", ""], [[f"**{k}**", v] for k, v in specs], "ll"), ""]
        if hw.notes:
            out += [*(f"- {n}" for n in hw.notes), ""]
    return "\n".join(out)


SIDEBAR = f"""**[Home](Home)**

**Results by model type**

- [Object Detection](Object-Detection)
- [Image Segmentation](Image-Segmentation)
- [Embedding](Embedding)

**Reference**

- [Hardware](Hardware)
- [Quickstart]({REPO}/blob/main/QUICKSTART.md)
- [README]({REPO}#readme)
"""


def main() -> None:
    """Write every page into the wiki clone."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmarks", type=Path, default=Path("benchmarks"), help="folder of per-machine results")
    parser.add_argument("--wiki", type=Path, required=True, help="clone of the batch-bench wiki to write into")
    args = parser.parse_args()
    if not (args.wiki / ".git").exists():
        parser.error(f"{args.wiki} is not a git clone; git clone {REPO}.wiki.git {args.wiki}")

    runs, hardware = load(args.benchmarks)
    pages = {"Home": home_page(runs, hardware), "Hardware": hardware_page(runs, hardware), "_Sidebar": SIDEBAR}
    pages |= {page: task_page(task, runs, hardware) for task, (page, _, _) in TASK_PAGES.items()}
    for name, text in pages.items():
        (args.wiki / f"{name}.md").write_text(text)
    print(f"Wrote {len(pages)} pages from {len(runs)} model runs to {args.wiki}")


if __name__ == "__main__":
    main()
