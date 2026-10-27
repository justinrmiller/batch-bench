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

from batch_bench.cli import DEFAULT_MEMORY_FRACTION
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
                ("GPU", "AMD Radeon AI PRO R9700: RDNA 4 (`gfx1201`), 64 compute units, 32 GB"),
                ("Host", "AMD Ryzen 5 5600X (6 cores, 12 threads), 64 GB RAM, Ubuntu 26.04.1 LTS"),
                ("Backend", "PyTorch ROCm 7.2 wheels (`uv sync --extra rocm`)"),
                ("batch-bench default workers", "11 (CPUs − 1)"),
            ),
            notes=(
                "MIOpen searches for kernels the first time it sees each batch shape. Warmup absorbs it, so the "
                "numbers are unaffected, but first runs take longer.",
                "The card also drives the desktop, so runs use 75% of it (`--memory-fraction`'s default), a "
                "23.9 GiB budget. Sweeps that filled it crashed GNOME Shell, and the ROCm runtime uses another "
                "1–4.6 GiB beyond PyTorch's cap.",
                "PyTorch's ROCm build drives the GPU through the `torch.cuda` API, so out-of-memory errors in the "
                "results tables read *CUDA out of memory*.",
                "The DataLoader workers share a 6-core CPU with the process that feeds the GPU, so input-bound "
                "models are limited by the host, not the card.",
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
    ("clip-vit-b32", "amd-r9700", "inference"): [
        "**Input-bound at every worker count.** With 4 workers the device waited on the DataLoader 38–65% "
        "of the time from batch 16 up; with 1 worker, 79–87%. From batch 128 up, 1 worker gets 436–495 "
        "img/s, 2 get 766–864 and 4 get 1,222–1,347, so each added worker supplies less, about 300–500 img/s.",
        "**The pick is batch 128 with 4 workers:** 1,347.0 img/s at a 36.4 ms step p50 and 0.7 GiB. Larger "
        "batches add latency but no throughput (1,329.6 img/s at 1024, 293.8 ms). The sweep stopped at 4 of "
        "the 11 default workers, and workers are the lever here, not batch size.",
        "**The device is fastest when kept busy.** Its own rate (Dev img/s) at batch 128 is 2,691.7 with 1 "
        "worker, 3,212.4 with 2 and 3,511.9 with 4. With one worker the GPU idles between steps, likely "
        "clocking down. Alone it peaks around 3,500 img/s, 2.6× what the pipeline delivers.",
        "**Batch 16 is slower than batch 8 at every worker count:** 298.4 vs. 504.1 img/s with 1 worker, "
        "588.8 vs. 942.5 with 2 and 958.5 vs. 1,076.1 with 4. The 2026-10-18 run showed the same dip with 1 "
        "and 2 workers, so it's repeatable. It comes from the input pipeline: data wait jumps (1.5% → 38.4% with "
        "4 workers) while Dev img/s rises.",
        "**Memory is no constraint:** 3.1 GiB (13% of the 23.9 GiB budget) at batch 1024. Host RAM costs "
        "more there: 9.8 GiB with 4 workers.",
        "**Against the M4 Pro:** 2.6× its best end-to-end result (518.1 img/s), and the device alone is 6.8× "
        "faster (3,511.9 vs. 519.3 Dev img/s). With one worker the two are about level (504.1 img/s here at "
        "batch 8). Precision differs: bf16 here, fp16 on the M4 Pro.",
    ],
    ("dinov2-small", "amd-r9700", "inference"): [
        "**Input-bound, like CLIP ViT-B/32.** The pick, batch 128 with 4 workers, gets 1,274.2 img/s and "
        "waited on the DataLoader 26.6% of the time. With 1 worker the wait is 58–81% from batch 8 up, at "
        "362–491 img/s.",
        "**Batch 128 is a clear peak with 4 workers:** 5.3% above batch 64 (1,209.5 img/s) and 7–8% above "
        "256–1024 (1,184.0–1,189.2), at a 71.1 ms step p50 and 0.9 GiB.",
        "**The device saturates early.** Dev img/s stays between 1,759 and 2,498 from batch 32 up. "
        "DINOv2-small has a quarter of CLIP ViT-B/32's parameters (22.1M vs. 87.8M) but about half its device "
        "rate, because patch size 14 gives 257 tokens per image against CLIP's 50.",
        "**Batch 16 dips with 2 and 4 workers** (604.4 vs. 856.4 and 883.9 vs. 1,020.2 img/s at batch 8), "
        "as it does for CLIP ViT-B/32.",
        "**Memory isn't the limit:** 5.4 GiB (23% of the budget) at batch 1024.",
    ],
    ("dinov2-base", "amd-r9700", "inference"): [
        "**GPU-bound with 4 workers.** Data wait is ≤1.4% at every batch size, and throughput is flat from "
        "batch 32 (1,106.8 img/s) to 1024 (1,120.2), apart from a dip at 64 (1,040.0). So the pick is 32, at "
        "a 27.6 ms step p50 and 0.8 GiB.",
        "**Fewer workers can't keep up.** 1 worker gets 417–486 img/s from batch 8 up, waiting 27–52% of the "
        "time; 2 workers get 595–891 img/s from batch 16 up, waiting 12–18%.",
        "**Same size as CLIP ViT-B/32, a third of its device rate:** about 0.9 ms per image (1,088–1,158 Dev "
        "img/s with 4 workers from batch 32) against 0.28 ms, since patch size 14 gives 257 tokens per image "
        "against CLIP's 50.",
        "**Memory:** 10.3 GiB (43% of the budget) at batch 1024, and nothing ran out.",
    ],
    ("siglip2-base", "amd-r9700", "inference"): [
        "**GPU-bound with 4 workers from batch 16 up** (data wait ≤1.6%, except 9.1% at batch 512). The "
        "pick, batch 128, gets 1,332.4 img/s at a 94.6 ms step p50, 2.4% below the fastest result, batch 1024 "
        "at 1,365.4.",
        "**About 0.73 ms of device time per image**, between CLIP ViT-B/32 (0.28 ms, 50 tokens) and "
        "DINOv2-base (0.9 ms, 257 tokens): patch size 16 at 224 px gives 196 tokens.",
        "**One or two workers leave it input-bound:** 1 worker gets 304–531 img/s from batch 8 up (22–66% "
        "wait), 2 workers 612–939 img/s.",
        "**Batch 32 dips with 1 and 2 workers** (304.4 and 612.0 img/s, against 457.4 and 795.2 at batch 16), "
        "an input-pipeline dip like the ones CLIP ViT-B/32 and DINOv2-small show at 16.",
        "**Memory:** 7.7 GiB (32% of the budget) at batch 1024.",
    ],
    ("clip-vit-l14", "amd-r9700", "inference"): [
        "**GPU-bound at every worker count.** Data wait is under 1% everywhere but one trial (8.7% at batch "
        "32 with 1 worker), and from batch 16 up throughput sits at 351–360 img/s whether 1, 2 or 4 workers "
        "feed it.",
        "**Batch 16 is the pick and the fastest:** 359.5 img/s at a 44.0 ms step p50, 4.9× batch 1. Past 16 "
        "it only adds latency: 2.9 s per step at 1024, for 2.4% less throughput.",
        "**About 2.8 ms of device time per image**, 10× CLIP ViT-B/32: 3.5× the parameters (304.0M vs. "
        "87.8M) and 257 tokens per image instead of 50.",
        "**One worker is enough.** Extra workers only add host RAM: 4.4 GiB with 1 worker at batch 1024, "
        "12.2 GiB with 4.",
        "**The heaviest embedding preset on memory:** 16.1 GiB (68% of the budget) at batch 1024, still "
        "without running out.",
    ],
    ("segformer-b0", "amd-r9700", "inference"): [
        "**Flat from batch 8 up with 2 or more workers:** 223.5–233.9 img/s, peaking at batch 16. The pick is "
        "batch 8 with 2 workers, 223.5 img/s at a 35.3 ms step p50, within 5% of the peak at half its memory "
        "(1.8 vs. 3.6 GiB).",
        "**GPU-bound with 2 or more workers:** data wait is ≤1.8% at every size.",
        "**One worker can't keep up past batch 4.** It peaks at 214.4 img/s at batch 4, then falls to "
        "126–139 img/s with the device waiting 16–40% of the time. Its device steps run slower too (150–165 "
        "Dev img/s at batch 8–32, against about 230 with 2 workers), consistent with the GPU clocking down "
        "between steps.",
        "**Memory sets the ceiling.** Each image adds about 225 MiB (14.1 GiB at batch 64), and batch 112, "
        "the largest that fit, used 22.9–23.6 GiB (96–99% of the 23.9 GiB budget). Batch 120 was projected "
        "to need 25.3 GiB and skipped.",
        "**Against the M4 Pro:** 2.4× its pick (223.5 vs. 94.4 img/s), and throughput holds as batches grow, "
        "where the M4 Pro's fell 65% by batch 128. Precision differs: bf16 here, fp16 on the M4 Pro.",
    ],
    ("segformer-b5", "amd-r9700", "inference"): [
        "**GPU-bound and flat:** 21.0–24.4 img/s at every batch size and worker count, with data wait ≤0.3%. "
        "Device time per image is 41–48 ms, about 10× SegFormer-B0's, from 22× the parameters (84.7M vs. "
        "3.8M) at 640 px instead of 512.",
        "**The pick is batch 4 with 1 worker:** 23.5 img/s at a 169.8 ms step p50 and 4.2 GiB. Batch 16 is "
        "the fastest at 24.4 img/s, 3.8% more for nearly 4× the latency and memory.",
        "**Workers don't matter:** one keeps the GPU fed, and 2 or 4 are within 3% of it.",
        "**Memory sets the ceiling at batch 24:** about 1 GiB per image, 22.9–23.7 GiB at 24 (96–99% of the "
        "budget). Batch 32 was projected to need 30.6–31.4 GiB and skipped.",
    ],
    ("upernet-convnext-small", "amd-r9700", "inference"): [
        "**GPU-bound and flat:** 40.3–48.6 img/s at every batch size and worker count, with data wait ≤0.5%. "
        "From batch 4 up, device time per image holds at 20.6–21.1 ms.",
        "**The pick is batch 4 with 1 worker:** 47.2 img/s at an 84.3 ms step p50 and 3.3 GiB, 2.9% below the "
        "fastest result (batch 16 with 4 workers, 48.6 img/s at 328.9 ms).",
        "**Workers don't matter:** 1, 2 and 4 are within 1% of each other at every batch size.",
        "**Memory sets the ceiling at batch 40:** 23.7–23.8 GiB, 99–100% of the budget. Batch 48 was "
        "projected to need 28.5 GiB and skipped.",
        "**Half SegFormer-B5's device time per image** (21 vs. 41 ms) at a similar size (81.9M vs. 84.7M "
        "parameters), on 512 px inputs instead of 640.",
    ],
    ("mask2former-swin-tiny", "amd-r9700", "inference"): [
        "**GPU-bound and flat from batch 8 up:** 88.2–91.5 img/s at every worker count, with data wait "
        "≤0.2%. Device time per image falls from 35 ms at batch 1 to about 11 ms at 8 and stays there.",
        "**The pick is batch 8 with 1 worker:** 89.0 img/s at an 89.5 ms step p50 and 1.7 GiB, 2.7% below "
        "the fastest result (batch 16 with 4 workers, 91.5 img/s). Batch 1 gets 28.1 img/s, so batching to 8 "
        "gains 3.2×.",
        "**Workers don't matter:** 1, 2 and 4 are within 2% of each other from batch 2 up.",
        "**Memory sets the ceiling at batch 120:** 23.3–23.6 GiB (97–99% of the budget). Batch 128 ran out of "
        "memory with 1 worker, where the projection from batch 64 put it just under the budget, and was "
        "skipped with 2 and 4 (projected 24.8–25.2 GiB).",
    ],
    ("detr-resnet-50", "amd-r9700", "inference"): [
        "**One worker can't keep up.** It peaks at batch 2 (44.4 img/s), then gets 33–36 img/s from batch 4 "
        "up while the device waits 27–47% of the time. DETR's preprocessing upsizes each 640×480 image to "
        "800×1066, so every batch is expensive to build.",
        "**With 4 workers it's GPU-bound** (data wait ≤0.5%) at 55.3–63.7 img/s. The pick is batch 8: 60.7 "
        "img/s at a 130.2 ms step p50 and 2.3 GiB, 4.7% below the fastest result, batch 80 at 63.7 img/s.",
        "**About 16 ms of device time per image** from batch 8 up (61–64 Dev img/s).",
        "**Memory sets the ceiling at batch 80:** 20.5–22.4 GiB (86–94% of the budget). Batch 88 was projected "
        "to need 24.6 GiB and skipped.",
        "**Host RAM is the bigger cost at large batches:** 24.1 GiB at batch 80 with 4 workers. Each image "
        "becomes about 17 MB on the way to the GPU, an 800×1066 float image plus an int64 padding mask.",
    ],
    ("rtdetr-v2-r18", "amd-r9700", "inference"): [
        "**GPU-bound with 2 or 4 workers from batch 8 up:** 176.8–194.8 img/s with data wait ≤0.4%. Device "
        "time settles at about 5.1 ms per image.",
        "**The pick is batch 16 with 4 workers:** 191.9 img/s at an 81.2 ms step p50 and 2.0 GiB, 1.5% below "
        "the fastest result (batch 128 with 2 workers, 194.8 img/s at 654.9 ms).",
        "**One worker can't keep up past batch 2:** 93–125 img/s from batch 4 up, waiting 16–35% of the time.",
        "**3× DETR's throughput** at half its parameters (20.2M vs. 41.5M), on fixed 640×640 inputs.",
        "**The ceiling is conservative.** The 1-worker pass, which ran each batch size first, used more memory "
        "than the later ones (19.7 vs. 14.5 GiB at batch 128), and its 22.1 GiB at batch 144 projected batch "
        "160 to 24.6 GiB, so 160 was skipped for every worker count. With 2 or 4 workers it would likely fit "
        "(16.3 GiB at 144).",
    ],
    ("rtdetr-r50", "amd-r9700", "inference"): [
        "**GPU-bound at every worker count:** data wait is ≤0.5%, and from batch 8 up throughput is flat at "
        "89.9–93.6 img/s, about 10.7 ms of device time per image.",
        "**The pick is batch 8 with 1 worker:** 92.0 img/s at an 86.5 ms step p50 and 1.7 GiB, 1.7% below the "
        "fastest result (batch 32 with 4 workers, 93.6 img/s).",
        "**Workers don't matter:** one keeps the GPU fed, and from batch 16 up 1, 2 and 4 workers are within "
        "0.5% of each other.",
        "**Half RT-DETRv2-R18's throughput** (93.6 vs. 194.8 img/s) with twice the parameters (42.9M vs. "
        "20.2M), on the same 640×640 inputs.",
        "**The ceiling is conservative, as for RT-DETRv2-R18.** The 1-worker pass used 23.1 GiB at batch 128 "
        "against 17.7 GiB with 2 and 4 workers, and projected batch 144 to 26.0 GiB, so 144 was skipped for "
        "every worker count.",
    ],
    ("dfine-small", "amd-r9700", "inference"): [
        "**The fastest detector on this machine:** 214.7 img/s at batch 32 with 4 workers (146.9 ms step p50, "
        "2.8 GiB), both the pick and the fastest result. That's 10% above RT-DETRv2-R18 with half its "
        "parameters (10.3M vs. 20.2M).",
        "**GPU-bound with 4 workers** (data wait ≤0.5%) at 185.4–214.7 img/s from batch 8 up, about 4.6–4.9 "
        "ms of device time per image from batch 16.",
        "**One worker can't keep up past batch 4:** 90–120 img/s from batch 8 up, waiting 24–44% of the time.",
        "**Nothing to gain past batch 32:** 198–209 img/s from 64 to 288 with 2 and 4 workers, at 2–9× the step time.",
        "**Memory sets the ceiling at batch 288:** 21.8–23.8 GiB (91–99% of the budget). Batch 320 was "
        "projected to need 24.2–26.4 GiB and skipped.",
    ],
    ("yolos-small", "amd-r9700", "inference"): [
        "**Input-bound past batch 2 at every worker count.** From batch 4 up the device waited 18–31% of the "
        "time with 4 workers, 18–45% with 2 and 50–66% with 1. YOLOS preprocesses like DETR, upsizing each "
        "640×480 image to 800×1066.",
        "**The pick is batch 2 with 4 workers:** 84.2 img/s at a 23.2 ms step p50 and 0.4 GiB, also the "
        "fastest result. Larger batches make the device faster (120.9 Dev img/s at 128 vs. 86.0 at 2), but "
        "4 workers deliver only 73.6–82.6 img/s from batch 4 up.",
        "**Host RAM is the ceiling with 2 and 4 workers:** 32.0 GiB at batch 192 with 2 and 29.3 GiB at 128 "
        "with 4. Batch 144 with 4 workers ran free host RAM into the 4 GiB reserve and was stopped, and larger "
        "sizes were skipped on projection. Device memory alone allowed batch 288 (22.9 GiB with 1 worker).",
        "**About 8–12 ms of device time per image:** a ViT-S over about 3,300 16×16 patches of an 800×1066 "
        "image. That's faster than DETR (16 ms) and close to RT-DETR R50 (10.7 ms), but about twice "
        "RT-DETRv2-R18 and D-FINE (5.1 and 4.6 ms), which take 640×640 inputs.",
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
        "host_memory_reserve": ("--host-memory-reserve", 0.1),
        "preprocess": ("--preprocess", "batch"),
        "mp_context": ("--mp-context", None),
    }
    # Runs recorded before a setting existed used its default.
    parts += [f"{flag} {s[key]}" for key, (flag, default) in defaults.items() if s.get(key, default) != default]
    # Unset caps MPS at 1.0 and other GPUs at DEFAULT_MEMORY_FRACTION. Runs from
    # before that default recorded an uncapped GPU as None, which is 1.0 today.
    if device["backend"] != "cpu":
        fraction = s.get("memory_fraction") or 1.0
        if fraction != (1.0 if device["backend"] == "mps" else DEFAULT_MEMORY_FRACTION):
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
                _bytes(r.get("peak_host_bytes")),  # absent from runs recorded before it was measured
                ", ".join(marks),
            ]
        )
    header = [
        "Batch",
        "Workers",
        "Status",
        "img/s",
        "Dev img/s",
        "p50 ms",
        "p95 ms",
        "Wait",
        "Peak mem",
        "Host mem",
        "",
    ]
    return _table(header, rows, "rrlrrrrrrrl")


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
