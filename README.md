# batch-bench

Find the DataLoader batch size that gets the most images per second out of
your GPU for HuggingFace object detection, image segmentation and embedding
models. It supports NVIDIA (CUDA), AMD (ROCm, including the Radeon AI PRO
R9700), Intel (XPU) and Apple Silicon (MPS).

**New here? Start with [QUICKSTART.md](QUICKSTART.md).** It takes you from
install to a recommendation in a few commands. This README is the reference.

Each trial runs the whole pipeline a real job runs: JPEG decoding and the
model's HuggingFace image processor in DataLoader worker processes, the copy
to the device, then a forward pass (or forward, backward and optimizer step).
For every batch size it reports:

- end-to-end throughput,
- device-only throughput,
- step latency,
- how long the GPU sat waiting for data,
- peak device memory,
- how much host RAM the trial took.

Timing the model alone overstates the useful batch size whenever the CPU side
can't keep up. This tool catches that case and tells you.

## Contents

- [How the batch size is chosen](#how-the-batch-size-is-chosen)
- [Install](#install)
- [Usage](#usage)
- [Reading the results](#reading-the-results)
- [Options](#options)
- [Model presets](#model-presets)
- [How it measures](#how-it-measures)
- [Backend notes](#backend-notes)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Development](#development)
- [License](#license)

## How the batch size is chosen

1. **Sweep.** Batch sizes double (1, 2, 4, …) until a trial runs out of memory
   or reaches `--max-batch-size`. Then up to `--refine-steps` bisection trials,
   in multiples of 8, narrow in on the largest batch that fits.
2. **Recommend.** The pick is the *smallest* batch within `--tolerance` (5%)
   of the best end-to-end throughput whose peak memory stays under
   `1 - --memory-headroom` (90%) of the *usable* memory. Past the knee of the
   curve, bigger batches add latency and memory but no throughput. The
   headroom leaves room for fragmentation and larger real-world inputs.
   - Usable memory is the device total, scaled by `--memory-fraction`, and
     no more than was free when the run started. Memory another process
     holds, such as a desktop session on the same GPU, doesn't count.
   - By default a run may use 75% of a CUDA, ROCm or XPU device. The sweep
     drives the GPU to out-of-memory on purpose, and a GPU that also drives
     the display needs some left: on the R9700, filling its 32 GB crashed
     GNOME Shell and every app with it. The cap covers only PyTorch's
     allocator, and the ROCm runtime used another 1–4.6 GiB beyond it, so
     85% still came within 150 MB of filling the card. On a headless
     machine, `--memory-fraction 1.0` uses the whole card.
3. **Diagnose.** If the device waited on the DataLoader for 15% or more of the
   run, the report calls the run *input-bound*. A bigger batch won't help
   there. More workers or cheaper preprocessing will. Pass several worker
   counts (`--num-workers 4,8,12`) to sweep the batch-size × worker grid.

## Install

You need [uv](https://docs.astral.sh/uv/). Choose the extra that matches
your hardware. The extras are mutually exclusive, and each one pulls torch
from the matching PyTorch wheel index.

| Hardware | Extra | PyTorch 2.14 build |
| --- | --- | --- |
| AMD Radeon / Instinct on Linux (e.g. Radeon AI PRO R9700) | `rocm` | ROCm 7.2 |
| NVIDIA, Turing (RTX 20-series) or newer | `cuda` | CUDA 13.0 |
| Intel Arc, Core Ultra iGPU, Data Center GPU Max | `xpu` | XPU |
| Apple Silicon | `mps` | macOS arm64 |
| No GPU (testing) | `cpu` | CPU only |

```bash
git clone https://github.com/justinrmiller/batch-bench.git
cd batch-bench
uv sync --extra rocm              # or cuda / xpu / mps / cpu
uv run batch-bench devices        # confirm the GPU is visible and supported
```

PyTorch is not a base dependency, because the right build depends on your
GPU and uv has no "default extra". So:

- **Always pass the extra to `uv sync`.** A plain `uv sync` installs no
  PyTorch, and it *uninstalls* one that an earlier `uv sync --extra rocm`
  installed. If that happens, `batch-bench` prints the install commands
  instead of a traceback.
- A plain `uv run` is safe. It never removes packages, so it leaves the
  installed build in place.
- To switch backends, run `uv sync --extra <other>`.
- `make dev BACKEND=rocm` does the sync. The other `make` targets never
  re-sync, so they can't swap the build.

## Usage

```bash
# Default model for each task (DETR, SegFormer-B0, CLIP ViT-B/32)
uv run batch-bench run

# One task, larger sweep
uv run batch-bench run -t embedding --max-batch-size 4096

# Specific presets, training instead of inference
uv run batch-bench run -m rtdetr-v2-r18 -m segformer-b5 --mode training

# Any Hub checkpoint or local directory (task inferred from its config)
uv run batch-bench run -m facebook/dinov2-large

# Your own images, sweeping worker counts as well
uv run batch-bench run -m mask2former-swin-tiny --image-dir ~/data/coco/val2017 --num-workers 4,8,12

# Only the sizes you care about
uv run batch-bench run -m clip-vit-b32 --batch-sizes 128,192,256,384

uv run batch-bench models         # list presets
uv run batch-bench devices        # list visible GPUs and build warnings
uv run batch-bench run --help     # all options, with examples
```

**Stopping early.** Ctrl-C ends the run but keeps what it has: the current
model's finished trials are printed with a recommendation and saved, and the
JSON gets `"interrupted": true`. Models that hadn't started are skipped.

**Exit status.** `0`: every model got a recommendation. `1`: at least one
didn't (it failed to load, or no batch size completed). `2`: usage error.
`130`: interrupted.

### Output files

Each run writes two files to `--output-dir` (default `results/`), named
`<UTC timestamp>-<backend><device index>`, e.g. `20261018-021500-rocm0`.
If two runs start in the same second, say one per GPU, the second gets a
`-2` suffix rather than overwriting the first.

- **`.json`**: everything. `device` holds name, memory, arch, runtime and
  build warnings. `environment` holds Python, torch and transformers
  versions. `settings` holds every option, including `memory_budget_bytes`.
  `models[]` has `results` (one entry per trial) and `recommendation`, which
  includes the notes. `interrupted` is set if you pressed Ctrl-C.
- **`.csv`**: one row per trial, ready for a spreadsheet or pandas.
  `recommended` marks the chosen row.

## Reading the results

| Column | Meaning |
| --- | --- |
| `img/s` | End-to-end images per second, including time spent waiting on the DataLoader. This is what a job actually gets. |
| `Dev img/s` | Images per second from the median device step alone, i.e. the ceiling if data loading cost nothing. |
| `p50 ms` / `p95 ms` | Device step time per batch: copy, forward, plus backward and optimizer step in training mode. |
| `Wait` | Share of wall time spent blocked on the DataLoader. |
| `Peak mem` | Peak memory held by the caching allocator, which is what runs out first, with its share of the usable memory. |
| `Host mem` | How far free host RAM fell during the trial: DataLoader workers, batches waiting in the queue, and anything else running at the time. On Apple Silicon it includes GPU memory, which is system memory there. |

Patterns worth recognising:

- **`img/s` rises, then flattens.** That's normal. The recommendation sits
  at the start of the flat part. Going further only costs memory and latency.
- **`Dev img/s` is far above `img/s`, and `Wait` is high.** The GPU is
  starved. Add workers (`--num-workers`), use faster storage or decoding, or
  move preprocessing to the GPU. A bigger batch won't help.
- **`p95` is well above `p50`.** Steps are uneven. Common causes are other
  processes on the GPU, thermal or power throttling, or memory pressure near
  the ceiling. Treat that batch size with suspicion.
- **`img/s` drops at the largest sizes.** The allocator is fighting
  fragmentation, or (on MPS) the system is under memory pressure. Stay below
  that point.
- **A trial shows `host-oom`.** Host RAM, not GPU memory, ran short, so the
  sweep stopped there as it would at an `oom`. See
  [Troubleshooting](#troubleshooting).
- **Notes under the table** flag runs that were input-bound, recommendations
  that skipped a faster run because of the memory headroom, sweeps that
  never hit out-of-memory (the ceiling is higher than you tested), and
  sweeps that host RAM cut short.

## Options

`batch-bench run --help` prints the same list with defaults and examples.

**Models**

| Option | Default | Meaning |
| --- | --- | --- |
| `-m, --model` | | Preset name, Hub id or local directory. Repeatable. |
| `-t, --task` | | `object-detection`, `image-segmentation`, `embedding` or `all`. Without `--model`, runs each task's default preset. With `--model`, sets the task for checkpoints whose config doesn't reveal it. Repeatable. |
| `--revision` | | Hub branch, tag or commit. |

With neither `-m` nor `-t`, the default preset of every task runs.

**Device**

| Option | Default | Meaning |
| --- | --- | --- |
| `--device` | `auto` | `cuda`, `rocm`, `xpu`, `mps` or `cpu`. `auto` picks the first visible GPU, else the CPU, and warns when it falls back. |
| `--device-index` | `0` | Which GPU, on multi-GPU machines. |
| `--precision` | `auto` | Autocast dtype. `auto` is bf16 where the GPU supports it natively (emulation doesn't count), otherwise fp16, and fp32 on CPU. |
| `--mode` | `inference` | `training` adds backward and an AdamW step. |
| `--memory-fraction` | `0.75`; `1.0` on MPS | Cap the process at this fraction of device memory, leaving room for a desktop on the same GPU. `1.0` uses all of it. |
| `--compile` | off | Wrap the model in `torch.compile`, recompiling for each batch size. |
| `--no-cudnn-benchmark` | | Skip cuDNN/MIOpen autotuning. Shortens the first trials on ROCm. Autotuning is also off automatically for `--image-dir` without `--image-size`. |

**Sweep**

| Option | Default | Meaning |
| --- | --- | --- |
| `--batch-sizes` | | Explicit list, e.g. `16,24,32,48`. Overrides the doubling sweep. |
| `--min-batch-size` / `--max-batch-size` | `1` / `1024` | Doubling-sweep bounds. |
| `--refine-steps` | `3` | Bisection trials after the first out-of-memory. `0` turns them off. |
| `--num-workers` | CPUs − 1 (min 1) | One or more DataLoader worker counts, e.g. `4,8,12`. |
| `--warmup-batches` | `3` | Untimed batches per trial. With workers, (prefetch factor + 1) × workers + 1 more follow; see [How it measures](#how-it-measures). |
| `--measure-batches` | `16` | Timed batches per trial. With workers, rounded up to a multiple of the worker count. |
| `--prefetch-factor` | `2` | Batches each worker prepares ahead. Also sets how long warmup takes to empty the queue. |
| `--tolerance` | `0.05` | How far below peak throughput the recommendation may be. |
| `--memory-headroom` | `0.10` | Share of usable memory the recommendation leaves free. |
| `--host-memory-reserve` | `0.10` | Share of host RAM to keep free. Batch sizes projected to need more are skipped, and a trial that cuts into it is stopped. Both show as `host-oom`. |

**Data**

| Option | Default | Meaning |
| --- | --- | --- |
| `--image-dir` | synthetic | Benchmark on your images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, recursive), read in a seeded random order. |
| `--image-size` | `640x480` | Size of synthetic images. With `--image-dir`, resize your images to it. |
| `--preprocess` | `batch` | Run the image processor per batch (in `collate_fn`, 32 images per call) or per sample (in `__getitem__`). |
| `--no-pin-memory` | | Disable pinned host memory (CUDA, ROCm, XPU). |
| `--mp-context` | platform default | Worker start method: `fork`, `spawn` or `forkserver`. |
| `--seed` | `0` | Seed for the synthetic images and for the order of `--image-dir` files. |

**Output**

| Option | Default | Meaning |
| --- | --- | --- |
| `--output-dir` | `results` | Where the JSON and CSV go. |
| `--no-save` | | Print only; write no files. |

## Model presets

| Task | Presets (default first) |
| --- | --- |
| Object detection | `detr-resnet-50`, `rtdetr-v2-r18`, `rtdetr-r50`, `dfine-small`, `yolos-small` |
| Image segmentation | `segformer-b0`, `segformer-b5`, `upernet-convnext-small`, `mask2former-swin-tiny` |
| Embedding | `clip-vit-b32`, `clip-vit-l14`, `siglip2-base`, `dinov2-small`, `dinov2-base` |

`batch-bench models` lists them with Hub ids and descriptions.

- **CLIP and SigLIP presets load the vision tower only.** That's what an
  embedding job keeps on the GPU.
- **Any other checkpoint loads with the class named in its config.** A full
  `CLIPModel` still works and runs through `get_image_features`, but its text
  tower counts against memory.
- **Gated checkpoints** such as DINOv3 need `HF_TOKEN` (or
  `uv run hf auth login`).

## How it measures

- **A fresh DataLoader per trial** with `drop_last=True`. Synthetic images are
  smooth random JPEGs that get decoded for every sample, so the decode cost
  is realistic without a dataset on disk.
- **Your images are read in a seeded random order.** Sorted order would give
  small batch sizes images from the first subdirectory only. In an
  ImageNet-style tree that's a single class.
- **Variable input shapes turn autotuning off.** Some models keep each
  image's aspect ratio and pad batches (the DETR and YOLOS presets), so mixed
  image sizes give a new tensor shape on almost every batch. cuDNN and MIOpen
  would re-tune kernels for each new shape inside the timed batches, so
  `--image-dir` without `--image-size` runs without autotuning.
- **Warmup empties the prefetch queue.** While a warmup step is slow
  (cuDNN/MIOpen autotuning, `torch.compile`, lazy initialization), the
  workers keep going and fill the queue with up to
  `prefetch_factor × num_workers` ready batches. Timing those would make an
  input-bound loader look faster than it is, by about 2× with 8 workers at
  the defaults. So with any workers, `--warmup-batches` is followed by
  `(prefetch_factor + 1) × num_workers + 1` more untimed batches. They
  consume whatever the queue holds, then one more round, so timing starts
  from a steady flow; they also keep worker start-up out of the timings.
  With 15 workers at the defaults that's 49 warmup batches. A lower
  `--prefetch-factor` shortens it.
- **Measurement in whole worker rounds.** Each worker builds a whole batch,
  so an input-bound loader delivers batches in rounds of `num_workers`. With
  any workers, the measured batch count rounds up to a whole number of
  rounds; a partial round would skew throughput either way.
- **Every step synchronizes the device**, so latency is real device time,
  not kernel-launch time. The first steps of a trial also absorb cuDNN/MIOpen
  autotuning and `torch.compile`.
- **Mixed precision via autocast**, with float32 weights.
- **Training mode** runs a real backward and AdamW step, using `lr=0` so
  weights stay identical across trials. The loss is a surrogate: the sum of
  the means of every floating-point output, so every head is exercised.
  Memory and compute match real training except for task-specific loss work,
  such as DETR's Hungarian matching (mostly CPU) and label loading.
- **Out-of-memory errors are caught per trial.** That covers
  `torch.OutOfMemoryError` plus the message-only errors from HIP, MIOpen,
  cuDNN/cuBLAS, MPS and Level Zero. The allocator cache is emptied before
  the next trial. Other errors are recorded as `error`, and the sweep
  carries on.
- **Host RAM is measured and guarded.** It grows with batch size × workers.
  Each worker builds a whole batch, and up to `prefetch_factor × num_workers`
  finished batches wait in the main process. On CUDA, ROCm and XPU those are
  pinned, and PyTorch rounds each pinned buffer up to a power of two, so a
  588 MiB batch takes 1 GiB. A sweep can therefore run out of host RAM long
  before the GPU fills, and the kernel's OOM killer may then take the desktop
  session down with it. To prevent that:
  - Free memory is read every 50 ms and after every step. It's system RAM,
    or the room left under a cgroup memory cap (a container's limit, or
    `systemd-run -p MemoryMax=`) when that's tighter. `Host mem` is how far
    it fell below its level at the start of the trial.
  - Before a larger batch runs, its need is projected from the largest
    successful trial with the same worker count. Most of it scales with the
    batch size; pinned batches grow in power-of-two steps, as the allocator
    rounds them (CUDA and ROCm report how much was pinned). If it won't fit
    above `--host-memory-reserve`, the trial is skipped.
  - If free memory falls into the reserve anyway, the trial's workers are
    killed at once.
  - Pinned buffers are released between trials, so each batch size doesn't
    keep its own cached set.
  - With `--preprocess batch`, workers decode and preprocess 32 images at a
    time and assemble the batch straight into the shared memory it's sent
    through. Each worker holds about twice its batch, where passing the
    whole batch to the processor at once held about seven times.
- **The timing logic is itself tested.** `tests/test_methodology.py` builds
  pipelines with known throughput (sleeping loaders and models) and checks
  that the measured images/s and data-wait share match theory. In practice
  they land within a few percent, against a 15% test tolerance.
- **Peak memory** comes from the caching allocator's counters on CUDA, ROCm
  and XPU. MPS has no peak counters, so `driver_allocated_memory()` is polled
  after every step. That's an approximation, and the budget it's compared to
  is Metal's recommended working set.

## Backend notes

### AMD Radeon AI PRO R9700 (ROCm)

- The R9700 is RDNA 4 (`gfx1201`) with 32 GB. You need a Linux kernel / amdgpu
  driver new enough for RDNA 4, and your user must be in the `render` and
  `video` groups. The PyTorch wheel bundles the ROCm user-space libraries.
- `batch-bench devices` should report `arch=gfx1201` with no warning. A
  warning means the installed wheel has no `gfx1201` kernels. In that case,
  install AMD's PyTorch wheels from repo.radeon.com.
- Don't set `HSA_OVERRIDE_GFX_VERSION`; that's for GPUs ROCm doesn't support,
  and gfx1201 is supported.
- ROCm runs through `torch.cuda`, so `--device rocm` or `auto` both work.
  `HIP_VISIBLE_DEVICES` picks the GPU.
- If the card also drives your displays, the desktop holds some of its
  memory. The run warns when more than 10% is in use at start, and keeps the
  headroom within what was free.
- MIOpen searches for convolution kernels the first time it sees a shape, and
  every batch size is a new shape. Warmup absorbs this, so the numbers are
  unaffected, but the first sweep takes longer. Results are cached for later
  runs. `MIOPEN_FIND_MODE=FAST` or `--no-cudnn-benchmark` shortens the search.
- `PYTORCH_TUNABLEOP_ENABLED=1` tunes GEMMs and can raise throughput. Expect
  slower first trials while it tunes.

### NVIDIA (CUDA)

- The `cuda` extra uses CUDA 13.0 wheels. Those need driver R580 or newer and
  dropped Maxwell, Pascal and Volta. For older GPUs, change the
  `pytorch-cu130` index URL in `pyproject.toml` to `.../whl/cu126` and
  re-lock.
- `batch-bench devices` warns if the wheel has no kernels for your GPU.
- Turing and older GPUs have no native bf16, so `--precision auto` uses fp16
  on them. torch would otherwise report bf16 as supported through slow
  emulation.

### Intel (XPU)

- This uses PyTorch's native XPU backend; IPEX isn't needed. Install Intel's
  GPU driver (Level Zero runtime) first.
- Some Intel GPUs reject any single allocation over 4 GB. That caps the batch
  size just like running out of memory, so it's reported as `oom`.

### Apple Silicon (MPS)

- MPS uses unified memory. By default PyTorch lets it allocate up to 1.7× the
  recommended working set, which is more than most Macs' RAM, so an oversized
  batch makes the system swap instead of failing. batch-bench caps MPS at 1.0×
  so the sweep gets a clean out-of-memory error. Override it with
  `--memory-fraction` (up to 1.0) or `PYTORCH_MPS_HIGH_WATERMARK_RATIO`.
- Pinned memory isn't supported on MPS. DataLoader workers start with
  `spawn` on macOS, so each one re-imports torch and transformers. They start
  in parallel, and warmup absorbs the delay.
- DataLoader workers share the same unified memory as the GPU. Very high
  worker counts leave less room for batches.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| `error: PyTorch is not installed` | Run `uv sync --extra rocm` (or `cuda` / `xpu` / `mps` / `cpu`). A plain `uv sync` removes PyTorch. |
| `warning: no GPU is visible … runs on the CPU` | Follow the hint printed with it. On ROCm, that usually means `/dev/kfd` permissions (the `render` and `video` groups) or `HIP_VISIBLE_DEVICES`. Run `batch-bench devices` to check. |
| `warning: This torch build has no kernels for gfx1201` | The wheel doesn't target your GPU. Install AMD's wheels (ROCm), or a CUDA build that supports the card. |
| `note: Input-bound` | The CPU side is the bottleneck. Try higher `--num-workers` counts, faster image storage, or smaller source images. |
| The first trials on ROCm are very slow | That's MIOpen's kernel search, and it's cached afterwards. Use `MIOPEN_FIND_MODE=FAST` or `--no-cudnn-benchmark`. |
| A Mac becomes sluggish at large batches | It's swapping. Make sure you haven't raised `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, and lower `--max-batch-size` or `--num-workers`. |
| Trials show `host-oom` | Host RAM ran short; the trial's message says whether it was skipped or stopped. Each worker holds a whole batch, and up to `--prefetch-factor` × workers more wait in the queue. Lower `--num-workers`, or pass `--prefetch-factor 1` or `--no-pin-memory`. On Linux, `systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 uv run batch-bench run …` adds a hard cap, so a runaway run can only kill itself. The guard sees the cap and works within it. |
| `failed to load: … 401` / gated repo | Accept the model's license on the Hub, then set `HF_TOKEN` or run `uv run hf auth login`. |
| `cannot infer the task for …` | Add `--task object-detection` (or `image-segmentation` / `embedding`). |
| Trials show `error` | Read the message in the table. In `--mode training`, batch size 1 can fail in layers that need more than one value per channel. The sweep moves on to larger sizes. If every size fails, try `--precision fp32` to rule out a dtype problem. |

## Limitations

- One GPU per run; no DDP.
- Preprocessing runs on the CPU in DataLoader workers. If a run is
  input-bound, GPU-side preprocessing (not implemented) is the next lever.
- The recommendation holds for the pipeline measured. Point `--image-dir` at
  your real data so decode cost and image sizes match production.

## Development

```bash
make dev          # uv sync --extra cpu (BACKEND=rocm etc. to test on a GPU)
make check        # ruff check + format check, ty, pytest with coverage
make pre-commit   # git hooks: ruff, ty, uv lock --check on commit; pytest on push
make test-hub     # every preset against real Hub checkpoints (downloads ~6 GB)
make wiki         # regenerate the GitHub wiki from benchmarks/ (WIKI=path to a wiki clone)
```

The unit tests need no GPU and no network. Models are tiny, randomly
initialized checkpoints saved to temp directories. The CUDA, ROCm, XPU and
MPS code paths run against fake `torch.cuda` / `torch.xpu` / `torch.mps`
namespaces. Coverage must stay at or above 90% (branch coverage included).
CI (`.github/workflows/ci.yml`) runs everything on Linux with the
`cpu` extra and on macOS with the `mps` extra.

```
src/batch_bench/
  launcher.py  console entry point; explains a missing PyTorch install
  cli.py       batch-bench run | devices | models
  devices.py   backend detection, Accelerator wrapper, OOM classification
  models.py    presets, loading, per-task forward pass and surrogate loss
  data.py      synthetic/folder image sources, preprocessing, DataLoader
  runner.py    one timed trial (batch size × workers)
  sweep.py     doubling + bisection search
  analysis.py  recommendation and diagnostics
  report.py    console tables, JSON and CSV
```

## License

[MIT](LICENSE)
