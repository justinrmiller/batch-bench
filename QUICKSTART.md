# Quick start

From a fresh checkout to a recommended batch size in about five minutes.
[README.md](README.md) has the full reference.

## 1. Prerequisites

- **uv:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **A working GPU driver:**

  | GPU | Requirement |
  | --- | --- |
  | AMD (e.g. Radeon AI PRO R9700) | Linux with an amdgpu driver that supports the card. Your user needs to be in the `render` and `video` groups: `sudo usermod -aG render,video $USER`, then log out and back in. |
  | NVIDIA | Driver R580 or newer (CUDA 13). |
  | Intel | Intel's GPU driver with the Level Zero runtime. |
  | Apple Silicon | macOS 14 or newer. |

## 2. Install the PyTorch build for your GPU

```bash
git clone https://github.com/justinrmiller/batch-bench.git
cd batch-bench
uv sync --extra rocm      # AMD on Linux, e.g. Radeon AI PRO R9700
# uv sync --extra cuda    # NVIDIA
# uv sync --extra xpu     # Intel
# uv sync --extra mps     # Apple Silicon
# uv sync --extra cpu     # no GPU, just trying it out
```

> **Always pass `--extra`.** A plain `uv sync` installs no PyTorch, and it
> removes one an extra installed. `uv run` is safe either way.

## 3. Check that the GPU is visible

```bash
uv run batch-bench devices
```

You want a `Device:` line naming your GPU, with the right backend and no
warning underneath. For the R9700 that means `backend=rocm` and
`arch=gfx1201`. If you only see `backend=cpu`, or a warning, read it: it says
what to fix. [Troubleshooting](README.md#troubleshooting) covers the common
causes.

## 4. Run a short first sweep

```bash
uv run batch-bench run -m segformer-b0 --max-batch-size 64
```

This downloads a 15 MB segmentation model and tries batch sizes 1–64. Each
line is one finished trial; a table and a recommendation follow:

```
segformer-b0: loading nvidia/segformer-b0-finetuned-ade-512-512
  batch     1  workers  2  ok     4.6 img/s  step p50 209.6 ms  data wait 1%  peak -
  batch     2  workers  2  ok     8.2 img/s  step p50 223.8 ms  data wait 1%  peak -
  batch     4  workers  2  ok     8.3 img/s  step p50 468.1 ms  data wait 1%  peak -
  batch     8  workers  2  ok     7.6 img/s  step p50 1029.0 ms  data wait 0%  peak -
segformer-b0 (image-segmentation, nvidia/segformer-b0-finetuned-ade-512-512)
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Batch ┃ Workers ┃ Status ┃ img/s ┃ Dev img/s ┃ p50 ms ┃ p95 ms ┃ Wait ┃ Peak mem ┃             ┃
┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│     1 │       2 │     ok │   4.6 │       4.8 │  209.6 │  260.8 │   1% │        - │             │
│     2 │       2 │     ok │   8.2 │       8.9 │  223.8 │  345.1 │   1% │        - │ recommended │
│     4 │       2 │     ok │   8.3 │       8.5 │  468.1 │  554.0 │   1% │        - │ fastest     │
│     8 │       2 │     ok │   7.6 │       7.8 │ 1029.0 │ 1241.6 │   0% │        - │             │
└───────┴─────────┴────────┴───────┴───────────┴────────┴────────┴──────┴──────────┴─────────────┘
Recommended: batch_size=2, num_workers=2 (8.2 img/s, peak -)
Largest batch that fit: 8
note: No out-of-memory up to batch 8; raise --max-batch-size (or list larger --batch-sizes) to find
the memory ceiling.
```

That sample is a real run on a CPU-only machine (two workers,
`--max-batch-size 8`), so the numbers are small. It still shows how the
pick is made. Batch 4 was fastest, but batch 2 got within 5% of it at half
the step latency, so batch 2 is recommended.

On a GPU, expect much higher `img/s` and a filled-in `Peak mem` column. You'll
usually also see an `oom` line where the card runs out of memory.

## 5. Read the result

- **`Recommended: batch_size=…, num_workers=…`** is the answer. It's the
  smallest batch within 5% of the best throughput that also leaves 10% of
  the usable GPU memory free.
- **`img/s` vs `Dev img/s`.** The first is what you'd really get. The second
  is what the GPU could do if data loading were free. A big gap means the
  DataLoader is the bottleneck, and a `note: Input-bound` line will say so.
- **`note:` lines** flag anything that affects the answer: a faster run that
  was too close to the memory limit, a sweep that never ran out of memory,
  and so on.
- Everything is also saved to `results/<timestamp>-<backend><index>.json` and
  `.csv`.

## 6. Benchmark what you actually run

```bash
uv run batch-bench run                                   # default model for each task
uv run batch-bench run -t embedding --max-batch-size 4096
uv run batch-bench run -m detr-resnet-50 --mode training
uv run batch-bench models                                # list the 14 presets
```

For numbers that match production, use your own model, your own images, and
a few worker counts:

```bash
uv run batch-bench run -m <hub-id-or-local-dir> --image-dir /path/to/images --num-workers 4,8,12
```

Ctrl-C stops early but keeps the trials that finished.

## If something is off

| You see | Do this |
| --- | --- |
| `error: PyTorch is not installed` | `uv sync --extra <your backend>` |
| `warning: no GPU is visible … runs on the CPU` | Follow the hint printed with it, then re-check with `batch-bench devices`. |
| `warning: … no kernels for gfx1201` | The wheel doesn't support your card; see [AMD notes](README.md#amd-radeon-ai-pro-r9700-rocm). |
| `note: Input-bound` | Add workers: `--num-workers 8,12,16`. |
| `failed to load: … 401` | Gated model. Accept its license on the Hub, then `uv run hf auth login`. |

More in [README.md → Troubleshooting](README.md#troubleshooting).
