# Benchmarks

Raw `batch-bench run` output from the machines behind the
[wiki](https://github.com/justinrmiller/batch-bench/wiki), one folder per
machine. The files are copied unchanged from `results/`, which stays
git-ignored so local runs don't show up in `git status`.

| Folder | Machine |
| --- | --- |
| [`apple-m4-pro/`](apple-m4-pro) | Apple M4 Pro (12-core CPU, 16-core GPU, 48 GB), MPS |

Each run is a pair of files named `<UTC timestamp>-<backend><device index>`:

- **`.json`**: device, environment, settings, and every trial with the
  recommendation and its notes.
- **`.csv`**: one row per trial. `recommended` marks the chosen row.

[Output files](../README.md#output-files) in the README describes both.
