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

## Adding results

1. Copy the run's `.json` and `.csv` from `results/` into the machine's
   folder. For a new machine, make a new folder and add it to `HARDWARE` in
   [`scripts/wiki.py`](../scripts/wiki.py) so its pages get a proper name
   and specs.
2. If the new run changes numbers quoted in `OBSERVATIONS` in the same
   script, update that commentary.
3. Regenerate the wiki, review it, and push it:

   ```bash
   git clone https://github.com/justinrmiller/batch-bench.wiki.git ../batch-bench.wiki  # once
   make wiki
   cd ../batch-bench.wiki && git diff
   git add -A && git commit -m "Update results" && git push
   ```
