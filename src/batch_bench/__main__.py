"""Allow ``python -m batch_bench``."""

from batch_bench.launcher import main

if __name__ == "__main__":
    raise SystemExit(main())
