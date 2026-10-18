.PHONY: help install dev lint format typecheck test test-hub check devices models bench wiki pre-commit clean

# Which PyTorch build `make install` / `make dev` install: cpu, cuda, rocm, xpu or mps.
BACKEND ?= cpu
# Extra arguments for `make bench`, e.g. ARGS="-m segformer-b5 --mode training".
ARGS ?=
# Clone of the GitHub wiki that `make wiki` writes into.
WIKI ?= ../batch-bench.wiki

# --no-sync: run with whatever `make dev BACKEND=...` installed, so a later
# target can never swap the torch build behind your back.
UV_RUN = uv run --no-sync

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies for BACKEND, e.g. make install BACKEND=rocm
	uv sync --no-dev --extra $(BACKEND)

dev: ## Install runtime and dev dependencies for BACKEND
	uv sync --extra $(BACKEND)

lint: ## Lint and check formatting with ruff
	$(UV_RUN) ruff check
	$(UV_RUN) ruff format --check

format: ## Auto-fix lint findings and format with ruff
	$(UV_RUN) ruff check --fix
	$(UV_RUN) ruff format

typecheck: ## Type-check with ty
	$(UV_RUN) ty check

test: ## Run the unit tests with coverage (no GPU or network needed)
	$(UV_RUN) pytest

test-hub: ## Run every preset against real Hub checkpoints (downloads several GB)
	$(UV_RUN) pytest -m hub --no-cov

check: lint typecheck test ## Lint, type-check and test

devices: ## Show the accelerators this torch build can see
	$(UV_RUN) batch-bench devices

models: ## List the model presets
	$(UV_RUN) batch-bench models

bench: ## Benchmark, e.g. make bench BACKEND=rocm ARGS="-t all --mode training"
	$(UV_RUN) batch-bench run $(ARGS)

wiki: ## Regenerate the wiki pages from benchmarks/ into WIKI, a clone of the wiki
	$(UV_RUN) python scripts/wiki.py --wiki $(WIKI)

pre-commit: ## Install the git hooks (ruff, ty and uv lock on commit; pytest on push)
	$(UV_RUN) pre-commit install --install-hooks -t pre-commit -t pre-push

clean: ## Remove caches and coverage data
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
