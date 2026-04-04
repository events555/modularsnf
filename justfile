# Run all checks (lint, typecheck, test+coverage, version parity)
check: lint typecheck test check-version

lint:
    uv run ruff check .

typecheck:
    uv run ty check modularsnf/ scripts/profile_snf.py tests/conftest.py

test:
    uv run pytest tests/ -x --backend python
    uv run pytest tests/ -x --backend rust

# Verify Cargo workspace and pyproject.toml versions match
check-version:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)"/\1/')
    py=$(grep -m1 '^version' pyproject.toml | sed 's/.*"\(.*\)"/\1/')
    if [ "$cargo" != "$py" ]; then echo "Version mismatch: Cargo=$cargo pyproject=$py"; exit 1; fi

# Build the Rust extension (debug)
build:
    uv run maturin develop

# Build the Rust extension (release, optimized)
build-release:
    uv run maturin develop --release
