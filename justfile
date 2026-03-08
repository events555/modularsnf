# Run all checks (lint, typecheck, test+coverage)
check: lint typecheck test

lint:
    uv run ruff check .

typecheck:
    uv run ty check modularsnf/

test:
    uv run pytest tests/ -x

# Build the Rust extension (debug)
build:
    uv run maturin develop

# Build the Rust extension (release, optimized)
build-release:
    uv run maturin develop --release
