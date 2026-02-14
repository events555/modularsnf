# Run all checks (lint, typecheck, test+coverage)
check: lint typecheck test

lint:
    uv run ruff check .

typecheck:
    uv run basedpyright modularsnf/

test:
    uv run pytest tests/ -x
