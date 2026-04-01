"""Shared fixtures for the modularsnf test suite."""

import random
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import modularsnf.diagonal as diagonal_mod
import modularsnf.ring as ring_mod
import modularsnf.snf as snf_mod
from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN

_RustRing: Any = None
_rust_diag: Any = None
_rust_merge: Any = None
_rust_snf: Any = None

try:
    from modularsnf._rust import RustRingZModN as _RustRing
    from modularsnf._rust import rust_merge_smith_blocks as _rust_merge
    from modularsnf._rust import rust_smith_from_diagonal as _rust_diag
    from modularsnf._rust import rust_smith_normal_form as _rust_snf
except ImportError:
    pass


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command-line options."""
    parser.addoption(
        "--backend",
        action="store",
        choices=("auto", "python", "rust"),
        default="auto",
        help="Execution backend for tests",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "perf: performance sanity checks (deselect with -m 'not perf')",
    )


@pytest.fixture(scope="session", autouse=True)
def configured_backend(pytestconfig: pytest.Config) -> Iterator[str]:
    """Configure the requested execution backend for the full test session."""
    has_rust = all(
        value is not None
        for value in (_RustRing, _rust_diag, _rust_merge, _rust_snf)
    )
    requested = pytestconfig.getoption("backend")

    if requested == "rust":
        if not has_rust:
            pytest.exit(
                "Rust backend requested for tests but modularsnf._rust is "
                "unavailable"
            )
        backend = "rust"
    elif requested == "python":
        backend = "python"
    elif has_rust:
        backend = "rust"
    else:
        backend = "python"

    old_ring = getattr(ring_mod, "_RustRing")
    old_diag = getattr(diagonal_mod, "_rust_diag")
    old_merge = getattr(diagonal_mod, "_rust_merge")
    old_snf = getattr(snf_mod, "_rust_snf")

    if backend == "rust":
        setattr(ring_mod, "_RustRing", _RustRing)
        setattr(diagonal_mod, "_rust_diag", _rust_diag)
        setattr(diagonal_mod, "_rust_merge", _rust_merge)
        setattr(snf_mod, "_rust_snf", _rust_snf)
    else:
        setattr(ring_mod, "_RustRing", None)
        setattr(diagonal_mod, "_rust_diag", None)
        setattr(diagonal_mod, "_rust_merge", None)
        setattr(snf_mod, "_rust_snf", None)

    yield backend

    setattr(ring_mod, "_RustRing", old_ring)
    setattr(diagonal_mod, "_rust_diag", old_diag)
    setattr(diagonal_mod, "_rust_merge", old_merge)
    setattr(snf_mod, "_rust_snf", old_snf)


@pytest.fixture
def seeded_rng(request: pytest.FixtureRequest) -> int:
    """Seed both stdlib random and numpy RNG.

    The seed is extracted from ``request.param`` when used with
    indirect parametrization, or defaults to 42.

    Returns the seed value for diagnostic printing.
    """
    seed = getattr(request, "param", 42)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def make_random_matrix(
    ring: RingZModN,
    nrows: int,
    ncols: int,
) -> RingMatrix:
    """Generate a random matrix over the given ring."""
    data = [
        [random.randint(0, ring.N - 1) for _ in range(ncols)]
        for _ in range(nrows)
    ]
    return RingMatrix.from_rows(ring, data)
