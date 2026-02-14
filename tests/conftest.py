"""Shared fixtures for the modularsnf test suite."""

import random

import numpy as np
import pytest

from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "perf: performance sanity checks (deselect with -m 'not perf')",
    )


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
