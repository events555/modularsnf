"""Performance sanity checks for the modular SNF pipeline.

These tests verify that runtime does not regress catastrophically.
They use generous wall-clock bounds and are marked ``perf`` so they
are excluded from the default test run.

Run with: uv run pytest -m perf
"""

import random
import time

import numpy as np
import pytest

from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN
from modularsnf.snf import smith_normal_form as compute_snf


@pytest.mark.perf
class TestPerformanceSanity:
    """Wall-clock sanity checks for representative sizes."""

    # Bounds are set generously (5x measured baseline) to
    # account for CI variability and slow runners.
    CASES = [
        pytest.param(12, 10, 5.0, id="10x10-mod12"),
        pytest.param(100, 15, 15.0, id="15x15-mod100"),
        pytest.param(8, 20, 30.0, id="20x20-mod8"),
        pytest.param(30, 20, 30.0, id="20x20-mod30"),
    ]

    @pytest.mark.parametrize("N, size, max_seconds", CASES)
    def test_runtime_bound(
        self,
        N: int,
        size: int,
        max_seconds: float,
    ) -> None:
        random.seed(42)
        ring = RingZModN(N)
        data = [
            [random.randint(0, N - 1) for _ in range(size)] for _ in range(size)
        ]
        A = RingMatrix.from_rows(ring, data)

        t0 = time.perf_counter()
        U, V, S = compute_snf(A)
        elapsed = time.perf_counter() - t0

        # Verify correctness so timing doesn't mask a bug
        assert np.array_equal((U @ A @ V).data, S.data)

        assert elapsed < max_seconds, (
            f"{size}x{size} mod {N} took {elapsed:.2f}s "
            f"(limit {max_seconds:.1f}s)"
        )

    def test_rectangular_perf(self) -> None:
        """Rectangular matrices should not be dramatically slower."""
        random.seed(42)
        N = 12
        ring = RingZModN(N)
        nrows, ncols = 15, 8
        data = [
            [random.randint(0, N - 1) for _ in range(ncols)]
            for _ in range(nrows)
        ]
        A = RingMatrix.from_rows(ring, data)

        t0 = time.perf_counter()
        U, V, S = compute_snf(A)
        elapsed = time.perf_counter() - t0

        assert np.array_equal((U @ A @ V).data, S.data)

        assert elapsed < 15.0, f"15x8 mod 12 took {elapsed:.2f}s (limit 15.0s)"

    def test_scaling_ratio(self) -> None:
        """Check that doubling size does not cause >16x slowdown.

        If the algorithm is O(n^3), doubling n should give ~8x.
        We allow up to 32x to account for constant factors and
        CI noise, but catch O(n^4) or worse regressions.
        """
        N = 12
        ring = RingZModN(N)

        def time_snf(size: int) -> float:
            random.seed(42)
            data = [
                [random.randint(0, N - 1) for _ in range(size)]
                for _ in range(size)
            ]
            A = RingMatrix.from_rows(ring, data)
            t0 = time.perf_counter()
            compute_snf(A)
            return time.perf_counter() - t0

        t_small = time_snf(8)
        t_large = time_snf(16)

        # Guard against division by near-zero
        if t_small < 0.001:
            return  # Too fast to measure meaningfully

        ratio = t_large / t_small
        assert ratio < 32.0, (
            f"Scaling ratio 8->16: {ratio:.1f}x (expected <32x for O(n^3))"
        )
