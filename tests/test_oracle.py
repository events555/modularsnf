"""Oracle comparison tests: modularsnf vs SymPy integer-domain SNF.

For each test case we:
1. Build a random integer matrix A over Z/NZ.
2. Compute SNF via the modular pipeline -> (U, V, S).
3. Verify four structural properties:
   a. S = U @ A @ V          (transform validity)
   b. S is diagonal           (diagonal structure)
   c. divisibility chain      (d_i | d_{i+1} as ideals)
   d. U, V unimodular         (det is unit mod N)
4. Compute integer-domain SNF via SymPy -> project to Z/N.
5. Compare invariant factors.
"""

import math
import random

import numpy as np
import pytest

sympy = pytest.importorskip("sympy")
from sympy import ZZ  # noqa: E402
from sympy.matrices.normalforms import (  # noqa: E402
    smith_normal_form as sympy_snf,
)

from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN
from modularsnf.snf import smith_normal_form as compute_snf
from modularsnf.snf import smith_normal_form_mod
from tests.helpers import det_ring_matrix, get_normalized_invariants


def _make_random_matrix(
    ring: RingZModN,
    nrows: int,
    ncols: int,
) -> RingMatrix:
    data = [
        [random.randint(0, ring.N - 1) for _ in range(ncols)]
        for _ in range(nrows)
    ]
    return RingMatrix.from_rows(ring, data)


# --- Parameter matrices ---

_ORACLE_PARAMS = [
    pytest.param(
        N,
        shape,
        seed,
        id=f"N={N}-{shape[0]}x{shape[1]}-seed={seed}",
    )
    for N in [2, 3, 5, 7, 4, 8, 9, 16, 6, 12, 30, 64, 97]
    for shape in [(4, 4), (6, 6), (5, 8), (8, 5)]
    for seed in [42, 137, 2025]
]

_PUBLIC_API_PARAMS = [
    pytest.param(
        N,
        shape,
        seed,
        id=f"api-N={N}-{shape[0]}x{shape[1]}-seed={seed}",
    )
    for N in [6, 12, 8, 97]
    for shape in [(4, 4), (5, 3)]
    for seed in [42, 2025]
]

_ZERO_DIVISOR_PARAMS = [
    pytest.param(
        N,
        seed,
        id=f"zerodiv-N={N}-seed={seed}",
    )
    for N in [4, 8, 16, 32, 64]
    for seed in [42, 137, 2025, 7919, 31337]
]


class TestOracleInternal:
    """Oracle tests against the internal compute_snf API."""

    @pytest.mark.parametrize("N, shape, seed", _ORACLE_PARAMS)
    def test_structural_properties(
        self,
        N: int,
        shape: tuple[int, int],
        seed: int,
    ) -> None:
        random.seed(seed)
        ring = RingZModN(N)
        nrows, ncols = shape
        A = _make_random_matrix(ring, nrows, ncols)

        U, V, S = compute_snf(A)

        # Property 1: S = U @ A @ V
        assert np.array_equal(
            (U @ A @ V).data,
            S.data,
        ), "Transform mismatch"

        # Property 2: S is diagonal
        for r in range(S.nrows):
            for c in range(S.ncols):
                if r != c:
                    assert ring.is_zero(S.data[r, c]), (
                        f"Off-diagonal ({r},{c}) = {S.data[r, c]}"
                    )

        # Property 3: Divisibility chain
        invariants = get_normalized_invariants(S)
        for i in range(len(invariants) - 1):
            d_curr, d_next = invariants[i], invariants[i + 1]
            if d_curr == 0:
                assert d_next == 0
            else:
                assert d_next % d_curr == 0, (
                    f"Chain break: {d_curr} !| {d_next}"
                )

        # Property 4: U, V unimodular (square inputs only;
        # rectangular crops lose unimodularity).
        if nrows == ncols:
            det_U = det_ring_matrix(U) % N
            assert math.gcd(det_U, N) == 1
            det_V = det_ring_matrix(V) % N
            assert math.gcd(det_V, N) == 1

    @pytest.mark.parametrize("N, shape, seed", _ORACLE_PARAMS)
    def test_invariants_match_sympy(
        self,
        N: int,
        shape: tuple[int, int],
        seed: int,
    ) -> None:
        random.seed(seed)
        ring = RingZModN(N)
        nrows, ncols = shape
        A = _make_random_matrix(ring, nrows, ncols)
        n_diag = min(nrows, ncols)

        # SymPy oracle: integer-domain SNF -> project to Z/N
        A_sym = A.to_sympy()
        S_sym = sympy_snf(A_sym, domain=ZZ)
        S_arr = np.array(S_sym.tolist(), dtype=int)
        sym_diags: list[int] = S_arr.diagonal().tolist()
        expected = sorted(math.gcd(d, N) for d in sym_diags)

        # Our pipeline
        _, _, S = compute_snf(A)
        actual = get_normalized_invariants(S)

        assert actual == expected, (
            f"Invariant mismatch for N={N}, shape={shape}, "
            f"seed={seed}\n"
            f"  Expected (SymPy): {expected}\n"
            f"  Actual (ours):    {actual}"
        )


class TestOraclePublicAPI:
    """Oracle tests against the public smith_normal_form_mod API."""

    @pytest.mark.parametrize("N, shape, seed", _PUBLIC_API_PARAMS)
    def test_public_api_structural(
        self,
        N: int,
        shape: tuple[int, int],
        seed: int,
    ) -> None:
        random.seed(seed)
        nrows, ncols = shape
        matrix = [
            [random.randint(0, N - 1) for _ in range(ncols)]
            for _ in range(nrows)
        ]

        result = smith_normal_form_mod(matrix, modulus=N)

        # Verify return type
        assert isinstance(result.S, list)
        assert isinstance(result.U, list)
        assert isinstance(result.V, list)

        # Reconstruct as RingMatrix for property checks
        ring = RingZModN(N)
        S = RingMatrix.from_rows(ring, result.S)
        U = RingMatrix.from_rows(ring, result.U)
        V = RingMatrix.from_rows(ring, result.V)
        A = RingMatrix.from_rows(ring, matrix)

        # S = U @ A @ V
        assert np.array_equal((U @ A @ V).data, S.data)

        # S is diagonal
        for r in range(S.nrows):
            for c in range(S.ncols):
                if r != c:
                    assert ring.is_zero(S.data[r, c])


class TestHighZeroDivisor:
    """Stress tests for moduli with many zero divisors."""

    @pytest.mark.parametrize("N, seed", _ZERO_DIVISOR_PARAMS)
    def test_power_of_2_moduli(
        self,
        N: int,
        seed: int,
    ) -> None:
        random.seed(seed)
        ring = RingZModN(N)
        A = _make_random_matrix(ring, 6, 6)

        U, V, S = compute_snf(A)

        assert np.array_equal((U @ A @ V).data, S.data)

        invariants = get_normalized_invariants(S)
        for i in range(len(invariants) - 1):
            d_curr, d_next = invariants[i], invariants[i + 1]
            if d_curr != 0:
                assert d_next % d_curr == 0
