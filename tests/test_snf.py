import math
import random

import numpy as np
import pytest

from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN
from modularsnf.snf import smith_normal_form as compute_snf
from tests.helpers import det_ring_matrix, get_normalized_invariants

# --- Test Configuration ---
RINGS_TO_TEST = [2, 4, 6, 9, 12, 100]
DIMENSIONS = [
    (4, 4),  # Square
    (4, 6),  # Fat
    (6, 4),  # Tall
    (5, 1),  # Vector column
]


class TestSmithNormalForm:
    @pytest.fixture
    def random_matrix(self, request):
        """Fixture to generate random matrices based on params."""
        N, rows, cols = request.param
        ring = RingZModN(N)
        data = [
            [random.randint(0, N - 1) for _ in range(cols)] for _ in range(rows)
        ]
        return RingMatrix.from_rows(ring, data)

    @pytest.mark.parametrize(
        "random_matrix",
        [(N, r, c) for N in [6, 12] for r, c in DIMENSIONS],
        indirect=True,
    )
    def test_snf_transform_validity(self, random_matrix):
        """
        Verify U * A * V == S.
        """
        A = random_matrix
        U, V, S = compute_snf(A)

        assert U.shape == (A.nrows, A.nrows)
        assert V.shape == (A.ncols, A.ncols)
        assert S.shape == A.shape

        LHS = U @ A @ V
        assert np.array_equal(LHS.data, S.data), (
            "Transform mismatch: U @ A @ V != S"
        )

    @pytest.mark.parametrize(
        "random_matrix",
        [(N, r, c) for N in [6, 12] for r, c in DIMENSIONS],
        indirect=True,
    )
    def test_snf_diagonal_structure(self, random_matrix):
        """
        Verify S_ij == 0 for all i != j.
        """
        A = random_matrix
        _, _, S = compute_snf(A)
        ring = S.ring

        for r in range(S.nrows):
            for c in range(S.ncols):
                if r != c:
                    assert ring.is_zero(S.data[r][c]), (
                        f"Non-zero off-diagonal at ({r},{c}): {S.data[r][c]}"
                    )

    @pytest.mark.parametrize(
        "random_matrix", [(N, 5, 5) for N in [12, 36, 100]], indirect=True
    )
    def test_snf_divisibility_chain(self, random_matrix):
        """
        Verify d_i | d_{i+1} in the sense of principal ideals.
        In Z/N, this means gcd(d_i, N) | gcd(d_{i+1}, N).
        """
        A = random_matrix
        _, _, S = compute_snf(A)

        invariants = get_normalized_invariants(S)

        for i in range(len(invariants) - 1):
            d_curr = invariants[i]
            d_next = invariants[i + 1]

            if d_curr == 0:
                assert d_next == 0, (
                    f"Divisibility break: 0 does not divide {d_next}"
                )
            else:
                assert d_next % d_curr == 0, (
                    f"Divisibility break at index {i}: {d_curr} does not divide {d_next}"
                )

    @pytest.mark.parametrize(
        "random_matrix", [(N, 4, 4) for N in [6, 9]], indirect=True
    )
    def test_transforms_are_unimodular(self, random_matrix):
        """
        Verify det(U) and det(V) are units in Z/N.
        i.e., gcd(det, N) == 1.
        """
        A = random_matrix
        N = A.ring.N
        U, V, _ = compute_snf(A)

        det_U = det_ring_matrix(U)
        det_V = det_ring_matrix(V)

        assert math.gcd(det_U, N) == 1, (
            f"U is not unimodular. det(U)={det_U} (mod {N})"
        )
        assert math.gcd(det_V, N) == 1, (
            f"V is not unimodular. det(V)={det_V} (mod {N})"
        )

    # Oracle comparison tests (formerly test_invariants_against_sympy)
    # have moved to tests/test_oracle.py with broader coverage.

    def test_zero_matrix(self):
        """Test strict zero matrix."""
        ring = RingZModN(12)
        A = RingMatrix.from_rows(ring, [[0, 0], [0, 0]])
        U, V, S = compute_snf(A)

        assert np.array_equal(S.data, np.zeros((2, 2), dtype=int))
        assert np.array_equal((U @ A @ V).data, S.data)

    def test_identity_matrix(self):
        """Test already diagonal matrix (identity)."""
        ring = RingZModN(12)
        A = RingMatrix.identity(ring, 3)
        U, V, S = compute_snf(A)

        for i in range(S.nrows):
            for j in range(S.ncols):
                if i != j:
                    assert ring.is_zero(S.data[i][j]), (
                        f"Non-zero off-diagonal at ({i},{j}): {S.data[i][j]}"
                    )

        invariants = get_normalized_invariants(S)
        assert invariants == [1, 1, 1]

    def test_already_diagonal_unsorted(self):
        """
        Test a diagonal matrix that violates divisibility chain.
        diag(2, 1) mod 4 -> should become diag(1, 2).
        """
        ring = RingZModN(4)
        A = RingMatrix.diagonal(ring, [2, 1])

        _, _, S = compute_snf(A)

        invariants = get_normalized_invariants(S)
        assert invariants == [1, 2]
        assert S.data[0][0] == 1 or S.data[0][0] == 3
        assert S.data[1][1] == 2
