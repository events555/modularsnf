"""Curated regression corpus for SNF edge cases.

Each case targets a historically problematic matrix class:
- repeated factors in the modulus
- rank-deficient blocks
- dense random with adversarial structure
- near-diagonal with adversarial superdiagonal entries
- rectangular shapes that exercise padding/cropping
- matrices that trigger the merge_smith_blocks permutation path
"""

import math
import random

import numpy as np
import pytest

from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN
from modularsnf.snf import smith_normal_form as compute_snf
from modularsnf.snf import smith_normal_form_mod
from tests.helpers import det_ring_matrix, get_normalized_invariants


def _assert_valid_snf(
    A: RingMatrix,
    U: RingMatrix,
    V: RingMatrix,
    S: RingMatrix,
    *,
    expected_invariants: list[int] | None = None,
) -> None:
    """Assert all four SNF structural properties.

    Optionally compares invariant factors against known values.
    """
    ring = A.ring
    N = ring.N

    # 1. Transform validity
    assert np.array_equal(
        (U @ A @ V).data,
        S.data,
    ), "S != U @ A @ V"

    # 2. Diagonal structure
    for r in range(S.nrows):
        for c in range(S.ncols):
            if r != c:
                assert ring.is_zero(S.data[r][c]), (
                    f"Off-diagonal ({r},{c}) = {S.data[r][c]}"
                )

    # 3. Divisibility chain
    inv = get_normalized_invariants(S)
    for i in range(len(inv) - 1):
        if inv[i] != 0:
            assert inv[i + 1] % inv[i] == 0, (
                f"Chain break: {inv[i]} !| {inv[i + 1]}"
            )

    # 4. Unimodularity (square inputs only)
    if A.nrows == A.ncols:
        if U.nrows == U.ncols and U.nrows > 0:
            assert math.gcd(det_ring_matrix(U) % N, N) == 1
        if V.nrows == V.ncols and V.nrows > 0:
            assert math.gcd(det_ring_matrix(V) % N, N) == 1

    # 5. Optional exact invariant check
    if expected_invariants is not None:
        assert inv == sorted(expected_invariants), (
            f"Invariants {inv} != expected {sorted(expected_invariants)}"
        )


class TestRepeatedFactors:
    """Matrices whose entries share factors with the modulus."""

    CASES = [
        pytest.param(
            [[2, 4], [6, 8]],
            12,
            [2, 4],
            id="2x2-all-even-mod12",
        ),
        pytest.param(
            [[3, 6, 9], [6, 3, 6], [9, 6, 3]],
            9,
            [3, 3, 3],
            id="3x3-multiples-of-3-mod9",
        ),
        pytest.param(
            [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            8,
            [4, 4, 4],
            id="3x3-scalar-4-mod8",
        ),
        pytest.param(
            [[2, 0], [0, 4]],
            8,
            [2, 4],
            id="2x2-diag-powers-of-2-mod8",
        ),
        pytest.param(
            [[6, 0, 0], [0, 10, 0], [0, 0, 15]],
            30,
            [1, 30, 30],
            id="3x3-diag-mod30-coprime-factors",
        ),
    ]

    @pytest.mark.parametrize("matrix, N, expected_inv", CASES)
    def test_repeated_factors(
        self,
        matrix: list[list[int]],
        N: int,
        expected_inv: list[int],
    ) -> None:
        ring = RingZModN(N)
        A = RingMatrix.from_rows(ring, matrix)
        U, V, S = compute_snf(A)
        _assert_valid_snf(
            A,
            U,
            V,
            S,
            expected_invariants=expected_inv,
        )


class TestRankDeficient:
    """Rank-deficient matrices targeting merge_smith_blocks paths."""

    CASES = [
        pytest.param(
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            12,
            [1, 12, 12],
            id="3x3-rank1-mod12",
        ),
        pytest.param(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 6, 0], [0, 0, 0, 4]],
            12,
            [2, 12, 12, 12],
            id="4x4-rank2-sparse-mod12",
        ),
        pytest.param(
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 6]],
            12,
            [2, 6, 12, 12],
            id="4x4-two-nonzero-diag-mod12",
        ),
        pytest.param(
            [[0, 0], [0, 0]],
            7,
            [7, 7],
            id="2x2-zero-mod-prime",
        ),
        pytest.param(
            [[1, 2, 3], [2, 4, 6], [3, 6, 9]],
            12,
            None,
            id="3x3-rank1-dense-mod12",
        ),
    ]

    @pytest.mark.parametrize("matrix, N, expected_inv", CASES)
    def test_rank_deficient(
        self,
        matrix: list[list[int]],
        N: int,
        expected_inv: list[int] | None,
    ) -> None:
        ring = RingZModN(N)
        A = RingMatrix.from_rows(ring, matrix)
        U, V, S = compute_snf(A)
        _assert_valid_snf(
            A,
            U,
            V,
            S,
            expected_invariants=expected_inv,
        )


class TestDenseRandom:
    """Seeded dense random matrices exercising the full pipeline."""

    @pytest.mark.parametrize(
        "N, size, seed",
        [
            pytest.param(12, 6, 42, id="dense-6x6-mod12-seed42"),
            pytest.param(8, 7, 137, id="dense-7x7-mod8-seed137"),
            pytest.param(100, 5, 2025, id="dense-5x5-mod100-seed2025"),
            pytest.param(64, 8, 7919, id="dense-8x8-mod64-seed7919"),
            pytest.param(30, 6, 31337, id="dense-6x6-mod30-seed31337"),
        ],
    )
    def test_dense_random(
        self,
        N: int,
        size: int,
        seed: int,
    ) -> None:
        random.seed(seed)
        ring = RingZModN(N)
        data = [
            [random.randint(0, N - 1) for _ in range(size)] for _ in range(size)
        ]
        A = RingMatrix.from_rows(ring, data)
        U, V, S = compute_snf(A)
        _assert_valid_snf(A, U, V, S)


class TestAdversarialSuperdiagonal:
    """Diagonal + adversarial superdiagonal entries."""

    CASES = [
        pytest.param(
            [[1, 11, 0], [0, 1, 0], [0, 0, 1]],
            12,
            [1, 1, 1],
            id="3x3-identity-plus-superdiag-mod12",
        ),
        pytest.param(
            [[2, 1, 0, 0], [0, 4, 1, 0], [0, 0, 6, 1], [0, 0, 0, 8]],
            12,
            None,
            id="4x4-staircase-mod12",
        ),
        pytest.param(
            [[3, 0, 0, 5], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 3]],
            9,
            None,
            id="4x4-scalar3-plus-corner-mod9",
        ),
        pytest.param(
            [[4, 2, 0], [0, 4, 2], [0, 0, 4]],
            8,
            None,
            id="3x3-upper-triangular-mod8",
        ),
        pytest.param(
            [
                [1, 0, 0, 0, 7],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            12,
            [1, 1, 1, 1, 1],
            id="5x5-identity-with-far-corner-mod12",
        ),
    ]

    @pytest.mark.parametrize("matrix, N, expected_inv", CASES)
    def test_adversarial_superdiagonal(
        self,
        matrix: list[list[int]],
        N: int,
        expected_inv: list[int] | None,
    ) -> None:
        ring = RingZModN(N)
        A = RingMatrix.from_rows(ring, matrix)
        U, V, S = compute_snf(A)
        _assert_valid_snf(
            A,
            U,
            V,
            S,
            expected_invariants=expected_inv,
        )


class TestRectangularRegression:
    """Rectangular matrices exercising padding/cropping paths."""

    CASES = [
        pytest.param(
            [[1, 2, 3, 4, 5]],
            12,
            id="1x5-single-row-mod12",
        ),
        pytest.param(
            [[1], [2], [3], [4], [5]],
            12,
            id="5x1-single-col-mod12",
        ),
        pytest.param(
            [[2, 4, 6], [8, 10, 0]],
            12,
            id="2x3-fat-mod12",
        ),
        pytest.param(
            [[3, 6], [9, 3], [6, 9]],
            9,
            id="3x2-tall-mod9",
        ),
    ]

    @pytest.mark.parametrize("matrix, N", CASES)
    def test_rectangular(
        self,
        matrix: list[list[int]],
        N: int,
    ) -> None:
        ring = RingZModN(N)
        A = RingMatrix.from_rows(ring, matrix)
        U, V, S = compute_snf(A)
        _assert_valid_snf(A, U, V, S)


class TestErrorHandling:
    """Exercise snf.py input validation paths."""

    def test_modulus_too_small(self) -> None:
        with pytest.raises(ValueError, match="Modulus must be"):
            smith_normal_form_mod([[1]], modulus=1)

    def test_modulus_zero(self) -> None:
        with pytest.raises(ValueError, match="Modulus must be"):
            smith_normal_form_mod([[1]], modulus=0)

    def test_modulus_negative(self) -> None:
        with pytest.raises(ValueError, match="Modulus must be"):
            smith_normal_form_mod([[1]], modulus=-5)

    def test_non_list_input(self) -> None:
        with pytest.raises(TypeError, match="list of lists"):
            smith_normal_form_mod("not a matrix", modulus=5)  # type: ignore[arg-type]

    def test_ragged_rows(self) -> None:
        with pytest.raises(ValueError, match="Ragged matrix"):
            smith_normal_form_mod([[1, 2], [3]], modulus=5)

    def test_empty_matrix(self) -> None:
        result = smith_normal_form_mod([], modulus=5)
        assert result.S == []
        assert result.U == []
        assert result.V == []


class TestCoverageTargeted:
    """Tests targeting specific uncovered code paths."""

    def test_merge_smith_blocks_permutation_path(self) -> None:
        """Target diagonal.py lines 221-249.

        diag(0, 6, 0, 4) mod 12 forces merge of blocks with
        partial ranks, triggering the permutation path in
        merge_smith_blocks.
        """
        ring = RingZModN(12)
        A = RingMatrix.diagonal(ring, [0, 6, 0, 4])
        U, V, S = compute_snf(A)
        _assert_valid_snf(
            A,
            U,
            V,
            S,
            expected_invariants=[2, 12, 12, 12],
        )

    def test_merge_smith_blocks_larger_rank_deficient(
        self,
    ) -> None:
        """Larger rank-deficient case for merge path."""
        ring = RingZModN(8)
        A = RingMatrix.diagonal(ring, [0, 0, 2, 0, 4, 0])
        U, V, S = compute_snf(A)
        _assert_valid_snf(A, U, V, S)

    def test_project_to_upper_bandwidth(self) -> None:
        """Target band.py project_to_upper_bandwidth (dead code)."""
        from modularsnf.band import project_to_upper_bandwidth

        ring = RingZModN(12)
        data = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 0],
            [1, 2, 3, 4],
        ]
        M = RingMatrix.from_rows(ring, data)
        P = project_to_upper_bandwidth(M, 2)

        # Only entries with 0 <= j - i < 2 should survive
        for i in range(4):
            for j in range(4):
                if 0 <= j - i < 2:
                    assert P.data[i][j] == M.data[i][j]
                else:
                    assert P.data[i][j] == 0

    def test_band_reduction_already_2banded(self) -> None:
        """band_reduction with b <= 2 is a no-op."""
        from modularsnf.band import band_reduction

        ring = RingZModN(12)
        A = RingMatrix.diagonal(ring, [1, 2, 3])
        A_red, U, V, b_new = band_reduction(A, 2)

        assert b_new == 2
        assert np.array_equal(A_red.data, A.data)
