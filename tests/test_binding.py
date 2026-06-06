"""PyO3 boundary tests for the modularsnf Python API.

Exhaustive algorithm correctness lives in the Rust crate (cargo test). These
tests cover what the Python layer is responsible for — input validation and
type/shape marshalling across the extension boundary — plus one end-to-end
oracle that validates the full stack against SymPy, an independent
implementation.
"""

import math

import numpy as np
import pytest
import sympy as sp
from sympy import ZZ
from sympy.matrices.normalforms import smith_normal_form as sympy_snf

from modularsnf import crt_snf, smith_normal_form_mod
from modularsnf.crt import factorize

ALGORITHMS = [smith_normal_form_mod, crt_snf]


def _check_contract(A, modulus, S, U, V):
    """S, U, V are list[list[int]] with the documented shapes and S = U A V."""
    rows = len(A)
    cols = len(A[0]) if rows else 0
    assert np.array(U).shape == (rows, rows)
    assert np.array(V).shape == (cols, cols)
    assert np.array(S).shape == (rows, cols)
    # Returned as plain Python ints, not numpy scalars.
    assert all(isinstance(x, int) for row in S for x in row)
    prod = (
        np.array(U, object) @ np.array(A, object) @ np.array(V, object)
    ) % modulus
    assert np.array_equal(prod, np.array(S, object) % modulus)


@pytest.mark.parametrize("fn", ALGORITHMS)
@pytest.mark.parametrize("rows,cols", [(3, 3), (4, 6), (6, 4), (5, 1), (1, 5)])
def test_shapes_and_contract(fn, rows, cols):
    modulus = 12
    A = [[(i * 7 + j * 3) % modulus for j in range(cols)] for i in range(rows)]
    S, U, V = fn(A, modulus)
    _check_contract(A, modulus, S, U, V)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_negative_entries_reduced(fn):
    """Negative inputs are reduced into [0, N) by the binding."""
    S, U, V = fn([[-1, -5], [-7, 2]], 12)
    assert all(0 <= x < 12 for row in S for x in row)
    _check_contract([[-1, -5], [-7, 2]], 12, S, U, V)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_tuple_input_accepted(fn):
    S, U, V = fn(((2, 4), (6, 8)), 12)
    assert np.array(S).shape == (2, 2)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_empty_matrix(fn):
    assert fn([], 12) == ([], [], [])


@pytest.mark.parametrize("fn", ALGORITHMS)
@pytest.mark.parametrize("modulus", [1, 0, -3])
def test_modulus_too_small(fn, modulus):
    with pytest.raises(ValueError):
        fn([[1, 2], [3, 4]], modulus)


@pytest.mark.parametrize("fn", ALGORITHMS)
@pytest.mark.parametrize("modulus", [True, 3.5])
def test_modulus_must_be_int(fn, modulus):
    with pytest.raises(ValueError):
        fn([[1]], modulus)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_modulus_out_of_int64(fn):
    with pytest.raises(OverflowError):
        fn([[1, 2], [3, 4]], 1 << 70)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_non_list_input(fn):
    with pytest.raises(TypeError):
        fn(42, 12)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_ragged_rows(fn):
    with pytest.raises(ValueError):
        fn([[1, 2, 3], [4, 5]], 12)


@pytest.mark.parametrize("fn", ALGORITHMS)
def test_entry_out_of_int64(fn):
    with pytest.raises(OverflowError):
        fn([[1 << 70]], 12)


def test_factorize():
    assert factorize(36) == [(2, 2), (3, 2)]
    assert factorize(30) == [(2, 1), (3, 1), (5, 1)]
    assert factorize(17) == [(17, 1)]


def test_crt_explicit_factors_match_auto():
    A = [[2, 4, 0], [6, 8, 3], [0, 3, 9]]
    assert crt_snf(A, 36).S == crt_snf(A, 36, factors=[(2, 2), (3, 2)]).S


def test_crt_bad_factors_rejected():
    with pytest.raises(ValueError):
        crt_snf([[1]], 36, factors=[(2, 1), (3, 2)])  # = 18, not 36


def test_crt_matches_default_on_small_case():
    """Sanity cross-check at the boundary (full parity is tested in Rust)."""
    A = [[2, 4, 0], [6, 8, 3], [0, 3, 9]]
    assert crt_snf(A, 36).S == smith_normal_form_mod(A, 36).S


def _normalized_invariants(diag, modulus):
    """gcd of each diagonal entry with the modulus, ascending."""
    r = min(len(diag), len(diag[0])) if diag else 0
    return sorted(math.gcd(int(diag[i][i]), modulus) for i in range(r))


def _sympy_invariants(A, modulus):
    """Invariant factors of the integer SNF, projected into Z/NZ."""
    D = sympy_snf(sp.Matrix(A), domain=ZZ)
    r = min(D.rows, D.cols)
    return sorted(math.gcd(int(D[i, i]), modulus) for i in range(r))


@pytest.mark.parametrize("fn", ALGORITHMS)
@pytest.mark.parametrize("modulus", [6, 12, 36, 8, 30, 100])
@pytest.mark.parametrize("rows,cols", [(3, 3), (4, 6), (6, 4), (5, 2), (1, 4)])
def test_invariants_match_sympy(fn, modulus, rows, cols):
    """The full stack agrees with SymPy on the invariant factors over Z/NZ."""
    rng = np.random.default_rng(modulus * 1000 + rows * 10 + cols)
    A = rng.integers(0, modulus, size=(rows, cols)).tolist()
    assert _normalized_invariants(
        fn(A, modulus).S, modulus
    ) == _sympy_invariants(A, modulus)
