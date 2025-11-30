"""Row-echelon utilities for modular matrices.

This module contains the row-reduction helpers used throughout the Smith normal
form pipeline over the ring ``Z/nZ``. The routines mirror the primitives in
Storjohann's dissertation:

* ``lemma_3_1`` constructs a unimodular left transform that drives a matrix to
  row-echelon form while tracking rank.
* ``index1_reduce_on_columns`` applies a targeted column-reduction step for the
  index-1 case in the band-reduction phase.

Each helper returns the unimodular matrix that witnesses the transformation so
callers can compose them into larger reduction sequences.
"""

from typing import Tuple

from .matrix import RingMatrix
from .ring import RingZModN


def lemma_3_1(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, int]:
    """Compute a row-echelon form using Storjohann's Lemma 3.1.

    The routine walks across the columns of ``A``. For each column it clears
    entries below the current pivot row by applying a 2×2 unimodular transform
    derived from the extended GCD of the pivot and the entry to eliminate. The
    accumulated unimodular matrix ``U`` records the full left transform such
    that ``T = U * A`` is in row-echelon form.

    Args:
        A: Input matrix over the modular ring ``Z/nZ``.

    Returns:
        Tuple containing the unimodular transform ``U``, the row-echelon matrix
        ``T``, and the detected rank.
    """

    ring: RingZModN = A.ring
    n_rows, n_cols = A.nrows, A.ncols

    U = RingMatrix.identity(ring, n_rows)
    T = A.copy()

    r = 0  # Tracks the pivot row index, which doubles as the current rank.

    for k in range(n_cols):
        if r >= n_rows:
            break

        # Eliminate entries below row r in column k.
        for i in range(r + 1, n_rows):
            a = T.data[r][k]
            b = T.data[i][k]

            if ring.is_zero(b):
                continue

            # Compute the unimodular 2×2 transform from gcdex.
            g, s, t, u, v = ring.gcdex(a, b)

            # Apply the paired row operations to both T and U.
            T.apply_row_2x2(r, i, s, t, u, v)
            U.apply_row_2x2(r, i, s, t, u, v)

        # If the pivot is nonzero, lock this row and move down.
        if not ring.is_zero(T.data[r][k]):
            r += 1

    return U, T, r


def index1_reduce_on_columns(A: RingMatrix, k: int) -> Tuple[RingMatrix, RingMatrix]:
    """Perform specialized index-1 reduction on the first ``k`` columns.

    This routine targets the "index-1" case from section 7.3 (step 9) of
    Storjohann's algorithm. It zeroes entries above the diagonal in the leading
    ``k`` columns by adding multiples of each pivot row to the rows above it.
    The unimodular matrix ``U`` captures the cumulative left transform such that
    ``T = U * A`` reflects the reduced structure.

    Args:
        A: Input matrix over the modular ring ``Z/nZ``.
        k: Column count to reduce (starting after column ``0``).

    Returns:
        Tuple containing the unimodular transform ``U`` and the reduced matrix
        ``T``.
    """

    ring = A.ring
    n = A.nrows
    U = RingMatrix.identity(ring, n)
    T = A.copy()

    for j in range(1, k):
        sj = T.data[j][j]
        for i in range(j):
            x = T.data[i][j]
            if ring.is_zero(x):
                continue
            phi = (-ring.quo(x, sj)) % ring.N
            # Vectorized row updates to clear the current entry.
            T.data[i, :] = (T.data[i, :] + phi * T.data[j, :]) % ring.N
            U.data[i, :] = (U.data[i, :] + phi * U.data[j, :]) % ring.N

    return U, T
