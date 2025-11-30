from typing import Tuple
from .matrix import RingMatrix
from .ring import RingZModN

def lemma_3_1(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, int]:
    """
    Implements Storjohann's Lemma 3.1.
    Returns (U, T, rank) such that T = U * A is in row echelon form.
    'rank' is the number of non-zero rows found (r).
    """
    ring: RingZModN = A.ring
    n_rows, n_cols = A.nrows, A.ncols

    U = RingMatrix.identity(ring, n_rows)
    T = A.copy()

    r = 0  # current pivot row count (rank)

    for k in range(n_cols):
        if r >= n_rows:
            break

        # eliminate entries below row r in column k
        for i in range(r + 1, n_rows):
            a = T.data[r][k]
            b = T.data[i][k]

            if ring.is_zero(b):
                continue

            # compute unimodular 2x2 transform
            g, s, t, u, v = ring.gcdex(a, b)

            # apply row operation to T and U
            T.apply_row_2x2(r, i, s, t, u, v)
            U.apply_row_2x2(r, i, s, t, u, v)

        # if the pivot is nonzero, we lock this row and move down
        if not ring.is_zero(T.data[r][k]):
            r += 1

    return U, T, r

def index1_reduce_on_columns(A: RingMatrix, k: int) -> Tuple[RingMatrix, RingMatrix]:
    """
    Specialized index-1 reduction for 7.3 step 9.
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
            for col in range(T.ncols):
                T.data[i][col] = ring.add(T.data[i][col], ring.mul(phi, T.data[j][col]))
            for col in range(U.ncols):
                U.data[i][col] = ring.add(U.data[i][col], ring.mul(phi, U.data[j][col]))
    return U, T