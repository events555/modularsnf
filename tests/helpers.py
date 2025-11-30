import math
import numpy as np
from modularsnf.matrix import RingMatrix
from itertools import product


def det_ring_matrix(M: RingMatrix) -> int:
    """
    Naive determinant for small square matrices over RingZModN.
    Returns an integer representative (mod N is implied by the ring).
    """
    ring = M.ring
    data = M.data
    n = M.nrows
    assert n == M.ncols

    if n == 1:
        return int(data[0, 0] % ring.N)
    if n == 2:
        a, b = data[0]
        c, d = data[1]
        ad = ring.mul(a, d)
        bc = ring.mul(b, c)
        return ring.sub(ad, bc)

    det = 0
    for j in range(n):
        sub_rows = np.concatenate((data[1:, :j], data[1:, j+1:]), axis=1)
        subM = RingMatrix(ring, sub_rows)
        sub_det = det_ring_matrix(subM)
        term = ring.mul(data[0, j], sub_det)
        if j % 2 == 0:
            det = ring.add(det, term)
        else:
            det = ring.sub(det, term)
    return det

def verify_echelon_structure(T: RingMatrix) -> bool:
    """
    Check:
    - For each non-zero row, the first non-zero column index strictly increases.
    - Once a zero row appears, all later rows are zero.
    """
    ring = T.ring
    nrows, ncols = T.nrows, T.ncols
    last_pivot_col = -1
    zero_row_seen = False

    for r in range(nrows):
        row = T.data[r]
        # find first non-zero entry in this row
        pivot_col = -1
        for c in range(ncols):
            if not ring.is_zero(row[c]):
                pivot_col = c
                break

        if pivot_col == -1:
            # zero row
            zero_row_seen = True
            # all subsequent rows must be zero
            if any(not all(ring.is_zero(x) for x in T.data[rr]) for rr in range(r+1, nrows)):
                return False
        else:
            # non-zero row; we must not have seen a zero row before
            if zero_row_seen:
                return False
            if pivot_col <= last_pivot_col:
                return False
            last_pivot_col = pivot_col

    return True

def row_span(M: RingMatrix) -> set[tuple[int, ...]]:
    """
    Compute the full row span of M over Z/NZ by brute force.
    Only for small matrices (e.g., r <= 3) in tests.
    """
    ring = M.ring
    N = ring.N
    r, c = M.nrows, M.ncols
    rows = M.data

    span = set()
    for coeffs in product(range(N), repeat=r):
        vec = np.zeros(c, dtype=int)
        for i, alpha in enumerate(coeffs):
            if alpha == 0:
                continue
            row = rows[i]
            vec = (vec + (alpha * row)) % N
        span.add(tuple(int(x % N) for x in vec))
    return span

def get_normalized_invariants(M: RingMatrix) -> list[int]:
    """
    Extracts diagonal entries and normalizes them to ideal generators.
    invariant = gcd(d_i, N).
    """
    n = min(M.nrows, M.ncols)
    diags = np.diag(M.data[:n, :n])
    invariants = [math.gcd(int(d), M.ring.N) for d in diags]
    invariants.sort()
    return invariants