"""Helpers for diagonalizing matrices into Smith normal form."""

from typing import Tuple

import numpy as np

from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN

try:
    from modularsnf._rust import (
        rust_smith_from_diagonal as _rust_diag,
        rust_merge_smith_blocks as _rust_merge,
    )
except ImportError:
    _rust_diag = None  # type: ignore[assignment]
    _rust_merge = None  # type: ignore[assignment]



def smith_from_diagonal(
    D: RingMatrix,
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Compute SNF transforms for a diagonal square matrix.

    Pads the matrix to a power-of-two dimension, runs the recursive merge
    routine, and then crops the result back to the original size.

    Args:
        D: Diagonal matrix to convert to Smith form.

    Returns:
        A tuple ``(U, V, S)`` where ``S`` is the Smith form of ``D`` and
        ``U`` and ``V`` are the unimodular transforms such that ``S = U D V``.

    Raises:
        ValueError: If ``D`` is not square.
    """

    if D.nrows != D.ncols:
        raise ValueError("smith_from_diagonal expects a square matrix")

    n = D.nrows
    if n == 0:
        ring = D.ring
        I0 = RingMatrix.identity(ring, 0)
        return I0, I0, D.copy()

    ring = D.ring

    if _rust_diag is not None:
        u_arr, v_arr, s_arr = _rust_diag(
            D.data.astype(np.int64), ring.N,
        )
        return (
            RingMatrix._from_ndarray(ring, u_arr[:n, :n].copy()),
            RingMatrix._from_ndarray(ring, v_arr[:n, :n].copy()),
            RingMatrix._from_ndarray(ring, s_arr[:n, :n].copy()),
        )

    # Python fallback: pad to power of two and run raw merge.
    p = max(n, 1)
    size = 1 << (p - 1).bit_length()
    pad_arr = np.zeros((size, size), dtype=int)
    pad_arr[:n, :n] = D.data

    U_arr, V_arr, S_arr = _smith_from_diagonal_raw(pad_arr, ring)

    U = RingMatrix._from_ndarray(ring, U_arr[:n, :n].copy())
    V = RingMatrix._from_ndarray(ring, V_arr[:n, :n].copy())
    S = RingMatrix._from_ndarray(ring, S_arr[:n, :n].copy())

    return U, V, S



def merge_smith_blocks(
    A: RingMatrix, B: RingMatrix
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Merge two SNF blocks of equal size.

    Implements the recursive merge step from Theorem 7.11. ``A`` and ``B``
    must be the same power-of-two dimension and already in Smith form.

    Args:
        A: Smith-form matrix to merge.
        B: Smith-form matrix to merge.

    Returns:
        A tuple ``(U, V, S)`` where ``S`` is the merged Smith form and ``U``
        and ``V`` are the associated unimodular transforms.

    Raises:
        ValueError: If the matrices use different rings or have mismatched
            sizes.
    """

    ring = A.ring
    n = A.nrows

    if B.ring is not ring:
        raise ValueError("merge_smith_blocks requires same ring")
    if n != A.ncols or n != B.nrows or n != B.ncols:
        raise ValueError(
            "merge_smith_blocks expects A,B to be same square size, "
            f"got {A.shape}, {B.shape}"
        )

    if _rust_merge is not None:
        u_arr, v_arr, s_arr = _rust_merge(
            A.data.astype(np.int64),
            B.data.astype(np.int64),
            ring.N,
        )
        return (
            RingMatrix._from_ndarray(ring, u_arr),
            RingMatrix._from_ndarray(ring, v_arr),
            RingMatrix._from_ndarray(ring, s_arr),
        )

    u_arr, v_arr, s_arr = _merge_raw(A.data, B.data, ring)
    return (
        RingMatrix._from_ndarray(ring, u_arr),
        RingMatrix._from_ndarray(ring, v_arr),
        RingMatrix._from_ndarray(ring, s_arr),
    )


def _merge_scalars(
    a: int, b: int, ring: RingZModN
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Merge two scalar SNF blocks.

    Args:
        a: Upper-left scalar entry.
        b: Lower-right scalar entry.
        ring: Underlying ring for the scalar operations.

    Returns:
        A tuple ``(U, V, S)`` where ``S`` is the Smith form of
        ``diag(a, b)`` and ``U`` and ``V`` are the unimodular transforms.
    """
    u_arr, v_arr, s_arr = _merge_scalars_raw(a, b, ring)
    return (
        RingMatrix._from_ndarray(ring, u_arr),
        RingMatrix._from_ndarray(ring, v_arr),
        RingMatrix._from_ndarray(ring, s_arr),
    )


# ---------------------------------------------------------------------------
# Raw numpy implementations — no RingMatrix construction in hot paths
# ---------------------------------------------------------------------------

_EYE2 = np.eye(2, dtype=int)
_ZERO2 = np.zeros((2, 2), dtype=int)


def _merge_scalars_raw(
    a: int, b: int, ring: RingZModN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge two scalar SNF blocks on raw arrays.

    Returns (U, V, S) as 2x2 numpy arrays reduced mod ring.N.
    """
    g, s, t, u, v = ring.gcdex(a, b)
    N = ring.N

    if g % N == 0:
        return _EYE2.copy(), _EYE2.copy(), _ZERO2.copy()

    tb = (t * b) % N
    q_raw = ring.div(tb, g)
    q = (-q_raw) % N

    u_arr = np.array([[s, t], [u, v]], dtype=int) % N
    v_arr = np.array([[1, q], [1, (1 + q) % N]], dtype=int)
    s_arr = (u_arr @ np.array([[a, 0], [0, b]], dtype=int) @ v_arr) % N
    return u_arr, v_arr, s_arr


def _is_zero_raw(arr: np.ndarray, N: int) -> bool:
    """Check if a diagonal SNF array is zero (first entry is zero mod N)."""
    if arr.shape[0] == 0:
        return True
    return arr[0, 0] % N == 0


def _get_rank_raw(arr: np.ndarray, N: int) -> int:
    """Count nonzero diagonal entries mod N."""
    n = min(arr.shape[0], arr.shape[1])
    rank = 0
    for i in range(n):
        if arr[i, i] % N != 0:
            rank += 1
    return rank


def _merge_raw(
    a_arr: np.ndarray, b_arr: np.ndarray, ring: RingZModN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw-array version of merge_smith_blocks.

    Args:
        a_arr: n x n SNF array.
        b_arr: n x n SNF array.
        ring: Ring for arithmetic.

    Returns:
        (U, V, S) as raw 2n x 2n numpy arrays.
    """
    N = ring.N
    n = a_arr.shape[0]

    if n == 0:
        e = np.zeros((0, 0), dtype=int)
        return e, e, e

    if n == 1:
        u2, v2, s2 = _merge_scalars_raw(int(a_arr[0, 0]), int(b_arr[0, 0]), ring)
        return u2, v2, s2

    t = n // 2
    NN = 2 * n

    A1 = a_arr[0:t, 0:t].copy()
    A2 = a_arr[t:n, t:n].copy()
    B1 = b_arr[0:t, 0:t].copy()
    B2 = b_arr[t:n, t:n].copy()

    U_total = np.eye(NN, dtype=int)
    V_total = np.eye(NN, dtype=int)

    index_map = (0, t, n, n + t)  # tuple for speed

    def apply_step(block1, block2, idx1, idx2):
        if _is_zero_raw(block1, N) and _is_zero_raw(block2, N):
            return block1, block2

        u_loc, v_loc, s_loc = _merge_raw(block1, block2, ring)

        b1_prime = s_loc[0:t, 0:t].copy()
        b2_prime = s_loc[t:2 * t, t:2 * t].copy()

        s1 = index_map[idx1]
        s2_ = index_map[idx2]

        # Left-apply U_loc block pair to U_total
        r1 = U_total[s1:s1 + t, :].copy()
        r2 = U_total[s2_:s2_ + t, :].copy()
        U_total[s1:s1 + t, :] = (u_loc[0:t, 0:t] @ r1 + u_loc[0:t, t:2 * t] @ r2) % N
        U_total[s2_:s2_ + t, :] = (u_loc[t:2 * t, 0:t] @ r1 + u_loc[t:2 * t, t:2 * t] @ r2) % N

        # Right-apply V_loc block pair to V_total
        c1 = V_total[:, s1:s1 + t].copy()
        c2 = V_total[:, s2_:s2_ + t].copy()
        V_total[:, s1:s1 + t] = (c1 @ v_loc[0:t, 0:t] + c2 @ v_loc[t:2 * t, 0:t]) % N
        V_total[:, s2_:s2_ + t] = (c1 @ v_loc[0:t, t:2 * t] + c2 @ v_loc[t:2 * t, t:2 * t]) % N

        return b1_prime, b2_prime

    A1, B1 = apply_step(A1, B1, 0, 2)
    A2, B2 = apply_step(A2, B2, 1, 3)
    A2, B1 = apply_step(A2, B1, 1, 2)
    B1, B2 = apply_step(B1, B2, 2, 3)

    if not _is_zero_raw(B2, N):
        r_B1 = _get_rank_raw(B1, N)

        if r_B1 < t:
            r_B2 = _get_rank_raw(B2, N)

            P_arr = np.zeros((2 * t, 2 * t), dtype=int)
            for i in range(r_B1):
                P_arr[i, i] = 1
            for i in range(r_B2):
                P_arr[r_B1 + i, t + i] = 1
            cr = r_B1 + r_B2
            for i in range(t - r_B1):
                P_arr[cr, r_B1 + i] = 1
                cr += 1
            for i in range(t - r_B2):
                P_arr[cr, t + r_B2 + i] = 1
                cr += 1

            # Build P_glob as identity with P_arr at (n, n)
            P_glob = np.eye(NN, dtype=int)
            P_glob[n:n + 2 * t, n:n + 2 * t] = P_arr
            P_glob_T = P_glob.T.copy()

            U_total = (P_glob @ U_total) % N
            V_total = (V_total @ P_glob_T) % N

            # Combine B1, B2 into block diag, permute
            B_comb = np.zeros((2 * t, 2 * t), dtype=int)
            B_comb[0:t, 0:t] = B1
            B_comb[t:2 * t, t:2 * t] = B2
            S_target = (P_arr @ B_comb @ P_arr.T) % N
            B1 = S_target[0:t, 0:t].copy()
            B2 = S_target[t:2 * t, t:2 * t].copy()

    # Assemble S_final as block_diag(A1, A2, B1, B2)
    S_final = np.zeros((NN, NN), dtype=int)
    S_final[0:t, 0:t] = A1
    S_final[t:n, t:n] = A2
    S_final[n:n + t, n:n + t] = B1
    S_final[n + t:NN, n + t:NN] = B2

    return U_total, V_total, S_final


def _smith_from_diagonal_raw(
    diag_arr: np.ndarray, ring: RingZModN
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw-array power-of-two diagonal SNF via bottom-up iterative merge.

    Args:
        diag_arr: n x n diagonal array with n a power of two.
        ring: Ring for arithmetic.

    Returns:
        (U, V, S) as n x n numpy arrays.
    """
    N = ring.N
    n = diag_arr.shape[0]

    if n <= 1:
        return np.eye(n, dtype=int), np.eye(n, dtype=int), diag_arr.copy()

    # Bottom-up: start with n 1x1 blocks, merge pairwise
    # Each element is (U, V, S) as raw arrays
    blocks = []
    for i in range(n):
        s = diag_arr[i:i + 1, i:i + 1].copy()
        blocks.append((np.ones((1, 1), dtype=int), np.ones((1, 1), dtype=int), s))

    size = 1
    while size < n:
        new_blocks = []
        for i in range(0, len(blocks), 2):
            U1, V1, A = blocks[i]
            U2, V2, B = blocks[i + 1]

            U_merge, V_merge, S = _merge_raw(A, B, ring)

            # U_block = block_diag(U1, U2), then U_total = U_merge @ U_block
            bsz = size
            U_block = np.zeros((2 * bsz, 2 * bsz), dtype=int)
            U_block[0:bsz, 0:bsz] = U1
            U_block[bsz:2 * bsz, bsz:2 * bsz] = U2
            U_total = (U_merge @ U_block) % N

            V_block = np.zeros((2 * bsz, 2 * bsz), dtype=int)
            V_block[0:bsz, 0:bsz] = V1
            V_block[bsz:2 * bsz, bsz:2 * bsz] = V2
            V_total = (V_block @ V_merge) % N

            new_blocks.append((U_total, V_total, S))

        blocks = new_blocks
        size *= 2

    return blocks[0]
