"""Helpers for diagonalizing matrices into Smith normal form."""

from typing import Tuple
from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN


def _is_zero_matrix(M: RingMatrix) -> bool:
    """Determine whether an SNF-form matrix contains only zeros.

    Args:
        M: Smith form matrix to examine.

    Returns:
        True if the matrix has no rows or columns or if its (0, 0) entry is
        zero; False otherwise.
    """

    if M.nrows == 0 or M.ncols == 0:
        return True
    return M.ring.is_zero(M.data[0][0])


def _get_rank(M: RingMatrix) -> int:
    """Count the diagonal rank of an SNF-form matrix.

    Args:
        M: Smith form matrix whose rank should be measured.

    Returns:
        The number of nonzero diagonal entries, limited by the shorter
        dimension of the matrix.
    """

    rank = 0
    diag_len = min(M.nrows, M.ncols)
    for i in range(diag_len):
        if not M.ring.is_zero(M.data[i][i]):
            rank += 1
    return rank


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

    # Step 1: Pad once to a power of two.
    D_pad = D.pad_to_square_power2()

    # Step 2: Run the power-of-two recursion.
    U_pad, V_pad, S_pad = _smith_from_diagonal_power2(D_pad)

    # Step 3: Crop back to original size. U_pad and V_pad are principal
    # transforms for D_pad, so the principal n x n submatrices are the
    # transforms for D.
    U = U_pad.submatrix(0, n, 0, n)
    V = V_pad.submatrix(0, n, 0, n)
    S = S_pad.submatrix(0, n, 0, n)

    return U, V, S


def _smith_from_diagonal_power2(
    D: RingMatrix,
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Diagonalize a power-of-two square matrix into Smith form.

    Args:
        D: Square matrix with dimension a power of two.

    Returns:
        A tuple ``(U, V, S)`` where ``S`` is the Smith form of ``D`` and
        ``U`` and ``V`` are the unimodular transforms such that ``S = U D V``.
    """

    ring: RingZModN = D.ring
    n = D.nrows

    if n == 1:
        I1 = RingMatrix.identity(ring, 1)
        return I1, I1, D.copy()

    t = n // 2
    D1 = D.submatrix(0, t, 0, t)
    D2 = D.submatrix(t, n, t, n)

    U1, V1, A = _smith_from_diagonal_power2(D1)
    U2, V2, B = _smith_from_diagonal_power2(D2)

    U_block = RingMatrix.block_diag(U1, U2)
    V_block = RingMatrix.block_diag(V1, V2)

    U_merge, V_merge, S = merge_smith_blocks(A, B)

    U_total = U_merge @ U_block
    V_total = V_block @ V_merge

    return U_total, V_total, S


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

    if n == 0:
        I0 = RingMatrix.identity(ring, 0)
        return I0, I0, I0

    if n == 1:
        a = A.data[0][0]
        b = B.data[0][0]
        return _merge_scalars(a, b, ring)

    t = n // 2
    N = 2 * n

    A1 = A.submatrix(0, t, 0, t)
    A2 = A.submatrix(t, n, t, n)
    B1 = B.submatrix(0, t, 0, t)
    B2 = B.submatrix(t, n, t, n)

    U_total = RingMatrix.identity(ring, N)
    V_total = RingMatrix.identity(ring, N)

    index_map = {0: 0, 1: t, 2: n, 3: n + t}

    def apply_merge_step(Block1, Block2, idx1, idx2):
        """Apply recursive merge and embed results in global transforms."""

        if _is_zero_matrix(Block1) and _is_zero_matrix(Block2):
            return Block1, Block2

        U_loc, V_loc, S_loc = merge_smith_blocks(Block1, Block2)

        Block1_prime = S_loc.submatrix(0, t, 0, t)
        Block2_prime = S_loc.submatrix(t, 2 * t, t, 2 * t)

        U_step = RingMatrix.identity(ring, N)
        V_step = RingMatrix.identity(ring, N)

        start1 = index_map[idx1]
        start2 = index_map[idx2]

        U_step.write_block(start1, start1, U_loc.submatrix(0, t, 0, t))
        U_step.write_block(start1, start2, U_loc.submatrix(0, t, t, 2 * t))
        U_step.write_block(start2, start1, U_loc.submatrix(t, 2 * t, 0, t))
        U_step.write_block(start2, start2, U_loc.submatrix(t, 2 * t, t, 2 * t))

        V_step.write_block(start1, start1, V_loc.submatrix(0, t, 0, t))
        V_step.write_block(start1, start2, V_loc.submatrix(0, t, t, 2 * t))
        V_step.write_block(start2, start1, V_loc.submatrix(t, 2 * t, 0, t))
        V_step.write_block(start2, start2, V_loc.submatrix(t, 2 * t, t, 2 * t))

        nonlocal U_total, V_total
        U_total = U_step @ U_total
        V_total = V_total @ V_step

        return Block1_prime, Block2_prime

    A1, B1 = apply_merge_step(A1, B1, 0, 2)
    A2, B2 = apply_merge_step(A2, B2, 1, 3)
    A2, B1 = apply_merge_step(A2, B1, 1, 2)
    B1, B2 = apply_merge_step(B1, B2, 2, 3)

    if not _is_zero_matrix(B2):
        r_B1 = _get_rank(B1)

        if r_B1 < t:
            r_B2 = _get_rank(B2)

            P_loc = RingMatrix(ring, [[0 for _ in range(2 * t)] for _ in range(2 * t)])
            one = 1

            for i in range(r_B1):
                P_loc.data[i][i] = one
            for i in range(r_B2):
                P_loc.data[r_B1 + i][t + i] = one

            current_row = r_B1 + r_B2
            for i in range(t - r_B1):
                P_loc.data[current_row][r_B1 + i] = one
                current_row += 1
            for i in range(t - r_B2):
                P_loc.data[current_row][t + r_B2 + i] = one
                current_row += 1

            P_glob = RingMatrix.identity(ring, N)
            P_glob.write_block(n, n, P_loc)
            P_glob_T = P_glob.transpose()

            U_total = P_glob @ U_total
            V_total = V_total @ P_glob_T

            B_combined = RingMatrix.block_diag(B1, B2)
            S_target = P_loc @ B_combined @ P_loc.transpose()
            B1 = S_target.submatrix(0, t, 0, t)
            B2 = S_target.submatrix(t, 2 * t, t, 2 * t)

    A_prime = RingMatrix.block_diag(A1, A2)
    B_prime = RingMatrix.block_diag(B1, B2)
    S_final = RingMatrix.block_diag(A_prime, B_prime)

    return U_total, V_total, S_final


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

    g, s, t, u, v = ring.gcdex(a, b)

    if ring.is_zero(g):
        I2 = RingMatrix.identity(ring, 2)
        Z2 = RingMatrix(ring, [[0, 0], [0, 0]])
        return I2, I2, Z2

    tb = ring.mul(t, b)
    q_raw = ring.div(tb, g)
    q = (-q_raw) % ring.N

    U = RingMatrix.from_rows(ring, [[s, t], [u, v]])
    V = RingMatrix.from_rows(ring, [[1, q], [1, (1 + q) % ring.N]])

    AB = RingMatrix.diagonal(ring, [a, b])
    S = U @ AB @ V
    return U, V, S