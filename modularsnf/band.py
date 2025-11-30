"""Band reduction transforms for Storjohann's block algorithms.

Provides helpers for triang and shift steps that manipulate banded blocks to
support Smith normal form reduction over modular rings.
"""

from typing import Tuple
from modularsnf.ring import RingZModN
from modularsnf.matrix import RingMatrix
from modularsnf.echelon import lemma_3_1

def triang(B: RingMatrix, b: int) -> Tuple[RingMatrix, RingMatrix]:
    """
    Triangulates the top-right block of an upper-b-banded matrix.

    Args:
        B: An n1 x n1 upper-b-banded block with s1 = floor(b/2), s2 = b - 1,
            and n1 = s1 + s2.
        b: Bandwidth parameter; must satisfy b > 2.

    Returns:
        A tuple ``(B_prime, W)`` where ``W`` is the s2 x s2 local right
        transform such that ``B_prime = B * diag(I_{s1}, W)`` and the
        top-right s1 x s2 block of ``B_prime`` has the triangulated structure
        induced by ``lemma_3_1`` on ``B2^T``.

    Raises:
        ValueError: If ``b <= 2`` or ``B`` is not an n1 x n1 block with
            ``n1 = s1 + s2``.
    """
    ring: RingZModN = B.ring
    if b <= 2:
        raise ValueError("Triang requires b > 2")

    s1 = b // 2
    s2 = b - 1
    n1 = s1 + s2

    if B.nrows != B.ncols or B.nrows != n1:
        raise ValueError(f"Triang expects an n1 x n1 block with n1 = s1 + s2 = {n1}")

    # 1. Extract B2: top-right s1 x s2 block
    row0, row1 = 0, s1
    col0, col1 = s1, s1 + s2
    B2 = B.submatrix(row0, row1, col0, col1)   # shape (s1, s2)

    # 2. Work with C = B2^T (shape s2 x s1)
    C = B2.transpose()

    # 3. Apply Lemma 3.1 to C: U_left * C = T_ech
    U_left, T_ech, _ = lemma_3_1(C)  # U_left: s2 x s2, T_ech: s2 x s1

    # 4. Local right transform is W = U_left^T
    W = U_left.transpose()        # W: s2 x s2

    # 5. Build local block-diagonal right transform V_loc = diag(I_{s1}, W)
    I_s1 = RingMatrix.identity(ring, s1)
    V_loc = RingMatrix.block_diag(I_s1, W)     # n1 x n1

    # 6. Compute B' = B * V_loc
    B_prime = B @ V_loc

    return B_prime, W

def shift(C: RingMatrix, b: int) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Applies Storjohann's Lemma 7.4 (Shift) to a local block.

    Args:
        C: An n2 x n2 block with ``s2 = b - 1`` and ``n2 = 2 * s2`` that is
            conceptually upper b-banded.
        b: Bandwidth parameter; must satisfy b > 2.

    Returns:
        A tuple ``(C_prime, U_block, V_block)`` where ``U_block`` and
        ``V_block`` are the s2 x s2 local left and right transforms embedded as
        ``diag(U_block, I_{s2})`` and ``diag(I_{s2}, V_block)`` such that
        ``C_prime = diag(U_block, I_{s2}) * C * diag(I_{s2}, V_block)`` is the
        shifted block from Lemma 7.4.

    Raises:
        ValueError: If ``b <= 2`` or ``C`` is not an n2 x n2 block with
            ``n2 = 2 * (b - 1)``.

    This function does not attempt to check or enforce bandedness; that is left
    to higher-level tests.
    """
    ring: RingZModN = C.ring
    if b <= 2:
        raise ValueError("Shift requires b > 2")

    s2 = b - 1
    n2 = 2 * s2

    if C.nrows != C.ncols or C.nrows != n2:
        raise ValueError(
            f"Shift expects an n2 x n2 block with n2 = 2*(b-1) = {n2}, "
            f"got shape {C.shape}"
        )

    # Block partition:
    # C = [ C1  C2 ]
    #     [ C3  C4 ]   (C4 may be irrelevant structurally, but it's there)
    #
    # Each block is s2 x s2.
    C1 = C.submatrix(0, s2, 0, s2)          # top-left
    C2 = C.submatrix(0, s2, s2, 2 * s2)     # top-right
    # C3 = C.submatrix(s2, 2 * s2, 0, s2)   # bottom-left (not used directly)
    # C4 = C.submatrix(s2, 2 * s2, s2, 2 * s2)

    # Step 1: Left triangularization of C1.
    #
    # lemma_3_1(C1) returns (U1, T1) with T1 = U1 * C1 in row-ech form.
    # We take U_block := U1, to be embedded on the left as diag(U1, I).
    U1, T1, _ = lemma_3_1(C1)      # shapes: U1 s2×s2, T1 s2×s2

    # Step 2: Right triangularization of (U1 * C2).
    #
    # First form C2' = U1 * C2, then apply lemma_3_1 to (C2')^T to get
    # a left transform U2; the corresponding right transform is V_block = U2^T,
    # so that C2' * V_block = (T2)^T is triangular.
    C2_prime = U1 @ C2                          # s2×s2

    C2_prime_T = C2_prime.transpose()           # s2×s2
    U2, T2, _ = lemma_3_1(C2_prime_T)              # U2 s2×s2, T2 s2×s2
    V_block = U2.transpose()                    # s2×s2

    # Step 3: Form full local left/right transforms and apply them to C.
    I_s2 = RingMatrix.identity(ring, s2)

    U_full = RingMatrix.block_diag(U1, I_s2)    # n2×n2 = diag(U1, I_s2)
    V_full = RingMatrix.block_diag(I_s2, V_block)  # n2×n2 = diag(I_s2, V_block)

    C_prime = U_full @ C @ V_full

    return C_prime, U1, V_block

# def band_reduction(A: RingMatrix, b: int, t: int = 0) -> Tuple[RingMatrix, int]:
#     """
#     Algorithm BandReduction(A, b) as in your screenshot.

#     Input:
#         A : an upper b-banded n×n RingMatrix with last t columns zero.
#         b : current bandwidth, b > 2.
#         t : number of trailing zero columns.

#     Output:
#         (A_reduced, b_new) where:
#             A_reduced : upper ((floor(b/2) + 1))-banded matrix equivalent to A
#                         and with last t columns still zero.
#             b_new     : floor(b/2) + 1
#     """
#     if b <= 2:
#         return A.copy(), b

#     ring: RingZModN = A.ring
#     n = A.nrows
#     if n != A.ncols:
#         raise ValueError("BandReduction expects a square matrix")

#     # s1 := floor(b/2)
#     s1 = b // 2
#     # n1 := floor(b/2) + b - 1
#     n1 = (b // 2) + b - 1
#     # s2 := b - 1
#     s2 = b - 1
#     # n2 := 2(b - 1)
#     n2 = 2 * (b - 1)

#     # B := copy of A augmented with 2b - t rows/cols of zeros
#     pad = 2 * b - t
#     N_big = n + pad

#     zero = 0
#     B_data = [[zero for _ in range(N_big)] for _ in range(N_big)]
#     for i in range(n):
#         for j in range(n):
#             B_data[i][j] = A.data[i][j]
#     B = RingMatrix(ring, B_data)

#     # for i = 0 to ceil((n - t)/s1) - 1 do
#     num_i = max(0, (n - t + s1 - 1) // s1)
#     for i in range(num_i):
#         # subB[is1, n1] means principal submatrix of size n1 starting at is1
#         top = i * s1
#         B_block = B.submatrix(top, top + n1, top, top + n1)
#         B_block_prime, _W = triang(B_block, b)
#         B.write_block(top, top, B_block_prime)

#         # for j = 0 to ceil((n - t - (i + 1)s1)/s2) - 1 do
#         numer = n - t - (i + 1) * s1
#         if numer <= 0:
#             continue
#         num_j = max(0, (numer + s2 - 1) // s2)

#         for j in range(num_j):
#             # subB[(i+1)s1 + js2, n2]
#             offset = (i + 1) * s1 + j * s2
#             C_block = B.submatrix(offset, offset + n2, offset, offset + n2)
#             C_block_prime, _U_block, _V_block = shift(C_block, b)
#             B.write_block(offset, offset, C_block_prime)

#     # return subB[0, n]
#     A_reduced = B.submatrix(0, n, 0, n)
#     b_new = (b // 2) + 1
#     return A_reduced, b_new

def band_reduction(
    A: RingMatrix,
    b: int,
    t: int = 0,
) -> Tuple[RingMatrix, RingMatrix, RingMatrix, int]:
    """
    Proposition 7.1 (band reduction) with transform tracking.

    Input:
        A : n x n upper-b-banded matrix (over a PIR)
        b : current upper bandwidth
        t : number of trailing zero columns (as in the paper, often 0 in practice)

    Output:
        A_reduced : n x n upper-b'-banded matrix, with b' = floor(b/2) + 1
        U_band    : n x n left transform
        V_band    : n x n right transform
        b_new     : new bandwidth (floor(b/2) + 1)

    Such that:
        A_reduced = U_band * A * V_band
    """
    ring = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("band_reduction expects a square matrix.")

    # Already at or below 2-banded: nothing to do.
    if b <= 2:
        I = RingMatrix.identity(ring, n)
        return A.copy(), I, I, b

    # Storjohann parameters
    s1 = b // 2
    s2 = b - 1
    n1 = s1 + s2     # size of the "triang" block
    n2 = 2 * s2      # size of each "shift" block

    # Padding size in both directions (as in the paper / your original code)
    pad = 2 * b - t
    N_big = n + pad

    # Build padded matrix B with A in the top-left n x n principal block
    B_data = [[0 for _ in range(N_big)] for _ in range(N_big)]
    for i in range(n):
        row_i = A.data[i]
        for j in range(n):
            B_data[i][j] = row_i[j]
    B = RingMatrix(ring, B_data)

    # Global transforms on the padded system
    U_big = RingMatrix.identity(ring, N_big)
    V_big = RingMatrix.identity(ring, N_big)

    # Number of outer iterations (as in the paper)
    num_i = max(0, (n - t + s1 - 1) // s1)

    for i in range(num_i):
        top = i * s1
        if top + n1 > N_big:
            break  # safety guard

        # --- Step: triang on n1 x n1 principal block starting at (top, top)
        B_block = B.submatrix(top, top + n1, top, top + n1)
        # triang returns B_block_prime, W with B_block_prime = B_block * diag(I_{s1}, W)
        B_block_prime, W = triang(B_block, b)

        # Build local right transform V_loc = diag(I_{s1}, W)
        I_s1 = RingMatrix.identity(ring, s1)
        V_loc = RingMatrix.block_diag(I_s1, W)   # n1 x n1

        # Embed into N_big x N_big
        V_step = RingMatrix.identity(ring, N_big)
        V_step.write_block(top, top, V_loc)

        # Apply globally
        B = B @ V_step
        V_big = V_big @ V_step

        # --- Inner "shift" blocks
        numer = n - t - (i + 1) * s1
        if numer <= 0:
            continue

        num_j = max(0, (numer + s2 - 1) // s2)

        for j in range(num_j):
            offset = (i + 1) * s1 + j * s2
            if offset + n2 > N_big:
                break  # safety guard

            # Take current 2*s2 x 2*s2 block from B
            C_block = B.submatrix(offset, offset + n2, offset, offset + n2)

            # shift returns C_prime, U_block, V_block with:
            # C_prime = diag(U_block, I_{s2}) * C_block * diag(I_{s2}, V_block)
            C_prime, U_block, V_block = shift(C_block, b)

            # Local 2s2 x 2s2 transforms
            I_s2 = RingMatrix.identity(ring, s2)
            U_loc = RingMatrix.block_diag(U_block, I_s2)
            V_loc2 = RingMatrix.block_diag(I_s2, V_block)

            # Embed into N_big x N_big
            U_step = RingMatrix.identity(ring, N_big)
            V_step2 = RingMatrix.identity(ring, N_big)

            U_step.write_block(offset, offset, U_loc)
            V_step2.write_block(offset, offset, V_loc2)

            # Apply globally
            B = U_step @ B @ V_step2
            U_big = U_step @ U_big
            V_big = V_big @ V_step2

    # Extract the reduced n x n principal block and the corresponding transforms
    A_reduced = B.submatrix(0, n, 0, n)
    U_band = U_big.submatrix(0, n, 0, n)
    V_band = V_big.submatrix(0, n, 0, n)

    b_new = (b // 2) + 1
    return A_reduced, U_band, V_band, b_new


def compute_upper_bandwidth(M: RingMatrix) -> int:
    """
    Compute the upper bandwidth 'b' in the sense:
        b = max{ j - i | M[i,j] ≠ 0, j ≥ i } + 1,
    or 0 if the matrix is identically zero.
    """
    ring = M.ring
    nrows, ncols = M.shape
    max_offset = -1
    for i in range(nrows):
        for j in range(ncols):
            val = M.data[i][j]
            if not ring.is_zero(val) and j >= i:
                offset = j - i
                if offset > max_offset:
                    max_offset = offset

    if max_offset < 0:
        return 0
    return max_offset + 1


def project_to_upper_bandwidth(M: RingMatrix, b: int) -> RingMatrix:
    """
    Zero out entries outside the upper bandwidth 'b':
        keep entries with 0 ≤ j - i < b,
        zero everything else.
    """
    ring = M.ring
    nrows, ncols = M.shape
    data = [[0 for _ in range(ncols)] for _ in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            if 0 <= j - i < b:
                data[i][j] = M.data[i][j]
    return RingMatrix(ring, data)