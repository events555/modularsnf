from typing import Tuple
from modularsnf.ring import RingZModN
from modularsnf.matrix import RingMatrix
from modularsnf.echelon import lemma_3_1

def triang(B: RingMatrix, b: int) -> Tuple[RingMatrix, RingMatrix]:
    """
    Lemma 7.3 (Triang) in block form.

    Input:
        B: n1 x n1 upper-b-banded block, where
           s1 = floor(b/2), s2 = b-1, n1 = s1 + s2.
        b: bandwidth parameter (b > 2).

    Output:
        (B_prime, W) where:
            - W is s2 x s2 local right transform.
            - B_prime = B * diag(I_{s1}, W).
            - The top-right s1 x s2 block of B_prime has the 'triangulated'
              structure induced by lemma_3_1 on B2^T.
    """
    ring: RingZModN = B.ring
    if b <= 2:
        raise ValueError("Triang requires b > 2")

    s1 = b // 2
    s2 = b - 1
    n1 = s1 + s2

    if B.nrows != B.ncols or B.nrows != n1:
        raise ValueError(f"Triang expects an n1 x n1 block with n1 = s1 + s2 = {n1}")

    # Extract the top-right s1 x s2 block.
    row0, row1 = 0, s1
    col0, col1 = s1, s1 + s2
    B2 = B.submatrix(row0, row1, col0, col1)   # shape (s1, s2)

    # Transpose B2 to work with the s2 x s1 view.
    C = B2.transpose()

    # Apply Lemma 3.1 to obtain the left transform for C.
    U_left, T_ech, _ = lemma_3_1(C)  # U_left: s2 x s2, T_ech: s2 x s1

    # Use the transpose of the left transform as the local right transform.
    W = U_left.transpose()        # W: s2 x s2

    # Form the block-diagonal right transform.
    I_s1 = RingMatrix.identity(ring, s1)
    V_loc = RingMatrix.block_diag(I_s1, W)     # n1 x n1

    # Multiply by the block-diagonal transform.
    B_prime = B @ V_loc

    return B_prime, W

def shift(C: RingMatrix, b: int) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Implements Storjohann's Lemma 7.4 (Shift) on a local block.

    Input:
        C : n2 x n2 RingMatrix, where
                s2 = b - 1,
                n2 = 2 * s2,
             and C is (conceptually) upper b-banded.
        b : bandwidth parameter, b > 2.

    Output:
        (C_prime, U_block, V_block) where:

            - U_block is s2 x s2, the *local left* transform
              to be embedded as diag(U_block, I_{s2}) on the left.

            - V_block is s2 x s2, the *local right* transform
              to be embedded as diag(I_{s2}, V_block) on the right.

            - C_prime = diag(U_block, I_{s2}) * C * diag(I_{s2}, V_block)
              is the shifted block C' from Lemma 7.4.

    This does *not* attempt to check or enforce bandedness; that is left
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

    # Block partition where each block is s2 x s2.
    C1 = C.submatrix(0, s2, 0, s2)          # top-left
    C2 = C.submatrix(0, s2, s2, 2 * s2)     # top-right
    # C3 = C.submatrix(s2, 2 * s2, 0, s2)   # bottom-left (not used directly)
    # C4 = C.submatrix(s2, 2 * s2, s2, 2 * s2)

    # Apply Lemma 3.1 to triangularize C1 on the left.
    U1, T1, _ = lemma_3_1(C1)      # shapes: U1 s2×s2, T1 s2×s2

    # Triangularize U1 * C2 on the right via Lemma 3.1 applied to its transpose.
    C2_prime = U1 @ C2                          # s2×s2

    C2_prime_T = C2_prime.transpose()           # s2×s2
    U2, T2, _ = lemma_3_1(C2_prime_T)              # U2 s2×s2, T2 s2×s2
    V_block = U2.transpose()                    # s2×s2

    # Assemble the local transforms and apply them to C.
    I_s2 = RingMatrix.identity(ring, s2)

    U_full = RingMatrix.block_diag(U1, I_s2)    # n2×n2 = diag(U1, I_s2)
    V_full = RingMatrix.block_diag(I_s2, V_block)  # n2×n2 = diag(I_s2, V_block)

    C_prime = U_full @ C @ V_full

    return C_prime, U1, V_block

# TODO: Implement band_reduction following Storjohann's BandReduction algorithm (see https://uwspace.uwaterloo.ca/items/b2a6ebc4-f0e2-40ab-93f1-7b5c7d4d1b7f).

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