# modularsnf/diagonal.py

from typing import Tuple
from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN

# --- Helper Functions ---

def _is_zero_matrix(M: RingMatrix) -> bool:
    """Checks if M is the zero matrix. Assumes M is in SNF."""
    if M.nrows == 0 or M.ncols == 0:
        return True
    # If M is in SNF, we only need to check the top-left element.
    return M.ring.is_zero(M.data[0][0])

def _get_rank(M: RingMatrix) -> int:
    """Determines the rank of a diagonal matrix (like SNF)."""
    rank = 0
    diag_len = min(M.nrows, M.ncols)
    for i in range(diag_len):
        if not M.ring.is_zero(M.data[i][i]):
            rank += 1
    return rank

# --- Main Functions (Proposition 7.7) ---

def smith_from_diagonal(D: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    if D.nrows != D.ncols:
        raise ValueError("smith_from_diagonal expects a square matrix")

    n = D.nrows
    if n == 0:
        ring = D.ring
        I0 = RingMatrix.identity(ring, 0)
        return I0, I0, D.copy()

    # 1. Pad once to a power of 2
    D_pad = D.pad_to_square_power2()

    # 2. Run the power-of-two recursion
    U_pad, V_pad, S_pad = _smith_from_diagonal_power2(D_pad)

    # 3. Crop back to original size
    # U_pad and V_pad are principal transforms for D_pad, so the principal
    # n x n submatrices are the transforms for D.
    U = U_pad.submatrix(0, n, 0, n)
    V = V_pad.submatrix(0, n, 0, n)
    S = S_pad.submatrix(0, n, 0, n)

    return U, V, S

def _smith_from_diagonal_power2(D: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Assumes D is square and size n is a power of 2.
    Returns U, V, S with S = U D V in Smith form.
    """
    ring: RingZModN = D.ring
    n = D.nrows

    if n == 1:
        I1 = RingMatrix.identity(ring, 1)
        return I1, I1, D.copy()

    t = n // 2
    D1 = D.submatrix(0, t, 0, t)
    D2 = D.submatrix(t, n, t, n)

    # Recursively Smith-ify each half:
    U1, V1, A = _smith_from_diagonal_power2(D1)
    U2, V2, B = _smith_from_diagonal_power2(D2)

    # Combine transforms up to this point:
    U_block = RingMatrix.block_diag(U1, U2)
    V_block = RingMatrix.block_diag(V1, V2)

    # Merge A and B into one Smith block S
    U_merge, V_merge, S = merge_smith_blocks(A, B)

    # Compose transforms: S = U_merge U_block D V_block V_merge
    U_total = U_merge @ U_block
    V_total = V_block @ V_merge

    return U_total, V_total, S

# --- Core Merge Logic (Theorem 7.11) ---

def merge_smith_blocks(A: RingMatrix, B: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Implements the recursive merge step (Theorem 7.11).
    Assumes A and B are the same size n (power of 2), and already in SNF.
    """
    ring = A.ring
    n = A.nrows

    if B.ring is not ring:
        raise ValueError("merge_smith_blocks requires same ring")
    if n != A.ncols or n != B.nrows or n != B.ncols:
         raise ValueError(f"merge_smith_blocks expects A,B to be same square size, got {A.shape}, {B.shape}")

    if n == 0:
        I0 = RingMatrix.identity(ring, 0)
        return I0, I0, I0

    # Base Case (Lemma 7.10)
    if n == 1:
        a = A.data[0][0]
        b = B.data[0][0]
        return _merge_scalars(a, b, ring)

    # Setup for recursion
    t = n // 2
    N = 2 * n  # Total size (4t)

    # 1. Split into 4 blocks (t x t each)
    A1 = A.submatrix(0, t, 0, t)
    A2 = A.submatrix(t, n, t, n)
    B1 = B.submatrix(0, t, 0, t)
    B2 = B.submatrix(t, n, t, n)

    U_total = RingMatrix.identity(ring, N)
    V_total = RingMatrix.identity(ring, N)

    # Index map for the 4 blocks: A1=0, A2=1, B1=2, B2=3
    # Used to calculate offsets for embedding the local transforms.
    index_map = {
        0: 0, 1: t, 2: n, 3: n + t
    }

    # Helper closure to apply a recursive merge step and update global transforms
    def apply_merge_step(Block1, Block2, idx1, idx2):
        # Optimization: If both are zero, skip.
        if _is_zero_matrix(Block1) and _is_zero_matrix(Block2):
             return Block1, Block2

        # Recursive call
        U_loc, V_loc, S_loc = merge_smith_blocks(Block1, Block2)

        # Extract results (t x t)
        Block1_prime = S_loc.submatrix(0, t, 0, t)
        Block2_prime = S_loc.submatrix(t, 2*t, t, 2*t)

        # Build global "star pattern" transformation matrices (N x N)
        U_step = RingMatrix.identity(ring, N)
        V_step = RingMatrix.identity(ring, N)

        start1 = index_map[idx1]
        start2 = index_map[idx2]

        # Embed U_loc (2t x 2t) into U_step
        U_step.write_block(start1, start1, U_loc.submatrix(0, t, 0, t)) # U11
        U_step.write_block(start1, start2, U_loc.submatrix(0, t, t, 2*t)) # U12
        U_step.write_block(start2, start1, U_loc.submatrix(t, 2*t, 0, t)) # U21
        U_step.write_block(start2, start2, U_loc.submatrix(t, 2*t, t, 2*t)) # U22

        # Embed V_loc (2t x 2t) into V_step
        V_step.write_block(start1, start1, V_loc.submatrix(0, t, 0, t)) # V11
        V_step.write_block(start1, start2, V_loc.submatrix(0, t, t, 2*t)) # V12
        V_step.write_block(start2, start1, V_loc.submatrix(t, 2*t, 0, t)) # V21
        V_step.write_block(start2, start2, V_loc.submatrix(t, 2*t, t, 2*t)) # V22

        # Update total transformations: U_total = U_step @ U_total; V_total = V_total @ V_step
        nonlocal U_total, V_total
        U_total = U_step @ U_total
        V_total = V_total @ V_step

        return Block1_prime, Block2_prime

    # --- The 5 Steps of Theorem 7.11 ---

    # Step 1: Merge (A1, B1). Indices (0, 2).
    A1, B1 = apply_merge_step(A1, B1, 0, 2)

    # Step 2: Merge (A2, B2). Indices (1, 3).
    A2, B2 = apply_merge_step(A2, B2, 1, 3)

    # Step 3: Merge (A2, B1). Indices (1, 2).
    A2, B1 = apply_merge_step(A2, B1, 1, 2)

    # Step 4: Merge (B1, B2). Indices (2, 3).
    B1, B2 = apply_merge_step(B1, B2, 2, 3)

    # Step 5: Cleanup Permutation (Handle zeros in PIRs)
    # If B2 is non-zero, B1 must not have trailing zeros.
    if not _is_zero_matrix(B2):
        r_B1 = _get_rank(B1)
        
        if r_B1 < t:
            # B1 has trailing zeros, B2 is non-zero. We must permute.
            r_B2 = _get_rank(B2) # We know r_B2 > 0

            # Construct the permutation matrix P_loc for the 2t x 2t block (B1, B2).
            # This performs a stable sort to move non-zeros ahead of zeros.
            P_loc = RingMatrix(ring, [[0 for _ in range(2*t)] for _ in range(2*t)])
            one = 1

            # P[new_row][old_row] = 1

            # 1. Map B1_nz (old rows 0..r_B1-1) to (new rows 0..r_B1-1)
            for i in range(r_B1):
                P_loc.data[i][i] = one
            # 2. Map B2_nz (old rows t..t+r_B2-1) to follow B1_nz (new rows r_B1..)
            for i in range(r_B2):
                P_loc.data[r_B1 + i][t + i] = one
            
            current_row = r_B1 + r_B2
            # 3. Map B1_z (old rows r_B1..t-1) to follow B2_nz
            for i in range(t - r_B1):
                P_loc.data[current_row][r_B1 + i] = one
                current_row += 1
            # 4. Map B2_z (old rows t+r_B2..) to the end
            for i in range(t - r_B2):
                P_loc.data[current_row][t + r_B2 + i] = one
                current_row += 1

            # Apply globally. P_loc applies to the last 2t block (indices n to 2n).
            P_glob = RingMatrix.identity(ring, N)
            P_glob.write_block(n, n, P_loc)
            P_glob_T = P_glob.transpose()

            # Update the total transformations: U = P @ U_total; V = V_total @ P^T
            U_total = P_glob @ U_total
            V_total = V_total @ P_glob_T

            # Update B1 and B2 to reflect the permutation for the final assembly.
            B_combined = RingMatrix.block_diag(B1, B2)
            # S_target = P_loc @ B_combined @ P_loc^T (Since B_combined is diagonal and P is permutation)
            S_target = P_loc @ B_combined @ P_loc.transpose()
            B1 = S_target.submatrix(0, t, 0, t)
            B2 = S_target.submatrix(t, 2*t, t, 2*t)

    # Final Assembly
    A_prime = RingMatrix.block_diag(A1, A2)
    B_prime = RingMatrix.block_diag(B1, B2)
    S_final = RingMatrix.block_diag(A_prime, B_prime)

    return U_total, V_total, S_final

def _merge_scalars(a: int, b: int, ring: RingZModN) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Base case (Lemma 7.10): A = [a], B = [b].
    Returns 2×2 U, V, S with S = U diag(a,b) V in Smith form.
    """
    g, s, t, u, v = ring.gcdex(a, b)

    if ring.is_zero(g):
        # If g=0 (which means a=b=0 in Z/NZ), the result is 0 matrix, transforms are I.
        I2 = RingMatrix.identity(ring, 2)
        Z2 = RingMatrix(ring, [[0, 0], [0, 0]])
        return I2, I2, Z2

    # q = -Div(t * b, g) (as per Lemma 7.10 proof)
    tb = ring.mul(t, b)
    q_raw = ring.div(tb, g)
    q = (-q_raw) % ring.N

    # Build U, V (Lemma 7.10)
    U = RingMatrix.from_rows(ring, [[s, t], [u, v]])
    # V = [1 q; 1 1+q]
    V = RingMatrix.from_rows(ring, [[1, q], [1, (1 + q) % ring.N]])

    AB = RingMatrix.diagonal(ring, [a, b])
    S = U @ AB @ V
    return U, V, S