from typing import Tuple

from modularsnf.band import band_reduction, compute_upper_bandwidth
from .matrix import RingMatrix
from .ring import RingZModN
from .diagonal import smith_from_diagonal
from .echelon import index1_reduce_on_columns, lemma_3_1

def smith_normal_form(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Generic Smith Normal Form front-end.

    - If A is square: compute SNF directly.
    - If A is rectangular: pad with zero rows/columns to a square matrix,
      compute SNF, then crop U, V, S back to the original shape.

    Returns (U, V, S) with S = U * A * V in Smith normal form.
    """
    ring = A.ring
    n = A.nrows
    m = A.ncols

    # Trivial cases
    if n == 0 or m == 0:
        U = RingMatrix.identity(ring, n)
        V = RingMatrix.identity(ring, m)
        return U, V, A.copy()

    # Square case: no padding needed
    if n == m:
        return _smith_square(A)

    # Rectangular case: zero-pad to square s x s
    s = max(n, m)
    data = [[0 for _ in range(s)] for _ in range(s)]
    for i in range(n):
        for j in range(m):
            data[i][j] = A.data[i][j]
    A_pad = RingMatrix(ring, data)

    U_pad, V_pad, S_pad = _smith_square(A_pad)

    # Crop back to original shape
    U = U_pad.submatrix(0, n, 0, n)
    V = V_pad.submatrix(0, m, 0, m)
    S = S_pad.submatrix(0, n, 0, m)

    return U, V, S

def _smith_square(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Smith Normal Form for a *square* matrix A over a PIR.

    Returns (U, V, S) with S = U * A * V in Smith form (diagonal, invariant factors).
    No padding here; this assumes A is square.
    """
    ring = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("Expected a square matrix in _smith_square.")

    if n == 0:
        I0 = RingMatrix.identity(ring, 0)
        return I0, I0, A.copy()

    # Step 1: Row echelon form (Lemma 3.1)
    # T = U_ech * A, U_ech invertible over the ring.
    U_ech, T, _rank = lemma_3_1(A)

    # Step 2: Reduce bandwidth (Proposition 7.1) until upper 2-banded.
    B = T.copy()
    b = compute_upper_bandwidth(B)

    U_band_total = RingMatrix.identity(ring, n)
    V_band_total = RingMatrix.identity(ring, n)

    while b > 2:
        B, U_step, V_step, b = band_reduction(B, b)
        U_band_total = U_step @ U_band_total
        V_band_total = V_band_total @ V_step


    U_snf, V_snf, S = smith_from_upper_2_banded(B)

    U_total = U_snf @ U_band_total @ U_ech
    V_total = V_band_total @ V_snf

    return U_total, V_total, S

def smith_from_upper_2_banded(A: RingMatrix):
    ring: RingZModN = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("Expected square matrix")
    if n <= 1:
        U = RingMatrix.identity(ring, n)
        V = RingMatrix.identity(ring, n)
        return U, V, A.copy()

    # 7.3 step 1: preconditioning sweep -> spike + two blocks
    U1, V1, A1, n1, n2 = _step1_split_with_spike(A)

    # 7.3 step 2: recursive SNF on principal & trailing blocks
    U2, V2, A2 = _step2_recursive_blocks(A1, n1, n2)

    # 7.3 step 3: permutation of rows & columns
    U3, V3, A3 = _step3_permute(A2, n1, n2)

    # 7.3 step 4: diagonal SNF on principal (n-1)x(n-1)
    U4, V4, A4 = _step4_smith_on_n_minus_1(A3)

    # 7.3 step 5–8: build gcd chain s1..sk
    U5, V5, A5, k = _step5_to_8_gcd_chain(A4)

    # 7.3 step 9: reverse first k rows and index reduction
    U6, V6, A6 = _step9_index_reduction(A5, k)

    # Compose transforms: U_total, V_total such that U_total A V_total = A6
    U_total = U6 @ U5 @ U4 @ U3 @ U2 @ U1
    V_total = V1 @ V2 @ V3 @ V4 @ V5 @ V6

    return U_total, V_total, A6

def _step1_split_with_spike(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix, int, int]:
    """
    Step 1 of Storjohann 7.3:
    - computes U s.t. A1 = U A
    - partitions into n1, 1, n2 (principal / middle / trailing)
    - returns (U, V, A1, n1, n2) where V is Identity.
    """
    ring = A.ring
    n = A.nrows
    n1 = (n - 1) // 2
    n2 = n - 1 - n1

    U = RingMatrix.identity(ring, n)
    T = A.copy()

    # first loop
    for k in range(n-1, n1, -1): 
        r0, r1 = k-1, k
        col = k
        a = T.data[r0][col]
        b = T.data[r1][col]
        g, s, t, u, v = ring.gcdex(a, b)
        T.apply_row_2x2(r0, r1, u, v, s, t)
        U.apply_row_2x2(r0, r1, u, v, s, t)

    # second loop
    for k in range(n1+1, n-1):
        r0, r1 = k, k+1
        col = k
        a = T.data[r0][col]
        b = T.data[r1][col]
        g, s, t, u, v = ring.gcdex(a, b)
        T.apply_row_2x2(r0, r1, s, t, u, v)
        U.apply_row_2x2(r0, r1, s, t, u, v)

    V = RingMatrix.identity(ring, n)

    return U, V, T, n1, n2


def _step2_recursive_blocks(A: RingMatrix, n1: int, n2: int):
    """
    Implements 7.3 Step 2.
    Recursively Smith-forms the top-left n1xn1 and bottom-right n2xn2 blocks.
    """
    ring = A.ring
    n = A.nrows
    
    # Extract the independent blocks
    # Block 1: Top-Left (0 to n1)
    B1 = A.submatrix(0, n1, 0, n1)
    
    # Block 2: Bottom-Right (n1+1 to n) -> SKIPPING the spike row/col at index n1
    B2 = A.submatrix(n1 + 1, n, n1 + 1, n)
    
    # Recursive Calls
    # These return diagonal matrices S1, S2
    U1_loc, V1_loc, S1 = smith_from_upper_2_banded(B1)
    U2_loc, V2_loc, S2 = smith_from_upper_2_banded(B2)
    
    # Embed into global transforms
    # U = diag(U1_loc, 1, U2_loc)
    # V = diag(V1_loc, 1, V2_loc)
    # The '1' is at index n1 (the spike)
    
    U = RingMatrix.identity(ring, n)
    V = RingMatrix.identity(ring, n)
    
    U.write_block(0, 0, U1_loc)
    U.write_block(n1 + 1, n1 + 1, U2_loc)
    
    V.write_block(0, 0, V1_loc)
    V.write_block(n1 + 1, n1 + 1, V2_loc)
    
    # Apply to A
    # Result: S1 and S2 are diagonal, but the spike row/col (n1) 
    # gets smeared by the U/V transforms.
    A_new = U @ A @ V
    
    return U, V, A_new

def _step3_permute(A: RingMatrix, n1: int, n2: int) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """
    Step 3 of 7.3.
    Constructs the permutation P that rotates rows/cols
    (n1, n1+1, ..., n-1) (0-based) to (n-1, n1, n1+1, ..., n-2).

    Returns (P, P_inv, A3) with A3 = P @ A @ P_inv.
    """
    ring = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("Expected square matrix")

    # Build permutation matrix P as n x n with P[new_i][old_i] = 1
    data = [[0 for _ in range(n)] for _ in range(n)]

    for old_i in range(n):
        if old_i < n1:
            new_i = old_i
        else:
            # old_i in [n1 .. n-1]
            if old_i == n1:
                new_i = n - 1
            else:
                new_i = old_i - 1
        data[new_i][old_i] = 1

    P = RingMatrix(ring, data)
    P_inv = P.transpose()  # permutation matrix => inverse = transpose

    A3 = P @ A @ P_inv
    return P, P_inv, A3



def _step4_smith_on_n_minus_1(A: RingMatrix):
    ring = A.ring
    n = A.nrows
    assert n == A.ncols

    # principal (n-1)x(n-1) block
    B = A.submatrix(0, n-1, 0, n-1)
    U_loc, V_loc, S_loc = smith_from_diagonal(B)  # your diagonal.py routine

    # embed in diag(U_loc, 1), diag(V_loc, 1)
    U = RingMatrix.identity(ring, n)
    V = RingMatrix.identity(ring, n)
    U.write_block(0, 0, U_loc)
    V.write_block(0, 0, V_loc)

    A_new = U @ A @ V
    return U, V, A_new


def _step5_to_8_gcd_chain(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix, int]:
    """
    Implements Steps 5, 6, 7, and 8 of Storjohann 7.3.
    
    Context: 
      The top-left (n-1)x(n-1) block is diagonal (Smith form). 
      The last column is arbitrary.
    
    Returns: (U, V, A_new, k)
      - k is the 1-based index from the paper (so k-1 is the 0-based index)
        where the dense block starts.
    """
    ring = A.ring
    n = A.nrows
    U = RingMatrix.identity(ring, n)
    V = RingMatrix.identity(ring, n)
    T = A.copy()

    idx_k = n - 1
    for i in range(n - 1):
        if ring.is_zero(T.data[i][i]):
            idx_k = i
            break
    
    last_col = n - 1
    
    # --- Step 5: Eliminate below k in last column ---
    for t in range(idx_k + 1, n):
        pivot_val = T.data[idx_k][last_col]
        target_val = T.data[t][last_col]

        if ring.is_zero(target_val):
            continue

        # Gcdex on (pivot, target)
        g, s, x, u, v = ring.gcdex(pivot_val, target_val)
        
        # Apply [s x; u v] to rows (idx_k, t)
        T.apply_row_2x2(idx_k, t, s, x, u, v)
        U.apply_row_2x2(idx_k, t, s, x, u, v)

    # --- Step 6: Permutation ---
    # "Switch columns k+1 and n" (1-based).
    # 0-based: Switch column `idx_k` (where the zeros start) and `last_col`.
    # If idx_k == last_col, no swap needed.
    
    if idx_k != last_col:
        P_swap = RingMatrix.identity(ring, n)
        # Swap columns idx_k and last_col in the Identity gives the right-mult matrix
        # P_swap[idx_k, idx_k] = 0, P_swap[idx_k, last_col] = 1, etc.
        
        # Manual column swap on P_swap (which starts as Identity)
        # Col idx_k becomes 0s with 1 at row last_col
        # Col last_col becomes 0s with 1 at row idx_k
        P_swap.data[idx_k][idx_k] = 0
        P_swap.data[last_col][last_col] = 0
        P_swap.data[idx_k][last_col] = 1
        P_swap.data[last_col][idx_k] = 1

        # Apply right transform
        T = T @ P_swap
        V = V @ P_swap

    # --- Step 7: Stabilize Loop (Ripple Up) ---
    col_target = idx_k 

    for i in range(idx_k - 1, -1, -1):
        # c := Stab(A[i, k], A[i+1, k], A[i, i])
        a_ik = T.data[i][col_target]
        a_i1k = T.data[i+1][col_target]
        a_ii = T.data[i][i] # The diagonal element s_i

        c = ring.stab(a_ik, a_i1k, a_ii)

        # q := -Div(c * A[i+1, i+1], A[i, i])
        # Note: A[i+1, i+1] is s_{i+1}. 
        # In a PID/PIR chain, s_i divides s_{i+1}, so division is valid.
        s_next = T.data[i+1][i+1]
        numerator = ring.mul(c, s_next)
        q_raw = ring.quo(numerator, a_ii)
        q = (-q_raw) % ring.N

        # Operation 1: Add c * Row[i+1] to Row[i]
        # Matrix: [1 c; 0 1] on rows (i, i+1)
        # s=1, t=c, u=0, v=1
        T.apply_row_2x2(i, i+1, 1, c, 0, 1)
        U.apply_row_2x2(i, i+1, 1, c, 0, 1)

        # Operation 2: Add q * Col[i] to Col[i+1]
        # This is a Right transform. 
        # Matrix acting on columns (i, i+1): [1 0; q 1]
        # Because T @ V_op, and V_op has q at (i, i+1).
        # We can implement this manually or via a helper. 
        # Let's do a manual column update for T and V.
        
        # Update T: Col[i+1] = Col[i+1] + q*Col[i]
        for r in range(n):
            val_i = T.data[r][i]
            val_ip1 = T.data[r][i+1]
            T.data[r][i+1] = ring.add(val_ip1, ring.mul(q, val_i))
        
        # Update V: same op
        for r in range(n):
            val_i = V.data[r][i]
            val_ip1 = V.data[r][i+1]
            V.data[r][i+1] = ring.add(val_ip1, ring.mul(q, val_i))

    # --- Step 8: GCD Reduction Loop (Ripple Down) ---
    
    for i in range(idx_k):
        # (g, s, t, u, v) := Gcdex(A[i, i], A[i, k])
        # A[i,k] is the element we dragged up in Step 7
        pivot = T.data[i][i]
        target = T.data[i][col_target] # col_target is idx_k
        
        if ring.is_zero(target):
            continue

        g, s, t, u, v = ring.gcdex(pivot, target)

        # Apply RIGHT transform on columns (i, k)
        # [ A[*, i]  A[*, k] ] * [s u; t v]
        
        # Manual column update
        # NewCol_i = s*Col_i + t*Col_k
        # NewCol_k = u*Col_i + v*Col_k
        
        for r in range(n):
            ci = T.data[r][i]
            ck = T.data[r][col_target]
            
            # T update
            T.data[r][i] = ring.add(ring.mul(s, ci), ring.mul(t, ck))
            T.data[r][col_target] = ring.add(ring.mul(u, ci), ring.mul(v, ck))
            
            # V update (same transform)
            vi = V.data[r][i]
            vk = V.data[r][col_target]
            V.data[r][i] = ring.add(ring.mul(s, vi), ring.mul(t, vk))
            V.data[r][col_target] = ring.add(ring.mul(u, vi), ring.mul(v, vk))

    # Return idx_k + 1 (1-based) to match paper's "k" for Step 9
    return U, V, T, (idx_k + 1)

def _step9_index_reduction(A: RingMatrix, k: int) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    ring = A.ring
    n = A.nrows
    idx_k = k - 1

    # Build P that reverses the first k indices
    P_data = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        if i <= idx_k:
            P_data[i][idx_k - i] = 1
        else:
            P_data[i][i] = 1
    P = RingMatrix(ring, P_data)

    # Conjugate by P to move the dense block to the top-left
    A_perm = P @ A @ P

    # Index-1 reduction on first k columns (row ops only)
    U_red, A_red = index1_reduce_on_columns(A_perm, k)

    # Net effect of:
    #   A_perm = P A P
    #   A_red  = U_red A_perm
    #   S      = P A_red P
    # is:
    #   S = (P U_red P) A
    U_final = P @ U_red @ P
    V_final = RingMatrix.identity(ring, n)

    A_final = U_final @ A  # no right transform

    return U_final, V_final, A_final
