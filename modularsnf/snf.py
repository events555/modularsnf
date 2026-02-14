"""Utilities for computing Smith normal forms over modular rings."""

from typing import NamedTuple, Tuple

from modularsnf.band import band_reduction, compute_upper_bandwidth
from .matrix import RingMatrix
from .ring import RingZModN
from .diagonal import smith_from_diagonal
from .echelon import index1_reduce_on_columns, lemma_3_1


class SNFResult(NamedTuple):
    """Result of a Smith Normal Form decomposition over Z/NZ.

    Attributes:
        S: The Smith Normal Form (diagonal matrix) as nested lists.
        U: Left unimodular transform as nested lists.
        V: Right unimodular transform as nested lists.

    The invariant ``S = U @ A @ V`` holds over Z/NZ, where ``@``
    denotes matrix multiplication modulo N.
    """

    S: list[list[int]]
    U: list[list[int]]
    V: list[list[int]]


def smith_normal_form_mod(
    matrix: list[list[int]],
    modulus: int,
) -> SNFResult:
    """Compute the Smith Normal Form of an integer matrix over Z/NZ.

    This is the primary user-facing entry point.  Given a matrix of
    integers and a modulus *N*, the function returns the decomposition
    ``S = U @ A @ V`` (mod *N*) where *S* is diagonal with the
    invariant-factor divisibility chain ``s_i | s_{i+1}`` and *U*, *V*
    are unimodular over Z/NZ.

    The return order ``(S, U, V)`` — diagonal form first — follows the
    dominant convention used by SymPy (``smith_normal_decomp``) and
    SageMath (``smith_form``).

    Args:
        matrix: A 2-D list of integers representing the input matrix.
            May be rectangular; empty inputs (``[]``) are accepted.
        modulus: A positive integer *N* >= 2 defining the ring Z/NZ.

    Returns:
        An :class:`SNFResult` ``(S, U, V)`` of plain Python integer
        lists satisfying ``S = U @ A @ V`` (mod *N*).

    Raises:
        ValueError: If *modulus* < 2 or if *matrix* rows have unequal
            lengths (ragged input).
        TypeError: If *matrix* is not a list of lists.

    Examples:
        >>> from modularsnf import smith_normal_form_mod
        >>> S, U, V = smith_normal_form_mod([[2, 4], [6, 8]], modulus=12)
    """
    if not isinstance(modulus, int) or modulus < 2:
        raise ValueError(
            f"Modulus must be an integer >= 2, got {modulus!r}"
        )

    if not isinstance(matrix, (list, tuple)):
        raise TypeError(
            "matrix must be a list of lists of integers"
        )

    # Empty matrix — return empty lists immediately.
    if len(matrix) == 0:
        return SNFResult(S=[], U=[], V=[])

    # Validate that all rows have the same length.
    ncols = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != ncols:
            raise ValueError(
                f"Ragged matrix: row 0 has {ncols} columns "
                f"but row {i} has {len(row)} columns"
            )

    ring = RingZModN(modulus)
    A = RingMatrix(ring, matrix)

    U_rm, V_rm, S_rm = smith_normal_form(A)

    return SNFResult(
        S=S_rm.data.tolist(),
        U=U_rm.data.tolist(),
        V=V_rm.data.tolist(),
    )


def smith_normal_form(
    A: RingMatrix,
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Computes the Smith normal form over a principal ideal ring.

    For rectangular matrices, the input is zero-padded to square before the
    Smith form is computed, then the resulting transforms are cropped back to
    the original shape.

    Args:
        A: Matrix over a principal ideal ring.

    Returns:
        A tuple ``(U, V, S)`` where ``S = U @ A @ V`` is in Smith normal form.
    """

    ring = A.ring
    n = A.nrows
    m = A.ncols

    # Handle empty matrices by returning identities of matching dimensions.
    if n == 0 or m == 0:
        U = RingMatrix.identity(ring, n)
        V = RingMatrix.identity(ring, m)
        return U, V, A.copy()

    # If the matrix is already square, proceed without padding.
    if n == m:
        return _smith_square(A)

    # Pad rectangular matrices to a square ``s x s`` matrix before reduction.
    s = max(n, m)
    data = [[0 for _ in range(s)] for _ in range(s)]
    for i in range(n):
        for j in range(m):
            data[i][j] = A.data[i][j]
    A_pad = RingMatrix(ring, data)

    U_pad, V_pad, S_pad = _smith_square(A_pad)

    # Restrict padded transforms back to the original shape.
    U = U_pad.submatrix(0, n, 0, n)
    V = V_pad.submatrix(0, m, 0, m)
    S = S_pad.submatrix(0, n, 0, m)

    return U, V, S


def _smith_square(A: RingMatrix) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Computes Smith normal form for a square matrix over a PIR.

    Args:
        A: Square matrix.

    Returns:
        ``(U, V, S)`` with ``S = U @ A @ V`` diagonal and containing the
        invariant factors of ``A``.

    Raises:
        ValueError: If ``A`` is not square.
    """

    ring = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("Expected a square matrix in _smith_square.")

    if n == 0:
        I0 = RingMatrix.identity(ring, 0)
        return I0, I0, A.copy()

    # Step 1 (Storjohann 7.3): Reduce to row echelon form ``T = U_ech * A``.
    U_ech, T, _rank = lemma_3_1(A)

    # Step 2 (Storjohann 7.3): Reduce bandwidth until the matrix is 2-banded.
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
    """Runs the Smith form pipeline for a 2-banded square matrix.

    Args:
        A: Square matrix with bandwidth at most two.

    Returns:
        A tuple ``(U, V, S)`` with ``S = U @ A @ V`` in Smith normal form.

    Raises:
        ValueError: If ``A`` is not square.
    """

    ring: RingZModN = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("Expected square matrix")
    if n <= 1:
        U = RingMatrix.identity(ring, n)
        V = RingMatrix.identity(ring, n)
        return U, V, A.copy()

    # Step 1 (Storjohann 7.3): Precondition into spike plus principal blocks.
    U1, V1, A1, n1, n2 = _step1_split_with_spike(A)

    # Step 2 (Storjohann 7.3): Run recursive Smith forms on both blocks.
    U2, V2, A2 = _step2_recursive_blocks(A1, n1, n2)

    # Step 3 (Storjohann 7.3): Permute the spike to the trailing position.
    U3, V3, A3 = _step3_permute(A2, n1, n2)

    # Step 4 (Storjohann 7.3): Diagonalize the principal ``(n-1) x (n-1)``.
    U4, V4, A4 = _step4_smith_on_n_minus_1(A3)

    # Steps 5–8 (Storjohann 7.3): Build the gcd chain ``s1..sk``.
    U5, V5, A5, k = _step5_to_8_gcd_chain(A4)

    # Step 9 (Storjohann 7.3): Reverse first ``k`` rows and run index reduction.
    U6, V6, A6 = _step9_index_reduction(A5, k)

    # Compose transforms: U_total, V_total such that U_total A V_total = A6
    U_total = U6 @ U5 @ U4 @ U3 @ U2 @ U1
    V_total = V1 @ V2 @ V3 @ V4 @ V5 @ V6

    return U_total, V_total, A6


def _step1_split_with_spike(
    A: RingMatrix,
) -> Tuple[RingMatrix, RingMatrix, RingMatrix, int, int]:
    """Splits a 2-banded matrix into principal, spike, and trailing blocks.

    Args:
        A: Square 2-banded matrix.

    Returns:
        ``(U, V, A1, n1, n2)`` where ``A1 = U @ A`` has block sizes ``n1`` and
        ``n2`` flanking the spike row/column and ``V`` is identity.
    """

    ring = A.ring
    n = A.nrows
    n1 = (n - 1) // 2
    n2 = n - 1 - n1

    U = RingMatrix.identity(ring, n)
    T = A.copy()

    # Sweep upward to isolate the spike column above ``n1``.
    for k in range(n - 1, n1, -1):
        r0, r1 = k - 1, k
        col = k
        a = T.data[r0][col]
        b = T.data[r1][col]
        g, s, t, u, v = ring.gcdex(a, b)
        T.apply_row_2x2(r0, r1, u, v, s, t)
        U.apply_row_2x2(r0, r1, u, v, s, t)

    # Sweep downward to clear the spike column below ``n1``.
    for k in range(n1 + 1, n - 1):
        r0, r1 = k, k + 1
        col = k
        a = T.data[r0][col]
        b = T.data[r1][col]
        g, s, t, u, v = ring.gcdex(a, b)
        T.apply_row_2x2(r0, r1, s, t, u, v)
        U.apply_row_2x2(r0, r1, s, t, u, v)

    V = RingMatrix.identity(ring, n)

    return U, V, T, n1, n2


def _step2_recursive_blocks(A: RingMatrix, n1: int, n2: int):
    """Recursively Smith-forms the principal and trailing diagonal blocks.

    Args:
        A: Matrix produced by the spike split.
        n1: Size of the top-left block.
        n2: Size of the bottom-right block.

    Returns:
        ``(U, V, A_new)`` where the principal and trailing blocks are diagonal
        but the spike row and column may be dense.
    """

    ring = A.ring
    n = A.nrows

    # Extract the independent blocks.
    # Block 1: Top-left (0 to n1).
    B1 = A.submatrix(0, n1, 0, n1)

    # Block 2: Bottom-right (n1+1 to n) skipping the spike at index ``n1``.
    B2 = A.submatrix(n1 + 1, n, n1 + 1, n)

    # Recursively diagonalize both independent blocks.
    U1_loc, V1_loc, S1 = smith_from_upper_2_banded(B1)
    U2_loc, V2_loc, S2 = smith_from_upper_2_banded(B2)

    # Embed the local transforms with an identity on the spike position.
    U = RingMatrix.identity(ring, n)
    V = RingMatrix.identity(ring, n)

    U.write_block(0, 0, U1_loc)
    U.write_block(n1 + 1, n1 + 1, U2_loc)

    V.write_block(0, 0, V1_loc)
    V.write_block(n1 + 1, n1 + 1, V2_loc)

    # Apply to A; S1 and S2 become diagonal while the spike spreads through the
    # transforms.
    A_new = U @ A @ V

    return U, V, A_new


def _step3_permute(
    A: RingMatrix, n1: int, n2: int
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Permutes the spike to the last row/column.

    Args:
        A: Matrix from the recursive block step.
        n1: Size of the principal block.
        n2: Size of the trailing block.

    Returns:
        ``(P, P_inv, A3)`` where ``A3 = P @ A @ P_inv`` and the spike sits in
        the last row and column.

    Raises:
        ValueError: If ``A`` is not square.
    """

    ring = A.ring
    n = A.nrows
    if n != A.ncols:
        raise ValueError("Expected square matrix")

    # Build permutation matrix P as n x n with P[new_i][old_i] = 1.
    data = [[0 for _ in range(n)] for _ in range(n)]

    for old_i in range(n):
        if old_i < n1:
            new_i = old_i
        else:
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
    """Diagonalizes the top-left ``(n-1) x (n-1)`` block.

    Args:
        A: Matrix with the spike in the last row and column.

    Returns:
        ``(U, V, A_new)`` where the principal block is diagonalized.
    """

    ring = A.ring
    n = A.nrows
    assert n == A.ncols

    # Principal ``(n-1) x (n-1)`` block.
    B = A.submatrix(0, n - 1, 0, n - 1)
    U_loc, V_loc, S_loc = smith_from_diagonal(B)

    # Embed the local transforms with a fixed spike position.
    U = RingMatrix.identity(ring, n)
    V = RingMatrix.identity(ring, n)
    U.write_block(0, 0, U_loc)
    V.write_block(0, 0, V_loc)

    A_new = U @ A @ V
    return U, V, A_new


def _step5_to_8_gcd_chain(
    A: RingMatrix,
) -> Tuple[RingMatrix, RingMatrix, RingMatrix, int]:
    """Builds the gcd chain that drives the final Smith reduction.

    Args:
        A: Matrix with a diagonal principal block and dense last column.

    Returns:
        ``(U, V, A_new, k)`` where ``k`` is the 1-based start index of the
        dense block as in Storjohann 7.3.
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

    # Step 5 (Storjohann 7.3): Eliminate entries below ``idx_k`` in last column.
    for t in range(idx_k + 1, n):
        pivot_val = T.data[idx_k][last_col]
        target_val = T.data[t][last_col]

        if ring.is_zero(target_val):
            continue

        # Compute extended gcd coefficients on ``(pivot, target)``.
        g, s, x, u, v = ring.gcdex(pivot_val, target_val)

        # Apply the 2x2 transform to rows ``idx_k`` and ``t``.
        T.apply_row_2x2(idx_k, t, s, x, u, v)
        U.apply_row_2x2(idx_k, t, s, x, u, v)

    # Step 6 (Storjohann 7.3): Swap the dense column into position ``idx_k``.

    if idx_k != last_col:
        P_swap = RingMatrix.identity(ring, n)
        # Swap columns ``idx_k`` and ``last_col`` in the identity to build the
        # right multiplication matrix.

        P_swap.data[idx_k][idx_k] = 0
        P_swap.data[last_col][last_col] = 0
        P_swap.data[idx_k][last_col] = 1
        P_swap.data[last_col][idx_k] = 1

        T = T @ P_swap
        V = V @ P_swap

    # Step 7 (Storjohann 7.3): Perform the stabilize loop (ripple up).
    col_target = idx_k

    for i in range(idx_k - 1, -1, -1):
        a_ik = T.data[i][col_target]
        a_i1k = T.data[i + 1][col_target]
        a_ii = T.data[i][i]  # The diagonal element s_i

        c = ring.stab(a_ik, a_i1k, a_ii)

        s_next = T.data[i + 1][i + 1]
        numerator = ring.mul(c, s_next)
        q_raw = ring.quo(numerator, a_ii)
        q = (-q_raw) % ring.N

        # Operation 1: Add ``c * Row[i+1]`` to ``Row[i]``.
        T.apply_row_2x2(i, i + 1, 1, c, 0, 1)
        U.apply_row_2x2(i, i + 1, 1, c, 0, 1)

        # Operation 2: Add ``q * Col[i]`` to ``Col[i+1]`` via right transform.
        for r in range(n):
            val_i = T.data[r][i]
            val_ip1 = T.data[r][i + 1]
            T.data[r][i + 1] = ring.add(val_ip1, ring.mul(q, val_i))

        # Update V with the same column operation.
        for r in range(n):
            val_i = V.data[r][i]
            val_ip1 = V.data[r][i + 1]
            V.data[r][i + 1] = ring.add(val_ip1, ring.mul(q, val_i))

    # Step 8 (Storjohann 7.3): Run the gcd reduction loop (ripple down).

    for i in range(idx_k):
        pivot = T.data[i][i]
        target = T.data[i][col_target]  # col_target is idx_k

        if ring.is_zero(target):
            continue

        g, s, t, u, v = ring.gcdex(pivot, target)

        for r in range(n):
            ci = T.data[r][i]
            ck = T.data[r][col_target]

            T.data[r][i] = ring.add(ring.mul(s, ci), ring.mul(t, ck))
            T.data[r][col_target] = ring.add(ring.mul(u, ci), ring.mul(v, ck))

            vi = V.data[r][i]
            vk = V.data[r][col_target]
            V.data[r][i] = ring.add(ring.mul(s, vi), ring.mul(t, vk))
            V.data[r][col_target] = ring.add(ring.mul(u, vi), ring.mul(v, vk))

    # Return idx_k + 1 (1-based) to match paper's "k" for Step 9
    return U, V, T, (idx_k + 1)


def _step9_index_reduction(
    A: RingMatrix, k: int
) -> Tuple[RingMatrix, RingMatrix, RingMatrix]:
    """Performs the final index-1 reduction on the dense block.

    Args:
        A: Matrix returned from the gcd chain step.
        k: 1-based index where the dense block starts.

    Returns:
        ``(U_final, V_final, A_final)`` where ``A_final`` is the Smith form of
        the input matrix.
    """

    ring = A.ring
    n = A.nrows
    idx_k = k - 1

    # Build ``P`` that reverses the first ``k`` indices.
    P_data = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        if i <= idx_k:
            P_data[i][idx_k - i] = 1
        else:
            P_data[i][i] = 1
    P = RingMatrix(ring, P_data)

    # Conjugate by ``P`` to move the dense block to the top-left.
    A_perm = P @ A @ P

    # Index-1 reduction on first ``k`` columns (row operations only).
    U_red, A_red = index1_reduce_on_columns(A_perm, k)

    # Net effect is ``S = (P U_red P) A``.
    U_final = P @ U_red @ P
    V_final = RingMatrix.identity(ring, n)

    A_final = U_final @ A  # no right transform

    return U_final, V_final, A_final
