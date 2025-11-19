import pytest
import copy
from modularsnf.ring import RingZModN
from modularsnf.bandreduction import BandReduction
from modularsnf.matrix import MatrixOps

# --- Helper Functions for Verification ---

def get_det_mod_n(matrix, ring):
    """
    Recursive determinant calculation for unit testing small matrices over Z/N.
    """
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    
    det = 0
    for c in range(n):
        # Laplace expansion along the first row
        element = matrix[0][c]
        if element == 0:
            continue
            
        # Minor matrix
        sub_matrix = [row[:c] + row[c+1:] for row in matrix[1:]]
        sub_det = get_det_mod_n(sub_matrix, ring)
        
        term = ring.mul(element, sub_det)
        
        if c % 2 == 1:
            det = ring.sub(det, term)
        else:
            det = ring.add(det, term)
            
    return det

def is_unit(val, ring):
    """
    Checks if val is a unit in Z/N (i.e., gcd(val, N) == 1).
    """
    # Simple Euclidean GCD for integers
    a, b = val, ring.N
    while b:
        a, b = b, a % b
    return a == 1

def is_echelon(A):
    """
    Verifies if matrix A is in row echelon form.
    (Pivot index strictly increases).
    """
    rows = len(A)
    cols = len(A[0])
    last_pivot = -1
    
    for r in range(rows):
        current_pivot = -1
        # Find first nonzero
        for c in range(cols):
            if A[r][c] != 0:
                current_pivot = c
                break
        
        if current_pivot == -1:
            # Zero row, ensure all subsequent rows are zero
            # (Not strictly required by all definitions, but standard for reduced forms)
            continue
            
        if current_pivot <= last_pivot:
            return False
        last_pivot = current_pivot
    return True

# --- Fixtures ---

@pytest.fixture
def setup_components():
    # Using Z/12 to ensure we test zero divisors (composite modulus)
    ring = RingZModN(12)
    reducer = BandReduction(ring)
    ops = MatrixOps(ring)
    return ring, reducer, ops

# --- Tests ---

def test_lemma_3_1_compliance(setup_components):
    """
    Strictly validates the requirements of Lemma 3.1.
    Input: Matrix A
    Output: Unimodular U, Echelon T
    Relation: UA = T
    """
    ring, reducer, ops = setup_components
    
    # Arbitrary 4x4 matrix in Z/12
    A = [
        [2, 4, 6, 8],
        [3, 6, 9, 0],
        [1, 2, 3, 4],
        [0, 4, 8, 2]
    ]
    
    U, T = ops.lemma_3_1_transform(A)
    
    # 1. Verify U is Unimodular (Determinant is a unit in Z/12)
    det_U = get_det_mod_n(U, ring)
    assert is_unit(det_U, ring), f"Transform U is not unimodular. Det: {det_U} in Z/{ring.N}"
    
    # 2. Verify Transformation: U * A = T
    UA = ops.mat_mul(U, A)
    assert UA == T, "Matrix multiplication UA did not yield T"
    
    # 3. Verify T is Echelon Form
    assert is_echelon(T), "Result T is not in echelon form"


def test_triang_lemma_7_3(setup_components):
    """
    Validates Lemma 7.3.
    Transforms an upper b-banded matrix B to B'.
    Focus is on the sub-block B2 (rows 0..s1-1, cols s1..n1-1).
    """
    ring, reducer, ops = setup_components
    
    # Setup specific constraints from Lemma 7.3
    # Let b=4. s1 = floor(4/2) = 2. s2 = 3 (b-1).
    # B dimensions: n1 x n1. Let n1 = 4.
    s1, s2 = 2, 2 # Using s2=2 to fit in 4x4
    
    # B2 is top-right 2x2 block (rows 0-1, cols 2-3)
    B = [
        [1, 0, 2, 3], # B1 | B2
        [0, 1, 4, 5], #    |
        [0, 0, 1, 0], # 0  | B3
        [0, 0, 0, 1]
    ]
    
    B_prime, V = reducer.triang(B, s1, s2)
    
    # 1. Verify V is Unimodular
    det_V = get_det_mod_n(V, ring)
    assert is_unit(det_V, ring), f"Transform V is not unimodular. Det: {det_V}"

    # 2. Verify B' = B * V
    BV = ops.mat_mul(B, V)
    assert BV == B_prime, "Transform mismatch B' != B * V"
    
    # 3. Verify Structure of B'
    # According to diagram in, the "corner" of B2 should be cleared.
    # Specifically, we performed column operations to triangularize B2^T.
    # This means B2 (in B') should be Lower Triangular.
    
    # Extract B2 from B_prime
    B2_prime = [row[s1:s1+s2] for row in B_prime[:s1]]
    
    # Check if B2_prime is Lower Triangular
    # (Everything above diagonal must be 0)
    for r in range(len(B2_prime)):
        for c in range(len(B2_prime[0])):
            if c > r: 
                assert B2_prime[r][c] == 0, f"B2' not lower triangular at {r},{c}"

def test_shift_lemma_7_4(setup_components):
    """
    Validates Lemma 7.4.
    Transforms C to C' via Left and Right transforms.
    """
    ring, reducer, ops = setup_components

    # Dimensions
    s2 = 2
    # C is n2 x n2. n2 = 4.
    C = [
        [0, 1, 3, 4], # C1 | C2
        [1, 0, 6, 7], #    |
        [0, 0, 1, 2], # 0  | C3
        [0, 0, 0, 5]
    ]

    C_prime, U, V = reducer.shift(C, s2)

    # 1. Verify Unimodularity
    assert is_unit(get_det_mod_n(U, ring), ring), "U is not unimodular"
    assert is_unit(get_det_mod_n(V, ring), ring), "V is not unimodular"

    # 2. Verify Transformation C' = U * C * V
    UC = ops.mat_mul(U, C)
    UCV = ops.mat_mul(UC, V)
    assert UCV == C_prime, "Transform mismatch C' != U * C * V"

    # 3. Verify Structure
    # The Shift algorithm ensures C1 becomes Upper Triangular (via U)
    # and (U*C2)*V becomes Lower Triangular.
    
    C1_prime = [row[:s2] for row in C_prime[:s2]]

    # Check 1: Did C1 change? (It should, because input was [[0,1],[1,0]])
    C1_original = [row[:s2] for row in C[:s2]]
    assert C1_prime != C1_original, "C1 block should have changed"

    # Check 2: Is C1_prime Upper Triangular?
    # The algorithm applies U such that U*C1 is Upper Triangular.
    # For a 2x2, this means the bottom-left entry must be 0.
    assert C1_prime[1][0] == 0, "C1 block should be Upper Triangular after shift"

def test_full_reduction_step(setup_components):
    """
    Tests one full pass of BandReduction (Proposition 7.1).
    If input is b-banded, output should be floor(b/2)+1 banded.
    """
    ring, reducer, ops = setup_components
    n = 8
    b = 4
    target_b = (b // 2) + 1 # Should be 3
    
    # Create an Upper 4-Banded Matrix
    # We fill it carefully to ensure it's NOT 3-banded initially
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            # Fill diagonal and up to b-1 super-diagonals
            if 0 <= j - i < b:
                val = (i * n + j) % 11 + 1 # Non-zero
                A[i][j] = val
            else:
                A[i][j] = 0
                
    # Verify initial state
    assert reducer.is_upper_b_banded(A, b)
    assert not reducer.is_upper_b_banded(A, target_b), "Test setup failed: Matrix is already reduced."

    # Run Reduction
    A_prime, U, V = reducer.reduce(A, b)

    # 1. Verify Bandwidth Reduction
    assert reducer.is_upper_b_banded(A_prime, target_b), \
        f"Matrix was not reduced to {target_b}-banded form."

    # 2. Verify Equivalence
    # A' = U * A * V
    U_A = ops.mat_mul(U, A)
    U_A_V = ops.mat_mul(U_A, V)
    assert U_A_V == A_prime, "Equivalence lost: A' != U * A * V"

    # 3. Verify Unimodularity
    assert is_unit(get_det_mod_n(U, ring), ring)
    assert is_unit(get_det_mod_n(V, ring), ring)

def test_reduction_boundary_padding(setup_components):
    """
    Tests that reduction works even when matrix dimensions 
    don't align perfectly with block sizes (requires padding).
    """
    ring, reducer, ops = setup_components
    n = 5
    b = 3 # s1=1, n1=3, s2=2, n2=4. 
    # n2 (4) is almost as big as n (5). Padding will be required.
    
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if 0 <= j - i < b: A[i][j] = 1

    A_prime, _, _ = reducer.reduce(A, b)
    
    target_b = (b // 2) + 1 # 2
    assert reducer.is_upper_b_banded(A_prime, target_b)