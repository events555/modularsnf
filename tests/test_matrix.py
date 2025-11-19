import pytest
import random
from modularsnf.ring import RingZModN
from modularsnf.matrix import MatrixOps

@pytest.fixture
def components():
    ring = RingZModN(12)
    ops = MatrixOps(ring)
    return ring, ops

# --- Helpers ---
def get_det(matrix, ring):
    """Recursive determinant for validation."""
    n = len(matrix)
    if n == 1: return matrix[0][0]
    det = 0
    for c in range(n):
        if matrix[0][c] == 0: continue
        sub = [row[:c] + row[c+1:] for row in matrix[1:]]
        term = ring.mul(matrix[0][c], get_det(sub, ring))
        det = ring.sub(det, term) if c % 2 else ring.add(det, term)
    return det

def is_unit(val, ring):
    a, b = val, ring.N
    while b: a, b = b, a % b
    return a == 1

def verify_echelon_structure(T):
    """
    Verifies property (r1) from Chapter 3.
    '0 = j0 < j1 < j2 ... < jr' (strictly increasing column indices)
    """
    rows, cols = len(T), len(T[0])
    last_pivot_col = -1
    
    for r in range(rows):
        # Find first nonzero entry in this row
        pivot_col = -1
        for c in range(cols):
            if T[r][c] != 0:
                pivot_col = c
                break
        
        if pivot_col == -1:
            # All zero row.
            # Ensure all subsequent rows are also zero (standard echelon definition)
            # Though  implies strictly increasing indices for nonzero rows only.
            # We just need to ensure we don't find a nonzero row *after* this.
            for r2 in range(r + 1, rows):
                for c2 in range(cols):
                    assert T[r2][c2] == 0, "Nonzero row found after zero row in Echelon form."
            break
        else:
            # Nonzero row. Must be strictly to the right of the previous pivot.
            assert pivot_col > last_pivot_col, \
                f"Pivots not strictly increasing. Row {r} pivot at {pivot_col}, previous at {last_pivot_col}"
            last_pivot_col = pivot_col
    return True

# --- Tests ---

def test_lemma_3_1_equivalency(components):
    """
    Validates that Lemma 3.1 produces a valid factorization UA = T.
    """
    ring, ops = components
    
    # Test on random matrices
    for _ in range(10):
        rows, cols = 5, 5
        A = [[random.randint(0, 11) for _ in range(cols)] for _ in range(rows)]
        
        U, T = ops.lemma_3_1_transform(A)
        
        # 1. Check Matrix Multiplication UA == T
        UA = ops.mat_mul(U, A)
        assert UA == T, "Product UA does not equal T"

def test_lemma_3_1_unimodularity(components):
    """
    Validates that the transform matrix U is unimodular.
    """
    ring, ops = components
    
    rows = 4 # Keep small for determinant speed
    A = [[random.randint(0, 11) for _ in range(rows)] for _ in range(rows)]
    
    U, _ = ops.lemma_3_1_transform(A)
    
    det = get_det(U, ring)
    assert is_unit(det, ring), f"Transform U is not unimodular. Det: {det}"

def test_lemma_3_1_echelon_form(components):
    """
    Validates that T satisfies condition (r1): increasing pivot indices.
    """
    ring, ops = components
    
    # Use a matrix likely to have dependent rows to force zero rows at bottom
    A = [
        [2, 4, 6],
        [1, 2, 3], # Dependent on row 0
        [0, 1, 5],
        [0, 0, 0]
    ]
    
    _, T = ops.lemma_3_1_transform(A)
    assert verify_echelon_structure(T), "Result T violates Echelon structure properties"

def test_lemma_3_1_rectangular(components):
    """
    Validates Lemma 3.1 on rectangular matrices (n != m).
    """
    ring, ops = components
    
    # Tall matrix
    A = [[random.randint(0, 11) for _ in range(3)] for _ in range(5)]
    U, T = ops.lemma_3_1_transform(A)
    
    assert ops.mat_mul(U, A) == T
    assert verify_echelon_structure(T)
    
    # Wide matrix
    A = [[random.randint(0, 11) for _ in range(5)] for _ in range(3)]
    U, T = ops.lemma_3_1_transform(A)
    
    assert ops.mat_mul(U, A) == T
    assert verify_echelon_structure(T)