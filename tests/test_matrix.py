import pytest
import random
from modularsnf.ring import RingZModN
from modularsnf.matrix import MatrixOps
from .helpers import get_det, is_unit, verify_echelon_structure

@pytest.fixture
def components():
    ring = RingZModN(12)
    ops = MatrixOps(ring)
    return ring, ops

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

def test_pad_matrix(components):
    """
    Verifies that pad_matrix adds zeros to the right and bottom.
    """
    ring, ops = components
    A = [
        [1, 2],
        [3, 4]
    ]
    pad_size = 2
    B = ops.pad_matrix(A, pad_size)
    
    # Check dimensions
    assert len(B) == 4
    assert len(B[0]) == 4
    
    # Check integrity of original data (top-left)
    assert B[0][0] == 1 and B[1][1] == 4
    
    # Check padding (bottom-right should be 0)
    assert B[2][2] == 0 and B[3][3] == 0
    assert B[0][3] == 0 # Right padding
    assert B[3][0] == 0 # Bottom padding

def test_embed_block(components):
    """
    Verifies that a small block is correctly written into a large matrix.
    """
    ring, ops = components
    # 4x4 Zero matrix
    Big = [[0]*4 for _ in range(4)]
    
    # 2x2 Block of ones
    Small = [
        [1, 1],
        [1, 1]
    ]
    
    # Embed at (1, 1) -> Center 
    ops.embed_block(Big, Small, 1, 1)
    
    expected = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ]
    assert Big == expected

def test_embed_identity(components):
    """
    Verifies the creation of a Global Transform matrix.
    It should be an Identity matrix EVERYWHERE except where the block is embedded.
    """
    ring, ops = components
    
    # We want a 4x4 global transform.
    # We have a 2x2 local transform acting on the middle rows/cols.
    LocalT = [
        [2, 3],
        [4, 5]
    ]
    
    # Embed at offset (1,1)
    GlobalT = ops.embed_identity(4, LocalT, 1, 1)
    
    # Check that indices OUTSIDE the block are Identity
    assert GlobalT[0][0] == 1
    assert GlobalT[3][3] == 1
    assert GlobalT[0][1] == 0
    assert GlobalT[3][0] == 0
    
    # Check that indices INSIDE the block match LocalT
    # GlobalT[1][1] should be LocalT[0][0]
    assert GlobalT[1][1] == 2
    assert GlobalT[1][2] == 3
    assert GlobalT[2][1] == 4
    assert GlobalT[2][2] == 5


def test_create_diagonal(components):
    """Ensures create_diagonal places entries on the diagonal and zeros elsewhere."""
    ring, ops = components
    entries = [3, 0, 5, 7]

    D = ops.create_diagonal(entries)

    n = len(entries)
    assert len(D) == n
    assert all(len(row) == n for row in D)

    for i in range(n):
        for j in range(n):
            if i == j:
                assert D[i][j] == entries[i]
            else:
                assert D[i][j] == 0