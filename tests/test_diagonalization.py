import pytest
import random
from modularsnf.ring import RingZModN
from modularsnf.matrix import MatrixOps
from modularsnf.diagonalization import DiagonalReduction
from .helpers import (
    get_det_mod_n,
    is_unit,
    is_divisible,
    is_associate,
    verify_smith_form_properties,
)

# --- Fixtures ---

@pytest.fixture
def components():
    # Z/12 is excellent for testing because it has zero divisors (2,3,4,6,8,10)
    ring = RingZModN(12)
    reducer = DiagonalReduction(ring)
    ops = MatrixOps(ring)
    return ring, reducer, ops

# --- Tests ---

def test_lemma_7_10_scalar_merge(components):
    """
    Tests the atomic merge of two scalars a, b.
    Input: a, b
    Output: U, V such that U * diag(a,b) * V = diag(g, x)
            where g = gcd(a,b) and g | x.
    """
    ring, reducer, ops = components
    
    # Test with zero divisors and coprimes
    test_pairs = [
        (2, 4),  # Divisible
        (3, 4),  # Coprime
        (4, 6),  # GCD is 2 (Zero divisors)
        (0, 5),  # Zero
        (0, 0)
    ]
    
    for a, b in test_pairs:
        U, V = reducer.scalar_merge(a, b)
        
        # 1. Unimodularity
        assert is_unit(get_det_mod_n(U, ring), ring)
        assert is_unit(get_det_mod_n(V, ring), ring)
        
        # 2. Equivalence
        D = ops.create_diagonal([a, b])
        UD = ops.mat_mul(U, D)
        UDV = ops.mat_mul(UD, V)
        
        # 3. Structure (Diagonal)
        assert UDV[0][1] == 0
        assert UDV[1][0] == 0
        
        # 4. Divisibility (g | x)
        g = UDV[0][0]
        x = UDV[1][1]
        assert is_divisible(g, x, ring), \
            f"Scalar merge failed divisibility for inputs {a},{b}. Result: {g}, {x}"


def test_theorem_7_11_merge_blocks(components):
    """
    Tests merging two diagonal blocks that are LOCALLY in Smith Form.
    D1 = diag(2, 4)  (2|4, good)
    D2 = diag(3, 9)  (3|9, good)
    
    Merge(D1, D2) -> S where S is 4x4 Smith Form.
    """
    ring, reducer, ops = components
    
    # Create two blocks already in SNF
    # D1 = [2, 4]
    # D2 = [3, 9]
    # Note: 2 does not divide 3. So sorting is required by the algorithm.
    D1 = ops.create_diagonal([2, 4])
    D2 = ops.create_diagonal([3, 9])
    
    S, U, V = reducer.merge_blocks(D1, D2)
    
    # 1. Unimodularity
    assert is_unit(get_det_mod_n(U, ring), ring)
    assert is_unit(get_det_mod_n(V, ring), ring)
    
    # 2. Equivalence S = U * diag(D1, D2) * V
    # Construct big diagonal
    BigD = ops.identity(4) # Placeholder size
    # Manual diagonal filling
    raw_diag = [2, 4, 3, 9]
    BigD = ops.create_diagonal(raw_diag)
    
    UD = ops.mat_mul(U, BigD)
    UDV = ops.mat_mul(UD, V)
    assert UDV == S
    
    # 3. Smith Form Properties
    is_snf, msg = verify_smith_form_properties(S, ring)
    assert is_snf, f"Merge result not in SNF: {msg}\nMatrix:\n{S}"


def test_prop_7_7_full_diagonal_reduction(components):
    """
    Tests reducing an arbitrary diagonal matrix to Smith Form.
    This exercises the full recursive tree.
    """
    ring, reducer, ops = components
    
    # A messy diagonal matrix in Z/12
    raw = [4, 6, 2, 0] 
    D = ops.create_diagonal(raw)
    
    S, U, V = reducer.reduce_diagonal(D)
    
    # 1. Equivalence
    UD = ops.mat_mul(U, D)
    UDV = ops.mat_mul(UD, V)
    assert UDV == S
    
    # 2. SNF Properties (Divisibility chain)
    is_snf, msg = verify_smith_form_properties(S, ring)
    assert is_snf, f"Reduction failed SNF check: {msg}\nResult:\n{S}"
    
    # 3. Determinant Check (Invariant up to unit)
    # Calculate product of diagonals (since matrices are diagonal)
    det_D = 1
    for x in raw: 
        det_D = ring.mul(det_D, x)
    
    det_S = 1
    for i in range(len(S)): 
        det_S = ring.mul(det_S, S[i][i])
        
    # In Z/12, det(D)=0 and det(S)=0, so this is trivial for this specific input.
    # But for non-singular inputs, this verifies they differ only by a unit.
    assert is_associate(det_D, det_S, ring), \
        f"Determinants not associated: det(D)={det_D}, det(S)={det_S}"


def test_randomized_diagonal_reduction(components):
    """
    Fuzz test for diagonal reduction with random inputs.
    """
    ring, reducer, ops = components
    
    for _ in range(5):
        n = 4
        raw = [random.randint(0, 11) for _ in range(n)]
        D = ops.create_diagonal(raw)
        
        S, U, V = reducer.reduce_diagonal(D)
        
        # Equivalence
        UD = ops.mat_mul(U, D)
        UDV = ops.mat_mul(UD, V)
        assert UDV == S
        
        # SNF
        is_snf, msg = verify_smith_form_properties(S, ring)
        assert is_snf, f"Random SNF failed for input {raw}: {msg}"