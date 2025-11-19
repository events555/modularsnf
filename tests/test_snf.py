import pytest
import random
from modularsnf.ring import RingZModN
from modularsnf.snf import BidiagonalToSmith
from modularsnf.matrix import MatrixOps
from modularsnf.band_reduction import BandReduction
from .helpers import verify_smith_form_properties

# Reuse helpers from previous tests (is_unit, verify_smith_form_properties, etc.)
# ...

@pytest.fixture
def components():
    ring = RingZModN(12)
    snf = BidiagonalToSmith(ring)
    ops = MatrixOps(ring)
    band = BandReduction(ring)
    return ring, snf, ops, band

def test_stab_atomic(components):
    """Verify the Stab logic from ring.py"""
    ring, _, _, _ = components
    # Example: gcd(2 + x*4, 6) = gcd(2, 4, 6) = 2
    # If x=1: gcd(6, 6) = 6 != 2
    # If x=2: gcd(10, 6) = 2. Success.
    x = ring.stab(2, 4, 6)
    assert ring.gcd(2 + x*4, ring.gcd(6, 12)) == 2

def test_bidiagonal_to_smith(components):
    """
    Validates Proposition 7.12.
    Input: Upper 2-Banded Matrix.
    Output: Smith Form.
    """
    ring, snf, ops, _ = components
    n = 4
    
    # Create a 2-banded matrix
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        A[i][i] = random.randint(0, 11)
        if i+1 < n:
            A[i][i+1] = random.randint(0, 11)
            
    S, U, V = snf.run(A)
    
    # 1. Equivalence
    UA = ops.mat_mul(U, A)
    UAV = ops.mat_mul(UA, V)
    assert UAV == S
    
    # 2. Structure
    # Check diagonal and divisibility
    is_valid, msg = verify_smith_form_properties(S, ring) # Assuming this helper is imported
    assert is_valid, msg

def test_full_pipeline_integration(components):
    """
    Tests Arbitrary Matrix -> BandReduce -> SNF
    """
    ring, snf, ops, band = components
    
    # Arbitrary Matrix
    A = [
        [2, 4, 6],
        [1, 3, 5],
        [0, 2, 4]
    ]
    
    # Phase 1: Bidiagonalize
    B, U1, V1 = band.bidiagonalize(A)
    
    # Phase 2: SNF
    S, U2, V2 = snf.run(B)
    
    # Combine Transforms
    # Final = U2 * U1 * A * V1 * V2
    U_total = ops.mat_mul(U2, U1)
    V_total = ops.mat_mul(V1, V2)
    
    UA = ops.mat_mul(U_total, A)
    UAV = ops.mat_mul(UA, V_total)
    
    assert UAV == S
    is_valid, msg = verify_smith_form_properties(S, ring)
    assert is_valid, msg