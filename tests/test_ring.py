import pytest
import random
from modularsnf.ring import RingZModN

@pytest.fixture
def ring():
    # Z/12 is chosen because it has zero divisors (2, 3, 4, 6, 8, 10)
    # providing a robust test for a Principal Ideal Ring (PIR).
    return RingZModN(12)

def is_unit(val, ring):
    """Checks if val is a unit in Z/N."""
    a, b = val, ring.N
    while b:
        a, b = b, a % b
    return a == 1

def test_gcdex_fundamental_identity(ring):
    """
    Validates that Gcdex produces a valid transformation.
    [[s, t], [u, v]] * [a, b]^T = [g, 0]^T
    See.
    """
    for _ in range(100):
        a = random.randint(0, 11)
        b = random.randint(0, 11)
        
        g, s, t, u, v = ring.gcdex(a, b)
        
        # Row 1 check: sa + tb = g
        res_g = ring.add(ring.mul(s, a), ring.mul(t, b))
        assert res_g == g, f"Row 1 failed for {a},{b}"
        
        # Row 2 check: ua + vb = 0 (The triangularization property)
        res_zero = ring.add(ring.mul(u, a), ring.mul(v, b))
        assert res_zero == 0, f"Row 2 failed for {a},{b}"

def test_gcdex_unimodularity(ring):
    """
    Validates that the transformation matrix is unimodular.
    sv - tu must be a unit.
    See.
    """
    for _ in range(100):
        a = random.randint(0, 11)
        b = random.randint(0, 11)
        
        _, s, t, u, v = ring.gcdex(a, b)
        
        # Determinant = sv - tu
        det = ring.sub(ring.mul(s, v), ring.mul(t, u))
        assert is_unit(det, ring), f"Matrix not unimodular for inputs {a},{b}. Det: {det}"

def test_gcdex_divisibility_condition(ring):
    """
    Validates the specific edge case requirement from the dissertation.
    'if b is divisible by a then s=v=1, t=0'
    See.
    """
    # Case 1: a=2, b=4 (4 is divisible by 2 in Z/12)
    a, b = 2, 4
    _, s, t, u, v = ring.gcdex(a, b)
    
    assert s == 1 and v == 1 and t == 0, \
        f"Failed divisibility constraint for {a}|{b}. Got s={s}, t={t}, v={v}"

    # Case 2: a=3, b=9
    a, b = 3, 9
    _, s, t, u, v = ring.gcdex(a, b)
    assert s == 1 and v == 1 and t == 0