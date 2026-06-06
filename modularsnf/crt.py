"""CRT-based Smith Normal Form fast path (experimental).

An alternative to :func:`modularsnf.smith_normal_form_mod`, specialized for the
small-modulus / large-matrix regime. It factors ``N = prod p^e``, solves the
SNF over each local ring ``Z/p^e`` by valuation-pivoted elimination, and
recombines via the Chinese Remainder Theorem. For general moduli prefer
``smith_normal_form_mod``. See ``docs/crt.md`` for the design.
"""

from typing import List, Optional, Tuple

from modularsnf._rust import rust_crt_snf as _rust_crt_snf

from ._core import to_int64_matrix, validate_modulus
from .snf import SNFResult


def factorize(n: int) -> List[Tuple[int, int]]:
    """Factor *n* into ``[(p, e), ...]`` by trial division.

    Intended for the small moduli this path targets; not a general-purpose
    integer factorizer.
    """
    if n < 2:
        raise ValueError(f"modulus must be >= 2 to factor, got {n}")
    factors: List[Tuple[int, int]] = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors


def _resolve_factors(
    modulus: int, factors: Optional[List[Tuple[int, int]]]
) -> List[Tuple[int, int]]:
    if factors is None:
        return factorize(modulus)
    prod = 1
    for p, e in factors:
        prod *= p**e
    if prod != modulus:
        raise ValueError(
            f"factors {factors} multiply to {prod}, not modulus {modulus}"
        )
    return factors


def crt_snf(
    matrix: list[list[int]],
    modulus: int,
    *,
    factors: Optional[List[Tuple[int, int]]] = None,
) -> SNFResult:
    """Compute the Smith Normal Form over Z/NZ via the CRT fast path.

    Returns ``(S, U, V)`` with ``S = U @ A @ V`` (mod *modulus*) — the same
    contract as :func:`modularsnf.smith_normal_form_mod`.

    Args:
        matrix: 2-D list of signed 64-bit integers. May be rectangular.
        modulus: Signed 64-bit integer *N* >= 2 defining the ring Z/NZ.
        factors: Prime factorization of *modulus* as ``[(p, e), ...]``. When
            omitted it is computed by trial division; pass it explicitly to
            amortize factoring across many calls with the same modulus.

    Raises:
        OverflowError: If *modulus* or any matrix entry is outside int64.
        ValueError: If *modulus* < 2, *factors* does not multiply to *modulus*,
            or rows have unequal lengths.
        TypeError: If *matrix* is not a list of lists.
    """
    modulus = validate_modulus(modulus)
    arr = to_int64_matrix(matrix, modulus)
    if arr is None:
        return SNFResult(S=[], U=[], V=[])

    factors = _resolve_factors(modulus, factors)
    u, v, s = _rust_crt_snf(arr, modulus, factors)
    return SNFResult(S=s.tolist(), U=u.tolist(), V=v.tolist())
