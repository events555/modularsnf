"""Smith Normal Form over Z/NZ via the native Storjohann band reduction.

A thin wrapper over ``modularsnf._rust.rust_smith_normal_form``; the reduction
itself lives in the Rust ``modularsnf`` crate. For the experimental CRT fast
path see ``modularsnf.crt``.
"""

from typing import NamedTuple

from modularsnf._rust import rust_smith_normal_form as _rust_snf

from ._core import to_int64_matrix, validate_modulus


class SNFResult(NamedTuple):
    """Result of a Smith Normal Form decomposition over Z/NZ.

    ``S`` is the diagonal Smith form, ``U`` and ``V`` the unimodular transforms,
    each a plain ``list[list[int]]``. The invariant ``S = U @ A @ V`` holds over
    Z/NZ.
    """

    S: list[list[int]]
    U: list[list[int]]
    V: list[list[int]]


def smith_normal_form_mod(
    matrix: list[list[int]],
    modulus: int,
) -> SNFResult:
    """Compute the Smith Normal Form of an integer matrix over Z/NZ.

    Returns ``(S, U, V)`` where ``S = U @ A @ V`` (mod *modulus*), *S* is
    diagonal with the divisibility chain ``s_i | s_{i+1}``, and *U*, *V* are
    unimodular. All three are plain Python ``list[list[int]]``.

    Args:
        matrix: 2-D list of signed 64-bit integers. May be rectangular.
        modulus: Signed 64-bit integer *N* >= 2 defining the ring Z/NZ.

    Raises:
        OverflowError: If *modulus* or any matrix entry is outside int64.
        ValueError: If *modulus* < 2 or rows have unequal lengths.
        TypeError: If *matrix* is not a list of lists.

    Examples:
        >>> S, U, V = smith_normal_form_mod([[2, 4], [6, 8]], modulus=12)
    """
    modulus = validate_modulus(modulus)
    arr = to_int64_matrix(matrix, modulus)
    if arr is None:
        return SNFResult(S=[], U=[], V=[])

    u, v, s = _rust_snf(arr, modulus)
    return SNFResult(S=s.tolist(), U=u.tolist(), V=v.tolist())
