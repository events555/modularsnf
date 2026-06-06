"""Input validation and marshalling for the modularsnf Python binding."""

from collections.abc import Sequence
from numbers import Integral
from typing import Optional

import numpy as np


def validate_modulus(modulus: object) -> int:
    """Coerce *modulus* to an int >= 2 or raise ValueError."""
    if isinstance(modulus, bool) or not isinstance(modulus, Integral):
        raise ValueError(f"Modulus must be an integer >= 2, got {modulus!r}")
    value = int(modulus)
    if value < 2:
        raise ValueError(f"Modulus must be an integer >= 2, got {modulus!r}")
    return value


def to_int64_matrix(
    matrix: Sequence[Sequence[int]], modulus: int
) -> Optional[np.ndarray]:
    """Validate and marshal *matrix* into a 2-D int64 array reduced mod *modulus*.

    Returns ``None`` for an empty matrix (zero rows).

    Raises:
        TypeError: If *matrix* is not a list/tuple of rows.
        ValueError: If rows have unequal lengths.
        OverflowError: If an entry does not fit in a signed 64-bit integer.
    """
    if not isinstance(matrix, (list, tuple)):
        raise TypeError("matrix must be a list of lists of integers")

    if len(matrix) == 0:
        return None

    ncols = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != ncols:
            raise ValueError(
                f"Ragged matrix: row 0 has {ncols} columns "
                f"but row {i} has {len(row)} columns"
            )

    # np.asarray raises OverflowError for entries outside int64.
    arr = np.asarray(matrix, dtype=np.int64)
    return arr % modulus
