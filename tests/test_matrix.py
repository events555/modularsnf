import numpy as np
import pytest

from modularsnf.matrix import RingMatrix, _is_within_modulus
from modularsnf.ring import RingZModN


def test_normalize_matrix_rejects_out_of_range_integers():
    """Matrix entries must fit in the signed 64-bit contract."""

    ring = RingZModN(70)
    huge = 10**100

    with pytest.raises(OverflowError, match="signed 64-bit"):
        RingMatrix.from_rows(ring, [[huge, 1], [2, 3]])


def test_is_within_modulus_checks_value_range():
    ring = RingZModN(11)
    assert _is_within_modulus(np.array([[0, 5, 10]], dtype=int), ring.N)
    assert not _is_within_modulus(np.array([[12, -1]], dtype=int), ring.N)


def test_post_init_applies_mod_when_needed():
    ring = RingZModN(5)
    matrix = RingMatrix(ring, [[-1, 9]])

    assert np.array_equal(matrix.data, np.array([[4, 4]]))
