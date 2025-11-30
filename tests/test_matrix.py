import numpy as np

from modularsnf.matrix import RingMatrix, _is_within_modulus
from modularsnf.ring import RingZModN


def test_normalize_matrix_handles_large_integers():
    """
    Constructing a ``RingMatrix`` from values larger than C long should not
    raise ``OverflowError``; the data should be reduced modulo N correctly.
    """

    ring = RingZModN(70)
    huge = 10 ** 100  # forces fallback to object dtype during normalization

    matrix = RingMatrix.from_rows(ring, [[huge, 1], [2, 3]])

    assert matrix.shape == (2, 2)
    expected = np.array([[huge % 70, 1], [2, 3]], dtype=int)
    assert matrix.data.dtype == expected.dtype
    assert np.array_equal(matrix.data, expected)


def test_is_within_modulus_checks_value_range():
    ring = RingZModN(11)
    assert _is_within_modulus(np.array([[0, 5, 10]], dtype=int), ring.N)
    assert not _is_within_modulus(np.array([[12, -1]], dtype=int), ring.N)


def test_post_init_applies_mod_when_needed():
    ring = RingZModN(5)
    matrix = RingMatrix(ring, [[-1, 9]])

    assert np.array_equal(matrix.data, np.array([[4, 4]]))
