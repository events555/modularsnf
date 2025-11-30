import numpy as np

from modularsnf.matrix import RingMatrix
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
