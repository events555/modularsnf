import math
import random

import numpy as np

import pytest
from modularsnf.ring import RingZModN
from modularsnf.matrix import RingMatrix
from modularsnf.echelon import lemma_3_1
from tests.helpers import det_ring_matrix, row_span, verify_echelon_structure

RINGS_TO_TEST = [2, 5, 9, 12]

@pytest.mark.parametrize("N", RINGS_TO_TEST)
def test_lemma_3_1_equation_and_row_module(N):
    ring = RingZModN(N)

    nrows, ncols = 5, 5
    A_rows = [
        [random.randint(0, N - 1) for _ in range(ncols)]
        for _ in range(nrows)
    ]
    A = RingMatrix.from_rows(ring, A_rows)

    U, T, _ = lemma_3_1(A)

    UA = U @ A
    assert np.array_equal(UA.data, T.data)

    span_A = row_span(A)
    span_T = row_span(T)
    assert span_A == span_T

@pytest.mark.parametrize("N", RINGS_TO_TEST)
def test_lemma_3_1_unimodular_U(N):
    ring = RingZModN(N)

    size = 5
    A_rows = [
        [random.randint(0, N - 1) for _ in range(size)]
        for _ in range(size)
    ]
    A = RingMatrix.from_rows(ring, A_rows)

    U, T, _ = lemma_3_1(A)

    det_U = det_ring_matrix(U) % N
    assert math.gcd(det_U, N) == 1, f"det(U)={det_U} not a unit mod {N}"

@pytest.mark.parametrize("N", RINGS_TO_TEST)
def test_lemma_3_1_echelon_structure(N):
    ring = RingZModN(N)

    examples = []

    examples.append([
        [2, 4, 6],
        [1, 2, 3],
        [0, 1, 5],
        [0, 0, 0],
    ])

    for _ in range(3):
        nrows, ncols = 4, 5
        rows = [
            [random.randint(0, N - 1) for _ in range(ncols)]
            for _ in range(nrows)
        ]
        examples.append(rows)

    for rows in examples:
        A = RingMatrix.from_rows(ring, rows)
        _, T, _ = lemma_3_1(A)

        assert verify_echelon_structure(T), (
            f"T is not in echelon form over Z/{N}: {T.data}"
        )

