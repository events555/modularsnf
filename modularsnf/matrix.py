"""Ring-aware matrix utilities for modular Smith normal form routines.

Defines the ``RingMatrix`` dataclass and helpers that normalize data, manage
block operations, and align shapes for modular arithmetic workflows.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Union

import numpy as np

from modularsnf.ring import RingZModN

def _normalize_matrix_data(data: Union[List[List[int]], np.ndarray]) -> np.ndarray:
    """
    Ensure matrix-like input is a 2D ``np.ndarray`` of ints.

    Allows callers to pass either a list-of-lists or an ndarray. Empty inputs
    become an explicit (0, 0) array so downstream shape logic remains simple.
    """
    if isinstance(data, np.ndarray):
        try:
            arr = np.array(data, dtype=int, copy=True)
        except OverflowError:
            # Fallback for extremely large Python ints (e.g., SymPy Integers)
            # that cannot be stored in a fixed-width dtype.
            arr = np.array(data, dtype=object, copy=True)
        if arr.size == 0:
            return arr.reshape((0, 0))
        if arr.ndim == 1:
            return arr.reshape(1, arr.size)
        if arr.ndim != 2:
            raise ValueError("Matrix data must be 2-dimensional")
        return arr

    rows = [list(row) for row in data]
    if not rows:
        return np.zeros((0, 0), dtype=int)

    ncols = len(rows[0])
    for row in rows:
        if len(row) != ncols:
            raise ValueError("All rows must have the same length")

    try:
        return np.array(rows, dtype=int)
    except OverflowError:
        # Same fallback as above for list inputs that include unbounded ints.
        return np.array(rows, dtype=object)


def _is_within_modulus(data: np.ndarray, N: int) -> bool:
    """Return True when ``data`` is provably in ``[0, N)``.

    Avoids expensive copies by only skipping the modulus reduction when values
    are demonstrably safe. Object-typed arrays or non-numeric dtypes always
    force reduction to avoid unsafe assumptions.
    """
    if data.size == 0:
        return True
    if data.dtype == object:
        return False
    if not np.issubdtype(data.dtype, np.integer):
        return False
    try:
        min_val = data.min()
        max_val = data.max()
    except (OverflowError, TypeError, ValueError):
        return False
    return 0 <= min_val and max_val < N


def _to_list_if_array(data: Union[List[List[int]], np.ndarray]) -> List[List[int]]:
    """Return nested Python lists, preserving list inputs for callers that expect them."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


@dataclass
class RingMatrix:
    ring: RingZModN
    data: np.ndarray
    _assume_reduced: bool = field(default=False, init=True, repr=False)

    def __post_init__(self):
        self.data = _normalize_matrix_data(self.data)
        N = self.ring.N

        # Fast path: internal callers that explicitly guarantee the array is
        # already reduced, or data we can verify is inside [0, N).
        if not (self._assume_reduced or _is_within_modulus(self.data, N)):
            self.data %= N
        # If we had to fall back to object dtype for normalization (e.g., from
        # extremely large Python ints), normalize back to a numeric dtype after
        # reduction. This keeps downstream NumPy ops (matmul, slicing, etc.) on
        # the cheaper fixed-width path when possible while still tolerating
        # arbitrarily large intermediate values. If the reduced values still
        # cannot fit, we continue with object dtype.
        if self.data.dtype == object:
            try:
                self.data = self.data.astype(int)
            except (OverflowError, ValueError, TypeError):
                # Remain object-typed; operations will still work, just a bit slower.
                pass

    @property
    def nrows(self) -> int:
        return self.data.shape[0]

    @property
    def ncols(self) -> int:
        return self.data.shape[1] if self.data.size else 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nrows, self.ncols

    @classmethod
    def from_rows(cls, ring: RingZModN, rows: List[List[int]]) -> "RingMatrix":
        return cls(ring=ring, data=rows)

    @classmethod
    def identity(cls, ring: RingZModN, n: int) -> "RingMatrix":
        return cls(ring, np.eye(n, dtype=int))

    @classmethod
    def diagonal(cls, ring: RingZModN, diag: List[int]) -> "RingMatrix":
        n = len(diag)
        rows = np.zeros((n, n), dtype=int)
        for i, v in enumerate(diag):
            rows[i, i] = v
        return cls(ring, rows)

    @classmethod
    def _from_ndarray(cls, ring: RingZModN, data: np.ndarray) -> "RingMatrix":
        """Internal fast constructor for pre-normalized arrays.

        Callers must ensure ``data`` is already 2-D, reduced modulo ``ring.N``,
        and uses an appropriate dtype. No copying or normalization is
        performed.
        """
        if data.ndim != 2:
            raise ValueError("Matrix data must be 2-dimensional")

        obj = cls.__new__(cls)
        obj.ring = ring
        obj.data = data
        return obj

    @classmethod
    def block_diag(cls, A: "RingMatrix", B: "RingMatrix") -> "RingMatrix":
        """
        Block-diagonal composition:
            [ A  0 ]
            [ 0  B ]

        A and B may be rectangular, but must share the same ring.
        """
        if A.ring is not B.ring:
            raise ValueError("Cannot form block diagonal over different rings")
        ring = A.ring
        a_rows, a_cols = A.shape
        b_rows, b_cols = B.shape

        total_rows = a_rows + b_rows
        total_cols = a_cols + b_cols

        data = np.zeros((total_rows, total_cols), dtype=int)

        # top-left: A
        data[:a_rows, :a_cols] = A.data

        # bottom-right: B
        data[a_rows:, a_cols:] = B.data

        return cls._from_ndarray(ring, data)

    def copy(self) -> "RingMatrix":
        return RingMatrix._from_ndarray(self.ring, np.copy(self.data))

    def transpose(self) -> "RingMatrix":
        return RingMatrix._from_ndarray(self.ring, np.transpose(self.data))

    def __matmul__(self, other: "RingMatrix") -> "RingMatrix":
        if self.ring is not other.ring:
            raise ValueError("Cannot multiply matrices over different rings")

        cA = self.ncols
        rB = other.nrows

        if cA != rB:
            raise ValueError(f"Dimension mismatch: {cA} != {rB}")

        ring = self.ring
        N = ring.N

        A = self.data
        B = other.data

        C = (A @ B) % N

        return RingMatrix._from_ndarray(ring, C)

    def pad_to(self, target_rows: int, target_cols: int) -> "RingMatrix":
        rows, cols = self.shape
        if rows == target_rows and cols == target_cols:
            return self.copy()
        N = self.ring.N
        new = np.zeros((target_rows, target_cols), dtype=int)
        new[:rows, :cols] = self.data[:rows, :cols] % N
        return RingMatrix._from_ndarray(self.ring, new)

    def pad_to_square_power2(self) -> "RingMatrix":
        n = max(self.nrows, self.ncols)
        if n <= 0:
            return self.copy()
        size = 1 << (n - 1).bit_length()
        return self.pad_to(size, size)

    def submatrix(self, row_start: int, row_end: int,
                  col_start: int, col_end: int) -> "RingMatrix":
        """
        Return a view copy of rows [row_start:row_end) and
        cols [col_start:col_end).
        """
        rows = self.data[row_start:row_end, col_start:col_end]
        return RingMatrix._from_ndarray(self.ring, np.copy(rows))

    def write_block(self, row_start: int, col_start: int,
                    block: "RingMatrix") -> None:
        """
        Overwrite a sub-block of self starting at (row_start, col_start)
        with the contents of 'block'. Rings must match.
        """
        if self.ring is not block.ring:
            raise ValueError("Cannot write block with different ring")
        b_rows, b_cols = block.shape
        self.data[row_start:row_start + b_rows, col_start:col_start + b_cols] = block.data


    def apply_row_2x2(self, r: int, i: int, s: int, t: int, u: int, v: int) -> None:
        """In-place:  [row_r; row_i] <- [s t; u v] [row_r; row_i]."""
        N = self.ring.N
        rows = self.data[[r, i], :]
        transform = np.array([[s, t], [u, v]], dtype=int) % N
        new_rows = (transform @ rows) % N
        self.data[r, :] = new_rows[0]
        self.data[i, :] = new_rows[1]

    def to_sympy(self):
        import sympy as sp
        return sp.Matrix(_to_list_if_array(self.data))

    def pprint(self):
        from sympy import pprint
        pprint(self.to_sympy())
