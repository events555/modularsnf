from dataclasses import dataclass
from typing import List, Tuple

from modularsnf.ring import RingZModN

@dataclass
class RingMatrix:
    ring: RingZModN
    data: List[List[int]]

    def __post_init__(self):
        if not self.data:
            return
        ncols = len(self.data[0])
        for row in self.data:
            if len(row) != ncols:
                raise ValueError("All rows must have the same length")
        N = self.ring.N
        self.data = [[x % N for x in row] for row in self.data]

    @property
    def nrows(self) -> int:
        return len(self.data)

    @property
    def ncols(self) -> int:
        return len(self.data[0]) if self.data else 0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.nrows, self.ncols

    @classmethod
    def from_rows(cls, ring: RingZModN, rows: List[List[int]]) -> "RingMatrix":
        return cls(ring=ring, data=rows)

    @classmethod
    def identity(cls, ring: RingZModN, n: int) -> "RingMatrix":
        rows = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(ring, rows)

    @classmethod
    def diagonal(cls, ring: RingZModN, diag: List[int]) -> "RingMatrix":
        n = len(diag)
        rows = [[0]*n for _ in range(n)]
        for i, v in enumerate(diag):
            rows[i][i] = v
        return cls(ring, rows)

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

        zero = 0
        data = [[zero for _ in range(total_cols)] for _ in range(total_rows)]

        # top-left: A
        for i in range(a_rows):
            for j in range(a_cols):
                data[i][j] = A.data[i][j]

        # bottom-right: B
        for i in range(b_rows):
            for j in range(b_cols):
                data[a_rows + i][a_cols + j] = B.data[i][j]

        return cls(ring, data)

    def copy(self) -> "RingMatrix":
        return RingMatrix(self.ring, [row[:] for row in self.data])

    def transpose(self) -> "RingMatrix":
        t = list(zip(*self.data))
        return RingMatrix(self.ring, [list(row) for row in t])

    def __matmul__(self, other: "RingMatrix") -> "RingMatrix":
        if self.ring is not other.ring:
            raise ValueError("Cannot multiply matrices over different rings")

        rA = len(self.data)
        cA = len(self.data[0]) if rA else 0
        rB = len(other.data)
        cB = len(other.data[0]) if rB else 0

        if cA != rB:
            raise ValueError(f"Dimension mismatch: {cA} != {rB}")

        ring = self.ring
        N = ring.N

        A = self.data
        B = other.data

        # Pre-allocate result
        C = [[0] * cB for _ in range(rA)]

        for i in range(rA):
            Ai = A[i]
            Ci = C[i]
            for k in range(cA):
                aik = Ai[k]
                if aik == 0:
                    continue
                Bk = B[k]
                for j in range(cB):
                    Ci[j] += aik * Bk[j]
            for j in range(cB):
                Ci[j] %= N

        return RingMatrix(ring, C)

    def pad_to(self, target_rows: int, target_cols: int) -> "RingMatrix":
        rows, cols = self.shape
        if rows == target_rows and cols == target_cols:
            return self.copy()
        N = self.ring.N
        new = [[0]*target_cols for _ in range(target_rows)]
        for r in range(rows):
            for c in range(cols):
                new[r][c] = self.data[r][c] % N
        return RingMatrix(self.ring, new)

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
        rows = [
            row[col_start:col_end]
            for row in self.data[row_start:row_end]
        ]
        return RingMatrix(self.ring, rows)

    def write_block(self, row_start: int, col_start: int,
                    block: "RingMatrix") -> None:
        """
        Overwrite a sub-block of self starting at (row_start, col_start)
        with the contents of 'block'. Rings must match.
        """
        if self.ring is not block.ring:
            raise ValueError("Cannot write block with different ring")
        b_rows, b_cols = block.shape
        for i in range(b_rows):
            for j in range(b_cols):
                self.data[row_start + i][col_start + j] = block.data[i][j]


    def apply_row_2x2(self, r: int, i: int, s: int, t: int, u: int, v: int) -> None:
        """In-place:  [row_r; row_i] <- [s t; u v] [row_r; row_i]."""
        ring = self.ring
        row_r = self.data[r]
        row_i = self.data[i]
        new_r = [
            ring.add(ring.mul(s, x), ring.mul(t, y))
            for x, y in zip(row_r, row_i)
        ]
        new_i = [
            ring.add(ring.mul(u, x), ring.mul(v, y))
            for x, y in zip(row_r, row_i)
        ]
        self.data[r] = new_r
        self.data[i] = new_i

    def to_sympy(self):
        import sympy as sp
        return sp.Matrix(self.data)

    def pprint(self):
        from sympy import pprint
        pprint(self.to_sympy())
