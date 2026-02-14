# Smith Normal form of Integer matrices mod N (Storjohann)

This is a Python module that follows the deterministic algorithms presented in Arne Storjohann's PhD Dissertation *Algorithms for Matrix Canonical Forms* (ETH No. 13922, 2000).

It implements the Lemmas and subsequent subroutines that are necessary for calculating the SNF without exponential intermediate values.

It validates against SymPy using a known equivalence between calculating the Smith Normal form of an integer matrix, and then taking mod N, compared to solving it natively in the ring.

The algorithm has two main phases:
1. **Band Reduction:** Transforming an arbitrary matrix into an upper bi-diagonal (2-banded) matrix.
2. **Diagonalization:** Transforming the bi-diagonal matrix into the canonical Smith Normal Form.

## Quick Start

Install and import the single entry-point function:

```python
from modularsnf import smith_normal_form_mod
```

### Square matrix over a composite modulus

```python
from modularsnf import smith_normal_form_mod

A = [[2, 4, 0],
     [6, 8, 3],
     [0, 3, 9]]

S, U, V = smith_normal_form_mod(A, modulus=36)

# S is the diagonal Smith Normal Form, U and V are unimodular transforms.
# The core invariant: S = U @ A @ V (mod 36).
```

The return order `(S, U, V)` — diagonal form first — matches the
convention used by SymPy (`smith_normal_decomp`) and SageMath
(`smith_form`).

### Verifying structural properties

Rather than checking exact values (the transforms `U` and `V` are not
unique), verify the mathematical invariants:

```python
import numpy as np
from modularsnf import smith_normal_form_mod

A = [[2, 4], [6, 8]]
N = 12
S, U, V = smith_normal_form_mod(A, modulus=N)

S, U, V = np.array(S), np.array(U), np.array(V)
A = np.array(A)

# 1. Transform equation: S = U @ A @ V (mod N)
assert np.array_equal(S, (U @ A @ V) % N)

# 2. Diagonal structure: all off-diagonal entries are zero.
for i in range(S.shape[0]):
    for j in range(S.shape[1]):
        if i != j:
            assert S[i][j] == 0

# 3. Divisibility chain: gcd(s_i, N) divides gcd(s_{i+1}, N).
from math import gcd
diag = [int(S[i][i]) for i in range(min(S.shape))]
for i in range(len(diag) - 1):
    assert gcd(diag[i], N) % gcd(diag[i + 1], N) == 0 or gcd(diag[i], N) == 0
```

### Rectangular matrices

Rectangular inputs are handled automatically — the matrix is internally
padded to square, reduced, and the result is cropped back:

```python
S, U, V = smith_normal_form_mod([[1, 2, 3], [4, 5, 6]], modulus=10)
# S is 2x3, U is 2x2, V is 3x3.
```

### Edge cases

```python
# 0x0 (empty matrix)
S, U, V = smith_normal_form_mod([], modulus=7)
assert S == [] and U == [] and V == []

# 1x1
S, U, V = smith_normal_form_mod([[5]], modulus=12)
# The lone diagonal entry generates the same ideal as gcd(5, 12).
# Verify: gcd(S[0][0], 12) == gcd(5, 12).

# All-zero matrix
S, U, V = smith_normal_form_mod([[0, 0], [0, 0]], modulus=6)
# S is all zeros; U and V are identity.
```

### Unimodularity over Z/NZ

A matrix $M$ is **unimodular** over $\mathbb{Z}/N\mathbb{Z}$ when its
determinant is a *unit* in the ring — that is,
$\gcd(\det(M),\, N) = 1$. Both transform matrices $U$ and $V$ returned
by `smith_normal_form_mod` satisfy this property. This is the modular
analogue of the integer requirement that $|\det(U)| = 1$.

### Advanced usage

For users who need `RingMatrix` objects instead of plain lists (e.g. to
compose with other ring operations), the lower-level API is available:

```python
from modularsnf.ring import RingZModN
from modularsnf.matrix import RingMatrix
from modularsnf.snf import smith_normal_form

ring = RingZModN(12)
A = RingMatrix(ring, [[2, 4], [6, 8]])
U, V, S = smith_normal_form(A)   # note: returns (U, V, S) order
```

## Mathematical Foundation

The algorithm operates over the Principal Ideal Ring $R = \mathbb{Z}/N\mathbb{Z}$. Since $R$ is not a field, we rely on unimodular transformations rather than simple Gaussian elimination.

* **The Ring:** We operate in $\mathbb{Z}/N\mathbb{Z}$. Operations must respect zero divisors.
* **Atomic Reduction (`Gcdex`):** Instead of division, we use the Extended Euclidean Algorithm to compute a unimodular matrix $M$ that eliminates entries. For any $a, b \in R$:

$$
\begin{bmatrix} s & t \\ 
u & v \end{bmatrix} 
\begin{bmatrix} a \\ 
b \end{bmatrix} = \begin{bmatrix} g \\ 
0 \end{bmatrix}
$$

   where $\det(M) = sv - tu$ is a unit in $R$.
* **Stabilizer (`Stab`):** When working with zero divisors, we compute a stabilizer $c$ such that $\text{gcd}(a+cb, N) = \text{gcd}(a, b, N)$ [Lemma 1.1].

## Triangularization

The core routine for the reduction is **Lemma 3.1**.

Given a matrix $A$, we compute a unimodular left-transform $U$ and an echelon form $T$ such that:

$$
UA = T
$$

This routine is used recursively to clear columns or rows within specific sub-blocks of the matrix during the Band Reduction phase.

## Band Reduction (Phase 1)

The goal of this phase is to take an upper $b$-banded matrix $A$ (where $A_{ij} = 0$ if $j < i$ or $j \ge i+b$) and transform it into an equivalent matrix $A'$ with bandwidth $\lfloor b/2 \rfloor + 1$. By iterating this process, the matrix becomes bi-diagonal ($b=2$).

This is achieved using two specific subroutines:

### 1. Subroutine `Triang` (Lemma 7.3)
This routine acts on a principal sub-block $B$. It eliminates the "upper triangle" of the block's top-right section using column operations (right multiplication).
* **Transformation:** $B' = B \cdot V$
* **Result:** The band is locally squeezed, but "fill-in" is created below the band.

### 2. Subroutine `Shift` (Lemma 7.4)
This routine chases the fill-in created by `Triang` down the diagonal to restore the banded structure.
* **Transformation:** $C' = U \cdot C \cdot V$
* **Step 1:** Applies a left transform $U$ (derived via Lemma 3.1) to clear the first column block.
* **Step 2:** Applies a right transform $V$ to restore upper-triangularity to the subsequent block.

## Diagonalization (Phase 2)

Once the matrix is upper bi-diagonal, we proceed to the final Smith Normal Form. This involves two steps: reducing the bi-diagonal matrix to a strictly diagonal one, and then fixing the divisibility of the diagonal entries.

### 1. Bi-diagonal to Diagonal (Proposition 7.12)
This routine eliminates the super-diagonal entries of the 2-banded matrix. It applies a specific sequence of `Gcdex`, `Stab`, and `Div` operations to "chase" non-zero off-diagonal entries off the matrix, resulting in a diagonal matrix $D$.

### 2. Diagonal to Smith Form (Proposition 7.7)
A diagonal matrix is only in Smith Normal Form if the diagonal entries satisfy the divisibility chain $d_i | d_{i+1}$. This routine enforces that property using a recursive divide-and-conquer approach:
* **Recursive Step:** The matrix is split into two halves, each is recursively solved.
* **Merge Step (Theorem 7.11):** The two solved halves are merged using a 5-step process that combines the invariants of the two blocks.
* **Base Case (Lemma 7.10):** For a $2 \times 2$ block, we apply atomic reductions to ensure the divisibility condition holds.

## Implementation Status

We have implemented the full pipeline to reach **Lemma 7.14** (Square SNF):

* **Ring Primitives (`modularsnf.ring`)**: Implemented over $\mathbb{Z}/N\mathbb{Z}$, including `Gcdex`, `Stab`, `Div`, and `Ann`.
* **Echelon Forms (`modularsnf.echelon`)**: Implemented **Lemma 3.1** for recursive column clearing.
* **Band Reduction (`modularsnf.snf`)**: Implemented **Lemma 7.3** (`Triang`) and **Lemma 7.4** (`Shift`) to reduce matrix bandwidth.
* **Diagonalization (`modularsnf.diagonal`)**: Implemented **Proposition 7.7** and **Theorem 7.11** to sort diagonal entries by divisibility.

## Development Workflow (modern Python)

This repository supports both standard `pip` workflows and `uv` workflows.

### Using uv

```bash
uv venv
uv sync --extra dev
uv run pytest
uv run ruff check .
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
ruff check .
```
