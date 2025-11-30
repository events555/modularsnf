# Smith Normal form of Integer matrices mod N (Storjohann)

This is a Python module that follows the deterministic algorithms presented in Arne Storjohann's PhD Dissertation *Algorithms for Matrix Canonical Forms* (ETH No. 13922, 2000).

It implements the Lemmas and subsequent subroutines that are necessary for calculating the SNF without exponential intermediate values.

It validates against SymPy using a known equivalence between calculating the Smith Normal form of an integer matrix, and then taking mod N, compared to solving it natively in the ring.

The algorithm has two main phases:
1. **Band Reduction:** Transforming an arbitrary matrix into an upper bi-diagonal (2-banded) matrix.
2. **Diagonalization:** Transforming the bi-diagonal matrix into the canonical Smith Normal Form.

## Mathematical Foundation

The algorithm operates over the Principal Ideal Ring $R = \mathbb{Z}/N\mathbb{Z}$. Since $R$ is not a field, we rely on unimodular transformations rather than simple Gaussian elimination.

* **The Ring:** We operate in $\mathbb{Z}/N\mathbb{Z}$. Operations must respect zero divisors.
* **Atomic Reduction (`Gcdex`):** Instead of division, we use the Extended Euclidean Algorithm to compute a unimodular matrix $M$ that eliminates entries. For any $a, b \in R$:
    $$
    \begin{bmatrix} s & t \\ u & v \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} g \\ 0 \end{bmatrix}
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