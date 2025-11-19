# Smith Normal form of Integer matrices mod N (Storjohann)

This is a Python module that implements the algorithm found in Storjohann's PhD dissertation on Algorithms for Matrix Canonical Forms.

It implements all the Lemmas and subsequent subroutines that are necessary.

It validates it by against SymPy using a known equivalence between calculating the Smith Normal form of an integer matrix, and then taking mod N, compared to solving it natively in the ring.

The algorithm has two phases:
1. **Band Reduction:** Transforming an arbitrary matrix into an upper bi-diagonal (2-banded) matrix.
2.  **Diagonalization:** Transforming the bi-diagonal matrix into the canonical Smith Normal Form.

## Mathematical Foundation

The algorithm operates over the Principal Ideal Ring $R = \mathbb{Z}/N\mathbb{Z}$. Since $R$ is not a field, we rely on unimodular transformations rather than simple Gaussian elimination.

* **The Ring:** We operate in $\mathbb{Z}/N\mathbb{Z}$. Operations must respect zero divisors.
* **Atomic Reduction (`Gcdex`):** Instead of division, we use the Extended Euclidean Algorithm to compute a unimodular matrix $M$ that eliminates entries. For any $a, b \in R$:
    $$
    \begin{bmatrix} s & t \\ u & v \end{bmatrix} \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} g \\ 0 \end{bmatrix}
    $$
   where $\det(M) = sv - tu$ is a unit in $R$.


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

## To be continued (WIP)
* Implementation of Phase 2: Diagonalization (Bi-diagonal to SNF).