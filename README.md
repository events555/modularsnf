# Smith Normal form of Integer matrices mod N (Storjohann)

This is a Python module that follows the deterministic algorithms presented in Arne Storjohann's PhD Dissertation *Algorithms for Matrix Canonical Forms* (ETH No. 13922, 2000).

It implements the Lemmas and subsequent subroutines that are necessary for calculating the SNF without exponential intermediate values.

It validates against SymPy using a known equivalence between calculating the Smith Normal form of an integer matrix, and then taking mod N, compared to solving it natively in the ring.

## Quick Start

```python
from modularsnf import smith_normal_form_mod

S, U, V = smith_normal_form_mod([[2, 4, 0],
                                  [6, 8, 3],
                                  [0, 3, 9]], modulus=36)
# S = U @ A @ V  (mod 36)
# S, U, V are plain Python list[list[int]].
```

`S` is the diagonal Smith Normal Form, `U` and `V` are unimodular
transforms over $\mathbb{Z}/N\mathbb{Z}$. The return order
`(S, U, V)` — diagonal first — follows the SymPy / SageMath convention.
Rectangular matrices are supported; `S`, `U`, `V` shapes match the
input dimensions.

A matrix is **unimodular** over $\mathbb{Z}/N\mathbb{Z}$ when
$\gcd(\det(M),\, N) = 1$, the modular analogue of $|\det(U)| = 1$
over $\mathbb{Z}$.

### Lower-level API

For direct access to `RingMatrix` objects:

```python
from modularsnf.ring import RingZModN
from modularsnf.matrix import RingMatrix
from modularsnf.snf import smith_normal_form

ring = RingZModN(12)
A = RingMatrix(ring, [[2, 4], [6, 8]])
U, V, S = smith_normal_form(A)   # note: (U, V, S) order
```

For details on the algorithm (band reduction, diagonalization, Storjohann's
lemmas), see [docs/algorithm.md](docs/algorithm.md).

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
