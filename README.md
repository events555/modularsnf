# Smith Normal form of Integer matrices mod N (Storjohann)

A Rust implementation (with a thin Python binding) of the deterministic algorithms presented in Arne Storjohann's PhD Dissertation *Algorithms for Matrix Canonical Forms* (ETH No. 13922, 2000).

It implements the Lemmas and subsequent subroutines that are necessary for calculating the SNF without exponential intermediate values. The algorithms live in the Rust `modularsnf` crate; the Python package is a thin wrapper over the native extension.

Correctness is validated in the Rust test suite: random inputs are checked against the Smith form contract ($S = UAV$, divisibility chain, unimodular transforms), and the default Storjohann path is cross-checked against the independent CRT path.

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

For details on the default algorithm (band reduction, diagonalization,
Storjohann's lemmas), see [docs/algorithm.md](docs/algorithm.md).

## Alternative: CRT fast path (experimental)

`smith_normal_form_mod` uses Storjohann's band reduction, which works for any
modulus `N >= 2`. For the **small-modulus, large-matrix** regime there is a
second, experimental algorithm that is several times faster: a CRT-based path
that factors `N = prod p^e`, solves the SNF over each local ring `Z/p^e` by
valuation-pivoted elimination, and recombines via the Chinese Remainder Theorem.

```python
from modularsnf import crt_snf

S, U, V = crt_snf([[2, 4, 0],
                   [6, 8, 3],
                   [0, 3, 9]], modulus=36)
# Same (S, U, V) contract as smith_normal_form_mod; S = U @ A @ V (mod 36).
```

It requires the prime factorization of `N`; pass it explicitly as
`factors=[(p, e), ...]` to amortize factoring across many calls with the same
modulus. Prefer `smith_normal_form_mod` for general moduli. See
[docs/crt.md](docs/crt.md) for the design and correctness basis.

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
