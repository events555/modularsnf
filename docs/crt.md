# CRT fast path (experimental)

An alternative SNF algorithm to the Storjohann band reduction
([`docs/algorithm.md`](algorithm.md)), specialized for the **small-modulus
(N up to a few hundred), large-matrix (n in the thousands)** regime. On that
class of inputs it is several times faster than the band-reduction path; for
general moduli, or when the prime factorization of N is unknown or expensive,
prefer the Storjohann path.

Implemented in `crates/modularsnf/src/crt.rs` (`crt_snf`) and exposed to Python
as `modularsnf.crt_snf`.

## Idea

Factor `N = prod p^e` (the factorization is passed in, so it is amortized
across many calls with the same modulus). The SNF over `Z/NZ` decomposes by the
Chinese Remainder Theorem into independent SNFs over each local ring `Z/p^e`:

1. **Local SNF.** Over a chain ring `Z/p^e` the conditioner can be a
   permutation — valuation-pivoted Gaussian elimination, with no `gcdex`/Bézout
   step. This yields the divisibility chain for free and produces the unimodular
   transforms `U_p, V_p` natively.
2. **Normalize.** Unit-normalize each prime's `V_p` so every prime realizes the
   same global invariant factors `d_i = prod_p p^{v_p(i)}`.
3. **Recombine.** CRT-recombine the per-prime `U_p, V_p` entrywise into
   `U, V` over `Z/NZ`.

The result satisfies `S = U A V (mod N)` with `S` diagonal and `U, V`
unimodular, identical in form to the band-reduction path.

## Local SNF internals

`local_snf` runs in two phases:

- **Phase L** — column elimination producing `U` and leaving `mat` upper
  triangular. A blocked right-looking LU clears the valuation-0 bulk: a panel is
  factored in place, then the trailing block is updated by TRSM + GEMM (i64
  accumulate with delayed reduction; i128 fallback for large `p^e`). A scalar
  minimal-valuation path finishes the higher-valuation and rank-deficient tail.
- **Phase R** — back-elimination of the super-diagonal producing `V`,
  processing pivots in reverse so fill lands in not-yet-processed rows.

The blocking exploits the fact that SNF/Howell over `Z/N` reduces to matrix
multiplication, so the trailing update is a modular GEMM. Exact integer GEMM
stays exact while `lambda * (p-1)^2` fits the accumulator, which for small `p^e`
means reductions are rare.

## Correctness basis

- Permutation conditioner over chain rings: Storjohann, dissertation §9.6
  (Prop 9.21, 9.23).
- SNF/Howell reduces to matrix multiplication: Storjohann, ESA.
- Local-then-recombine: Wilkening–Yu, *local Smith form*.

The scalar `local_snf` is the oracle for the blocked path: a randomized Cargo
test suite (`local_snf_is_valid_smith_form`, `local_snf_blocked_paths`) checks
`U A V == diag(p^vals)`, ascending valuations, and unimodularity of `U, V`
across single/multi-panel, rank-deficient, `p | A`, and rectangular cases.

## References

- Storjohann, *Algorithms for Matrix Canonical Forms* (diss., §9.6):
  cs.uwaterloo.ca/~astorjoh/diss2up.pdf
- Storjohann, *Howell/Smith reduce to matmul* (ESA):
  cs.uwaterloo.ca/~astorjoh/esa.pdf
- Wilkening, Yu, *local Smith form*:
  math.berkeley.edu/~wilken/papers/smith.pdf
- Dumas, Giorgi, Pernet, *FFLAS-FFPACK* (exact GEMM, delayed reduction):
  hal.science/hal-00018223v2
