# PLAN: Optimize the CRT-based Smith Normal Form path

## Goal

Turn the CRT SNF path (`crates/modularsnf/src/crt.rs`, `rust_crt_snf`) into a
fully-optimized, large-`n`, multi-core implementation for the **small-modulus
(N ≤ a few hundred), large-matrix (n in the thousands)** regime.

Baseline established (head-to-head, rust-vs-rust, branch `bench/headtohead`):
the CRT path already beats the recursive Storjohann baseline 4.4–35×. This plan
attacks its remaining bottlenecks: i128 in the hot loop, scalar elimination,
and single-threaded execution.

## Why this design is correct (research-validated)

- Over a chain/valuation ring `Z/p^e`, the SNF conditioner can be a permutation
  matrix — valuation-pivoted Gaussian elimination, **no gcdex/Bézout**
  (Storjohann diss. Prop 9.21; ESA). This is exactly what `local_snf` does.
- Local-then-recombine is independently validated (Wilkening–Yu).
- SNF/Howell over `Z/N` reduces to **matrix multiplication** — so "make it fast"
  means "turn the trailing update into GEMM."

## Library strategy (call-into vs hand-roll)

| Component | Verdict | Rationale |
|---|---|---|
| Local elimination (Z/p^e) | **Hand-roll (Rust)** | Simple chain-ring pivot; no lib does Smith-with-transforms over Z/p^e; license-clean. |
| Modular GEMM (trailing update) | **Hand-roll thin layer + BLAS/faer** | Accumulate f64/i64, reduce mod p^e; FFLAS does this but is LGPL + field-only. |
| CRT recombination | **Hand-roll (Rust)** | Already done; O(n²·np), not the bottleneck. |
| External SNF lib | **Reference only** | FLINT `nmod_mat_howell_form` = Howell≠Smith, single-limb, **LGPL**. PARI = **GPL → incompatible** with Apache-2.0. |

Decisive constraint: LGPL relinking is fragile in a static manylinux/abi3 wheel;
GPL is out. Hand-roll; use FFLAS-FFPACK / FLINT as algorithmic references.
Prefer **faer** (MIT/Apache) if a Rust GEMM backend is needed.

Key arithmetic fact for our regime: exact integer GEMM stays exact while
`λ·(p−1)² < 2^53`. For p^e ≈ 256 that's λ ≈ 1.4e8 accumulations before a
reduction is needed — so reductions are essentially free and BLAS `dgemm` /
`faer` can be the inner kernel. (Use only this single-level bound; the nested
Strassen-Winograd `kWinograd` bound did not survive verification.)

## Roadmap (phased, each verified against the head-to-head harness)

### Phase 0 — Characterize  *(in progress)*
- Env-gated timing in `crt_snf`: measure `local_snf` total vs recombination vs
  unit-normalization at large n (500, 1000, 2000) and representative moduli.
- Confirm the `local_snf` i128 inner loop dominates. Record the split.

### Phase 1 — Kill i128 in the hot loop (single-core constant win)  *(next)*
- Add an i64 fast-path reduction (valid when `p^e < 2^31`, i128 fallback else),
  mirroring `optimize-modulus`'s `mul_mod`.
- Apply to `local_snf` row/column clears + pivot normalization (`mulmod`).
- Verify: invariant factors + `U·A·V == S (mod N)` unchanged; measure speedup.

### Phase 2 — Blocked right-looking LU + modular GEMM (asymptotic win)

**2a — two-phase split (DONE).** `local_snf` now does Phase L (minimal-valuation
column elimination → U, upper-triangular `mat`) + Phase R (reverse-order back-
elimination → V). Groundwork for blocking; also removed wasted work (old inline
row-clear touched all n rows of each column). ~1.1–1.3× over Phase 1. Gated by a
randomized cargo oracle test (`local_snf_is_valid_smith_form`, 500+ cases:
`U·A·V==diag(p^vals)`, ascending vals, `U,V` unimodular) + head-to-head vs dev.

**2b — blocked GEMM trailing update (DONE).** Implemented as right-looking
blocked LU over the valuation-0 bulk (panel B=48): in-place panel factorization,
TRSM for the deferred panel rows, GEMM trailing update (i64 accumulate + delayed
reduction) on `mat` and `u`; scalar min-valuation path finishes the higher-
valuation/rank-deficient tail; final diagonal normalization. ~1.3–1.5× over 2a
at n=400, 1.6–2.6× over Phase 1 at n=1000–2000 (cumulative ~3× over original
CRT). Gated by two cargo tests (600+ cases incl. multi-panel, rank-deficiency
mod p, p|A, rectangular) + head-to-head vs dev.
**2c — packed cache-friendly GEMM micro-kernel (DONE).** The 2b GEMM read panel
rows with stride-m across the contraction index (~pb cache lines per output).
Fix: pack U12 into a contiguous (pb×ncols) buffer, then contiguous-axpy each
trailing row (autovectorizes); i128 fallback for large q. ~1.10–1.34× over 2b
across single- and multi-prime moduli. (An f64 ndarray/matrixmultiply GEMM was
tried first but regressed multi-prime — n*n product allocation + thin K=48 — so
the packed i64 kernel was kept.) Cumulative ~3.5–4× over original CRT.
NEXT: a true tiled/register-blocked or BLAS/faer kernel could push large-n
further (n=2000 still ~1.1× since the micro-kernel isn't register-tiled), but
diminishing returns vs Phase 3 (rayon).

Original design notes (for reference):
Key constraint:
minimal-valuation pivoting needs the *global* min over the trailing block, which
defeats naive column-panel/lazy blocking. The blockable formulation is
**valuation-level staging**: process pivots by p-adic level ℓ = 0..e; within a
level every pivot is a unit mod p, so it degenerates to standard blocked LU.
Implementation plan for Phase L:
- Outer loop: `lev` = current global min valuation in `mat[k.., k..]`; `pl=p^lev`.
- Panel (width B≈48) of valuation-`lev` pivots. Bookkeeping per standard
  right-looking blocked LU:
  - factor panel columns `[pstart,pend)`: full updates to **panel rows** (so the
    `U12` block `mat[pstart..pend, pend..]` is correct) and to **panel columns**
    for all rows (so pivot search + the `L21` multipliers `mat[pend.., pstart..pend]`
    are correct); defer non-panel trailing columns.
  - **GEMM trailing update**: `mat[pend.., pend..] -= L21 @ U12 (mod p^e)`, and
    the analogous `u[pend.., :] -= L21 @ u[pstart..pend, :]`.
  - column-deflation: a panel column with no valuation-`lev` unit → swap out to
    the active-region tail (track in V), revisit at a higher level.
- GEMM kernel: i64 accumulate with delayed reduction (small `p^e` ⇒ block
  `λ·(p−1)² < 2^53` is huge ⇒ reduce rarely); i128 acc fallback for large `p^e`.
  Later swap inner kernel for `faer` (MIT/Apache) or BLAS dgemm.
- RISK: exact-LU bookkeeping (L21/U12 split, panel-row vs below-panel updates,
  deflation) is subtle; keep the scalar 2a `local_snf` as the cargo-test oracle
  and only switch the default once the blocked version passes the full suite.
- Apply the same blocking to Phase R's V-update (triangular column reduction →
  one TRMM/GEMM) once Phase L lands.

### Phase 3 — Multi-core (rayon)
- Parallelize the trailing-update GEMM first (real scaling).
- Add across-primes parallelism (cheap, low ceiling: np = 1–3).
- **Re-characterize scaling empirically** — published speedups (e.g. 13.6×/32
  cores, FFLAS Euro-Par 2014) are over prime *fields* Z/pZ, not Z/p^e.

### Phase 4 — GPU (deferred, literature-gated)
- No reusable GPU kernel for modular LU/SNF over Z/p(^e) was found. Revisit only
  after rayon scaling is measured.

## Open questions / risks
- Field-vs-chain-ring: parallel/GEMM benchmark evidence is Z/pZ; Z/p^e parallel
  behavior is unmeasured — expect to measure it ourselves.
- Integer-vs-polynomial: Storjohann/Wilkening-Yu state results over K[x]/(p^k);
  Z/p^e transfer is structurally sound extrapolation.
- Evaluate `faer` early in Phase 2 as the license-clean GEMM backend.

## References
- Storjohann, dissertation §9.6 (Prop 9.21, 9.23): cs.uwaterloo.ca/~astorjoh/diss2up.pdf
- Storjohann, ESA (Howell/Smith reduce to matmul): cs.uwaterloo.ca/~astorjoh/esa.pdf
- Dumas/Gautier/Pernet/Sultan, parallel echelon forms (Euro-Par 2014): arxiv.org/abs/1402.3501
- Dumas/Giorgi/Pernet, FFLAS-FFPACK (exact GEMM, delayed reduction): hal.science/hal-00018223v2
- Wilkening–Yu, local Smith form: math.berkeley.edu/~wilken/papers/smith.pdf
- FLINT nmod_mat (Howell form over Z/N): flintlib.org/doc/nmod_mat.html
