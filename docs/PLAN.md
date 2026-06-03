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
- Restructure `local_snf` into panels; trailing update becomes one matmul
  `T -= L·U_panel (mod p^e)` (Storjohann reduction-to-matmul).
- Build the f64/i64-accumulate → reduce-mod-p^e GEMM layer; evaluate `faer`
  vs ndarray+BLAS vs hand-rolled as the inner kernel.

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
