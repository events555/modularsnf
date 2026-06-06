# Agent Instructions

Smith Normal Form over Z/NZ following Storjohann's dissertation (ETH No. 13922).
Algorithm details live in [`docs/algorithm.md`](docs/algorithm.md).

## Code Style

Google Python Style Guide with these adjustments:

* **Math names:** single-letter variables (`U`, `V`, `A`, `n`) are preferred
  when they match the paper's notation.
* **Formatting:** 4-space indent, 80-char soft limit, no whitespace inside
  parens.
* **Types:** mandatory on function signatures, Python 3.10+ syntax.
* **Strings:** f-strings only.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add smith_normal_form_mod convenience API
fix: handle zero-row edge case in band reduction
docs: move algorithm writeup to docs/
refactor: extract _merge_scalars from merge_smith_blocks
test: add seeded random SNF parity checks
chore: bump ruff, update pyproject metadata
```

## Ring Arithmetic

All operations over Z/NZ. No floats, no `/`.
Use `modularsnf.ring` primitives: `gcdex`, `div`, `quo`, `stab`, `ann`.

## Testing

* Algorithm correctness lives in the Rust crate (`cargo test`): structural
  assertions (`S = U A V`, diagonal shape, divisibility chain, unimodularity)
  over random inputs, plus a Storjohann-vs-CRT cross-check.
* The Python suite (`tests/`) covers only the PyO3 boundary: input validation,
  type/shape marshalling, and the public-API contract on small cases.
* Run `cargo test` and `just check` (`just lint`, `just typecheck`, `just test`)
  before finalizing.

## File Structure

The SNF algorithms live in the Rust workspace; the Python package is a thin
wrapper over the `modularsnf._rust` extension.

* `crates/modularsnf/` — Rust lib crate: ring primitives, echelon/band
  reduction, diagonalization (`snf.rs`/`band.rs`/`echelon.rs`/`diagonal.rs`),
  and the CRT fast path (`crt.rs`).
* `crates/modularsnf-py/` — PyO3 bindings exposed as `modularsnf._rust`.
* `modularsnf/ring.py` — `RingZModN`, forwarding ring primitives to Rust.
* `modularsnf/matrix.py` — `RingMatrix` data structure.
* `modularsnf/diagonal.py` — diagonalization wrappers.
* `modularsnf/snf.py` — public `smith_normal_form_mod` API (default path).
* `modularsnf/crt.py` — public `crt_snf` API (experimental fast path).
* `docs/algorithm.md` — default algorithm (Storjohann band reduction).
* `docs/crt.md` — CRT fast path design and correctness basis.
