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

* Structural assertions (`S = U A V`, diagonal shape, divisibility chain,
  unimodularity) over hardcoded array equality.
* Validate against SymPy integer-domain projection where feasible.
* Run `uv run pytest` and `uv run ruff check .` before finalizing.

## File Structure

* `modularsnf/ring.py` — ring primitives (Gcdex, Stab, Div, Ann).
* `modularsnf/matrix.py` — `RingMatrix` data structure.
* `modularsnf/echelon.py` — triangularization (Lemma 3.1).
* `modularsnf/band.py` — band reduction (Lemmas 7.3, 7.4, Prop 7.1).
* `modularsnf/diagonal.py` — diagonalization (Prop 7.7, Theorem 7.11).
* `modularsnf/snf.py` — master pipeline (Lemma 7.14) + public
  `smith_normal_form_mod` API.
* `docs/algorithm.md` — mathematical foundation and algorithm description.
* `docs/PLAN.md` — current ExecPlan.

## ExecPlans

For work spanning multiple modules or changing public APIs, write a plan in
`docs/PLAN.md` before implementing. Users may say "use an ExecPlan" as
shorthand.
