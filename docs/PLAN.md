# PLAN: Feature Parity Validation + PyPI Readiness

This ExecPlan focuses on two outcomes:

1. Deliver a generally available SNF entry point that accepts a user-supplied
   modulus.
2. Build confidence through parity checks against existing SNF implementations,
   then package the project for reliable PyPI release.

## A. Generally Available API (user supplies modulus)

### A1. Add a top-level convenience API (done)

- Add a function such as
  `smith_normal_form_mod(matrix: list[list[int]], modulus: int)` that:
  - Builds `RingZModN(modulus)`.
  - Converts input into `RingMatrix`.
  - Calls the existing `smith_normal_form` pipeline.
  - Returns `(U, S, V)` in plain Python integer arrays (or a small dataclass).
- Keep current lower-level APIs unchanged for advanced users.

**Acceptance criteria**

- Users can call one function with `matrix + modulus` and obtain SNF outputs.
- Invalid modulus (`<= 0`) and ragged matrices raise clear `ValueError`s.
- Rectangular matrix behavior is documented (padding/cropping semantics).

### A2. API documentation and examples (done)

- Add usage examples to `README.md`:
  - square matrix over composite modulus,
  - rectangular matrix,
  - edge cases (`0x0`, `1x1`, all-zero matrix).
- Document what “unimodular” means in `Z/NZ` (determinant is a unit mod `N`).

**Acceptance criteria**

- README has copy-pasteable examples with expected structural properties.

## B. Feature Parity Validation Strategy

### B1. Oracle comparison harness (SymPy-backed) (done)

- Build tests that:
  - Generate random integer matrices `M` and modulus `N`.
  - Compute integer-domain SNF using SymPy, then project to mod `N`.
  - Compare invariants against modular pipeline result.
- Prefer property checks over direct array equality:
  - `S = U A V`,
  - `S` diagonal,
  - divisibility chain on diagonal entries,
  - `U`, `V` unimodular over `Z/NZ`.

**Acceptance criteria**

- Deterministic seeded test suite across multiple moduli:
  - prime `N`,
  - composite `N`,
  - high zero-divisor cases (e.g., powers like `2^k`).

### B2. Curated regression corpus (done)

- Add a small fixed corpus of matrices that historically break SNF routines:
  - repeated factors,
  - rank-deficient blocks,
  - dense random,
  - near-diagonal with adversarial superdiagonal entries.
- Store fixtures and expected structural predicates.

**Acceptance criteria**

- Corpus runs in CI and guards against algorithmic regressions.

### B3. Performance sanity checks (confidence, not benchmarking suite) (done)

- Add a lightweight timing script/test (optional in CI) for medium sizes.
- Track runtime/memory trends to catch accidental complexity blowups.

**Acceptance criteria**

- No major regression versus current baseline for representative sizes.

## C. PyPI Deployability Checklist

### C1. Packaging metadata hardening

- Update `pyproject.toml`:
  - raise `requires-python` to the tested floor (e.g., `>=3.10` if true),
  - add project URLs (`Homepage`, `Repository`, `Issues`),
  - add keywords and more classifiers,
  - ensure dependency bounds are explicit.
- Add missing repo artifacts:
  - `LICENSE` file (classifier already claims MIT),
  - `CHANGELOG.md`,
  - optional `CONTRIBUTING.md`.

**Acceptance criteria**

- `python -m build` succeeds for sdist and wheel.
- `twine check dist/*` passes with no warnings.

### C2. CI for release confidence

- Add CI matrix (Linux, Python versions supported).
- Required jobs:
  - unit + property tests,
  - packaging build,
  - lint/type checks (if adopted).
- Add a release workflow that publishes on tags after checks pass.

**Acceptance criteria**

- Green CI on default branch and on release tag candidate.

### C3. Versioning and release process

- Choose versioning policy (SemVer recommended).
- Define release steps in `RELEASING.md`:
  - bump version,
  - update changelog,
  - build/check artifacts,
  - publish (trusted publisher or API token).

**Acceptance criteria**

- A dry-run release to TestPyPI completes successfully.

## D. Recommended Execution Order (minimal-to-confidence)

1. **GA API wrapper + README examples** (fastest user-facing improvement).
2. **Parity/property tests + regression corpus** (confidence gate).
3. **Packaging hardening (`LICENSE`, metadata, build/twine checks)**.
4. **CI + TestPyPI dry run**.
5. **PyPI production release**.

## E. Definition of Done

- A user can provide `matrix + modulus` in one call and get valid SNF outputs.
- Test suite demonstrates parity-level confidence via structural invariants.
- Repository builds distributable artifacts cleanly and passes PyPI checks.
- Release workflow is documented and repeatable.
