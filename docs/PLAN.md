# PLAN: 0.2.0 Release Hardening

## Goal

Ship `0.2.0` with a coherent machine-integer contract, safer Rust hot paths,
and release automation that publishes binary wheels for mainstream platforms.

## Scope

1. Make the public Python API explicitly `int64`-only.
2. Keep Rust as the primary accelerated path, but remove accidental claims of
   arbitrary-precision support.
3. Use wider intermediates in Rust where `i64` multiplication or accumulation
   can overflow before modular reduction.
4. Publish `abi3` wheels for:
   - Linux `x86_64`
   - Linux `aarch64`
   - macOS Intel
   - macOS Apple Silicon
   - Windows `x86_64`
5. Clean up the profiling script so backend selection is explicit and the
   reported instrumentation matches the code path being exercised.
6. Make `just check` validate the full test suite against both the pure-Python
   and Rust backends without duplicating test files.
7. Finish with `just check`.

## Work Items

### 1. Python Contract

- Add explicit signed-64-bit validation for:
  - modulus values
  - matrix entries
- Remove object-dtype fallback paths that implied arbitrary-precision support.
- Update tests to assert the new failure mode for out-of-range inputs.

### 2. Rust Arithmetic Safety

- Audit multiplication and accumulation sites in the Rust core.
- Use `i128` intermediates only where needed before reducing back to `i64`.
- Preserve `i64` storage and FFI types.

### 3. Packaging And Release Automation

- Bump package versions to `0.2.0`.
- Switch PyO3 packaging to `abi3-py310`.
- Update GitHub Actions to build and publish wheels for the release matrix.
- Keep sdist generation, accepting that source builds require Rust.

### 4. Profiling Tooling

- Make backend choice explicit in `scripts/profile_snf.py`.
- Ensure Python-only instrumentation is not presented as if it profiles the
  Rust top-level fast path.
- Clean up style issues so it passes the repo checks.

### 5. Validation

- Add a pytest backend switch that can force either implementation for an
  entire test session.
- Run the existing test suite twice under `just test`:
  - once with `--backend python`
  - once with `--backend rust`
- Keep the ring and matrix tests as exact-behavior checks.
- Keep the SNF tests focused on structural properties and oracle comparisons
  rather than raw `U`, `V`, `S` tuple equality.
- Run `just check`.
- Fix lint, type, and test failures introduced by the release hardening.

## Notes

- The intended contract for `0.2.0` is machine integers, not arbitrary-size
  Python integers.
- Using `i128` internally does not expand the public numeric contract; it only
  avoids incorrect overflow in modular arithmetic on `i64` inputs.
