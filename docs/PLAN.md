# PLAN: Feature Parity Validation + PyPI Readiness

All items complete.

## What was delivered

- **Convenience API**: `smith_normal_form_mod(matrix, modulus)` — one-call SNF
  over Z/NZ with input validation and plain-integer output.
- **Parity tests**: Oracle comparison against SymPy (prime, composite, and
  power-of-two moduli), curated regression corpus, and performance sanity
  checks.
- **Packaging**: MIT license, hardened `pyproject.toml` metadata, explicit
  dependency bounds, `setuptools` build, `twine check` clean.
- **CI/CD**: GitHub Actions for lint, type-check, and test matrix; OIDC-based
  publish workflow (TestPyPI on `dev`, PyPI on `main`).
