# Agent Instructions

This repository implements the deterministic Smith Normal Form algorithms described in Arne Storjohann's PhD Dissertation (ETH No. 13922).

The following guidelines outline the coding standards, algorithmic sources of truth, and architectural boundaries for this project.

## Algorithmic Source of Truth

The `README.md` file serves as the **primary requirements specification**. It details the specific Lemmas, Propositions, and the 4-stage pipeline required to implement the solver.

* **Strict Adherence:** Implement algorithms exactly as described in the `README.md`. Do not substitute generic Smith Normal Form algorithms (e.g., standard Gaussian elimination) unless explicitly requested.
* **Pipeline Hierarchy:** Respect the dependency chain: Triangularization $\to$ Band Reduction $\to$ Diagonalization $\to$ SNF.
* **Notation:** Variable names in code must correspond to the mathematical notation used in the `README.md` (e.g., matrices `A`, `U`, `V`, `S`).

## Code Style & Naming

Follow the **Google Python Style Guide**, with one specific exception regarding variable names.

* **Mathematical Naming Exception:** Single-letter variable names are **permitted and encouraged** when they match the mathematical notation of the reference paper/README (e.g., `U`, `V`, `A`, `n`, `m`). Do not force verbose names like `left_transform_matrix`.
* **Formatting:**
    * Use 4 spaces for indentation.
    * Soft line limit of 80 characters (allow exceptions for complex math formulas).
    * No whitespace inside parentheses.
* **Type Annotations:** Mandatory for all function signatures. Use Python 3.10+ syntax (e.g., `int | None`, `list[int]`).
* **Strings:** Use f-strings exclusively.

## Implementation Constraints

### Ring Arithmetic
All operations must occur over the ring $\mathbb{Z}/N\mathbb{Z}$.
* **No Floating Point:** Never use standard division `/` or floats.
* **Primitives:** Use the `modularsnf.ring` module for all arithmetic.
    * Use `ring.div()` or `ring.quo()` for division.
    * Use `ring.gcdex()` for atomic reductions.
* **Zero Divisors:** Use `ring.stab()` (Stabilizer) when handling zero divisors, as described in **Lemma 1.1** of the README.

### Testing
* **Verification:** Where possible, validate results against SymPy using integer-to-modular projection.
* **Structural Assertions:** Tests should verify mathematical properties (e.g., "Matrix is upper triangular", "Determinant is a unit") rather than just checking for equality against a hardcoded array.

## File Structure & Organization

Code should be placed in the module corresponding to its algorithmic phase:

* `modularsnf/ring.py`: Ring primitives (`Gcdex`, `Stab`, `Div`, `Ann`).
* `modularsnf/matrix.py`: The `RingMatrix` data structure.
* `modularsnf/echelon.py`: **Phase 1** (Triangularization / Lemma 3.1).
* `modularsnf/snf.py`: **Phase 2** (Band Reduction) and **Phase 4** (Master Algorithm / Lemma 7.14).
* `modularsnf/diagonal.py`: **Phase 3** (Diagonalization / Prop 7.7 & 7.12).