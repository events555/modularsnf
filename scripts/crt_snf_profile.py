"""Profile CRT-style SNF prototype vs the existing implementation.

Compares three code paths at increasing n for a composite modulus:
  * existing pipeline, Rust backend (the current production fast path)
  * existing pipeline, pure-Python backend
  * CRT prototype (pure-Python, object dtype)
"""

from __future__ import annotations

import time

import numpy as np
from crt_snf_prototype import crt_snf

import modularsnf.diagonal as diagonal_mod
import modularsnf.ring as ring_mod
import modularsnf.snf as snf_mod
from modularsnf import smith_normal_form_mod

# Capture real Rust hooks so we can toggle the backend.
_R = {
    "ring": ring_mod._RustRing,
    "diag": diagonal_mod._rust_diag,
    "merge": diagonal_mod._rust_merge,
    "snf": snf_mod._rust_snf,
}


def set_backend(rust: bool) -> None:
    ring_mod._RustRing = _R["ring"] if rust else None
    diagonal_mod._rust_diag = _R["diag"] if rust else None
    diagonal_mod._rust_merge = _R["merge"] if rust else None
    snf_mod._rust_snf = _R["snf"] if rust else None


def timeit(fn, *args, repeat: int = 3) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    N = 720  # 2^4 * 3^2 * 5  — composite with zero divisors
    sizes = [4, 8, 16, 24, 32, 48, 64]
    rng = np.random.default_rng(0)

    print(f"modulus N = {N}  (factor = 2^4 * 3^2 * 5)")
    print(f"{'n':>4} | {'rust (ms)':>12} | {'python (ms)':>12} | {'crt (ms)':>12}")
    print("-" * 52)

    for n in sizes:
        A = rng.integers(0, N, size=(n, n))
        Al = A.tolist()

        set_backend(True)
        t_rust = timeit(lambda: smith_normal_form_mod(Al, modulus=N))

        set_backend(False)
        t_py = timeit(lambda: smith_normal_form_mod(Al, modulus=N), repeat=1)

        t_crt = timeit(lambda: crt_snf(A, N), repeat=1)

        print(
            f"{n:>4} | {t_rust*1e3:>12.2f} | {t_py*1e3:>12.2f} | {t_crt*1e3:>12.2f}"
        )

    set_backend(True)


if __name__ == "__main__":
    main()
