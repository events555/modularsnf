"""Verify the CRT-style SNF prototype against the existing implementation."""

from __future__ import annotations

import random
from math import gcd

import numpy as np
from crt_snf_prototype import crt_snf, is_unimodular

from modularsnf import smith_normal_form_mod


def check_one(A: np.ndarray, N: int) -> tuple[bool, str]:
    n, m = A.shape
    U, V, S = crt_snf(A, N)

    # 1. Self-certifying factorization equation.
    prod = (U.astype(object) @ A.astype(object) @ V.astype(object)) % N
    if not np.array_equal(prod % N, S % N):
        return False, "U@A@V != S mod N"

    # 2. S diagonal.
    for i in range(n):
        for j in range(m):
            if i != j and int(S[i, j]) % N != 0:
                return False, "S not diagonal"

    # 3. Divisibility chain d_i | d_{i+1} (as ideals: gcd(d_i,N) | gcd(d_{i+1},N)).
    r = min(n, m)
    g = [gcd(int(S[i, i]) % N, N) for i in range(r)]
    for i in range(r - 1):
        if g[i + 1] % g[i] != 0:
            return False, f"divisibility chain broken: {g}"

    # 4. Unimodularity of U and V.
    if not is_unimodular(U, N):
        return False, "U not unimodular"
    if not is_unimodular(V, N):
        return False, "V not unimodular"

    # 5. Diagonal matches the oracle (SNF diagonal is unique up to gcd-with-N).
    So, _, _ = smith_normal_form_mod(A.tolist(), modulus=N)
    So = np.array(So, dtype=object)
    g_oracle = [gcd(int(So[i, i]) % N, N) for i in range(r)]
    if g != g_oracle:
        return False, f"diagonal mismatch: mine={g} oracle={g_oracle}"

    return True, "ok"


def main() -> None:
    random.seed(12345)
    np.random.seed(12345)

    moduli = [2, 3, 4, 6, 8, 9, 12, 16, 30, 36, 60, 36, 100, 210, 720, 1024]
    sizes = [1, 2, 3, 4, 5, 6]

    total = 0
    failures = 0
    for N in moduli:
        for n in sizes:
            for _ in range(30):
                m = n  # square; rectangular handled by same padding upstream
                A = np.random.randint(0, N, size=(n, m))
                total += 1
                ok, msg = check_one(A, N)
                if not ok:
                    failures += 1
                    if failures <= 10:
                        print(f"FAIL N={N} shape={A.shape}: {msg}")
                        print(A)

    # Edge cases.
    edge = [
        (np.zeros((3, 3), dtype=int), 12),
        (np.eye(3, dtype=int), 12),
        (np.array([[2, 4, 0], [6, 8, 3], [0, 3, 9]]), 36),
        (np.array([[6]]), 12),
        (np.full((4, 4), 6), 36),
        (np.array([[4, 0], [0, 9]]), 36),
    ]
    for A, N in edge:
        total += 1
        ok, msg = check_one(A, N)
        if not ok:
            failures += 1
            print(f"FAIL edge N={N} shape={A.shape}: {msg}\n{A}")

    print(f"\n{total - failures}/{total} passed, {failures} failures")


if __name__ == "__main__":
    main()
