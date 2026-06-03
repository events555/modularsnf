"""Prototype: CRT-style Smith Normal Form over Z/NZ with transforms.

Non-recursive. Factor N = prod p^e, compute SNF over each local ring Z/p^e
by valuation-pivoted Gaussian elimination (which yields the divisibility
chain for free and unimodular transforms natively), then CRT-recombine the
per-prime transforms into U, V over Z/NZ.

This is a correctness/feasibility prototype in pure Python (object dtype to
avoid overflow), validated against the existing implementation as oracle.
"""

from __future__ import annotations

from math import gcd

import numpy as np


def factorize(n: int) -> dict[int, int]:
    factors: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def pval(x: int, p: int, cap: int) -> int:
    """p-adic valuation of x, capped at cap; x == 0 -> cap."""
    x %= p**cap
    if x == 0:
        return cap
    v = 0
    while v < cap and x % p == 0:
        x //= p
        v += 1
    return v


def egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return (a, 1, 0)
    g, x, y = egcd(b, a % b)
    return (g, y, x - (a // b) * y)


def inv_mod(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    assert g == 1, f"{a} not invertible mod {m}"
    return x % m


def local_snf(A: np.ndarray, p: int, e: int):
    """SNF of A over the local ring Z/p^e via valuation pivoting.

    Returns (U, V, diag_vals) with U @ A @ V == diag(p^vals) mod p^e,
    U (n x n) and V (m x m) unimodular, vals ascending (divisibility chain).
    """
    q = p**e
    M = (A.astype(object)) % q
    n, m = M.shape
    U = np.eye(n, dtype=object)
    V = np.eye(m, dtype=object)
    r = min(n, m)

    for k in range(r):
        # Find pivot in M[k:, k:] with minimal p-adic valuation.
        best = None
        bestval = e + 1
        for i in range(k, n):
            for j in range(k, m):
                if int(M[i, j]) % q != 0:
                    v = pval(int(M[i, j]), p, e)
                    if v < bestval:
                        bestval, best = v, (i, j)
                        if v == 0:
                            break
            if bestval == 0:
                break
        if best is None:
            break  # entire trailing block is zero

        pi, pj = best
        if pi != k:
            M[[k, pi], :] = M[[pi, k], :]
            U[[k, pi], :] = U[[pi, k], :]
        if pj != k:
            M[:, [k, pj]] = M[:, [pj, k]]
            V[:, [k, pj]] = V[:, [pj, k]]

        v = bestval
        pv = p**v
        # Normalize pivot to exactly p^v by scaling row k by the unit inverse.
        pivot = int(M[k, k]) % q
        unit = pivot // pv  # coprime to p
        uinv = inv_mod(unit, q)
        M[k, :] = (M[k, :] * uinv) % q
        U[k, :] = (U[k, :] * uinv) % q

        # Clear column k below the pivot.
        for i in range(k + 1, n):
            if int(M[i, k]) % q != 0:
                c = (int(M[i, k]) % q) // pv  # exact: val(M[i,k]) >= v
                M[i, :] = (M[i, :] - c * M[k, :]) % q
                U[i, :] = (U[i, :] - c * U[k, :]) % q

        # Clear row k to the right of the pivot.
        for j in range(k + 1, m):
            if int(M[k, j]) % q != 0:
                c = (int(M[k, j]) % q) // pv
                M[:, j] = (M[:, j] - c * M[:, k]) % q
                V[:, j] = (V[:, j] - c * V[:, k]) % q

    vals = [pval(int(M[i, i]), p, e) for i in range(r)]
    return U, V, vals


def crt_combine(residues: list[int], moduli: list[int]) -> int:
    """Combine residues mod pairwise-coprime moduli into a value mod prod."""
    x = 0
    M = 1
    for r, m in zip(residues, moduli):
        # solve x ≡ x (mod M), x ≡ r (mod m)
        g, s, _ = egcd(M, m)
        assert g == 1
        x = (x + M * ((r - x) * s % m)) % (M * m)
        M *= m
    return x


def crt_snf(A: np.ndarray, N: int):
    """CRT-style SNF over Z/NZ. Returns (U, V, S) with S = U @ A @ V mod N."""
    A = A.astype(object) % N
    n, m = A.shape
    r = min(n, m)
    fac = factorize(N)
    primes = sorted(fac)
    qs = [p ** fac[p] for p in primes]

    per_prime = {}
    vals = {}
    for p in primes:
        e = fac[p]
        Up, Vp, vp = local_snf(A, p, e)
        per_prime[p] = (Up, Vp)
        vals[p] = vp

    # Global invariant factors d_i = prod_p p^{vals_p[i]} (divides N).
    d = []
    for i in range(r):
        di = 1
        for p in primes:
            di *= p ** vals[p][i]
        d.append(di % N)

    # Unit-normalize each prime's V so all primes realize the SAME global d_i.
    for p in primes:
        e = fac[p]
        q = p**e
        Vp = per_prime[p][1]
        for i in range(r):
            # w = prod_{p'!=p} p'^{vals_{p'}[i]} (a unit mod q)
            w = 1
            for p2 in primes:
                if p2 != p:
                    w *= p2 ** vals[p2][i]
            w %= q
            Vp[:, i] = (Vp[:, i] * w) % q

    # CRT-recombine U and V entrywise across primes.
    U = np.zeros((n, n), dtype=object)
    V = np.zeros((m, m), dtype=object)
    for a in range(n):
        for b in range(n):
            U[a, b] = crt_combine(
                [int(per_prime[p][0][a, b]) for p in primes], qs
            )
    for a in range(m):
        for b in range(m):
            V[a, b] = crt_combine(
                [int(per_prime[p][1][a, b]) for p in primes], qs
            )

    S = np.zeros((n, m), dtype=object)
    for i in range(r):
        S[i, i] = d[i] % N
    return U, V, S


def det_mod(M: np.ndarray, N: int) -> int:
    """Determinant of integer matrix mod N via fraction-free (Bareiss-ish)."""
    M = M.astype(object) % N
    n = M.shape[0]
    if n == 0:
        return 1 % N
    # Plain cofactor for small n is fine for tests; use integer det via numpy on python ints.
    from sympy import Matrix

    return int(Matrix(M.tolist()).det()) % N


def is_unimodular(M: np.ndarray, N: int) -> bool:
    return gcd(det_mod(M, N) % N, N) == 1
