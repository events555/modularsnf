#!/usr/bin/env python3
"""Microbenchmarks for modular arithmetic cost in the SNF pipeline.

Suites:
  matmul   -- Break down (A @ B) % N into matmul vs mod cost.
  scalar   -- Scalar ring ops (add, mul, gcd, gcdex) via Python and Rust.
  dtype    -- Compare mod reduction across int16 / int32 / int64.
  growth   -- Characterize intermediate accumulator values for lazy reduction
              safety analysis.  Includes both synthetic and real-SNF measurements.
  strategy -- Compare eager vs lazy reduction in pure-Python loops.

Usage:
  python scripts/bench_modulus.py                          # run all suites
  python scripts/bench_modulus.py --suite matmul growth    # selected suites
  python scripts/bench_modulus.py --suite growth --sizes 8,16,32 --moduli 7,127,255
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

I64_MAX = (1 << 63) - 1
I128_MAX = (1 << 127) - 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bootstrap_repo() -> None:
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def fmt_time(secs: float) -> str:
    if secs < 1e-6:
        return f"{secs * 1e9:.1f}ns"
    if secs < 1e-3:
        return f"{secs * 1e6:.1f}us"
    if secs < 1.0:
        return f"{secs * 1e3:.1f}ms"
    return f"{secs:.3f}s"


def fmt_int(n: int | float) -> str:
    if n >= 1e15:
        return f"{n:.2e}"
    return f"{int(n):,}"


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]


def _timeit(fn, iters: int) -> tuple[float, float]:
    """Return (mean, std) of *fn()* over *iters* repetitions."""
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Suite 1: matmul breakdown
# ---------------------------------------------------------------------------

def suite_matmul(sizes: list[int], moduli: list[int], iters: int, seed: int) -> None:
    print("\n=== MATMUL COST BREAKDOWN: (A @ B) % N ===")
    print(f"{'size':>6} {'N':>6} {'total':>10} {'matmul':>10} {'mod':>10} {'mod%':>7}")
    print(f"{'----':>6} {'--':>6} {'-----':>10} {'------':>10} {'---':>10} {'----':>7}")

    rng = np.random.default_rng(seed)
    for size in sizes:
        for N in moduli:
            A = rng.integers(0, N, size=(size, size), dtype=np.int64)
            B = rng.integers(0, N, size=(size, size), dtype=np.int64)

            # Warmup
            _ = (A @ B) % N

            # Time combined
            total_mean, _ = _timeit(lambda: (A @ B) % N, iters)

            # Time matmul only
            matmul_mean, _ = _timeit(lambda: A @ B, iters)

            # Time mod only (on a pre-computed product)
            C = A @ B
            mod_mean, _ = _timeit(lambda: C % N, iters)

            mod_frac = mod_mean / total_mean * 100 if total_mean > 0 else 0
            print(
                f"{size:>6} {N:>6} {fmt_time(total_mean):>10} "
                f"{fmt_time(matmul_mean):>10} {fmt_time(mod_mean):>10} "
                f"{mod_frac:>6.1f}%"
            )


# ---------------------------------------------------------------------------
# Suite 2: scalar ring ops
# ---------------------------------------------------------------------------

def suite_scalar(moduli: list[int], count: int = 100_000) -> None:
    bootstrap_repo()
    from modularsnf.ring import RingZModN

    print(f"\n=== SCALAR RING OPS ({count:,} pairs per measurement) ===")

    # Check Rust availability
    try:
        from modularsnf._rust import RustRingZModN
        has_rust = True
    except ImportError:
        RustRingZModN = None
        has_rust = False

    backends = ["python"]
    if has_rust:
        backends.append("rust")

    print(f"{'backend':>8} {'N':>6} {'add':>10} {'mul':>10} {'gcd':>10} {'gcdex':>10}")
    print(f"{'-------':>8} {'--':>6} {'---':>10} {'---':>10} {'---':>10} {'-----':>10}")

    rng = np.random.default_rng(42)

    for N in moduli:
        pairs_a = rng.integers(0, N, size=count).tolist()
        pairs_b = rng.integers(0, N, size=count).tolist()

        for backend in backends:
            if backend == "rust":
                ring = RustRingZModN(N)
            else:
                ring = RingZModN(N)
                # Force Python path
                ring._rust = None

            ops = {}
            for op_name in ("add", "mul", "gcd", "gcdex"):
                fn = getattr(ring, op_name)
                t0 = time.perf_counter()
                for a, b in zip(pairs_a, pairs_b):
                    fn(a, b)
                elapsed = time.perf_counter() - t0
                ops[op_name] = elapsed / count * 1e9  # ns/op

            print(
                f"{backend:>8} {N:>6} "
                f"{ops['add']:>8.0f}ns {ops['mul']:>8.0f}ns "
                f"{ops['gcd']:>8.0f}ns {ops['gcdex']:>8.0f}ns"
            )


# ---------------------------------------------------------------------------
# Suite 3: dtype comparison
# ---------------------------------------------------------------------------

def suite_dtype(sizes: list[int], moduli: list[int], iters: int, seed: int) -> None:
    print("\n=== DTYPE COMPARISON: time for C % N with different dtypes ===")
    print(f"{'size':>6} {'N':>6} {'int16':>10} {'int32':>10} {'int64':>10}")
    print(f"{'----':>6} {'--':>6} {'-----':>10} {'-----':>10} {'-----':>10}")

    rng = np.random.default_rng(seed)
    for size in sizes:
        for N in moduli:
            # C represents a matmul result: values up to (N-1)^2 * size
            max_val = min((N - 1) ** 2 * size, I64_MAX)

            times = {}
            for dtype in (np.int16, np.int32, np.int64):
                info = np.iinfo(dtype)
                if max_val > info.max:
                    times[dtype.__name__] = "overflow"
                    continue
                C = rng.integers(0, min(max_val + 1, info.max), size=(size, size), dtype=dtype)
                _ = C % np.dtype(dtype).type(N)  # warmup
                mean, _ = _timeit(lambda C=C, dt=dtype: C % dt(N), iters)
                times[dtype.__name__] = fmt_time(mean)

            print(
                f"{size:>6} {N:>6} {times.get('int16', 'n/a'):>10} "
                f"{times.get('int32', 'n/a'):>10} {times.get('int64', 'n/a'):>10}"
            )


# ---------------------------------------------------------------------------
# Suite 4: accumulator growth characterization
# ---------------------------------------------------------------------------

def suite_growth(sizes: list[int], moduli: list[int], seed: int) -> None:
    print("\n=== ACCUMULATOR GROWTH ANALYSIS ===")
    print("Determines whether lazy reduction (accumulate without mod, reduce once) is safe.\n")

    # Part A: Theoretical bounds
    print("--- Part A: Theoretical worst-case ---")
    print(
        f"{'size':>6} {'N':>6} {'max_product':>14} {'worst_accum':>14} "
        f"{'safe_k_i64':>14} {'safe_k_i128':>14}"
    )
    print(
        f"{'----':>6} {'--':>6} {'───────────':>14} {'───────────':>14} "
        f"{'──────────':>14} {'───────────':>14}"
    )
    for size in sizes:
        for N in moduli:
            max_product = (N - 1) ** 2
            worst_accum = max_product * size
            safe_k_i64 = I64_MAX // max_product if max_product > 0 else float("inf")
            safe_k_i128 = I128_MAX // max_product if max_product > 0 else float("inf")
            print(
                f"{size:>6} {N:>6} {fmt_int(max_product):>14} {fmt_int(worst_accum):>14} "
                f"{fmt_int(safe_k_i64):>14} {fmt_int(safe_k_i128):>14}"
            )

    # Part B: Empirical from random matmul
    print("\n--- Part B: Actual max accumulator from random matmul (manual dot product) ---")
    print(
        f"{'size':>6} {'N':>6} {'actual_max':>14} {'theoretical':>14} "
        f"{'ratio':>8} {'fits_i64':>10}"
    )
    print(
        f"{'----':>6} {'--':>6} {'──────────':>14} {'───────────':>14} "
        f"{'─────':>8} {'────────':>10}"
    )

    rng = np.random.default_rng(seed)
    # Only run manual dot product on small sizes (it's O(n^3) in Python)
    small_sizes = [s for s in sizes if s <= 64]
    for size in small_sizes:
        for N in moduli:
            A = rng.integers(0, N, size=(size, size), dtype=np.int64)
            B = rng.integers(0, N, size=(size, size), dtype=np.int64)

            max_acc = 0
            for i in range(size):
                for j in range(size):
                    acc = 0
                    for k in range(size):
                        acc += int(A[i, k]) * int(B[k, j])
                        max_acc = max(max_acc, abs(acc))

            theoretical = (N - 1) ** 2 * size
            ratio = max_acc / theoretical if theoretical > 0 else 0
            fits_i64 = "yes" if max_acc <= I64_MAX else "NO"
            print(
                f"{size:>6} {N:>6} {fmt_int(max_acc):>14} {fmt_int(theoretical):>14} "
                f"{ratio:>7.2f}x {fits_i64:>10}"
            )

    # Part C: Empirical from real SNF computation
    print("\n--- Part C: Actual max accumulator during real SNF computation ---")
    _measure_snf_accumulators(small_sizes, moduli, seed)


def _measure_snf_accumulators(sizes: list[int], moduli: list[int], seed: int) -> None:
    """Monkey-patch RingMatrix to track accumulator growth during actual SNF."""
    bootstrap_repo()
    from modularsnf.matrix import RingMatrix
    from modularsnf.ring import RingZModN
    from modularsnf.snf import smith_normal_form

    max_accumulator = 0
    matmul_count = 0

    def _tracking_matmul(self, other):
        """Replacement __matmul__ that tracks accumulator growth."""
        nonlocal max_accumulator, matmul_count
        N = self.ring.N
        A = self.data
        B = other.data
        rows, inner = A.shape
        _, cols = B.shape

        # Track growth: manual dot product on a sample of output elements
        # (full tracking would be too slow for larger matrices)
        sample_limit = min(rows * cols, 64)
        sample_indices = []
        step = max(1, (rows * cols) // sample_limit)
        for idx in range(0, rows * cols, step):
            sample_indices.append((idx // cols, idx % cols))
            if len(sample_indices) >= sample_limit:
                break

        for i, j in sample_indices:
            acc = 0
            for k in range(inner):
                acc += int(A[i, k]) * int(B[k, j])
                if abs(acc) > max_accumulator:
                    max_accumulator = abs(acc)

        matmul_count += 1
        # Perform the actual matmul
        result_data = (A @ B) % N
        return RingMatrix(self.ring, result_data, _assume_reduced=True)

    print(
        f"{'size':>6} {'N':>6} {'matmuls':>8} {'max_accum':>14} "
        f"{'fits_i64':>10} {'fits_i128':>10}"
    )
    print(
        f"{'----':>6} {'--':>6} {'───────':>8} {'─────────':>14} "
        f"{'────────':>10} {'─────────':>10}"
    )

    # Only run on small sizes since we're doing per-element tracking
    bench_sizes = [s for s in sizes if s <= 32]

    original_matmul = RingMatrix.__matmul__
    try:
        RingMatrix.__matmul__ = _tracking_matmul

        rng_py = __import__("random").Random(seed)
        for size in bench_sizes:
            for N in moduli:
                max_accumulator = 0
                matmul_count = 0

                ring = RingZModN(N)
                data = [[rng_py.randint(0, N - 1) for _ in range(size)] for _ in range(size)]
                A = RingMatrix.from_rows(ring, data)
                try:
                    smith_normal_form(A)
                except Exception:
                    print(f"{size:>6} {N:>6}  (SNF failed, skipping)")
                    continue

                fits_i64 = "yes" if max_accumulator <= I64_MAX else "NO"
                fits_i128 = "yes" if max_accumulator <= I128_MAX else "NO"
                print(
                    f"{size:>6} {N:>6} {matmul_count:>8} "
                    f"{fmt_int(max_accumulator):>14} "
                    f"{fits_i64:>10} {fits_i128:>10}"
                )
    finally:
        RingMatrix.__matmul__ = original_matmul


# ---------------------------------------------------------------------------
# Suite 5: eager vs lazy reduction strategy comparison
# ---------------------------------------------------------------------------

def suite_strategy(sizes: list[int], moduli: list[int], iters: int, seed: int) -> None:
    print("\n=== REDUCTION STRATEGY COMPARISON ===")
    print("Pure-Python loops to isolate mod cost. Limited to small sizes.\n")

    # Cap at 48 for pure-Python loops
    bench_sizes = [s for s in sizes if s <= 48]
    if not bench_sizes:
        print("  (all sizes > 48, skipping pure-Python strategy comparison)")
        return

    print("--- Pure-Python loops (isolating mod overhead) ---")
    print(
        f"{'size':>6} {'N':>6} {'eager':>10} {'lazy':>10} "
        f"{'speedup':>8}"
    )
    print(
        f"{'----':>6} {'--':>6} {'─────':>10} {'────':>10} "
        f"{'───────':>8}"
    )

    rng = np.random.default_rng(seed)
    for size in bench_sizes:
        for N in moduli:
            A = rng.integers(0, N, size=(size, size), dtype=np.int64)
            B = rng.integers(0, N, size=(size, size), dtype=np.int64)

            # Convert to Python lists for pure-Python loops
            Al = A.tolist()
            Bl = B.tolist()

            def eager():
                C = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        acc = 0
                        for k in range(size):
                            acc = (acc + Al[i][k] * Bl[k][j]) % N
                        C[i][j] = acc
                return C

            def lazy():
                C = [[0] * size for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        acc = 0
                        for k in range(size):
                            acc += Al[i][k] * Bl[k][j]
                        C[i][j] = acc % N
                return C

            # Warmup
            eager()
            lazy()

            eager_mean, _ = _timeit(eager, max(1, iters // 2))
            lazy_mean, _ = _timeit(lazy, max(1, iters // 2))

            speedup = eager_mean / lazy_mean if lazy_mean > 0 else 0
            print(
                f"{size:>6} {N:>6} {fmt_time(eager_mean):>10} "
                f"{fmt_time(lazy_mean):>10} {speedup:>7.2f}x"
            )

    # Also show numpy-level for reference
    print("\n--- NumPy-level comparison ---")
    print(
        f"{'size':>6} {'N':>6} {'(A@B)%N':>10} {'A@B only':>10} "
        f"{'mod_frac':>9}"
    )
    print(
        f"{'----':>6} {'--':>6} {'───────':>10} {'───────':>10} "
        f"{'────────':>9}"
    )
    for size in sizes:
        for N in moduli:
            A = rng.integers(0, N, size=(size, size), dtype=np.int64)
            B = rng.integers(0, N, size=(size, size), dtype=np.int64)
            _ = (A @ B) % N
            total_mean, _ = _timeit(lambda: (A @ B) % N, iters)
            matmul_mean, _ = _timeit(lambda: A @ B, iters)
            mod_frac = (total_mean - matmul_mean) / total_mean * 100 if total_mean > 0 else 0
            print(
                f"{size:>6} {N:>6} {fmt_time(total_mean):>10} "
                f"{fmt_time(matmul_mean):>10} {mod_frac:>8.1f}%"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SUITES = ("matmul", "scalar", "dtype", "growth", "strategy")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmarks for modular arithmetic cost."
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=list(SUITES) + ["all"],
        default=["all"],
        help="Which benchmark suite(s) to run.",
    )
    parser.add_argument(
        "--sizes", type=str, default="8,16,32,64,128",
        help="Comma-separated matrix sizes.",
    )
    parser.add_argument(
        "--moduli", type=str, default="7,12,127,255",
        help="Comma-separated modulus values.",
    )
    parser.add_argument("--iters", type=int, default=5, help="Timing repetitions.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()

    sizes = _parse_int_list(args.sizes)
    moduli = _parse_int_list(args.moduli)
    run_suites = set(SUITES) if "all" in args.suite else set(args.suite)

    print(f"Sizes: {sizes}  Moduli: {moduli}  Iters: {args.iters}  Seed: {args.seed}")

    if "matmul" in run_suites:
        suite_matmul(sizes, moduli, args.iters, args.seed)

    if "scalar" in run_suites:
        suite_scalar(moduli)

    if "dtype" in run_suites:
        suite_dtype(sizes, moduli, args.iters, args.seed)

    if "growth" in run_suites:
        suite_growth(sizes, moduli, args.seed)

    if "strategy" in run_suites:
        suite_strategy(sizes, moduli, args.iters, args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()
