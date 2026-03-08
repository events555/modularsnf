#!/usr/bin/env python3
"""Profile the modular SNF pipeline to characterize performance bottlenecks.

Monkey-patches library functions with lightweight instrumentation (timers and
call counters) and runs across multiple matrix sizes to produce scaling curves.

No library source files are modified.

Usage:
    python scripts/profile_snf.py                           # defaults
    python scripts/profile_snf.py --sizes 50,100,200        # custom sizes
    python scripts/profile_snf.py --modulus 1000             # large modulus
    python scripts/profile_snf.py --sizes 100 --cprofile     # dump .prof file
    python scripts/profile_snf.py --sizes 100 --flamegraph   # py-spy flamegraph
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import random
import shutil
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable when running from the repo root.
# ---------------------------------------------------------------------------
_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

import modularsnf.band as band_mod  # noqa: E402
import modularsnf.diagonal as diag_mod  # noqa: E402
import modularsnf.echelon as ech_mod  # noqa: E402
import modularsnf.snf as snf_mod  # noqa: E402
from modularsnf.matrix import RingMatrix  # noqa: E402
from modularsnf.ring import RingZModN  # noqa: E402
from modularsnf.snf import smith_normal_form  # noqa: E402

# ===================================================================
# Stats accumulator
# ===================================================================


@dataclass
class Stats:
    """Collects timing, call-count, and memory data from a single run."""

    timers: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    matmul_flops: int = 0
    matmul_sizes_small: int = 0  # dim < 64
    matmul_sizes_large: int = 0  # dim >= 64
    stab_iters: list[int] = field(default_factory=list)
    peak_memory_bytes: int = 0

    def reset(self) -> None:
        self.timers.clear()
        self.counts.clear()
        self.matmul_flops = 0
        self.matmul_sizes_small = 0
        self.matmul_sizes_large = 0
        self.stab_iters.clear()
        self.peak_memory_bytes = 0


# ===================================================================
# Wrapper factories
# ===================================================================


def timed_wrapper(
    stats: Stats, name: str, original_fn: Callable
) -> Callable:
    """Wrap *original_fn* to accumulate wall-clock time under *name*."""

    @functools.wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = original_fn(*args, **kwargs)
        stats.timers[name].append(time.perf_counter() - t0)
        return result

    return wrapper


def counted_wrapper(
    stats: Stats, name: str, original_fn: Callable
) -> Callable:
    """Wrap *original_fn* to count invocations only (minimal overhead)."""

    @functools.wraps(original_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        stats.counts[name] += 1
        return original_fn(*args, **kwargs)

    return wrapper


def matmul_wrapper(stats: Stats, original_fn: Callable) -> Callable:
    """Wrap ``__matmul__`` to count calls and accumulate flop estimates."""

    @functools.wraps(original_fn)
    def wrapper(self: RingMatrix, other: RingMatrix) -> RingMatrix:
        n, m, k = self.nrows, self.ncols, other.ncols
        stats.counts["matmul"] += 1
        stats.matmul_flops += n * m * k
        if max(n, m, k) < 64:
            stats.matmul_sizes_small += 1
        else:
            stats.matmul_sizes_large += 1
        return original_fn(self, other)

    return wrapper


def stab_wrapper(stats: Stats, original_fn: Callable) -> Callable:
    """Wrap ``stab`` to record iteration counts per call."""

    @functools.wraps(original_fn)
    def wrapper(self: RingZModN, a: int, b: int, c: int) -> int:
        a_val, b_val, c_val = a % self.N, b % self.N, c % self.N
        target = self.gcd(a_val, self.gcd(b_val, c_val))
        for x in range(self.N):
            candidate = (a_val + x * b_val) % self.N
            current = self.gcd(candidate, c_val)
            if current == target:
                stats.counts["stab"] += 1
                stats.stab_iters.append(x + 1)  # 1-based iteration count
                return x
        raise ValueError(
            f"Stab failed for a={a_val}, b={b_val}, c={c_val} in Z/{self.N}"
        )

    return wrapper


# ===================================================================
# Patch / unpatch
# ===================================================================

# Store originals for restoration.
_originals: list[tuple[Any, str, Any]] = []


def patch_library(stats: Stats) -> None:
    """Monkey-patch library functions with instrumentation."""
    global _originals
    _originals = []

    def _patch_func(module: Any, name: str, wrapper_fn: Callable) -> None:
        original = getattr(module, name)
        _originals.append((module, name, original))
        setattr(module, name, wrapper_fn(stats, name, original))

    def _patch_method(
        cls: type, name: str, wrapper_fn: Callable
    ) -> None:
        original = getattr(cls, name)
        _originals.append((cls, name, original))
        setattr(cls, name, wrapper_fn(stats, name, original))

    # --- Phase timers (module-level functions) ---
    _patch_func(snf_mod, "_smith_square", timed_wrapper)
    _patch_func(ech_mod, "lemma_3_1", timed_wrapper)
    _patch_func(band_mod, "band_reduction", timed_wrapper)
    _patch_func(band_mod, "triang", timed_wrapper)
    _patch_func(band_mod, "shift", timed_wrapper)
    _patch_func(snf_mod, "smith_from_upper_2_banded", timed_wrapper)
    _patch_func(snf_mod, "_step1_split_with_spike", timed_wrapper)
    _patch_func(snf_mod, "_step2_recursive_blocks", timed_wrapper)
    _patch_func(snf_mod, "_step3_permute", timed_wrapper)
    _patch_func(snf_mod, "_step4_smith_on_n_minus_1", timed_wrapper)
    _patch_func(snf_mod, "_step5_to_8_gcd_chain", timed_wrapper)
    _patch_func(snf_mod, "_step9_index_reduction", timed_wrapper)
    _patch_func(diag_mod, "smith_from_diagonal", timed_wrapper)
    _patch_func(diag_mod, "merge_smith_blocks", timed_wrapper)

    # --- Call counters (class methods) ---
    _patch_method(RingZModN, "gcdex", counted_wrapper)
    _patch_method(RingZModN, "gcd", counted_wrapper)
    _patch_method(RingMatrix, "apply_row_2x2", counted_wrapper)
    _patch_method(RingMatrix, "submatrix", counted_wrapper)
    _patch_method(RingMatrix, "copy", counted_wrapper)

    # --- Special wrappers ---
    # stab: replace with iteration-tracking version
    orig_stab = RingZModN.stab
    _originals.append((RingZModN, "stab", orig_stab))
    RingZModN.stab = stab_wrapper(stats, orig_stab)

    # __matmul__: track dimensions and flops
    orig_matmul = RingMatrix.__matmul__
    _originals.append((RingMatrix, "__matmul__", orig_matmul))
    RingMatrix.__matmul__ = matmul_wrapper(stats, orig_matmul)


def unpatch_library() -> None:
    """Restore all original functions."""
    for obj, name, original in reversed(_originals):
        setattr(obj, name, original)
    _originals.clear()


# ===================================================================
# Matrix generation
# ===================================================================


def make_random_matrix(
    n: int, modulus: int, seed: int = 42
) -> RingMatrix:
    """Create a deterministic random n x n matrix over Z/modulus."""
    rng = random.Random(seed)
    ring = RingZModN(modulus)
    data = [[rng.randint(0, modulus - 1) for _ in range(n)] for _ in range(n)]
    return RingMatrix.from_rows(ring, data)


# ===================================================================
# Single run
# ===================================================================


def run_one(
    n: int,
    modulus: int,
    stats: Stats,
    timeout: float = 300.0,
) -> tuple[float, bool]:
    """Run SNF on a random n x n matrix, returning (elapsed, correctness)."""
    stats.reset()
    A = make_random_matrix(n, modulus)

    tracemalloc.start()
    t0 = time.perf_counter()
    U, V, S = smith_normal_form(A)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    stats.peak_memory_bytes = peak

    correct = check_snf_correctness(A, U, V, S)

    return elapsed, correct


def check_snf_correctness(
    A: RingMatrix, U: RingMatrix, V: RingMatrix, S: RingMatrix
) -> bool:
    """Verify SNF result: factorization, diagonal form, and divisibility."""
    N = A.ring.N
    n = min(S.nrows, S.ncols)

    # 1. Factorization: U @ A @ V == S
    if not np.array_equal((U @ A @ V).data, S.data):
        return False

    # 2. S is diagonal
    for i in range(S.nrows):
        for j in range(S.ncols):
            if i != j and S.data[i, j] % N != 0:
                return False

    # 3. Divisibility chain: s_i | s_{i+1} in Z/NZ
    import math
    for i in range(n - 1):
        si = int(S.data[i, i]) % N
        si1 = int(S.data[i + 1, i + 1]) % N
        g = math.gcd(si, N)
        if g != 0 and si1 % g != 0:
            return False

    return True


# ===================================================================
# Report formatting
# ===================================================================


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:,.0f}"
    return f"{n:,}"


def _fmt_time(t: float) -> str:
    if t < 0.001:
        return f"{t * 1_000_000:.0f}us"
    if t < 1.0:
        return f"{t * 1_000:.1f}ms"
    return f"{t:.3f}s"


def _fmt_bytes(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    if b < 1024**2:
        return f"{b / 1024:.1f}KB"
    return f"{b / 1024**2:.1f}MB"


def print_single_report(
    n: int,
    modulus: int,
    elapsed: float,
    correct: bool,
    stats: Stats,
) -> None:
    """Print the detailed breakdown for a single (n, modulus) run."""
    print(f"\n{'=' * 60}")
    print(f"  n={n}  modulus={modulus}  total={_fmt_time(elapsed)}  "
          f"correct={'YES' if correct else 'NO'}")
    print(f"{'=' * 60}")

    # Phase breakdown
    phase_order = [
        "_smith_square",
        "lemma_3_1",
        "band_reduction",
        "triang",
        "shift",
        "smith_from_upper_2_banded",
        "_step1_split_with_spike",
        "_step2_recursive_blocks",
        "_step3_permute",
        "_step4_smith_on_n_minus_1",
        "_step5_to_8_gcd_chain",
        "_step9_index_reduction",
        "smith_from_diagonal",
        "merge_smith_blocks",
    ]

    print("\n  Phase Breakdown:")
    for name in phase_order:
        durations = stats.timers.get(name, [])
        if not durations:
            continue
        total = sum(durations)
        calls = len(durations)
        pct = (total / elapsed * 100) if elapsed > 0 else 0
        indent = "    " if name.startswith("_step") else "  "
        print(
            f"  {indent}{name:<30s}  "
            f"{_fmt_time(total):>10s}  ({pct:5.1f}%)  "
            f"calls: {calls}"
        )

    # Ring operation counts
    print("\n  Ring Operations:")
    for name in ["gcdex", "stab", "gcd"]:
        count = stats.counts.get(name, 0)
        if count:
            print(f"    {name:<20s}  {_fmt_count(count):>12s}")

    # Stab iteration histogram
    if stats.stab_iters:
        iters = stats.stab_iters
        print(f"\n  Stab Iteration Distribution (N={modulus}):")
        print(f"    min={min(iters)}  median={statistics.median(iters):.0f}  "
              f"mean={statistics.mean(iters):.1f}  "
              f"p95={sorted(iters)[int(len(iters) * 0.95)]}"
              f"  max={max(iters)}  total_iters={sum(iters):,}")

    # Matrix operation counts
    print("\n  Matrix Operations:")
    print(f"    matmul (@)          {_fmt_count(stats.counts.get('matmul', 0)):>12s}"
          f"   flops: {_fmt_count(stats.matmul_flops)}")
    print(f"      small (<64)       {_fmt_count(stats.matmul_sizes_small):>12s}")
    print(f"      large (>=64)      {_fmt_count(stats.matmul_sizes_large):>12s}")
    print(f"    apply_row_2x2       {_fmt_count(stats.counts.get('apply_row_2x2', 0)):>12s}")
    print(f"    submatrix           {_fmt_count(stats.counts.get('submatrix', 0)):>12s}")
    print(f"    copy                {_fmt_count(stats.counts.get('copy', 0)):>12s}")

    # Memory
    print(f"\n  Memory: peak {_fmt_bytes(stats.peak_memory_bytes)}")


def print_scaling_summary(
    results: list[dict],
) -> None:
    """Print the scaling summary table and fit exponents."""
    if len(results) < 2:
        return

    print(f"\n{'=' * 80}")
    print("  SCALING SUMMARY")
    print(f"{'=' * 80}")

    # Header
    print(f"  {'n':>6s}  {'time':>10s}  {'matmul_flops':>14s}  "
          f"{'gcdex':>10s}  {'stab':>8s}  {'matmul_calls':>12s}  "
          f"{'peak_mem':>10s}")
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 14}  "
          f"{'─' * 10}  {'─' * 8}  {'─' * 12}  {'─' * 10}")

    for r in results:
        print(
            f"  {r['n']:>6d}  {_fmt_time(r['elapsed']):>10s}  "
            f"{_fmt_count(r['matmul_flops']):>14s}  "
            f"{_fmt_count(r['gcdex']):>10s}  "
            f"{_fmt_count(r['stab']):>8s}  "
            f"{_fmt_count(r['matmul_calls']):>12s}  "
            f"{_fmt_bytes(r['peak_mem']):>10s}"
        )

    # Fit scaling exponents (log-log linear regression)
    valid = [(r['n'], r['elapsed']) for r in results if r['elapsed'] > 1e-6]
    if len(valid) >= 2:
        log_n = np.log([v[0] for v in valid])
        log_t = np.log([v[1] for v in valid])
        coeffs = np.polyfit(log_n, log_t, 1)
        print(f"\n  Empirical scaling exponent (total time): n^{coeffs[0]:.2f}")

    valid_flops = [
        (r['n'], r['matmul_flops']) for r in results if r['matmul_flops'] > 0
    ]
    if len(valid_flops) >= 2:
        log_n = np.log([v[0] for v in valid_flops])
        log_f = np.log([v[1] for v in valid_flops])
        coeffs = np.polyfit(log_n, log_f, 1)
        print(f"  Empirical scaling exponent (matmul flops): n^{coeffs[0]:.2f}")

    valid_gcdex = [
        (r['n'], r['gcdex']) for r in results if r['gcdex'] > 0
    ]
    if len(valid_gcdex) >= 2:
        log_n = np.log([v[0] for v in valid_gcdex])
        log_g = np.log([v[1] for v in valid_gcdex])
        coeffs = np.polyfit(log_n, log_g, 1)
        print(f"  Empirical scaling exponent (gcdex calls): n^{coeffs[0]:.2f}")

    print()


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the modular SNF pipeline."
    )
    parser.add_argument(
        "--sizes",
        default="10,20,50,100,200",
        help="Comma-separated matrix sizes (default: 10,20,50,100,200)",
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=12,
        help="Ring modulus N (default: 12)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-size timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Dump a cProfile .prof file (only first size)",
    )
    parser.add_argument(
        "--flamegraph",
        action="store_true",
        help="Generate a py-spy flamegraph SVG (includes native/Rust frames)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    print(f"Profiling SNF: sizes={sizes}, modulus={args.modulus}, seed={args.seed}")

    # Detect backend
    try:
        from modularsnf._rust import rust_smith_normal_form  # noqa: F401
        backend = "rust"
    except ImportError:
        backend = "python"
    print(f"Backend: {backend}")

    # cProfile mode: just dump a .prof and exit.
    if args.cprofile:
        n = sizes[0]
        print(f"\nRunning cProfile on n={n}, modulus={args.modulus}...")
        A = make_random_matrix(n, args.modulus, seed=args.seed)
        prof = cProfile.Profile()
        prof.enable()
        smith_normal_form(A)
        prof.disable()
        outfile = f"profile_n{n}_mod{args.modulus}.prof"
        prof.dump_stats(outfile)
        print(f"Saved to {outfile}")
        print("Explore with: python -m pstats " + outfile)
        print("  or: pip install snakeviz && snakeviz " + outfile)
        return

    # Flamegraph mode: spawn py-spy on a child process.
    if args.flamegraph:
        py_spy = shutil.which("py-spy")
        if py_spy is None:
            # Check in venv
            venv_spy = Path(sys.prefix) / "bin" / "py-spy"
            if venv_spy.exists():
                py_spy = str(venv_spy)
        if py_spy is None:
            print("ERROR: py-spy not found. Install with: pip install py-spy")
            sys.exit(1)

        n = sizes[0]
        outfile = f"flamegraph_n{n}_mod{args.modulus}.svg"
        print(f"\nGenerating flamegraph for n={n}, modulus={args.modulus}...")
        print(f"Using py-spy at: {py_spy}")

        # Build a child script that runs enough iterations for py-spy
        # to collect meaningful samples (target ~5 seconds).
        child_script = f"""
import sys, random, time
sys.path.insert(0, {str(_repo)!r})
from modularsnf.ring import RingZModN
from modularsnf.matrix import RingMatrix
from modularsnf.snf import smith_normal_form

rng = random.Random({args.seed})
ring = RingZModN({args.modulus})
data = [[rng.randint(0, {args.modulus - 1}) for _ in range({n})] for _ in range({n})]
A = RingMatrix.from_rows(ring, data)

# Warmup
smith_normal_form(A)

# Time one run to estimate iterations needed
t0 = time.perf_counter()
smith_normal_form(A)
dt = time.perf_counter() - t0

iters = max(1, int(5.0 / max(dt, 1e-6)))
print(f"Running {{iters}} iterations ({{dt*1000:.1f}}ms each)...", flush=True)
for _ in range(iters):
    smith_normal_form(A)
"""
        # Write to temp file so py-spy can track it properly.
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write(child_script)
            child_path = f.name

        cmd = [
            py_spy, "record",
            "--native",
            "--output", outfile,
            "--rate", "197",
            "--", sys.executable, child_path,
        ]
        print(f"Command: {' '.join(cmd[:6])} ...")
        subprocess.run(cmd, text=True)
        Path(child_path).unlink(missing_ok=True)
        outpath = Path(outfile)
        if outpath.exists() and outpath.stat().st_size > 0:
            print(f"Wrote flamegraph to {outfile}")
            print(f"Open in browser: file://{outpath.resolve()}")
        else:
            print("ERROR: flamegraph was not generated.")
            sys.exit(1)
        return

    stats = Stats()
    patch_library(stats)

    all_results: list[dict] = []

    try:
        for n in sizes:
            print(f"\n--- Running n={n} ---")
            t0_wall = time.perf_counter()
            elapsed, correct = run_one(n, args.modulus, stats, args.timeout)
            wall = time.perf_counter() - t0_wall

            if wall > args.timeout:
                print(f"  SKIPPED: n={n} exceeded timeout ({args.timeout}s)")
                break

            result = {
                "n": n,
                "elapsed": elapsed,
                "correct": correct,
                "matmul_flops": stats.matmul_flops,
                "gcdex": stats.counts.get("gcdex", 0),
                "stab": stats.counts.get("stab", 0),
                "matmul_calls": stats.counts.get("matmul", 0),
                "peak_mem": stats.peak_memory_bytes,
            }
            all_results.append(result)

            print_single_report(n, args.modulus, elapsed, correct, stats)

        print_scaling_summary(all_results)

    finally:
        unpatch_library()


if __name__ == "__main__":
    main()
