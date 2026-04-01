#!/usr/bin/env python3
"""Benchmark and profile the modular SNF pipeline.

The default mode reports wall-clock time, a rough ``n^3 / time`` throughput,
and peak Python memory for a sequence of square matrix sizes.

Backend selection is explicit:
- ``--backend rust`` benchmarks the native PyO3 fast path.
- ``--backend python`` benchmarks the pure-Python implementation.
- ``--backend auto`` prefers Rust when available.

For deeper profiling:
- ``--cprofile`` records a Python profiler snapshot for the first size.
- ``--flamegraph`` runs ``py-spy --native`` on a child process so Rust frames
  remain visible.
"""

from __future__ import annotations

import argparse
import cProfile
import random
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Modules:
    """Loaded modularsnf modules and optional Rust entry points."""

    ring_mod: Any
    diagonal_mod: Any
    snf_mod: Any
    ring_matrix_cls: Any
    ring_cls: Any
    smith_normal_form: Any
    rust_ring_cls: Any
    rust_diag_fn: Any
    rust_merge_fn: Any
    rust_snf_fn: Any


@dataclass
class BenchmarkResult:
    """Summary for one benchmarked matrix size."""

    n: int
    elapsed: float
    rough_n3_per_sec: float
    peak_memory_bytes: int
    correct: bool


def bootstrap_repo() -> None:
    """Ensure the repository root is importable."""
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_modules() -> Modules:
    """Import modularsnf lazily after bootstrapping the repo path."""
    bootstrap_repo()

    import modularsnf.diagonal as diagonal_mod
    import modularsnf.ring as ring_mod
    import modularsnf.snf as snf_mod
    from modularsnf.matrix import RingMatrix
    from modularsnf.ring import RingZModN
    from modularsnf.snf import smith_normal_form

    rust_ring_cls: Any = None
    rust_diag_fn: Any = None
    rust_merge_fn: Any = None
    rust_snf_fn: Any = None

    try:
        from modularsnf._rust import RustRingZModN as rust_ring_cls
        from modularsnf._rust import rust_merge_smith_blocks as rust_merge_fn
        from modularsnf._rust import rust_smith_from_diagonal as rust_diag_fn
        from modularsnf._rust import rust_smith_normal_form as rust_snf_fn
    except ImportError:
        pass

    return Modules(
        ring_mod=ring_mod,
        diagonal_mod=diagonal_mod,
        snf_mod=snf_mod,
        ring_matrix_cls=RingMatrix,
        ring_cls=RingZModN,
        smith_normal_form=smith_normal_form,
        rust_ring_cls=rust_ring_cls,
        rust_diag_fn=rust_diag_fn,
        rust_merge_fn=rust_merge_fn,
        rust_snf_fn=rust_snf_fn,
    )


def rust_available(modules: Modules) -> bool:
    """Return whether the native extension is fully importable."""
    return all(
        value is not None
        for value in (
            modules.rust_ring_cls,
            modules.rust_diag_fn,
            modules.rust_merge_fn,
            modules.rust_snf_fn,
        )
    )


def configure_backend(modules: Modules, requested: str) -> str:
    """Select and activate the requested backend."""
    has_rust = rust_available(modules)

    if requested == "rust":
        if not has_rust:
            raise RuntimeError(
                "Rust backend requested but modularsnf._rust is unavailable"
            )
        backend = "rust"
    elif requested == "python":
        backend = "python"
    elif has_rust:
        backend = "rust"
    else:
        backend = "python"

    if backend == "rust":
        modules.ring_mod._RustRing = modules.rust_ring_cls
        modules.diagonal_mod._rust_diag = modules.rust_diag_fn
        modules.diagonal_mod._rust_merge = modules.rust_merge_fn
        modules.snf_mod._rust_snf = modules.rust_snf_fn
    else:
        modules.ring_mod._RustRing = None
        modules.diagonal_mod._rust_diag = None
        modules.diagonal_mod._rust_merge = None
        modules.snf_mod._rust_snf = None

    return backend


def make_random_matrix(
    modules: Modules,
    n: int,
    modulus: int,
    seed: int,
) -> Any:
    """Create a deterministic random square matrix over Z/modulus."""
    rng = random.Random(seed)
    ring = modules.ring_cls(modulus)
    data = [[rng.randint(0, modulus - 1) for _ in range(n)] for _ in range(n)]
    return modules.ring_matrix_cls.from_rows(ring, data)


def check_snf_correctness(A: Any, U: Any, V: Any, S: Any) -> bool:
    """Verify transform validity, diagonal form, and divisibility chain."""
    modulus = A.ring.N
    diag_len = min(S.nrows, S.ncols)

    if not np.array_equal((U @ A @ V).data, S.data):
        return False

    for i in range(S.nrows):
        for j in range(S.ncols):
            if i != j and S.data[i, j] % modulus != 0:
                return False

    for i in range(diag_len - 1):
        cur = int(S.data[i, i]) % modulus
        nxt = int(S.data[i + 1, i + 1]) % modulus
        g = np.gcd(cur, modulus)
        if g != 0 and nxt % g != 0:
            return False

    return True


def run_once(
    modules: Modules,
    n: int,
    modulus: int,
    seed: int,
    verify: bool,
) -> BenchmarkResult:
    """Benchmark one square matrix size."""
    matrix = make_random_matrix(modules, n, modulus, seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    U, V, S = modules.smith_normal_form(matrix)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    correct = check_snf_correctness(matrix, U, V, S) if verify else True
    rough_n3_per_sec = (n**3 / elapsed) if elapsed > 0 else 0.0
    return BenchmarkResult(
        n=n,
        elapsed=elapsed,
        rough_n3_per_sec=rough_n3_per_sec,
        peak_memory_bytes=peak,
        correct=correct,
    )


def fmt_time(seconds: float) -> str:
    """Format seconds into a compact human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.3f}s"


def fmt_bytes(num_bytes: int) -> str:
    """Format bytes into a compact human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes}B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f}KB"
    return f"{num_bytes / 1024**2:.1f}MB"


def fmt_rate(rate: float) -> str:
    """Format the rough ``n^3 / time`` throughput metric."""
    if rate >= 1_000_000_000:
        return f"{rate / 1_000_000_000:.2f}G"
    if rate >= 1_000_000:
        return f"{rate / 1_000_000:.2f}M"
    if rate >= 1_000:
        return f"{rate / 1_000:.2f}K"
    return f"{rate:.0f}"


def print_result(result: BenchmarkResult) -> None:
    """Print the outcome for one matrix size."""
    status = "YES" if result.correct else "NO"
    print(
        f"  n={result.n:<5d} time={fmt_time(result.elapsed):>10s} "
        f"rough n^3/s={fmt_rate(result.rough_n3_per_sec):>8s} "
        f"peak_mem={fmt_bytes(result.peak_memory_bytes):>8s} "
        f"correct={status}"
    )


def print_scaling_summary(results: list[BenchmarkResult]) -> None:
    """Print a compact scaling table and fitted exponent."""
    if not results:
        return

    print(f"\n{'=' * 72}")
    print("  SCALING SUMMARY")
    print(f"{'=' * 72}")
    print(
        f"  {'n':>6s}  {'time':>10s}  {'rough n^3/s':>12s}  "
        f"{'peak_mem':>10s}  {'correct':>7s}"
    )
    print(f"  {'─' * 6}  {'─' * 10}  {'─' * 12}  {'─' * 10}  {'─' * 7}")

    for result in results:
        print(
            f"  {result.n:>6d}  {fmt_time(result.elapsed):>10s}  "
            f"{fmt_rate(result.rough_n3_per_sec):>12s}  "
            f"{fmt_bytes(result.peak_memory_bytes):>10s}  "
            f"{'YES' if result.correct else 'NO':>7s}"
        )

    if len(results) >= 2:
        log_n = np.log([result.n for result in results])
        log_t = np.log([result.elapsed for result in results])
        coeffs = np.polyfit(log_n, log_t, 1)
        print(f"\n  Empirical scaling exponent: n^{coeffs[0]:.2f}")

    print()


def run_cprofile(
    modules: Modules,
    backend: str,
    n: int,
    modulus: int,
    seed: int,
) -> None:
    """Record a Python cProfile snapshot for the first size."""
    print(f"\nRunning cProfile on n={n}, modulus={modulus}...")
    if backend == "rust":
        print(
            "Note: cProfile will mostly see the Python wrapper around native code."
        )

    matrix = make_random_matrix(modules, n, modulus, seed)
    profiler = cProfile.Profile()
    profiler.enable()
    modules.smith_normal_form(matrix)
    profiler.disable()

    outfile = f"profile_n{n}_mod{modulus}_{backend}.prof"
    profiler.dump_stats(outfile)
    print(f"Saved to {outfile}")
    print(f"Explore with: python -m pstats {outfile}")
    print(f"  or: pip install snakeviz && snakeviz {outfile}")


def run_flamegraph(
    backend: str,
    n: int,
    modulus: int,
    seed: int,
) -> None:
    """Generate a native flamegraph with py-spy."""
    py_spy = shutil.which("py-spy")
    if py_spy is None:
        venv_spy = Path(sys.prefix) / "bin" / "py-spy"
        if venv_spy.exists():
            py_spy = str(venv_spy)
    if py_spy is None:
        raise RuntimeError("py-spy not found. Install with: pip install py-spy")

    child_script = f"""
import random
import sys
import time

repo_root = {str(REPO_ROOT)!r}
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import modularsnf.diagonal as diagonal_mod
import modularsnf.ring as ring_mod
import modularsnf.snf as snf_mod
from modularsnf.matrix import RingMatrix
from modularsnf.ring import RingZModN
from modularsnf.snf import smith_normal_form

USE_RUST = {backend == "rust"!r}
if not USE_RUST:
    ring_mod._RustRing = None
    diagonal_mod._rust_diag = None
    diagonal_mod._rust_merge = None
    snf_mod._rust_snf = None

rng = random.Random({seed})
ring = RingZModN({modulus})
data = [[rng.randint(0, {modulus - 1}) for _ in range({n})] for _ in range({n})]
A = RingMatrix.from_rows(ring, data)

smith_normal_form(A)

t0 = time.perf_counter()
smith_normal_form(A)
dt = time.perf_counter() - t0
iters = max(1, int(5.0 / max(dt, 1e-6)))
print(f"Running {{iters}} iterations ({{dt*1000:.1f}}ms each)...", flush=True)
for _ in range(iters):
    smith_normal_form(A)
"""

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
    ) as handle:
        handle.write(child_script)
        child_path = handle.name

    outfile = f"flamegraph_n{n}_mod{modulus}_{backend}.svg"
    cmd = [
        py_spy,
        "record",
        "--native",
        "--output",
        outfile,
        "--rate",
        "197",
        "--",
        sys.executable,
        child_path,
    ]

    try:
        print(f"\nGenerating flamegraph for n={n}, modulus={modulus}...")
        print(f"Using py-spy at: {py_spy}")
        subprocess.run(cmd, check=True, text=True)
    finally:
        Path(child_path).unlink(missing_ok=True)

    outpath = Path(outfile)
    if not outpath.exists() or outpath.stat().st_size == 0:
        raise RuntimeError("flamegraph was not generated")
    print(f"Wrote flamegraph to {outfile}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark and profile the modular SNF pipeline."
    )
    parser.add_argument(
        "--sizes",
        default="10,20,50,100,200",
        help="Comma-separated square matrix sizes",
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=12,
        help="Ring modulus N (default: 12)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "python", "rust"),
        default="auto",
        help="Execution backend (default: auto)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Stop after a run exceeds this wall-clock time in seconds",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the structural SNF correctness check",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Dump a cProfile .prof file for the first size",
    )
    parser.add_argument(
        "--flamegraph",
        action="store_true",
        help="Generate a py-spy flamegraph SVG for the first size",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    sizes = [
        int(part.strip()) for part in args.sizes.split(",") if part.strip()
    ]

    modules = load_modules()
    backend = configure_backend(modules, args.backend)

    print(
        f"Benchmarking SNF: sizes={sizes}, modulus={args.modulus}, "
        f"seed={args.seed}, backend={backend}"
    )
    if backend == "rust":
        print("Note: detailed Python-level counters are intentionally omitted.")
        print(
            "      Use --flamegraph for native profiling and --backend python"
        )
        print("      if you specifically want the pure-Python path.")

    if args.cprofile:
        run_cprofile(modules, backend, sizes[0], args.modulus, args.seed)
        return

    if args.flamegraph:
        run_flamegraph(backend, sizes[0], args.modulus, args.seed)
        return

    verify = not args.skip_check
    results: list[BenchmarkResult] = []
    for n in sizes:
        result = run_once(modules, n, args.modulus, args.seed, verify)
        print_result(result)
        results.append(result)
        if result.elapsed > args.timeout:
            print(
                f"  Stopping after n={n}: exceeded timeout {args.timeout:.1f}s"
            )
            break

    print_scaling_summary(results)


if __name__ == "__main__":
    main()
