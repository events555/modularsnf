//! Performance characterization of mod reduction strategies and merge costs.
//!
//! Run: cargo bench -p modularsnf --bench bench_barrett -- --warm-up-time 1 --measurement-time 3

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use modularsnf::diagonal::{matmul_mod, merge_raw, smith_from_diagonal};
use modularsnf::ring::RingZModN;
use modularsnf::snf::smith_square;

// ---------------------------------------------------------------------------
// Barrett reduction: scalar and SIMD implementations
// ---------------------------------------------------------------------------

/// Precomputed Barrett constants for a given modulus.
struct Barrett {
    n: i64,
    /// m = floor(2^s / n)
    m: u64,
    /// shift amount
    s: u32,
}

impl Barrett {
    fn new(n: i64) -> Self {
        assert!(n > 1 && n < (1 << 16));
        // Choose s such that 2^s / n has enough precision.
        // For n < 2^16, s = 48 gives m < 2^32, and products fit in u64.
        let s = 48u32;
        let m = ((1u128 << s) / n as u128) as u64;
        Barrett { n, m, s }
    }

    /// Barrett reduction: compute x mod n for x in [0, 2*(n-1)^2].
    #[inline(always)]
    fn reduce_unsigned(&self, x: u64) -> i64 {
        let q = ((x as u128 * self.m as u128) >> self.s) as u64;
        let mut r = x - q * self.n as u64;
        if r >= self.n as u64 {
            r -= self.n as u64;
        }
        r as i64
    }

    /// Barrett reduction for signed values in [-(n-1)^2, 2*(n-1)^2].
    /// Shifts input to unsigned, reduces, then adjusts.
    #[inline(always)]
    fn reduce_signed(&self, x: i64) -> i64 {
        // Shift to non-negative range by adding n * ceil((n-1)^2 / n)
        // For simplicity, add n^2 (always enough for our range).
        let n = self.n;
        let shifted = (x + n * n) as u64;
        self.reduce_unsigned(shifted)
    }
}

// ---------------------------------------------------------------------------
// Micro-benchmark: mod reduction methods on batches of values
// ---------------------------------------------------------------------------

fn bench_mod_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("mod_reduction");
    let n: i64 = 127;
    let ring = RingZModN::new(n).unwrap();
    let barrett = Barrett::new(n);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate test values in the range that apply_row_2x2 produces:
    // s*a + t*b where s,a,t,b ∈ [0, n). Range: [-(n-1)^2, 2*(n-1)^2].
    let count = 10_000usize;
    let signed_vals: Vec<i64> = (0..count)
        .map(|_| {
            let s = rng.gen_range(0..n);
            let a = rng.gen_range(0..n);
            let t = rng.gen_range(0..n);
            let b = rng.gen_range(0..n);
            s * a + t * b
        })
        .collect();

    let unsigned_vals: Vec<i64> = (0..count)
        .map(|_| {
            let a = rng.gen_range(0..n);
            let b = rng.gen_range(0..n);
            a * b
        })
        .collect();

    // Method 1: i64 % + conditional add (baseline)
    group.bench_function("i64_mod_signed", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for &v in &signed_vals {
                let r = v % n;
                sum += if r < 0 { r + n } else { r };
            }
            black_box(sum)
        });
    });

    // Method 2: LUT (our current fast path)
    group.bench_function("lut_signed", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for &v in &signed_vals {
                sum += ring.fast_mod(black_box(v));
            }
            black_box(sum)
        });
    });

    // Method 3: Barrett scalar (signed)
    group.bench_function("barrett_scalar_signed", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for &v in &signed_vals {
                sum += barrett.reduce_signed(black_box(v));
            }
            black_box(sum)
        });
    });

    // Method 4: Barrett scalar (unsigned, for products already non-negative)
    group.bench_function("barrett_scalar_unsigned", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for &v in &unsigned_vals {
                sum += barrett.reduce_unsigned(black_box(v as u64));
            }
            black_box(sum)
        });
    });

    // Method 5: i64 % unsigned (for comparison)
    group.bench_function("i64_mod_unsigned", |bench| {
        bench.iter(|| {
            let mut sum = 0i64;
            for &v in &unsigned_vals {
                sum += v % n;
            }
            black_box(sum)
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Simulate the hot loop: (s*a + t*b) mod n for a column
// ---------------------------------------------------------------------------

fn bench_row_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_transform");
    let n: i64 = 127;
    let ring = RingZModN::new(n).unwrap();
    let barrett = Barrett::new(n);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for &cols in &[16, 64, 256, 1024] {
        let row_a: Vec<i64> = (0..cols).map(|_| rng.gen_range(0..n)).collect();
        let row_b: Vec<i64> = (0..cols).map(|_| rng.gen_range(0..n)).collect();
        let s = rng.gen_range(0..n);
        let t = rng.gen_range(0..n);
        let u = rng.gen_range(0..n);
        let v = rng.gen_range(0..n);

        // Method 1: LUT (current implementation)
        group.bench_with_input(
            BenchmarkId::new("lut", cols),
            &cols,
            |bench, _| {
                let mut out_a = row_a.clone();
                let mut out_b = row_b.clone();
                bench.iter(|| {
                    for j in 0..cols {
                        let a = black_box(row_a[j]);
                        let b = black_box(row_b[j]);
                        out_a[j] = ring.fast_mod(s * a + t * b);
                        out_b[j] = ring.fast_mod(u * a + v * b);
                    }
                    black_box(&out_a);
                    black_box(&out_b);
                });
            },
        );

        // Method 2: i64 %
        group.bench_with_input(
            BenchmarkId::new("i64_mod", cols),
            &cols,
            |bench, _| {
                let mut out_a = row_a.clone();
                let mut out_b = row_b.clone();
                bench.iter(|| {
                    for j in 0..cols {
                        let a = black_box(row_a[j]);
                        let b = black_box(row_b[j]);
                        let va = (s * a + t * b) % n;
                        let vb = (u * a + v * b) % n;
                        out_a[j] = if va < 0 { va + n } else { va };
                        out_b[j] = if vb < 0 { vb + n } else { vb };
                    }
                    black_box(&out_a);
                    black_box(&out_b);
                });
            },
        );

        // Method 3: Barrett scalar
        group.bench_with_input(
            BenchmarkId::new("barrett", cols),
            &cols,
            |bench, _| {
                let mut out_a = row_a.clone();
                let mut out_b = row_b.clone();
                bench.iter(|| {
                    for j in 0..cols {
                        let a = black_box(row_a[j]);
                        let b = black_box(row_b[j]);
                        out_a[j] = barrett.reduce_signed(s * a + t * b);
                        out_b[j] = barrett.reduce_signed(u * a + v * b);
                    }
                    black_box(&out_a);
                    black_box(&out_b);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Component benchmarks: merge_raw at various block sizes
// ---------------------------------------------------------------------------

fn random_diagonal_matrix(size: usize, n: i64, rng: &mut ChaCha8Rng) -> Array2<i64> {
    let mut m = Array2::zeros((size, size));
    for i in 0..size {
        m[[i, i]] = rng.gen_range(0..n);
    }
    m
}

fn bench_merge_raw(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_raw");
    let n: i64 = 127;
    let ring = RingZModN::new(n).unwrap();

    for &size in &[1, 2, 4, 8, 16, 32, 64] {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let a = random_diagonal_matrix(size, n, &mut rng);
        let b = random_diagonal_matrix(size, n, &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bench, _| {
                bench.iter(|| merge_raw(black_box(&a), black_box(&b), black_box(&ring)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Component benchmarks: smith_from_diagonal at various sizes
// ---------------------------------------------------------------------------

fn bench_smith_from_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("smith_from_diagonal");
    group.sample_size(10);
    let n: i64 = 127;
    let ring = RingZModN::new(n).unwrap();

    for &size in &[16, 32, 64, 128, 256, 512, 1024] {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let diag = random_diagonal_matrix(size, n, &mut rng);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bench, _| {
                bench.iter(|| smith_from_diagonal(black_box(&diag), black_box(&ring)));
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Full SNF pipeline at various sizes
// ---------------------------------------------------------------------------

fn bench_full_snf(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_snf");
    group.sample_size(10);
    let n: i64 = 127;

    for &size in &[32, 64, 128, 256, 512, 1024] {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let a = Array2::from_shape_fn((size, size), |_| rng.gen_range(0..n));
        let ring = RingZModN::new(n).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &size,
            |bench, _| {
                bench.iter(|| smith_square(black_box(&a), black_box(&ring)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mod_reduction,
    bench_row_transform,
    bench_merge_raw,
    bench_smith_from_diagonal,
    bench_full_snf,
);
criterion_main!(benches);
