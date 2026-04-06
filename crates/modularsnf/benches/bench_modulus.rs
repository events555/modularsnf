//! Benchmarks for modular arithmetic and matrix operations.
//!
//! Run with: cargo bench -p modularsnf
//! Filter:   cargo bench -p modularsnf -- matmul_mod
//!           cargo bench -p modularsnf -- scalar
//!           cargo bench -p modularsnf -- snf

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;

use modularsnf::diagonal::matmul_mod;
use modularsnf::ring::RingZModN;

/// Create a random matrix with entries in [0, n).
fn random_matrix(rows: usize, cols: usize, n: i64, rng: &mut ChaCha8Rng) -> Array2<i64> {
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(0..n))
}

// ---------------------------------------------------------------------------
// Group 1: matmul_mod — the core hot path
// ---------------------------------------------------------------------------

fn bench_matmul_mod(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_mod");
    let sizes: &[usize] = &[8, 16, 32, 64, 128, 256, 512];
    let moduli: &[i64] = &[7, 127, 255];

    for &size in sizes {
        for &n in moduli {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let a = random_matrix(size, size, n, &mut rng);
            let b = random_matrix(size, size, n, &mut rng);

            group.bench_with_input(
                BenchmarkId::new(format!("n{n}"), size),
                &size,
                |bench, _| {
                    bench.iter(|| matmul_mod(black_box(&a), black_box(&b), black_box(n)));
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: Scalar ring operations
// ---------------------------------------------------------------------------

fn bench_scalar_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar");
    let moduli: &[i64] = &[7, 12, 127, 255];
    let count = 10_000usize;

    for &n in moduli {
        let ring = RingZModN::new(n).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let pairs: Vec<(i64, i64)> = (0..count)
            .map(|_| (rng.gen_range(0..n), rng.gen_range(0..n)))
            .collect();

        group.bench_with_input(BenchmarkId::new("add", n), &pairs, |bench, pairs| {
            bench.iter(|| {
                let mut acc = 0i64;
                for &(a, b) in pairs {
                    acc = ring.add(black_box(a), black_box(b));
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("mul", n), &pairs, |bench, pairs| {
            bench.iter(|| {
                let mut acc = 0i64;
                for &(a, b) in pairs {
                    acc = ring.mul(black_box(a), black_box(b));
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("gcd", n), &pairs, |bench, pairs| {
            bench.iter(|| {
                let mut acc = 0i64;
                for &(a, b) in pairs {
                    acc = ring.gcd(black_box(a), black_box(b));
                }
                black_box(acc)
            });
        });

        group.bench_with_input(BenchmarkId::new("gcdex", n), &pairs, |bench, pairs| {
            bench.iter(|| {
                let mut acc = (0i64, 0i64, 0i64, 0i64, 0i64);
                for &(a, b) in pairs {
                    acc = ring.gcdex(black_box(a), black_box(b));
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: Full SNF pipeline
// ---------------------------------------------------------------------------

fn bench_snf(c: &mut Criterion) {
    let mut group = c.benchmark_group("snf");
    // Keep sizes small — SNF is O(n^3) with large constants.
    let sizes: &[usize] = &[8, 16];
    let moduli: &[i64] = &[7, 12, 127];

    for &size in sizes {
        for &n in moduli {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let a = random_matrix(size, size, n, &mut rng);

            group.bench_with_input(
                BenchmarkId::new(format!("n{n}"), size),
                &size,
                |bench, _| {
                    bench.iter(|| {
                        modularsnf::smith_normal_form(black_box(&a), black_box(n)).unwrap()
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_matmul_mod, bench_scalar_ops, bench_snf);
criterion_main!(benches);
