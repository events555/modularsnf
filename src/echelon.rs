//! Row-echelon utilities — Rust port of modularsnf/echelon.py.

use numpy::ndarray::Array2;

use crate::ring::{posmod_i128, RustRingZModN};

/// Apply a 2x2 row transform to rows r0, r1 of both matrices.
/// [s t] [row_r0]   [new_r0]
/// [u v] [row_r1] = [new_r1]
#[inline]
fn apply_row_2x2(
    m: &mut Array2<i64>,
    r0: usize,
    r1: usize,
    s: i64,
    t: i64,
    u: i64,
    v: i64,
    n: i64,
) {
    let cols = m.ncols();
    for j in 0..cols {
        let a = m[[r0, j]];
        let b = m[[r1, j]];
        m[[r0, j]] = posmod_i128((s as i128) * (a as i128) + (t as i128) * (b as i128), n);
        m[[r1, j]] = posmod_i128((u as i128) * (a as i128) + (v as i128) * (b as i128), n);
    }
}

/// Apply a 2x2 row transform to two matrices simultaneously.
#[inline]
pub fn apply_row_2x2_pair(
    m1: &mut Array2<i64>,
    m2: &mut Array2<i64>,
    r0: usize,
    r1: usize,
    s: i64,
    t: i64,
    u: i64,
    v: i64,
    n: i64,
) {
    apply_row_2x2(m1, r0, r1, s, t, u, v, n);
    apply_row_2x2(m2, r0, r1, s, t, u, v, n);
}

/// Lemma 3.1: row-echelon form via extended GCD elimination.
/// Returns (U, T, rank).
pub fn lemma_3_1(a: &Array2<i64>, ring: &RustRingZModN) -> (Array2<i64>, Array2<i64>, usize) {
    let n_mod = ring.n();
    let n_rows = a.nrows();
    let n_cols = a.ncols();

    let mut u = Array2::<i64>::eye(n_rows);
    let mut t = a.clone();

    let mut r = 0usize;

    for k in 0..n_cols {
        if r >= n_rows {
            break;
        }

        for i in (r + 1)..n_rows {
            let a_val = t[[r, k]];
            let b_val = t[[i, k]];

            if ring.is_zero_internal(b_val) {
                continue;
            }

            let (_, s, tv, uv, v) = ring.gcdex_internal(a_val, b_val);
            apply_row_2x2_pair(&mut t, &mut u, r, i, s, tv, uv, v, n_mod);
        }

        if !ring.is_zero_internal(t[[r, k]]) {
            r += 1;
        }
    }

    (u, t, r)
}

/// Index-1 reduction on first k columns (Storjohann 7.3 step 9).
/// Returns (U, T).
pub fn index1_reduce_on_columns(
    a: &Array2<i64>,
    k: usize,
    ring: &RustRingZModN,
) -> (Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();

    let mut u = Array2::<i64>::eye(n);
    let mut t = a.clone();

    for j in 1..k {
        let sj = t[[j, j]];
        for i in 0..j {
            let x = t[[i, j]];
            if ring.is_zero_internal(x) {
                continue;
            }
            let rem = {
                let b_ass = ring.gcd_internal(sj, 0);
                let x_val = ((x % n_mod) + n_mod) % n_mod;
                if b_ass == 0 {
                    x_val
                } else {
                    x_val % b_ass
                }
            };
            let diff = ((x - rem) % n_mod + n_mod) % n_mod;
            let quo = ring.div_internal(diff, sj).unwrap_or(0);
            let phi = ((-quo) % n_mod + n_mod) % n_mod;

            let cols = t.ncols();
            for c in 0..cols {
                t[[i, c]] = posmod_i128(
                    (t[[i, c]] as i128) + (phi as i128) * (t[[j, c]] as i128),
                    n_mod,
                );
                u[[i, c]] = posmod_i128(
                    (u[[i, c]] as i128) + (phi as i128) * (u[[j, c]] as i128),
                    n_mod,
                );
            }
        }
    }

    (u, t)
}
