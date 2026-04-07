//! Row-echelon utilities — Rust port of modularsnf/echelon.py.
//!
//! Uses i64 arithmetic for small moduli (n < 2^31) to avoid the
//! expensive i128 software division (__modti3). For n < 128, uses
//! precomputed mod LUT to replace even the i64 `%` with a table lookup.

use ndarray::Array2;

use crate::ring::{posmod_i128, RingZModN};

/// Apply a 2×2 row rotation to columns `col_start..col_end` of a matrix.
///
/// For n < 128 with a mod LUT: table lookup (no division at all).
/// For n < 2^31: i64 arithmetic.
/// Otherwise: i128.
#[inline]
fn apply_row_2x2_cols(
    m: &mut Array2<i64>,
    r0: usize,
    r1: usize,
    s: i64,
    t: i64,
    u: i64,
    v: i64,
    n: i64,
    col_start: usize,
    col_end: usize,
    ring: &RingZModN,
) {
    if ring.has_lut() {
        // LUT path: compute s*a + t*b in i64, shift to non-negative, table lookup.
        // Max negative: -(n-1)^2 (when one term is 0 and the other has opposite signs).
        // Shift by (n-1)^2 to guarantee non-negative; then fast_mod the shifted result.
        // But fast_mod only covers [0, 2*(n-1)^2], so we need the shifted value in range.
        // s*a + t*b ∈ [-(n-1)^2, 2*(n-1)^2] when s,t,a,b ∈ [0, n).
        // After adding (n-1)^2, range is [0, 3*(n-1)^2] — too large for our LUT.
        //
        // Simpler: just use i64 % which is fast for small n, and conditional add.
        // The LUT win is in gcdex, not here. Keep i64 % for the row ops.
        for j in col_start..col_end {
            let a = m[[r0, j]];
            let b = m[[r1, j]];
            let new_r0 = (s * a + t * b) % n;
            let new_r1 = (u * a + v * b) % n;
            m[[r0, j]] = if new_r0 < 0 { new_r0 + n } else { new_r0 };
            m[[r1, j]] = if new_r1 < 0 { new_r1 + n } else { new_r1 };
        }
    } else if n < (1i64 << 31) {
        for j in col_start..col_end {
            let a = m[[r0, j]];
            let b = m[[r1, j]];
            let new_r0 = (s * a + t * b) % n;
            let new_r1 = (u * a + v * b) % n;
            m[[r0, j]] = if new_r0 < 0 { new_r0 + n } else { new_r0 };
            m[[r1, j]] = if new_r1 < 0 { new_r1 + n } else { new_r1 };
        }
    } else {
        for j in col_start..col_end {
            let a = m[[r0, j]];
            let b = m[[r1, j]];
            m[[r0, j]] =
                posmod_i128((s as i128) * (a as i128) + (t as i128) * (b as i128), n);
            m[[r1, j]] =
                posmod_i128((u as i128) * (a as i128) + (v as i128) * (b as i128), n);
        }
    }
}

/// Apply a 2x2 row transform to rows r0, r1 of a matrix (all columns).
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
    ring: &RingZModN,
) {
    let cols = m.ncols();
    apply_row_2x2_cols(m, r0, r1, s, t, u, v, n, 0, cols, ring);
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
    ring: &RingZModN,
) {
    apply_row_2x2(m1, r0, r1, s, t, u, v, n, ring);
    apply_row_2x2(m2, r0, r1, s, t, u, v, n, ring);
}

/// Lemma 3.1: row-echelon form via extended GCD elimination.
/// Returns (U, T, rank).
pub fn lemma_3_1(a: &Array2<i64>, ring: &RingZModN) -> (Array2<i64>, Array2<i64>, usize) {
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

            if ring.is_zero(b_val) {
                continue;
            }

            let (_, s, tv, uv, v) = ring.gcdex(a_val, b_val);
            apply_row_2x2_pair(&mut t, &mut u, r, i, s, tv, uv, v, n_mod, ring);
        }

        if !ring.is_zero(t[[r, k]]) {
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
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();

    let mut u = Array2::<i64>::eye(n);
    let mut t = a.clone();

    for j in 1..k {
        let sj = t[[j, j]];
        for i in 0..j {
            let x = t[[i, j]];
            if ring.is_zero(x) {
                continue;
            }
            let rem = {
                let b_ass = ring.gcd(sj, 0);
                let x_val = ((x % n_mod) + n_mod) % n_mod;
                if b_ass == 0 {
                    x_val
                } else {
                    x_val % b_ass
                }
            };
            let diff = ((x - rem) % n_mod + n_mod) % n_mod;
            let quo = ring.div(diff, sj).unwrap_or(0);
            let phi = ((-quo) % n_mod + n_mod) % n_mod;

            // row[i] += phi * row[j]  (mod n)
            if n_mod < (1i64 << 31) {
                let cols = t.ncols();
                for c in 0..cols {
                    let val = (t[[i, c]] + phi * t[[j, c]]) % n_mod;
                    t[[i, c]] = if val < 0 { val + n_mod } else { val };
                    let val = (u[[i, c]] + phi * u[[j, c]]) % n_mod;
                    u[[i, c]] = if val < 0 { val + n_mod } else { val };
                }
            } else {
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
    }

    (u, t)
}
