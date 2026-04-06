//! Row-echelon utilities — Rust port of modularsnf/echelon.py.
//!
//! Includes a blocked panel-factorization variant of Lemma 3.1 that
//! accumulates Gcdex rotations on a narrow panel, then applies the
//! accumulated transform to trailing columns via a single BLAS GEMM.

use ndarray::{s, Array2};

use crate::diagonal::matmul_mod;
use crate::ring::{posmod_i128, RingZModN};

/// Crossover: trailing blocks smaller than this use the naive path.
const BLOCKED_CROSSOVER: usize = 64;

/// Panel width for blocked elimination.
const PANEL_WIDTH: usize = 32;

/// Apply a 2×2 row rotation to columns `col_start..col_end` of a matrix.
///
/// For small n (< 2^31), uses i64 arithmetic instead of i128.
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
) {
    if n < (1i64 << 31) {
        // For small n, s*a + t*b fits in i64 when a,b ∈ [0,n) and s,t ∈ [0,n).
        // Max value: 2*(n-1)^2 < 2 * 2^62 = 2^63 — but could overflow signed i64
        // for n close to 2^31. Use wrapping arithmetic + single i64 mod.
        // Actually, max = 2*(n-1)^2. For n < 2^31, (n-1)^2 < 2^62, so 2*(n-1)^2 < 2^63.
        // This fits in i64 (max 2^63 - 1). Safe.
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
) {
    let cols = m.ncols();
    apply_row_2x2_cols(m, r0, r1, s, t, u, v, n, 0, cols);
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
///
/// Uses blocked panel factorization when the trailing submatrix is large
/// enough to benefit from BLAS GEMM. Falls back to naive element-wise
/// application for small matrices.
pub fn lemma_3_1(a: &Array2<i64>, ring: &RingZModN) -> (Array2<i64>, Array2<i64>, usize) {
    let n_mod = ring.n();
    let n_rows = a.nrows();
    let n_cols = a.ncols();

    let mut u = Array2::<i64>::eye(n_rows);
    let mut t = a.clone();

    let mut r = 0usize; // current pivot row

    // Process columns in panels of PANEL_WIDTH.
    let mut col = 0usize;
    while col < n_cols && r < n_rows {
        let panel_end = (col + PANEL_WIDTH).min(n_cols);
        let n_trailing = n_cols.saturating_sub(panel_end);
        let n_active = n_rows - r;

        let use_blocked = n_trailing >= BLOCKED_CROSSOVER && n_active >= BLOCKED_CROSSOVER;

        if use_blocked {
            // --- Blocked path: accumulate rotations, apply via GEMM ---

            // U_acc tracks the accumulated left-transform on active rows [r..n_rows].
            let mut u_acc = Array2::<i64>::eye(n_active);

            let r_start = r; // save starting pivot row for this panel

            for k in col..panel_end {
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

                    // Apply rotation to panel columns [col..panel_end] of T
                    apply_row_2x2_cols(&mut t, r, i, s, tv, uv, v, n_mod, col, panel_end);

                    // Apply same rotation to ALL columns of U (U tracks the full transform)
                    apply_row_2x2_cols(&mut u, r, i, s, tv, uv, v, n_mod, 0, n_rows);

                    // Accumulate into U_acc: rows (r - r_start) and (i - r_start)
                    let r0_local = r - r_start;
                    let r1_local = i - r_start;
                    apply_row_2x2_cols(
                        &mut u_acc, r0_local, r1_local, s, tv, uv, v, n_mod, 0, n_active,
                    );
                }

                if !ring.is_zero(t[[r, k]]) {
                    r += 1;
                }
            }

            // Apply U_acc to trailing columns [panel_end..n_cols] via GEMM.
            let trailing = t.slice(s![r_start..n_rows, panel_end..n_cols]).to_owned();
            let updated = matmul_mod(&u_acc, &trailing, n_mod);
            t.slice_mut(s![r_start..n_rows, panel_end..n_cols])
                .assign(&updated);
        } else {
            // --- Naive path: apply each rotation to full row width ---
            for k in col..panel_end {
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
                    apply_row_2x2_pair(&mut t, &mut u, r, i, s, tv, uv, v, n_mod);
                }

                if !ring.is_zero(t[[r, k]]) {
                    r += 1;
                }
            }
        }

        col = panel_end;
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
