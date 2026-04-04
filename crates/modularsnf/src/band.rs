//! Band reduction — Rust port of modularsnf/band.py.

use ndarray::{s, Array2};

use crate::diagonal::matmul_mod;
use crate::echelon::lemma_3_1;
use crate::ring::RingZModN;

/// Positive modulo helper.
#[inline]
fn posmod(a: i64, n: i64) -> i64 {
    ((a % n) + n) % n
}

/// Right-apply a block: M[:, start..start+k] = M[:, start..start+k] @ block
fn right_apply_block(m: &mut Array2<i64>, block: &Array2<i64>, start: usize, n_mod: i64) {
    let k = block.nrows();
    let cols = m.slice(s![.., start..start + k]).to_owned();
    let new_cols = matmul_mod(&cols, block, n_mod);
    m.slice_mut(s![.., start..start + k]).assign(&new_cols);
}

/// Left-apply a block: M[start..start+k, :] = block @ M[start..start+k, :]
fn left_apply_block(m: &mut Array2<i64>, block: &Array2<i64>, start: usize, n_mod: i64) {
    let k = block.nrows();
    let rows = m.slice(s![start..start + k, ..]).to_owned();
    let new_rows = matmul_mod(block, &rows, n_mod);
    m.slice_mut(s![start..start + k, ..]).assign(&new_rows);
}

/// Triang step: triangulate top-right block of upper-b-banded matrix.
/// Returns (B_prime, W) where W is the s2 x s2 right transform.
fn triang(b_mat: &Array2<i64>, b: usize, ring: &RingZModN) -> (Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let s1 = b / 2;
    let s2 = b - 1;

    // Extract top-right s1 x s2 block, transpose it
    let b2 = b_mat.slice(s![0..s1, s1..s1 + s2]).to_owned();
    let c = b2.t().to_owned();

    // lemma_3_1 on the transposed block
    let (u_left, _, _) = lemma_3_1(&c, ring);
    let w = u_left.t().to_owned();

    // B_prime = B @ block_diag(I_s1, W)
    // Only columns s1..s1+s2 are affected
    let n1 = s1 + s2;
    let mut b_prime = b_mat.clone();
    let cols = b_prime.slice(s![.., s1..n1]).to_owned();
    let new_cols = matmul_mod(&cols, &w, n_mod);
    b_prime.slice_mut(s![.., s1..n1]).assign(&new_cols);

    (b_prime, w)
}

/// Shift step: Storjohann's Lemma 7.4.
/// Returns (C_prime, U_block, V_block).
fn shift(
    c_mat: &Array2<i64>,
    b: usize,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let s2 = b - 1;

    // Block partition: C1 = top-left, C2 = top-right
    let c1 = c_mat.slice(s![0..s2, 0..s2]).to_owned();
    let c2 = c_mat.slice(s![0..s2, s2..2 * s2]).to_owned();

    let (u1, _, _) = lemma_3_1(&c1, ring);

    // C2_prime = U1 @ C2
    let c2_prime = matmul_mod(&u1, &c2, n_mod);

    // Triangulate C2_prime^T
    let c2_prime_t = c2_prime.t().to_owned();
    let (u2, _, _) = lemma_3_1(&c2_prime_t, ring);
    let v_block = u2.t().to_owned();

    // C_prime = block_diag(U1, I) @ C @ block_diag(I, V_block)
    let mut c_prime = c_mat.clone();
    // Left-apply U1 to top rows
    left_apply_block(&mut c_prime, &u1, 0, n_mod);
    // Right-apply V_block to right columns
    right_apply_block(&mut c_prime, &v_block, s2, n_mod);

    (c_prime, u1, v_block)
}

/// Compute upper bandwidth of a matrix.
pub fn compute_upper_bandwidth(m: &Array2<i64>, n_mod: i64) -> usize {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let mut max_offset: Option<usize> = None;
    for i in 0..nrows {
        for j in i..ncols {
            if posmod(m[[i, j]], n_mod) != 0 {
                let offset = j - i;
                match max_offset {
                    None => max_offset = Some(offset),
                    Some(cur) => {
                        if offset > cur {
                            max_offset = Some(offset);
                        }
                    }
                }
            }
        }
    }
    match max_offset {
        None => 0,
        Some(o) => o + 1,
    }
}

/// Full band reduction: reduce upper bandwidth from b to floor(b/2)+1.
/// Returns (A_reduced, U_band, V_band, b_new).
pub fn band_reduction(
    a: &Array2<i64>,
    b: usize,
    t_param: usize,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>, usize) {
    let n_mod = ring.n();
    let n = a.nrows();

    if b <= 2 {
        return (a.clone(), Array2::eye(n), Array2::eye(n), b);
    }

    let s1 = b / 2;
    let s2 = b - 1;
    let n1 = s1 + s2;
    let n2 = 2 * s2;

    let pad = 2 * b - t_param;
    let n_big = n + pad;

    // Padded matrix
    let mut b_mat = Array2::zeros((n_big, n_big));
    b_mat.slice_mut(s![..n, ..n]).assign(a);

    let mut u_big = Array2::<i64>::eye(n_big);
    let mut v_big = Array2::<i64>::eye(n_big);

    let num_i = if n > t_param {
        (n - t_param + s1 - 1) / s1
    } else {
        0
    };

    for i in 0..num_i {
        let top = i * s1;
        if top + n1 > n_big {
            break;
        }

        // Triang step
        let b_block = b_mat.slice(s![top..top + n1, top..top + n1]).to_owned();
        let (_, w) = triang(&b_block, b, ring);

        let w_start = top + s1;
        right_apply_block(&mut b_mat, &w, w_start, n_mod);
        right_apply_block(&mut v_big, &w, w_start, n_mod);

        // Shift steps
        let numer = if n > t_param + (i + 1) * s1 {
            n - t_param - (i + 1) * s1
        } else {
            continue;
        };

        let num_j = (numer + s2 - 1) / s2;

        for j in 0..num_j {
            let offset = (i + 1) * s1 + j * s2;
            if offset + n2 > n_big {
                break;
            }

            let c_block = b_mat
                .slice(s![offset..offset + n2, offset..offset + n2])
                .to_owned();
            let (_, u_block, v_block) = shift(&c_block, b, ring);

            left_apply_block(&mut b_mat, &u_block, offset, n_mod);
            right_apply_block(&mut b_mat, &v_block, offset + s2, n_mod);
            left_apply_block(&mut u_big, &u_block, offset, n_mod);
            right_apply_block(&mut v_big, &v_block, offset + s2, n_mod);
        }
    }

    let a_reduced = b_mat.slice(s![..n, ..n]).to_owned();
    let u_band = u_big.slice(s![..n, ..n]).to_owned();
    let v_band = v_big.slice(s![..n, ..n]).to_owned();

    let b_new = b / 2 + 1;
    (a_reduced, u_band, v_band, b_new)
}
