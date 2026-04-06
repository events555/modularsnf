//! Diagonal merge for Smith Normal Form — Rust port of modularsnf/diagonal.py.
//!
//! Implements _merge_scalars_raw, _merge_raw, and _smith_from_diagonal_raw
//! entirely in Rust.

use ndarray::{s, Array2};

use crate::ring::{mul_mod, posmod_i128, RingZModN};

/// Positive modulo.
#[inline]
fn posmod(a: i64, n: i64) -> i64 {
    let r = a % n;
    if r < 0 { r + n } else { r }
}

/// Merge two scalar SNF entries. Returns (U, V, S) as 2x2 arrays.
fn merge_scalars(a: i64, b: i64, ring: &RingZModN) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n = ring.n();
    let (g, s, t, u, v) = ring.gcdex(a, b);

    if g % n == 0 {
        return (Array2::eye(2), Array2::eye(2), Array2::zeros((2, 2)));
    }

    let tb = mul_mod(t, b, n);
    let q_raw = ring.div(tb, g).unwrap_or(0);
    let q = posmod(-q_raw, n);

    let u_arr = Array2::from_shape_vec(
        (2, 2),
        vec![posmod(s, n), posmod(t, n), posmod(u, n), posmod(v, n)],
    )
    .unwrap();

    let v_arr = Array2::from_shape_vec((2, 2), vec![1, q, 1, posmod(1 + q, n)]).unwrap();

    // S = U @ diag(a, b) @ V
    let ab = Array2::from_shape_vec((2, 2), vec![a, 0, 0, b]).unwrap();
    let s_arr = matmul_mod(&matmul_mod(&u_arr, &ab, n), &v_arr, n);

    (u_arr, v_arr, s_arr)
}

/// Maximum value of a dot-product element that float64 can represent exactly.
///
/// IEEE 754 float64 has a 53-bit mantissa, so any integer with absolute value
/// ≤ 2^53 is represented exactly.  For a dot product `sum_k(a[k]*b[k])` where
/// `a, b ∈ [0, n)`, each product is at most `(n-1)^2` and the sum of `inner`
/// terms is at most `(n-1)^2 * inner`.  The float64 path is exact when this
/// bound fits in 2^53.
#[cfg(feature = "blas")]
const F64_EXACT_MAX: u128 = 1u128 << 53;

/// Return true when `matmul_mod` can safely delegate to float64 BLAS.
///
/// The condition is: `(n-1)^2 * inner ≤ 2^53`, ensuring every intermediate
/// sum in the dot product is representable exactly as a float64.
#[cfg(feature = "blas")]
#[inline]
fn blas_safe(n: i64, inner: usize) -> bool {
    let max_entry = (n - 1) as u128;
    let max_product = max_entry * max_entry;
    // Check overflow-safe: max_product * inner ≤ F64_EXACT_MAX
    match max_product.checked_mul(inner as u128) {
        Some(bound) => bound <= F64_EXACT_MAX,
        None => false,
    }
}

/// Matrix multiply with mod reduction. C = (A @ B) % n.
///
/// When the `blas` feature is enabled and the modulus is small enough for
/// float64 to be exact, delegates to BLAS `dgemm` for a dramatic speedup
/// (100–400× at large sizes).  Otherwise falls back to a safe integer loop.
pub fn matmul_mod(a: &Array2<i64>, b: &Array2<i64>, n: i64) -> Array2<i64> {
    #[cfg(feature = "blas")]
    {
        if blas_safe(n, a.ncols()) {
            return matmul_mod_blas(a, b, n);
        }
    }
    matmul_mod_integer(a, b, n)
}

/// BLAS-backed matmul_mod: cast to f64, call dgemm, cast back, reduce mod n.
///
/// All inputs are in [0, n), so all dot-product results are non-negative.
/// We use simple i64 `%` instead of expensive i128 posmod.
#[cfg(feature = "blas")]
fn matmul_mod_blas(a: &Array2<i64>, b: &Array2<i64>, n: i64) -> Array2<i64> {
    use ndarray::Array2 as A2;

    // Convert i64 → f64 (exact for values in [0, n) where n < 2^53).
    let a_f: A2<f64> = a.mapv(|v| v as f64);
    let b_f: A2<f64> = b.mapv(|v| v as f64);

    // BLAS dgemm via ndarray's .dot() — this is the fast path.
    let c_f: A2<f64> = a_f.dot(&b_f);

    // Convert back to i64 and reduce mod n.
    // Results are non-negative (inputs in [0, n)), so plain % suffices.
    c_f.mapv(|v| (v as i64) % n)
}

/// Pure-integer matmul_mod fallback.
///
/// Uses lazy reduction: accumulates the full dot product in i128, then
/// reduces mod n once per output element. For small n (< 256) and any
/// practical matrix dimension, the i128 accumulator never overflows
/// (worst case: (n-1)^2 * inner_dim ≪ 2^127).
fn matmul_mod_integer(a: &Array2<i64>, b: &Array2<i64>, n: i64) -> Array2<i64> {
    let rows = a.nrows();
    let cols = b.ncols();
    let inner = a.ncols();
    let mut c = Array2::zeros((rows, cols));
    for i in 0..rows {
        let a_row = a.row(i);
        for j in 0..cols {
            let mut acc: i128 = 0;
            for k in 0..inner {
                acc += a_row[k] as i128 * b[[k, j]] as i128;
            }
            c[[i, j]] = posmod_i128(acc, n);
        }
    }
    c
}

/// Check if diagonal SNF array is zero (first entry is 0 mod n).
#[inline]
fn is_zero(arr: &Array2<i64>, n: i64) -> bool {
    if arr.nrows() == 0 {
        return true;
    }
    arr[[0, 0]] % n == 0
}

/// Count nonzero diagonal entries mod n.
fn get_rank(arr: &Array2<i64>, n: i64) -> usize {
    let dim = arr.nrows().min(arr.ncols());
    let mut rank = 0;
    for i in 0..dim {
        if arr[[i, i]] % n != 0 {
            rank += 1;
        }
    }
    rank
}

/// Extract a square subblock from a matrix (copy).
fn subblock(arr: &Array2<i64>, r0: usize, r1: usize, c0: usize, c1: usize) -> Array2<i64> {
    arr.slice(s![r0..r1, c0..c1]).to_owned()
}

/// Write a block into a matrix at position (r0, c0).
fn write_block(dst: &mut Array2<i64>, r0: usize, c0: usize, src: &Array2<i64>) {
    let rows = src.nrows();
    let cols = src.ncols();
    dst.slice_mut(s![r0..r0 + rows, c0..c0 + cols]).assign(src);
}

/// Left-apply a 2-block transform: rows [s1..s1+t] and [s2..s2+t] of M.
fn left_apply_block_pair(
    m: &mut Array2<i64>,
    u00: &Array2<i64>,
    u01: &Array2<i64>,
    u10: &Array2<i64>,
    u11: &Array2<i64>,
    s1: usize,
    s2: usize,
    t: usize,
    n: i64,
) {
    let r1 = m.slice(s![s1..s1 + t, ..]).to_owned();
    let r2 = m.slice(s![s2..s2 + t, ..]).to_owned();
    let new_r1 = matmul_mod_add(&matmul_mod(u00, &r1, n), &matmul_mod(u01, &r2, n), n);
    let new_r2 = matmul_mod_add(&matmul_mod(u10, &r1, n), &matmul_mod(u11, &r2, n), n);
    m.slice_mut(s![s1..s1 + t, ..]).assign(&new_r1);
    m.slice_mut(s![s2..s2 + t, ..]).assign(&new_r2);
}

/// Right-apply a 2-block transform: cols [s1..s1+t] and [s2..s2+t] of M.
fn right_apply_block_pair(
    m: &mut Array2<i64>,
    v00: &Array2<i64>,
    v01: &Array2<i64>,
    v10: &Array2<i64>,
    v11: &Array2<i64>,
    s1: usize,
    s2: usize,
    t: usize,
    n: i64,
) {
    let c1 = m.slice(s![.., s1..s1 + t]).to_owned();
    let c2 = m.slice(s![.., s2..s2 + t]).to_owned();
    let new_c1 = matmul_mod_add(&matmul_mod(&c1, v00, n), &matmul_mod(&c2, v10, n), n);
    let new_c2 = matmul_mod_add(&matmul_mod(&c1, v01, n), &matmul_mod(&c2, v11, n), n);
    m.slice_mut(s![.., s1..s1 + t]).assign(&new_c1);
    m.slice_mut(s![.., s2..s2 + t]).assign(&new_c2);
}

/// Element-wise (A + B) % n.
///
/// Inputs are already in [0, n), so sum is in [0, 2n-2].
/// A conditional subtraction replaces i128 division.
fn matmul_mod_add(a: &Array2<i64>, b: &Array2<i64>, n: i64) -> Array2<i64> {
    let mut c = a.clone();
    c.zip_mut_with(b, |x, &y| {
        let s = *x + y;
        *x = if s >= n { s - n } else { s };
    });
    c
}

/// Recursive merge of two SNF blocks. Returns (U, V, S) as 2n x 2n arrays.
pub fn merge_raw(
    a_arr: &Array2<i64>,
    b_arr: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a_arr.nrows();

    if n == 0 {
        let e = Array2::zeros((0, 0));
        return (e.clone(), e.clone(), e);
    }

    if n == 1 {
        return merge_scalars(a_arr[[0, 0]], b_arr[[0, 0]], ring);
    }

    let t = n / 2;
    let nn = 2 * n;

    let mut a1 = subblock(a_arr, 0, t, 0, t);
    let mut a2 = subblock(a_arr, t, n, t, n);
    let mut b1 = subblock(b_arr, 0, t, 0, t);
    let mut b2 = subblock(b_arr, t, n, t, n);

    let mut u_total = Array2::eye(nn);
    let mut v_total = Array2::eye(nn);

    let index_map = [0, t, n, n + t];

    // Apply merge step helper (inlined as a closure-like pattern)
    macro_rules! apply_step {
        ($block1:expr, $block2:expr, $idx1:expr, $idx2:expr) => {{
            if !(is_zero(&$block1, n_mod) && is_zero(&$block2, n_mod)) {
                let (u_loc, v_loc, s_loc) = merge_raw(&$block1, &$block2, ring);
                $block1 = subblock(&s_loc, 0, t, 0, t);
                $block2 = subblock(&s_loc, t, 2 * t, t, 2 * t);

                let s1 = index_map[$idx1];
                let s2 = index_map[$idx2];

                left_apply_block_pair(
                    &mut u_total,
                    &subblock(&u_loc, 0, t, 0, t),
                    &subblock(&u_loc, 0, t, t, 2 * t),
                    &subblock(&u_loc, t, 2 * t, 0, t),
                    &subblock(&u_loc, t, 2 * t, t, 2 * t),
                    s1,
                    s2,
                    t,
                    n_mod,
                );
                right_apply_block_pair(
                    &mut v_total,
                    &subblock(&v_loc, 0, t, 0, t),
                    &subblock(&v_loc, 0, t, t, 2 * t),
                    &subblock(&v_loc, t, 2 * t, 0, t),
                    &subblock(&v_loc, t, 2 * t, t, 2 * t),
                    s1,
                    s2,
                    t,
                    n_mod,
                );
            }
        }};
    }

    apply_step!(a1, b1, 0, 2);
    apply_step!(a2, b2, 1, 3);
    apply_step!(a2, b1, 1, 2);
    apply_step!(b1, b2, 2, 3);

    if !is_zero(&b2, n_mod) {
        let r_b1 = get_rank(&b1, n_mod);
        if r_b1 < t {
            let r_b2 = get_rank(&b2, n_mod);

            let mut p_arr = Array2::zeros((2 * t, 2 * t));
            for i in 0..r_b1 {
                p_arr[[i, i]] = 1;
            }
            for i in 0..r_b2 {
                p_arr[[r_b1 + i, t + i]] = 1;
            }
            let mut cr = r_b1 + r_b2;
            for i in 0..(t - r_b1) {
                p_arr[[cr, r_b1 + i]] = 1;
                cr += 1;
            }
            for i in 0..(t - r_b2) {
                p_arr[[cr, t + r_b2 + i]] = 1;
                cr += 1;
            }

            // P_glob = identity with p_arr at (n, n)
            let mut p_glob = Array2::<i64>::eye(nn);
            write_block(&mut p_glob, n, n, &p_arr);
            let p_glob_t = p_glob.t().to_owned();

            u_total = matmul_mod(&p_glob, &u_total, n_mod);
            v_total = matmul_mod(&v_total, &p_glob_t, n_mod);

            // Permute B blocks
            let mut b_comb = Array2::zeros((2 * t, 2 * t));
            write_block(&mut b_comb, 0, 0, &b1);
            write_block(&mut b_comb, t, t, &b2);
            let p_arr_t = p_arr.t().to_owned();
            let s_target = matmul_mod(&matmul_mod(&p_arr, &b_comb, n_mod), &p_arr_t, n_mod);
            b1 = subblock(&s_target, 0, t, 0, t);
            b2 = subblock(&s_target, t, 2 * t, t, 2 * t);
        }
    }

    // Assemble S_final = block_diag(A1, A2, B1, B2)
    let mut s_final = Array2::zeros((nn, nn));
    write_block(&mut s_final, 0, 0, &a1);
    write_block(&mut s_final, t, t, &a2);
    write_block(&mut s_final, n, n, &b1);
    write_block(&mut s_final, n + t, n + t, &b2);

    (u_total, v_total, s_final)
}

/// Bottom-up iterative diagonal SNF for power-of-two matrices.
fn smith_from_diagonal_raw(
    diag: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = diag.nrows();

    if n <= 1 {
        return (Array2::eye(n), Array2::eye(n), diag.clone());
    }

    // Start with n 1x1 blocks
    let mut blocks: Vec<(Array2<i64>, Array2<i64>, Array2<i64>)> = (0..n)
        .map(|i| {
            let s = Array2::from_elem((1, 1), diag[[i, i]]);
            (Array2::eye(1), Array2::eye(1), s)
        })
        .collect();

    let mut size = 1;
    while size < n {
        let mut new_blocks = Vec::with_capacity(blocks.len() / 2);
        for pair in blocks.chunks_exact(2) {
            let (u1, v1, a) = &pair[0];
            let (u2, v2, b) = &pair[1];

            let (u_merge, v_merge, s) = merge_raw(a, b, ring);

            let bsz = size;
            let mut u_block = Array2::zeros((2 * bsz, 2 * bsz));
            write_block(&mut u_block, 0, 0, u1);
            write_block(&mut u_block, bsz, bsz, u2);
            let u_total = matmul_mod(&u_merge, &u_block, n_mod);

            let mut v_block = Array2::zeros((2 * bsz, 2 * bsz));
            write_block(&mut v_block, 0, 0, v1);
            write_block(&mut v_block, bsz, bsz, v2);
            let v_total = matmul_mod(&v_block, &v_merge, n_mod);

            new_blocks.push((u_total, v_total, s));
        }
        blocks = new_blocks;
        size *= 2;
    }

    blocks.into_iter().next().unwrap()
}

/// Compute SNF of a diagonal matrix.
/// Pads to power-of-two, runs merge, crops back.
pub fn smith_from_diagonal(
    diag: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n = diag.nrows();
    if n <= 1 {
        return (Array2::eye(n), Array2::eye(n), diag.clone());
    }
    let size = (n as u64).next_power_of_two() as usize;
    let mut pad = Array2::zeros((size, size));
    pad.slice_mut(s![..n, ..n]).assign(diag);

    let (u, v, s) = smith_from_diagonal_raw(&pad, ring);

    (
        subblock(&u, 0, n, 0, n),
        subblock(&v, 0, n, 0, n),
        subblock(&s, 0, n, 0, n),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that matmul_mod produces correct results for various sizes and moduli.
    #[test]
    fn test_matmul_mod_correctness() {
        // Small hand-computed case: [[1,2],[3,4]] @ [[5,6],[7,8]] mod 10
        // = [[19,22],[43,50]] mod 10 = [[9,2],[3,0]]
        let a = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![5, 6, 7, 8]).unwrap();
        let c = matmul_mod(&a, &b, 10);
        assert_eq!(c, Array2::from_shape_vec((2, 2), vec![9, 2, 3, 0]).unwrap());
    }

    /// Compare matmul_mod against a naive reference implementation.
    #[test]
    fn test_matmul_mod_vs_naive() {
        for n in [7, 127, 255] {
            for size in [4, 16, 64] {
                let a = Array2::from_shape_fn((size, size), |(i, j)| {
                    ((i * 7 + j * 13 + 3) as i64) % n
                });
                let b = Array2::from_shape_fn((size, size), |(i, j)| {
                    ((i * 11 + j * 5 + 7) as i64) % n
                });

                let c = matmul_mod(&a, &b, n);

                // Naive reference
                let mut expected = Array2::zeros((size, size));
                for i in 0..size {
                    for j in 0..size {
                        let mut acc: i128 = 0;
                        for k in 0..size {
                            acc += a[[i, k]] as i128 * b[[k, j]] as i128;
                        }
                        expected[[i, j]] = posmod_i128(acc, n);
                    }
                }

                assert_eq!(c, expected, "mismatch at size={size}, n={n}");
            }
        }
    }
}
