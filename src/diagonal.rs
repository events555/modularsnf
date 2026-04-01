//! Diagonal merge for Smith Normal Form — Rust port of modularsnf/diagonal.py.
//!
//! Implements _merge_scalars_raw, _merge_raw, and _smith_from_diagonal_raw
//! entirely in Rust, returning results as numpy arrays to Python.

use numpy::ndarray::{s, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::ring::{mul_mod, posmod_i128, RustRingZModN};

/// Positive modulo.
#[inline]
fn posmod(a: i64, n: i64) -> i64 {
    posmod_i128(a as i128, n)
}

/// Merge two scalar SNF entries. Returns (U, V, S) as 2x2 arrays.
fn merge_scalars(a: i64, b: i64, ring: &RustRingZModN) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n = ring.n();
    let (g, s, t, u, v) = ring.gcdex_internal(a, b);

    if g % n == 0 {
        return (Array2::eye(2), Array2::eye(2), Array2::zeros((2, 2)));
    }

    let tb = mul_mod(t, b, n);
    let q_raw = ring.div_internal(tb, g).unwrap_or(0);
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

/// Matrix multiply with mod reduction. C = (A @ B) % n.
///
/// Uses row-major iteration with slices to avoid per-element bounds checks.
/// Cannot use BLAS (.dot()) because we need integer arithmetic with mod.
pub fn matmul_mod(a: &Array2<i64>, b: &Array2<i64>, n: i64) -> Array2<i64> {
    let rows = a.nrows();
    let cols = b.ncols();
    let inner = a.ncols();
    let mut c = Array2::zeros((rows, cols));
    for i in 0..rows {
        let a_row = a.row(i);
        let mut c_row = c.row_mut(i);
        for k in 0..inner {
            let a_ik = a_row[k];
            if a_ik == 0 {
                continue;
            }
            let b_row = b.row(k);
            for j in 0..cols {
                c_row[j] = posmod_i128((c_row[j] as i128) + (a_ik as i128) * (b_row[j] as i128), n);
            }
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
fn matmul_mod_add(a: &Array2<i64>, b: &Array2<i64>, n: i64) -> Array2<i64> {
    let mut c = a.clone();
    c.zip_mut_with(b, |x, &y| {
        *x = posmod_i128((*x as i128) + (y as i128), n);
    });
    c
}

/// Recursive merge of two SNF blocks. Returns (U, V, S) as 2n x 2n arrays.
fn merge_raw(
    a_arr: &Array2<i64>,
    b_arr: &Array2<i64>,
    ring: &RustRingZModN,
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
    ring: &RustRingZModN,
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

/// Public entry for diagonal SNF from other Rust modules.
/// Pads to power-of-two, runs merge, crops back.
pub fn smith_from_diagonal_internal(
    diag: &Array2<i64>,
    ring: &RustRingZModN,
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

// ---- PyO3 exports ----

/// Compute SNF of a diagonal matrix. Takes a flat diagonal array and modulus.
/// Returns (U, V, S) as numpy arrays, cropped to original size.
#[pyfunction]
pub fn rust_smith_from_diagonal<'py>(
    py: Python<'py>,
    diag_data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let ring = RustRingZModN::new_internal(modulus)?;
    let arr = diag_data.as_array().to_owned();
    let n = arr.nrows();

    // Pad to power of two
    let size = if n <= 1 {
        n
    } else {
        (n as u64).next_power_of_two() as usize
    };
    let mut pad = Array2::zeros((size, size));
    for i in 0..n {
        for j in 0..n {
            pad[[i, j]] = arr[[i, j]];
        }
    }

    let (u, v, s) = smith_from_diagonal_raw(&pad, &ring);

    // Crop to original size
    let u_crop = subblock(&u, 0, n, 0, n);
    let v_crop = subblock(&v, 0, n, 0, n);
    let s_crop = subblock(&s, 0, n, 0, n);

    Ok((
        u_crop.into_pyarray(py),
        v_crop.into_pyarray(py),
        s_crop.into_pyarray(py),
    ))
}

/// Merge two SNF blocks. Returns (U, V, S) as numpy arrays.
#[pyfunction]
pub fn rust_merge_smith_blocks<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray2<'py, i64>,
    b_data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let ring = RustRingZModN::new_internal(modulus)?;
    let a = a_data.as_array().to_owned();
    let b = b_data.as_array().to_owned();

    let (u, v, s) = merge_raw(&a, &b, &ring);

    Ok((u.into_pyarray(py), v.into_pyarray(py), s.into_pyarray(py)))
}
