//! Smith Normal Form pipeline — Rust port of modularsnf/snf.py.

use ndarray::{s, Array2};

use crate::band::{band_reduction, compute_upper_bandwidth};
use crate::diagonal::matmul_mod;
use crate::echelon::{apply_row_2x2_pair, index1_reduce_on_columns, lemma_3_1};
use crate::ring::{mul_mod, posmod_i128, RingZModN};

/// Positive modulo.
#[inline]
fn posmod(a: i64, n: i64) -> i64 {
    let r = a % n;
    if r < 0 { r + n } else { r }
}

/// Apply a 2×2 column transform to columns c0, c1 of a matrix.
#[inline]
fn apply_col_2x2(
    m: &mut Array2<i64>,
    c0: usize,
    c1: usize,
    s: i64,
    t: i64,
    u: i64,
    v: i64,
    n: i64,
    ring: &RingZModN,
) {
    let rows = m.nrows();
    if ring.has_lut() {
        for row in 0..rows {
            let a = m[[row, c0]];
            let b = m[[row, c1]];
            m[[row, c0]] = ring.fast_mod(s * a + t * b);
            m[[row, c1]] = ring.fast_mod(u * a + v * b);
        }
    } else if n < (1i64 << 31) {
        for row in 0..rows {
            let a = m[[row, c0]];
            let b = m[[row, c1]];
            let new_c0 = (s * a + t * b) % n;
            let new_c1 = (u * a + v * b) % n;
            m[[row, c0]] = if new_c0 < 0 { new_c0 + n } else { new_c0 };
            m[[row, c1]] = if new_c1 < 0 { new_c1 + n } else { new_c1 };
        }
    } else {
        for row in 0..rows {
            let a = m[[row, c0]];
            let b = m[[row, c1]];
            m[[row, c0]] =
                posmod_i128((s as i128) * (a as i128) + (t as i128) * (b as i128), n);
            m[[row, c1]] =
                posmod_i128((u as i128) * (a as i128) + (v as i128) * (b as i128), n);
        }
    }
}

/// Add phi * col[src] to col[dst].
#[inline]
fn add_col_scaled(
    m: &mut Array2<i64>,
    dst: usize,
    src: usize,
    phi: i64,
    n: i64,
    ring: &RingZModN,
) {
    let rows = m.nrows();
    if ring.has_lut() {
        for row in 0..rows {
            m[[row, dst]] = ring.fast_mod(m[[row, dst]] + phi * m[[row, src]]);
        }
    } else if n < (1i64 << 31) {
        for row in 0..rows {
            let val = (m[[row, dst]] + phi * m[[row, src]]) % n;
            m[[row, dst]] = if val < 0 { val + n } else { val };
        }
    } else {
        for row in 0..rows {
            m[[row, dst]] = posmod_i128(
                (m[[row, dst]] as i128) + (phi as i128) * (m[[row, src]] as i128),
                n,
            );
        }
    }
}

/// Top-level SNF for a square matrix.
/// Returns (U, V, S) such that S = U @ A @ V.
pub fn smith_square(
    a: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();

    if n == 0 {
        return (Array2::eye(0), Array2::eye(0), a.clone());
    }

    // Step 1: row echelon
    let (u_ech, t, _rank) = lemma_3_1(a, ring);

    // Step 2: band reduction
    let mut b_mat = t;
    let mut b = compute_upper_bandwidth(&b_mat, n_mod);

    let mut u_band_total = Array2::<i64>::eye(n);
    let mut v_band_total = Array2::<i64>::eye(n);

    while b > 2 {
        let (b_new_mat, u_step, v_step, b_new) = band_reduction(&b_mat, b, 0, ring);
        b_mat = b_new_mat;
        u_band_total = matmul_mod(&u_step, &u_band_total, n_mod);
        v_band_total = matmul_mod(&v_band_total, &v_step, n_mod);
        b = b_new;
    }

    let (u_snf, v_snf, s) = smith_from_upper_2_banded(&b_mat, ring);

    let u_total = matmul_mod(&matmul_mod(&u_snf, &u_band_total, n_mod), &u_ech, n_mod);
    let v_total = matmul_mod(&v_band_total, &v_snf, n_mod);

    (u_total, v_total, s)
}

/// Smith form for a 2-banded square matrix.
fn smith_from_upper_2_banded(
    a: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();

    if n <= 1 {
        return (Array2::eye(n), Array2::eye(n), a.clone());
    }

    // Step 1: split with spike
    let (u1, v1, a1, n1, n2) = step1_split_with_spike(a, ring);

    // Step 2: recursive blocks
    let (u2, v2, a2) = step2_recursive_blocks(&a1, n1, n2, ring);

    // Step 3: permute spike to last position
    let (u3, v3, a3) = step3_permute(&a2, n1);

    // Step 4: diagonalize (n-1) x (n-1) block
    let (u4, v4, a4) = step4_smith_on_n_minus_1(&a3, ring);

    // Steps 5-8: gcd chain
    let (u5, v5, a5, k) = step5_to_8_gcd_chain(&a4, ring);

    // Step 9: index reduction
    let (u6, v6, a6) = step9_index_reduction(&a5, k, ring);

    let u_total = matmul_mod(
        &matmul_mod(
            &matmul_mod(
                &matmul_mod(&matmul_mod(&u6, &u5, n_mod), &u4, n_mod),
                &u3,
                n_mod,
            ),
            &u2,
            n_mod,
        ),
        &u1,
        n_mod,
    );
    let v_total = matmul_mod(
        &matmul_mod(
            &matmul_mod(
                &matmul_mod(&matmul_mod(&v1, &v2, n_mod), &v3, n_mod),
                &v4,
                n_mod,
            ),
            &v5,
            n_mod,
        ),
        &v6,
        n_mod,
    );

    (u_total, v_total, a6)
}

/// Step 1: split 2-banded matrix into spike + principal blocks.
fn step1_split_with_spike(
    a: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>, usize, usize) {
    let n_mod = ring.n();
    let n = a.nrows();
    let n1 = (n - 1) / 2;
    let n2 = n - 1 - n1;

    let mut u = Array2::<i64>::eye(n);
    let mut t = a.clone();

    // Sweep upward
    for k in (n1 + 1..n).rev() {
        let (r0, r1) = (k - 1, k);
        let a_val = t[[r0, k]];
        let b_val = t[[r1, k]];
        let (_, s, tv, uv, v) = ring.gcdex(a_val, b_val);
        // Note: swapped order (u,v,s,t) vs (s,t,u,v) per Python code
        apply_row_2x2_pair(&mut t, &mut u, r0, r1, uv, v, s, tv, n_mod, ring);
    }

    // Sweep downward
    for k in n1 + 1..n - 1 {
        let (r0, r1) = (k, k + 1);
        let a_val = t[[r0, k]];
        let b_val = t[[r1, k]];
        let (_, s, tv, uv, v) = ring.gcdex(a_val, b_val);
        apply_row_2x2_pair(&mut t, &mut u, r0, r1, s, tv, uv, v, n_mod, ring);
    }

    let v = Array2::<i64>::eye(n);
    (u, v, t, n1, n2)
}

/// Step 2: recursively Smith-form principal and trailing blocks.
fn step2_recursive_blocks(
    a: &Array2<i64>,
    n1: usize,
    _n2: usize,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();

    let b1 = a.slice(s![..n1, ..n1]).to_owned();
    let b2 = a.slice(s![n1 + 1..n, n1 + 1..n]).to_owned();

    let (u1_loc, v1_loc, _s1) = smith_from_upper_2_banded(&b1, ring);
    let (u2_loc, v2_loc, _s2) = smith_from_upper_2_banded(&b2, ring);

    let mut u = Array2::<i64>::eye(n);
    let mut v = Array2::<i64>::eye(n);

    // Embed local transforms
    u.slice_mut(s![..n1, ..n1]).assign(&u1_loc);
    u.slice_mut(s![n1 + 1..n, n1 + 1..n]).assign(&u2_loc);
    v.slice_mut(s![..n1, ..n1]).assign(&v1_loc);
    v.slice_mut(s![n1 + 1..n, n1 + 1..n]).assign(&v2_loc);

    let a_new = matmul_mod(&matmul_mod(&u, a, n_mod), &v, n_mod);
    (u, v, a_new)
}

/// Step 3: permute spike to last row/column.
///
/// The permutation moves row/col n1 to the end and shifts
/// rows/cols [n1+1..n) up by one. This is just an index remap.
fn step3_permute(a: &Array2<i64>, n1: usize) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n = a.nrows();

    // Build permutation mapping: new_index → old_index
    let mut fwd = vec![0usize; n]; // fwd[new] = old
    for i in 0..n1 {
        fwd[i] = i;
    }
    fwd[n - 1] = n1;
    for i in n1 + 1..n {
        fwd[i - 1] = i;
    }

    // Build inverse: old_index → new_index
    let mut inv = vec![0usize; n];
    for (new_i, &old_i) in fwd.iter().enumerate() {
        inv[old_i] = new_i;
    }

    // A3[i, j] = A[fwd[i], fwd[j]] — O(n²) index remap
    let mut a3 = Array2::zeros((n, n));
    for i in 0..n {
        let old_i = fwd[i];
        for j in 0..n {
            a3[[i, j]] = a[[old_i, fwd[j]]];
        }
    }

    // Build permutation matrices (needed by callers for transform composition)
    let mut perm = Array2::zeros((n, n));
    for (new_i, &old_i) in fwd.iter().enumerate() {
        perm[[new_i, old_i]] = 1;
    }
    let perm_t = perm.t().to_owned();

    (perm, perm_t, a3)
}

/// Step 4: diagonalize (n-1) x (n-1) principal block.
fn step4_smith_on_n_minus_1(
    a: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();

    let b = a.slice(s![..n - 1, ..n - 1]).to_owned();

    // Use diagonal SNF
    use crate::diagonal::smith_from_diagonal;
    let (u_loc, v_loc, _s_loc) = smith_from_diagonal(&b, ring);

    let mut u = Array2::<i64>::eye(n);
    let mut v = Array2::<i64>::eye(n);
    u.slice_mut(s![..n - 1, ..n - 1]).assign(&u_loc);
    v.slice_mut(s![..n - 1, ..n - 1]).assign(&v_loc);

    let a_new = matmul_mod(&matmul_mod(&u, a, n_mod), &v, n_mod);
    (u, v, a_new)
}

/// Steps 5-8: gcd chain.
fn step5_to_8_gcd_chain(
    a: &Array2<i64>,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>, usize) {
    let n_mod = ring.n();
    let n = a.nrows();

    let mut u = Array2::<i64>::eye(n);
    let mut v = Array2::<i64>::eye(n);
    let mut t = a.clone();

    // Find idx_k: first zero diagonal entry
    let mut idx_k = n - 1;
    for i in 0..n - 1 {
        if ring.is_zero(t[[i, i]]) {
            idx_k = i;
            break;
        }
    }

    let last_col = n - 1;

    // Step 5: eliminate entries below idx_k in last column
    for row in idx_k + 1..n {
        let target_val = t[[row, last_col]];
        if ring.is_zero(target_val) {
            continue;
        }
        let pivot_val = t[[idx_k, last_col]];
        let (_, s, tv, uv, vv) = ring.gcdex(pivot_val, target_val);
        apply_row_2x2_pair(&mut t, &mut u, idx_k, row, s, tv, uv, vv, n_mod, ring);
    }

    // Step 6: swap columns idx_k and last_col
    if idx_k != last_col {
        for row in 0..n {
            let tmp = t[[row, idx_k]];
            t[[row, idx_k]] = t[[row, last_col]];
            t[[row, last_col]] = tmp;

            let tmp = v[[row, idx_k]];
            v[[row, idx_k]] = v[[row, last_col]];
            v[[row, last_col]] = tmp;
        }
    }

    let col_target = idx_k;

    // Step 7: stabilize loop (ripple up)
    for i in (0..idx_k).rev() {
        let a_ik = t[[i, col_target]];
        let a_i1k = t[[i + 1, col_target]];
        let a_ii = t[[i, i]];

        // stab
        let c = {
            let target_gcd = ring.gcd(a_ik, ring.gcd(a_i1k, a_ii));
            let mut found = 0i64;
            for x in 0..n_mod {
                let candidate = mul_mod(x, a_i1k, n_mod);
                let candidate = posmod((a_ik as i64).wrapping_add(candidate), n_mod);
                let current = ring.gcd(candidate, a_ii);
                if current == target_gcd {
                    found = x;
                    break;
                }
            }
            found
        };

        let s_next = t[[i + 1, i + 1]];
        let numerator = mul_mod(c, s_next, n_mod);

        // quo
        let a_ii_ass = ring.gcd(a_ii, 0);
        let num_mod = posmod(numerator, n_mod);
        let rem = if a_ii_ass == 0 {
            num_mod
        } else {
            num_mod % a_ii_ass
        };
        let diff = posmod(num_mod - rem, n_mod);
        let q_raw = ring.div(diff, a_ii).unwrap_or(0);
        let q = posmod(-q_raw, n_mod);

        // Op 1: add c * row[i+1] to row[i]
        apply_row_2x2_pair(&mut t, &mut u, i, i + 1, 1, c, 0, 1, n_mod, ring);

        // Op 2: add q * col[i] to col[i+1]
        add_col_scaled(&mut t, i + 1, i, q, n_mod, ring);
        add_col_scaled(&mut v, i + 1, i, q, n_mod, ring);
    }

    // Step 8: gcd reduction loop (ripple down)
    for i in 0..idx_k {
        let pivot = t[[i, i]];
        let target = t[[i, col_target]];

        if ring.is_zero(target) {
            continue;
        }

        let (_, s, tv, uv, vv) = ring.gcdex(pivot, target);

        // Column operations on both t and v
        apply_col_2x2(&mut t, i, col_target, s, tv, uv, vv, n_mod, ring);
        apply_col_2x2(&mut v, i, col_target, s, tv, uv, vv, n_mod, ring);
    }

    (u, v, t, idx_k + 1)
}

/// Step 9: final index reduction.
fn step9_index_reduction(
    a: &Array2<i64>,
    k: usize,
    ring: &RingZModN,
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n_mod = ring.n();
    let n = a.nrows();
    let idx_k = k - 1;

    // Build reversal permutation P
    let mut p = Array2::zeros((n, n));
    for i in 0..n {
        if i <= idx_k {
            p[[i, idx_k - i]] = 1;
        } else {
            p[[i, i]] = 1;
        }
    }

    // A_perm = P @ A @ P
    let a_perm = matmul_mod(&matmul_mod(&p, a, n_mod), &p, n_mod);

    // Index-1 reduction
    let (u_red, _a_red) = index1_reduce_on_columns(&a_perm, k, ring);

    // U_final = P @ U_red @ P
    let u_final = matmul_mod(&matmul_mod(&p, &u_red, n_mod), &p, n_mod);
    let v_final = Array2::<i64>::eye(n);

    let a_final = matmul_mod(&u_final, a, n_mod);

    (u_final, v_final, a_final)
}
