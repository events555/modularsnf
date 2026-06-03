//! CRT-style Smith Normal Form over Z/NZ — non-recursive.
//!
//! Factor `N = prod p^e` (passed in, so factoring is amortized across many
//! calls with the same modulus), compute the SNF over each local ring
//! `Z/p^e` by valuation-pivoted Gaussian elimination — which yields the
//! divisibility chain for free and the unimodular transforms natively — then
//! CRT-recombine the per-prime transforms into `U, V` over `Z/NZ`.

use ndarray::Array2;

use crate::ring::posmod_i128;

#[inline]
fn pmod(a: i64, n: i64) -> i64 {
    posmod_i128(a as i128, n)
}

/// Threshold below which q^2 (and a - b*c with a,b,c in [0,q)) fits in i64,
/// so the hot loops avoid i128. q < 2^31 => q^2 < 2^62, comfortably in range.
const I64_FAST_MAX: i64 = 1i64 << 31;

#[inline]
fn mulmod(a: i64, b: i64, n: i64) -> i64 {
    if n < I64_FAST_MAX {
        let r = (a * b) % n;
        if r < 0 { r + n } else { r }
    } else {
        posmod_i128(a as i128 * b as i128, n)
    }
}

/// Reduce `a - b*c` into [0, q). Inputs `a, b, c` are in [0, q).
/// i64 fast path when q < 2^31; widens to i128 for larger prime-power moduli.
#[inline]
fn sub_mul_mod(a: i64, b: i64, c: i64, q: i64) -> i64 {
    if q < I64_FAST_MAX {
        let r = (a - b * c) % q;
        if r < 0 { r + q } else { r }
    } else {
        posmod_i128(a as i128 - b as i128 * c as i128, q)
    }
}

fn egcd_i128(a: i128, b: i128) -> (i128, i128, i128) {
    let (mut r0, mut r1) = (a, b);
    let (mut s0, mut s1) = (1i128, 0i128);
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        let t = r1;
        r1 = r0 - q * r1;
        r0 = t;
        let t = s1;
        s1 = s0 - q * s1;
        s0 = t;
        let t = t1;
        t1 = t0 - q * t1;
        t0 = t;
    }
    (r0, s0, t0)
}

fn inv_mod(a: i64, m: i64) -> i64 {
    let (_g, x, _) = egcd_i128(pmod(a, m) as i128, m as i128);
    let r = ((x % m as i128) + m as i128) % m as i128;
    r as i64
}

/// p-adic valuation of `x` in `[0, p^cap)`, capped at `cap`; `x == 0 -> cap`.
fn pval(mut x: i64, p: i64, cap: u32) -> u32 {
    if x == 0 {
        return cap;
    }
    let mut v = 0u32;
    while v < cap && x % p == 0 {
        x /= p;
        v += 1;
    }
    v
}

/// Block width for the right-looking blocked LU (Phase L, unit stage).
const PANEL_B: usize = 48;
/// The i64 GEMM accumulator is safe while `PANEL_B * (q-1)^2 < 2^63`. q < 2^28
/// gives 48 * 2^56 < 2^62, so use the i64 micro-kernel below it, i128 above.
const GEMM_I64_MAX: i64 = 1i64 << 28;

/// Minimal p-adic valuation over `mat[k.., k..]` (returns `e+1` if all zero).
fn min_valuation(mat: &Array2<i64>, k: usize, n: usize, m: usize, p: i64, e: u32) -> u32 {
    let mut lev = e + 1;
    for i in k..n {
        for j in k..m {
            let val = mat[[i, j]];
            if val != 0 {
                let vv = pval(val, p, e);
                if vv < lev {
                    if vv == 0 {
                        return 0;
                    }
                    lev = vv;
                }
            }
        }
    }
    lev
}

/// Apply a deferred panel update to `target` over columns `[c0, c1)`:
/// first TRSM the panel rows `[k, k+pb)` by the unit-lower-triangular `L11`,
/// then GEMM the trailing rows `[k+pb, n)` as `-= L21 @ panel (mod q)`.
/// `l11[t*pb + s]` (s < t) and `l21[i*pb + s]` are the stored multipliers.
fn apply_block_update(
    target: &mut Array2<i64>,
    l11: &[i64],
    l21: &[i64],
    k: usize,
    pb: usize,
    c0: usize,
    c1: usize,
    q: i64,
) {
    if c1 <= c0 {
        return;
    }
    let n = target.nrows();
    let pend = k + pb;
    // TRSM: forward substitution against unit-lower-triangular L11.
    for t in 1..pb {
        for s in 0..t {
            let f = l11[t * pb + s];
            if f != 0 {
                for col in c0..c1 {
                    target[[k + t, col]] =
                        sub_mul_mod(target[[k + t, col]], f, target[[k + s, col]], q);
                }
            }
        }
    }
    // GEMM trailing update: trailing rows -= L21 @ U12 (mod q).
    // Pack U12 (post-TRSM panel rows) into a contiguous (pb x ncols) buffer so
    // the inner axpy streams contiguously instead of striding by row length.
    let trail = n - pend;
    if trail == 0 {
        return;
    }
    let ncols = c1 - c0;
    let mut rpack = vec![0i64; pb * ncols];
    for s in 0..pb {
        let base = s * ncols;
        for j in 0..ncols {
            rpack[base + j] = target[[k + s, c0 + j]];
        }
    }
    if q < GEMM_I64_MAX {
        let mut acc = vec![0i64; ncols];
        for i in 0..trail {
            let row = pend + i;
            for j in 0..ncols {
                acc[j] = target[[row, c0 + j]];
            }
            for s in 0..pb {
                let l = l21[i * pb + s];
                if l != 0 {
                    let base = s * ncols;
                    for j in 0..ncols {
                        acc[j] -= l * rpack[base + j]; // contiguous, autovectorizes
                    }
                }
            }
            for j in 0..ncols {
                let rr = acc[j] % q;
                target[[row, c0 + j]] = if rr < 0 { rr + q } else { rr };
            }
        }
    } else {
        let mut acc = vec![0i128; ncols];
        for i in 0..trail {
            let row = pend + i;
            for j in 0..ncols {
                acc[j] = target[[row, c0 + j]] as i128;
            }
            for s in 0..pb {
                let l = l21[i * pb + s] as i128;
                if l != 0 {
                    let base = s * ncols;
                    for j in 0..ncols {
                        acc[j] -= l * rpack[base + j] as i128;
                    }
                }
            }
            for j in 0..ncols {
                target[[row, c0 + j]] = posmod_i128(acc[j], q);
            }
        }
    }
}

/// Blocked right-looking LU over the unit (valuation-0) part of `mat`.
/// Eliminates leading columns whose pivot is a unit (mod p), updating `mat`
/// and `u`; returns the first pivot index it could not place (a column with no
/// unit), which the scalar path then finishes. `v` is untouched (no column
/// swaps here). Pre-condition: `min_valuation(mat, 0, ..) == 0`.
fn phase_l_blocked_unit(
    mat: &mut Array2<i64>,
    u: &mut Array2<i64>,
    p: i64,
    e: u32,
    q: i64,
    n: usize,
    m: usize,
    r: usize,
) -> usize {
    let mut k = 0;
    while k < r {
        let pend_max = (k + PANEL_B).min(r);

        // ---- factor the panel columns [k, pend_max): unit row-pivoting ----
        let mut kk = k;
        while kk < pend_max {
            // find a unit (valuation 0) in column kk, rows [kk, n)
            let mut prow = None;
            for i in kk..n {
                if mat[[i, kk]] != 0 && pval(mat[[i, kk]], p, e) == 0 {
                    prow = Some(i);
                    break;
                }
            }
            let pr = match prow {
                Some(x) => x,
                None => break, // column kk has no unit -> end panel (then scalar)
            };
            if pr != kk {
                // full-width row swap keeps the deferred block + u consistent
                for col in 0..m {
                    mat.swap([kk, col], [pr, col]);
                }
                for col in 0..n {
                    u.swap([kk, col], [pr, col]);
                }
            }
            // Divide-by-pivot multipliers (the pivot is a unit). The pivot row
            // is NOT normalized here; the diagonal is normalized once at the end
            // so that mat and u undergo identical row operations — u's elimination
            // is deferred (TRSM/GEMM via the stored L), and a normalize-then-
            // eliminate / eliminate-then-normalize mismatch would corrupt it.
            let uinv = inv_mod(mat[[kk, kk]], q);
            for i in (kk + 1)..n {
                if mat[[i, kk]] != 0 {
                    let c = mulmod(mat[[i, kk]], uinv, q); // = mat[i,kk] / mat[kk,kk]
                    for col in (kk + 1)..pend_max {
                        mat[[i, col]] = sub_mul_mod(mat[[i, col]], c, mat[[kk, col]], q);
                    }
                    mat[[i, kk]] = c; // store the multiplier (L)
                }
            }
            kk += 1;
        }
        let pend = kk;
        if pend == k {
            return k; // no unit in column k -> hand the rest to the scalar path
        }
        let pb = pend - k;

        // ---- extract L11 (unit lower-tri) and L21 (trailing-row) multipliers ----
        let trail = n - pend;
        let mut l11 = vec![0i64; pb * pb];
        for t in 0..pb {
            for s in 0..t {
                l11[t * pb + s] = mat[[k + t, k + s]];
            }
        }
        let mut l21 = vec![0i64; trail * pb];
        for i in 0..trail {
            for s in 0..pb {
                l21[i * pb + s] = mat[[pend + i, k + s]];
            }
        }

        // ---- deferred TRSM + GEMM on mat (cols [pend_max, m)) and u (all cols) ----
        apply_block_update(mat, &l11, &l21, k, pb, pend_max, m, q);
        apply_block_update(u, &l11, &l21, k, pb, 0, n, q);

        // ---- zero the L storage below the diagonal of the pivot columns ----
        for t in 0..pb {
            for i in (k + t + 1)..n {
                mat[[i, k + t]] = 0;
            }
        }
        k = pend;
    }
    k
}

/// Scalar Phase L (minimal-valuation column elimination), continuing from
/// `k_start`. Fully general: global min-valuation pivot + row/column swaps.
/// Produces `u` (and column swaps tracked in `v`); leaves `mat` upper-triangular.
fn phase_l_scalar(
    mat: &mut Array2<i64>,
    u: &mut Array2<i64>,
    v: &mut Array2<i64>,
    p: i64,
    e: u32,
    q: i64,
    n: usize,
    m: usize,
    r: usize,
    k_start: usize,
) {
    for k in k_start..r {
        // Find the pivot in mat[k.., k..] with minimal p-adic valuation.
        let mut best: Option<(usize, usize)> = None;
        let mut bestval = e + 1;
        'search: for i in k..n {
            for j in k..m {
                let val = mat[[i, j]];
                if val != 0 {
                    let vv = pval(val, p, e);
                    if vv < bestval {
                        bestval = vv;
                        best = Some((i, j));
                        if vv == 0 {
                            break 'search;
                        }
                    }
                }
            }
        }
        let (pi, pj) = match best {
            Some(x) => x,
            None => break, // trailing block is entirely zero
        };

        if pi != k {
            for col in 0..m {
                mat.swap([k, col], [pi, col]);
            }
            for col in 0..n {
                u.swap([k, col], [pi, col]);
            }
        }
        if pj != k {
            for row in 0..n {
                mat.swap([row, k], [row, pj]);
            }
            for row in 0..m {
                v.swap([row, k], [row, pj]);
            }
        }

        let pv = p.pow(bestval);

        // Normalize the pivot to exactly p^vv by scaling row k by a unit.
        let unit = mat[[k, k]] / pv;
        let uinv = inv_mod(unit, q);
        for col in 0..m {
            mat[[k, col]] = mulmod(mat[[k, col]], uinv, q);
        }
        for col in 0..n {
            u[[k, col]] = mulmod(u[[k, col]], uinv, q);
        }

        // Clear column k below the pivot (Schur update of the trailing block).
        for i in (k + 1)..n {
            let val = mat[[i, k]];
            if val != 0 {
                let c = val / pv;
                for col in (k + 1)..m {
                    mat[[i, col]] = sub_mul_mod(mat[[i, col]], c, mat[[k, col]], q);
                }
                mat[[i, k]] = 0;
                for col in 0..n {
                    u[[i, col]] = sub_mul_mod(u[[i, col]], c, u[[k, col]], q);
                }
            }
        }
    }
}

/// SNF of `A` over the local ring `Z/p^e` via valuation pivoting.
///
/// Returns `(U, V, vals)` with `U @ A @ V == diag(p^vals) (mod p^e)`,
/// `U` (n x n) and `V` (m x m) unimodular and `vals` ascending.
fn local_snf(a: &Array2<i64>, p: i64, e: u32) -> (Array2<i64>, Array2<i64>, Vec<u32>) {
    let q = p.pow(e);
    let n = a.nrows();
    let m = a.ncols();
    let r = n.min(m);

    let mut mat = a.mapv(|x| pmod(x, q));
    let mut u = Array2::<i64>::eye(n);
    let mut v = Array2::<i64>::eye(m);

    // ---- Phase L: column elimination -> U, mat becomes upper-triangular ----
    // Blocked LU clears the valuation-0 bulk (trailing update = GEMM); the
    // scalar path finishes the higher-valuation / column-swap tail.
    let k0 = if min_valuation(&mat, 0, n, m, p, e) == 0 {
        phase_l_blocked_unit(&mut mat, &mut u, p, e, q, n, m, r)
    } else {
        0
    };
    phase_l_scalar(&mut mat, &mut u, &mut v, p, e, q, n, m, r, k0);

    let vals: Vec<u32> = (0..r).map(|i| pval(mat[[i, i]], p, e)).collect();

    // Normalize each pivot diagonal to exactly p^vals[k]: blocked pivots were
    // left as units; scalar pivots are already p^vals so this no-ops for them.
    for k in 0..r {
        if mat[[k, k]] == 0 {
            continue; // zero invariant factor (p^e ≡ 0)
        }
        let pvk = p.pow(vals[k]);
        if mat[[k, k]] != pvk {
            let unit = mat[[k, k]] / pvk; // exact: pivot = p^vals * unit
            let sc = inv_mod(unit, q);
            for col in 0..m {
                mat[[k, col]] = mulmod(mat[[k, col]], sc, q);
            }
            for col in 0..n {
                u[[k, col]] = mulmod(u[[k, col]], sc, q);
            }
        }
    }

    // ---- Phase R: diagonalize the upper-triangular mat -> V ----
    // Clear super-diagonal entries by column operations, processing pivots from
    // last to first so that fill created above row k lands in not-yet-processed
    // rows (and is cleared when those pivots are reached). col k of the upper-
    // triangular mat is nonzero only in rows 0..=k.
    for k in (0..r).rev() {
        let pv = p.pow(vals[k]);
        for j in (k + 1)..m {
            let val = mat[[k, j]];
            if val != 0 {
                let c = val / pv; // exact: mat[k,j] divisible by pv
                for row in 0..=k {
                    mat[[row, j]] = sub_mul_mod(mat[[row, j]], c, mat[[row, k]], q);
                }
                for row in 0..m {
                    v[[row, j]] = sub_mul_mod(v[[row, j]], c, v[[row, k]], q);
                }
            }
        }
    }

    (u, v, vals)
}

/// Incremental CRT: combine `residues` mod pairwise-coprime `moduli`.
fn crt_combine(residues: &[i64], moduli: &[i64]) -> i64 {
    let mut x: i128 = 0;
    let mut m_acc: i128 = 1;
    for (&r, &m) in residues.iter().zip(moduli) {
        let m128 = m as i128;
        let m_mod = ((m_acc % m128) + m128) % m128;
        let (_g, s, _) = egcd_i128(m_mod, m128);
        let inv = ((s % m128) + m128) % m128;
        let diff = (((r as i128 - x) % m128) + m128) % m128;
        let t = (diff * inv) % m128;
        x += m_acc * t;
        m_acc *= m128;
        x = ((x % m_acc) + m_acc) % m_acc;
    }
    x as i64
}

/// CRT-style SNF over `Z/NZ`. `factors` is the prime factorization of
/// `modulus` as `(p, e)` pairs (passed in so it can be amortized).
///
/// Returns `(U, V, S)` with `S = U @ A @ V (mod N)`, `S` diagonal with the
/// divisibility chain and `U, V` unimodular.
pub fn crt_snf(
    a: &Array2<i64>,
    modulus: i64,
    factors: &[(i64, u32)],
) -> (Array2<i64>, Array2<i64>, Array2<i64>) {
    let n = a.nrows();
    let m = a.ncols();
    let r = n.min(m);
    let np = factors.len();

    let qs: Vec<i64> = factors.iter().map(|&(p, e)| p.pow(e)).collect();

    // Phase 0 profiling: set CRT_PROFILE=1 to print a phase-time breakdown.
    let prof = std::env::var("CRT_PROFILE").is_ok();
    let t_local = std::time::Instant::now();

    let mut locals: Vec<(Array2<i64>, Array2<i64>, Vec<u32>)> =
        factors.iter().map(|&(p, e)| local_snf(a, p, e)).collect();
    let d_local = t_local.elapsed();

    // Global invariant factors d_i = prod_p p^{vals_p[i]} (divides N).
    let mut d = vec![0i64; r];
    for (i, di_slot) in d.iter_mut().enumerate() {
        let mut di: i128 = 1;
        for (pi, &(p, _)) in factors.iter().enumerate() {
            di *= (p as i128).pow(locals[pi].2[i]);
        }
        *di_slot = (di % modulus as i128) as i64;
    }

    // Unit-normalize each prime's V so all primes realize the same d_i.
    let t_norm = std::time::Instant::now();
    for pi in 0..np {
        let q = qs[pi];
        for i in 0..r {
            let mut w: i128 = 1;
            for (pj, &(p, _)) in factors.iter().enumerate() {
                if pj != pi {
                    w = (w * (p as i128).pow(locals[pj].2[i])) % q as i128;
                }
            }
            let w = w as i64;
            for row in 0..m {
                let val = locals[pi].1[[row, i]];
                locals[pi].1[[row, i]] = mulmod(val, w, q);
            }
        }
    }

    let d_norm = t_norm.elapsed();

    // CRT-recombine U (n x n) and V (m x m) entrywise across primes.
    let t_recomb = std::time::Instant::now();
    let mut u = Array2::<i64>::zeros((n, n));
    let mut resid = vec![0i64; np];
    for ar in 0..n {
        for br in 0..n {
            for pi in 0..np {
                resid[pi] = locals[pi].0[[ar, br]];
            }
            u[[ar, br]] = crt_combine(&resid, &qs);
        }
    }
    let mut v = Array2::<i64>::zeros((m, m));
    for ar in 0..m {
        for br in 0..m {
            for pi in 0..np {
                resid[pi] = locals[pi].1[[ar, br]];
            }
            v[[ar, br]] = crt_combine(&resid, &qs);
        }
    }

    let d_recomb = t_recomb.elapsed();

    let mut s = Array2::<i64>::zeros((n, m));
    for (i, &di) in d.iter().enumerate() {
        s[[i, i]] = di;
    }

    if prof {
        let total = d_local + d_norm + d_recomb;
        eprintln!(
            "[CRT_PROFILE] n={n} m={m} N={modulus} np={np} | \
             local_snf={:.3}ms ({:.0}%) normalize={:.3}ms ({:.0}%) \
             recombine={:.3}ms ({:.0}%) | total={:.3}ms",
            d_local.as_secs_f64() * 1e3,
            100.0 * d_local.as_secs_f64() / total.as_secs_f64(),
            d_norm.as_secs_f64() * 1e3,
            100.0 * d_norm.as_secs_f64() / total.as_secs_f64(),
            d_recomb.as_secs_f64() * 1e3,
            100.0 * d_recomb.as_secs_f64() / total.as_secs_f64(),
            total.as_secs_f64() * 1e3,
        );
    }

    (u, v, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Deterministic LCG so tests need no external rng dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 16
        }
        fn below(&mut self, n: i64) -> i64 {
            (self.next() % n as u64) as i64
        }
    }

    fn matmul_mod(a: &Array2<i64>, b: &Array2<i64>, q: i64) -> Array2<i64> {
        let (n, k) = (a.nrows(), a.ncols());
        let m = b.ncols();
        let mut c = Array2::<i64>::zeros((n, m));
        for i in 0..n {
            for j in 0..m {
                let mut acc: i128 = 0;
                for t in 0..k {
                    acc += a[[i, t]] as i128 * b[[t, j]] as i128;
                }
                c[[i, j]] = posmod_i128(acc, q);
            }
        }
        c
    }

    /// True iff det(M mod p) != 0 over the field F_p (=> M unimodular mod p^e).
    fn invertible_mod_p(mm: &Array2<i64>, p: i64) -> bool {
        let n = mm.nrows();
        let mut a = mm.mapv(|x| pmod(x, p));
        for k in 0..n {
            // find a nonzero pivot in column k at/below row k
            let mut piv = None;
            for i in k..n {
                if a[[i, k]] != 0 {
                    piv = Some(i);
                    break;
                }
            }
            let pr = match piv {
                Some(x) => x,
                None => return false, // singular mod p
            };
            if pr != k {
                for col in 0..n {
                    a.swap([k, col], [pr, col]);
                }
            }
            let inv = inv_mod(a[[k, k]], p);
            for i in (k + 1)..n {
                if a[[i, k]] != 0 {
                    let f = pmod(a[[i, k]] * inv, p);
                    for col in k..n {
                        a[[i, col]] = pmod(a[[i, col]] - f * a[[k, col]], p);
                    }
                }
            }
        }
        true
    }

    #[test]
    fn local_snf_is_valid_smith_form() {
        let cases = [(2u32, 1u32), (2, 3), (3, 2), (5, 1), (5, 2), (7, 3)];
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let mut checked = 0;
        for &(p, e) in &cases {
            let p = p as i64;
            let q = p.pow(e);
            for n in [1usize, 2, 5, 9, 16, 23] {
                for m in [1usize, 3, 5, 9, 16] {
                    for _trial in 0..6 {
                        let a = Array2::from_shape_fn((n, m), |_| rng.below(q));
                        assert_valid_snf(&a, p, e);
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 500, "expected many cases, got {checked}");
    }

    /// Assert local_snf(a, p, e) is a valid Smith form: U@A@V == diag(p^vals),
    /// ascending valuations, U,V unimodular.
    fn assert_valid_snf(a: &Array2<i64>, p: i64, e: u32) {
        let q = p.pow(e);
        let (n, m) = (a.nrows(), a.ncols());
        let r = n.min(m);
        let (uu, vv, vals) = local_snf(a, p, e);
        let prod = matmul_mod(&matmul_mod(&uu, a, q), &vv, q);
        for i in 0..n {
            for j in 0..m {
                let expect = if i == j && i < r { pmod(p.pow(vals[i]), q) } else { 0 };
                assert_eq!(prod[[i, j]], expect, "U*A*V at ({i},{j}) p={p} e={e} n={n} m={m}");
            }
        }
        for i in 1..r {
            assert!(vals[i - 1] <= vals[i], "vals not ascending {vals:?} p={p} e={e}");
        }
        assert!(invertible_mod_p(&uu, p), "U not unimodular p={p} e={e} n={n} m={m}");
        assert!(invertible_mod_p(&vv, p), "V not unimodular p={p} e={e} n={n} m={m}");
    }

    /// Exercise the blocked-LU paths: large n (multi-panel), rank-deficiency
    /// mod p (forces the blocked -> scalar handoff), and p|A (blocked skipped).
    #[test]
    fn local_snf_blocked_paths() {
        let cases = [(2u32, 1u32), (2, 4), (3, 2), (3, 3), (5, 2), (7, 2)];
        let mut rng = Lcg(0xdead_beef_0bad_f00d);
        let mut checked = 0;
        for &(p, e) in &cases {
            let p = p as i64;
            let q = p.pow(e);
            // Larger square sizes spanning the panel width (PANEL_B = 48).
            for &n in &[40usize, 49, 64, 97] {
                // (1) dense random: generically full-rank mod p (single long blocked run)
                let a = Array2::from_shape_fn((n, n), |_| rng.below(q));
                assert_valid_snf(&a, p, e);
                checked += 1;

                // (2) low rank mod p: A = B @ C with inner dim rk << n, so the
                //     blocked unit pass exhausts units early and hands to scalar.
                for &rk in &[1usize, 3, 7] {
                    let b = Array2::from_shape_fn((n, rk), |_| rng.below(q));
                    let c = Array2::from_shape_fn((rk, n), |_| rng.below(q));
                    let a = matmul_mod(&b, &c, q);
                    assert_valid_snf(&a, p, e);
                    checked += 1;
                }

                // (3) p | A everywhere: min valuation >= 1, blocked pass skipped.
                let a = Array2::from_shape_fn((n, n), |_| (rng.below(q / p.max(1)) * p) % q);
                assert_valid_snf(&a, p, e);
                checked += 1;

                // (4) mixed: a unit block plus a p-scaled block (multi-level).
                let a = Array2::from_shape_fn((n, n), |(i, j)| {
                    if (i + j) % 3 == 0 { rng.below(q) } else { (rng.below(q) * p) % q }
                });
                assert_valid_snf(&a, p, e);
                checked += 1;

                // (5) rectangular, both orientations.
                let a = Array2::from_shape_fn((n, n / 2 + 1), |_| rng.below(q));
                assert_valid_snf(&a, p, e);
                let a = Array2::from_shape_fn((n / 2 + 1, n), |_| rng.below(q));
                assert_valid_snf(&a, p, e);
                checked += 2;
            }
        }
        assert!(checked > 100, "expected many cases, got {checked}");
    }
}
