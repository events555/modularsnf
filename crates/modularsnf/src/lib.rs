pub mod band;
pub mod crt;
pub mod diagonal;
pub mod echelon;
pub mod ring;
pub mod snf;

pub use ring::RingZModN;

use ndarray::Array2;

/// Full Smith Normal Form: takes an n x m matrix and modulus,
/// returns (U, V, S) with S = U @ A @ V (mod N).
pub fn smith_normal_form(
    a: &Array2<i64>,
    modulus: i64,
) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>), String> {
    let r = RingZModN::new(modulus)?;
    let n = a.nrows();
    let m = a.ncols();

    if n == 0 || m == 0 {
        let u = Array2::<i64>::eye(n);
        let v = Array2::<i64>::eye(m);
        return Ok((u, v, a.clone()));
    }

    // Tall matrices (n > m) are routed through the transpose: padding columns
    // and cropping V can leave V non-unimodular for column-rank-deficient
    // inputs (e.g. zero columns), whereas the wide path crops U, which stays
    // unimodular. From `U' Aᵀ V' = S'` we get `(V'ᵀ) A (U'ᵀ) = S'ᵀ`.
    if n > m {
        let at = a.t().to_owned();
        let (u_t, v_t, s_t) = smith_normal_form(&at, modulus)?;
        return Ok((v_t.t().to_owned(), u_t.t().to_owned(), s_t.t().to_owned()));
    }

    // Pad to square if needed
    let s = n.max(m);
    let mut a_pad = Array2::zeros((s, s));
    a_pad.slice_mut(ndarray::s![..n, ..m]).assign(a);

    let (u_pad, v_pad, s_pad) = snf::smith_square(&a_pad, &r);

    // Crop back
    let u = u_pad.slice(ndarray::s![..n, ..n]).to_owned();
    let v = v_pad.slice(ndarray::s![..m, ..m]).to_owned();
    let s_mat = s_pad.slice(ndarray::s![..n, ..m]).to_owned();

    Ok((u, v, s_mat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crt::crt_snf;
    use crate::ring::posmod_i128;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            let t = b;
            b = a % b;
            a = t;
        }
        a.abs()
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

    /// Exact integer determinant by cofactor expansion (small n only).
    fn det_i128(a: &Array2<i64>) -> i128 {
        let n = a.nrows();
        let mut m: Vec<Vec<i128>> = (0..n)
            .map(|i| (0..n).map(|j| a[[i, j]] as i128).collect())
            .collect();
        det_recur(&mut m)
    }

    fn det_recur(m: &mut [Vec<i128>]) -> i128 {
        let n = m.len();
        if n == 1 {
            return m[0][0];
        }
        if n == 2 {
            return m[0][0] * m[1][1] - m[0][1] * m[1][0];
        }
        let mut det = 0i128;
        for j in 0..n {
            let mut sub: Vec<Vec<i128>> = m[1..]
                .iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .filter(|&(c, _)| c != j)
                        .map(|(_, &v)| v)
                        .collect()
                })
                .collect();
            let sign = if j % 2 == 0 { 1 } else { -1 };
            det += sign * m[0][j] * det_recur(&mut sub);
        }
        det
    }

    /// gcd(det(M) mod N, N) == 1: M is unimodular over Z/NZ.
    fn is_unimodular(m: &Array2<i64>, n: i64) -> bool {
        let d = posmod_i128(det_i128(m), n);
        gcd_i64(d, n) == 1
    }

    /// Normalized invariant factors: gcd(diag, N), ascending.
    fn normalized_invariants(s: &Array2<i64>, n: i64) -> Vec<i64> {
        let r = s.nrows().min(s.ncols());
        let mut inv: Vec<i64> = (0..r).map(|i| gcd_i64(s[[i, i]], n)).collect();
        inv.sort_unstable();
        inv
    }

    fn assert_valid_snf(a: &Array2<i64>, n: i64, check_unimodular: bool) {
        let (rows, cols) = (a.nrows(), a.ncols());
        let (u, v, s) = smith_normal_form(a, n).expect("smith_normal_form");

        assert_eq!(u.dim(), (rows, rows));
        assert_eq!(v.dim(), (cols, cols));
        assert_eq!(s.dim(), (rows, cols));

        // U A V == S (mod N)
        let prod = matmul_mod(&matmul_mod(&u, a, n), &v, n);
        for i in 0..rows {
            for j in 0..cols {
                assert_eq!(
                    prod[[i, j]],
                    posmod_i128(s[[i, j]] as i128, n),
                    "U A V != S at ({i},{j}) N={n} shape={rows}x{cols}"
                );
            }
        }

        // S diagonal.
        for i in 0..rows {
            for j in 0..cols {
                if i != j {
                    assert_eq!(s[[i, j]], 0, "S not diagonal at ({i},{j})");
                }
            }
        }

        // Divisibility chain on ideal generators: g_i | g_{i+1}.
        let inv = normalized_invariants(&s, n);
        for w in inv.windows(2) {
            assert_eq!(w[1] % w[0], 0, "invariants not a chain: {inv:?} N={n}");
        }

        if check_unimodular {
            assert!(is_unimodular(&u, n), "U not unimodular N={n}");
            assert!(is_unimodular(&v, n), "V not unimodular N={n}");
        }
    }

    #[test]
    fn smith_normal_form_is_valid() {
        let moduli = [2i64, 4, 6, 8, 9, 12, 30, 36, 100];
        let shapes = [(1, 1), (3, 3), (4, 6), (6, 4), (5, 1), (1, 5), (7, 7)];
        let mut rng = StdRng::seed_from_u64(0x0bad_c0de_1234_5678);
        let mut checked = 0;
        for &n in &moduli {
            for &(rows, cols) in &shapes {
                for _ in 0..4 {
                    let a = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(0..n));
                    // det helper is O(k!); only check unimodularity for small k.
                    assert_valid_snf(&a, n, rows.max(cols) <= 7);
                    checked += 1;
                }
            }
        }
        assert!(checked > 200, "expected many cases, got {checked}");
    }

    #[test]
    fn smith_normal_form_matches_crt() {
        let cases: [(i64, Vec<(i64, u32)>); 6] = [
            (6, vec![(2, 1), (3, 1)]),
            (12, vec![(2, 2), (3, 1)]),
            (36, vec![(2, 2), (3, 2)]),
            (8, vec![(2, 3)]),
            (30, vec![(2, 1), (3, 1), (5, 1)]),
            (100, vec![(2, 2), (5, 2)]),
        ];
        let shapes = [(3, 3), (4, 6), (6, 4), (5, 5), (1, 4)];
        let mut rng = StdRng::seed_from_u64(0xfeed_face_dead_beef);
        for (n, factors) in &cases {
            for &(rows, cols) in &shapes {
                for _ in 0..4 {
                    let a = Array2::from_shape_fn((rows, cols), |_| rng.gen_range(0..*n));
                    let (_, _, s_storj) = smith_normal_form(&a, *n).expect("storjohann");
                    let (_, _, s_crt) = crt_snf(&a, *n, factors);
                    assert_eq!(
                        normalized_invariants(&s_storj, *n),
                        normalized_invariants(&s_crt, *n),
                        "invariant mismatch N={n} shape={rows}x{cols}"
                    );
                }
            }
        }
    }
}
