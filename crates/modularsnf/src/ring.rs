//! Modular arithmetic for Z/NZ — Rust port of modularsnf/ring.py.
//!
//! All operations match Storjohann's Dissertation Section 1.1.
//!
//! For small moduli (N < 128), precomputes lookup tables for gcdex
//! (all N² input pairs) and modular reduction, eliminating computation
//! from the hot path entirely.

/// Maximum modulus for which we precompute gcdex and mod LUTs.
const LUT_MAX: i64 = 128;

/// Extended GCD: returns (g, x, y) such that a*x + b*y = g.
#[inline]
fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
    let (mut r0, mut r1) = (a, b);
    let (mut s0, mut s1) = (1i64, 0i64);
    let (mut t0, mut t1) = (0i64, 1i64);
    while r1 != 0 {
        let q = r0 / r1;
        let tmp = r1;
        r1 = r0 - q * r1;
        r0 = tmp;
        let tmp = s1;
        s1 = s0 - q * s1;
        s0 = tmp;
        let tmp = t1;
        t1 = t0 - q * t1;
        t0 = tmp;
    }
    (r0, s0, t0)
}

#[inline]
fn gcd_raw(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Positive modulo — always returns value in [0, n).
#[inline]
fn posmod(a: i64, n: i64) -> i64 {
    let r = a % n;
    if r < 0 { r + n } else { r }
}

/// Positive modulo for widened intermediates.
#[inline]
pub(crate) fn posmod_i128(a: i128, n: i64) -> i64 {
    let n128 = n as i128;
    let r = a % n128;
    (if r < 0 { r + n128 } else { r }) as i64
}

/// Modular addition for inputs already in [0, n).
#[inline]
pub(crate) fn add_mod(a: i64, b: i64, n: i64) -> i64 {
    let s = a + b;
    if s >= n { s - n } else { s }
}

/// Modular subtraction for inputs already in [0, n).
#[inline]
pub(crate) fn sub_mod(a: i64, b: i64, n: i64) -> i64 {
    let d = a - b;
    if d < 0 { d + n } else { d }
}

/// Modular multiplication for inputs already in [0, n).
#[inline]
pub(crate) fn mul_mod(a: i64, b: i64, n: i64) -> i64 {
    if n < (1i64 << 31) {
        (a * b) % n
    } else {
        posmod_i128((a as i128) * (b as i128), n)
    }
}

/// Precomputed gcdex entry: (g, s, t, u, v).
#[derive(Clone, Copy)]
struct GcdexEntry {
    g: i64,
    s: i64,
    t: i64,
    u: i64,
    v: i64,
}

/// Precomputed lookup tables for small moduli.
struct SmallModLut {
    /// Gcdex LUT: gcdex_lut[a * n + b] = gcdex(a, b) for a, b ∈ [0, n).
    gcdex_lut: Vec<GcdexEntry>,
    /// Modular reduction LUT: mod_lut[x] = x % n for x ∈ [0, 2*(n-1)²].
    /// Used for fast reduction of s*a + t*b products.
    mod_lut: Vec<i64>,
    /// Size of the mod LUT.
    mod_lut_size: usize,
}

pub struct RingZModN {
    n: i64,
    lut: Option<SmallModLut>,
}

/// Compute gcdex for (a, b) in Z/n without using the LUT.
/// This is the "slow path" used during LUT construction and for large n.
fn gcdex_slow(a_val: i64, b_val: i64, n: i64) -> (i64, i64, i64, i64, i64) {
    // Fast path: b is a multiple of a in Z/N
    if a_val != 0 && (b_val % gcd_raw(a_val, n) == 0) {
        let b_mod = posmod(b_val, n);
        let a_mod = posmod(a_val, n);
        let (g, x, _) = egcd(a_mod, n);
        if g != 0 && b_mod % g == 0 {
            let q = posmod_i128((x as i128) * ((b_mod / g) as i128), n / g);
            return (a_mod, 1, 0, posmod(-q, n), 1);
        }
    }

    // Standard extended Euclidean
    let (mut r0, mut r1) = (a_val, b_val);
    let (mut s0, mut s1) = (1i64, 0i64);
    let (mut t0, mut t1) = (0i64, 1i64);

    while r1 != 0 {
        let q = r0 / r1;
        let tmp = r1;
        r1 = r0 - q * r1;
        r0 = tmp;
        let tmp = s1;
        s1 = s0 - q * s1;
        s0 = tmp;
        let tmp = t1;
        t1 = t0 - q * t1;
        t0 = tmp;
    }

    if r0 == 0 {
        return (0, 1, 0, 0, 1);
    }

    let u = -(b_val / r0);
    let v = a_val / r0;

    (
        posmod(r0, n),
        posmod(s0, n),
        posmod(t0, n),
        posmod(u, n),
        posmod(v, n),
    )
}

impl RingZModN {
    pub fn new(n: i64) -> Result<Self, String> {
        if n < 2 {
            return Err("Modulus N must be >= 2".to_string());
        }

        let lut = if n < LUT_MAX {
            let n_usize = n as usize;

            // Build gcdex LUT: all N² pairs
            let mut gcdex_lut = Vec::with_capacity(n_usize * n_usize);
            for a in 0..n {
                for b in 0..n {
                    let (g, s, t, u, v) = gcdex_slow(a, b, n);
                    gcdex_lut.push(GcdexEntry { g, s, t, u, v });
                }
            }

            // Build mod LUT for fast reduction of products.
            // For apply_row_2x2: s*a + t*b where s,a,t,b ∈ [0, n).
            // Max positive value: 2*(n-1)² ≈ 32258 for n=128.
            // Min negative value: after i64 %, could be -(2*(n-1)²).
            // We store mod results for [0, 2*(n-1)²].
            let max_val = 2 * (n - 1) * (n - 1);
            let mod_lut_size = (max_val + 1) as usize;
            let mut mod_lut = Vec::with_capacity(mod_lut_size);
            for i in 0..mod_lut_size {
                mod_lut.push((i as i64) % n);
            }

            Some(SmallModLut {
                gcdex_lut,
                mod_lut,
                mod_lut_size,
            })
        } else {
            None
        };

        Ok(Self { n, lut })
    }

    #[inline]
    pub fn n(&self) -> i64 {
        self.n
    }

    /// Fast modular reduction using LUT when available.
    /// Input must be non-negative and < mod_lut_size.
    #[inline]
    pub fn fast_mod(&self, val: i64) -> i64 {
        if let Some(ref lut) = self.lut {
            let v = val as usize;
            if v < lut.mod_lut_size {
                return unsafe { *lut.mod_lut.get_unchecked(v) };
            }
        }
        val % self.n
    }

    /// Returns true if this ring has precomputed LUTs.
    #[inline]
    pub fn has_lut(&self) -> bool {
        self.lut.is_some()
    }

    pub fn add(&self, a: i64, b: i64) -> i64 {
        add_mod(a, b, self.n)
    }

    pub fn sub(&self, a: i64, b: i64) -> i64 {
        sub_mod(a, b, self.n)
    }

    pub fn mul(&self, a: i64, b: i64) -> i64 {
        mul_mod(a, b, self.n)
    }

    pub fn is_zero(&self, a: i64) -> bool {
        posmod(a, self.n) == 0
    }

    pub fn gcd(&self, a: i64, b: i64) -> i64 {
        gcd_raw(posmod(a, self.n), gcd_raw(posmod(b, self.n), self.n))
    }

    pub fn ass(&self, a: i64) -> i64 {
        gcd_raw(posmod(a, self.n), self.n)
    }

    pub fn ann(&self, a: i64) -> i64 {
        let g = gcd_raw(posmod(a, self.n), self.n);
        posmod(self.n / g, self.n)
    }

    pub fn rem(&self, a: i64, b: i64) -> i64 {
        let b_ass = self.ass(b);
        let a_val = posmod(a, self.n);
        if b_ass == 0 {
            a_val
        } else {
            a_val % b_ass
        }
    }

    pub fn quo(&self, a: i64, b: i64) -> Result<i64, String> {
        let r = self.rem(a, b);
        let diff = self.sub(a, r);
        self.div(diff, b)
    }

    pub fn div(&self, a: i64, b: i64) -> Result<i64, String> {
        let a_val = posmod(a, self.n);
        let b_val = posmod(b, self.n);
        let (g, x, _) = egcd(b_val, self.n);
        if g == 0 || a_val % g != 0 {
            return Err(format!(
                "{a} not divisible by {b} in Z/{}",
                self.n
            ));
        }
        Ok(posmod_i128((x as i128) * ((a_val / g) as i128), self.n / g))
    }

    pub fn unit(&self, a: i64) -> i64 {
        let a_val = posmod(a, self.n);
        let (_, u, _) = egcd(a_val, self.n);
        posmod(u, self.n)
    }

    pub fn gcdex(&self, a: i64, b: i64) -> (i64, i64, i64, i64, i64) {
        let a_val = posmod(a, self.n);
        let b_val = posmod(b, self.n);

        // LUT path for small moduli
        if let Some(ref lut) = self.lut {
            let idx = a_val as usize * self.n as usize + b_val as usize;
            let e = unsafe { lut.gcdex_lut.get_unchecked(idx) };
            return (e.g, e.s, e.t, e.u, e.v);
        }

        gcdex_slow(a_val, b_val, self.n)
    }

    pub fn stab(&self, a: i64, b: i64, c: i64) -> Result<i64, String> {
        let a = posmod(a, self.n);
        let b = posmod(b, self.n);
        let c = posmod(c, self.n);
        let target = self.gcd(a, self.gcd(b, c));
        for x in 0..self.n {
            let candidate = posmod_i128((a as i128) + (x as i128) * (b as i128), self.n);
            let current = self.gcd(candidate, c);
            if current == target {
                return Ok(x);
            }
        }
        Err(format!(
            "Stab failed for a={a}, b={b}, c={c} in Z/{}",
            self.n
        ))
    }
}
