//! Modular arithmetic for Z/NZ — Rust port of modularsnf/ring.py.
//!
//! All operations match Storjohann's Dissertation Section 1.1.

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
///
/// Since a, b ∈ [0, n), their sum is in [0, 2n-2].
/// A single conditional subtraction replaces i128 division.
#[inline]
pub(crate) fn add_mod(a: i64, b: i64, n: i64) -> i64 {
    let s = a + b;
    if s >= n { s - n } else { s }
}

/// Modular subtraction for inputs already in [0, n).
///
/// Since a, b ∈ [0, n), their difference is in [-(n-1), n-1].
/// A single conditional addition replaces i128 division.
#[inline]
pub(crate) fn sub_mod(a: i64, b: i64, n: i64) -> i64 {
    let d = a - b;
    if d < 0 { d + n } else { d }
}

/// Modular multiplication for inputs already in [0, n).
///
/// For n < 2^31 (covers all practical moduli), a*b fits in i64.
/// Uses i64 `%` instead of expensive i128 `__modti3`.
#[inline]
pub(crate) fn mul_mod(a: i64, b: i64, n: i64) -> i64 {
    // For n < 2^31, (n-1)^2 < 2^62 which fits in i64.
    // For larger n, widen to i128 to avoid overflow.
    if n < (1i64 << 31) {
        (a * b) % n
    } else {
        posmod_i128((a as i128) * (b as i128), n)
    }
}

pub struct RingZModN {
    n: i64,
}

impl RingZModN {
    pub fn new(n: i64) -> Result<Self, String> {
        if n < 2 {
            return Err("Modulus N must be >= 2".to_string());
        }
        Ok(Self { n })
    }

    #[inline]
    pub fn n(&self) -> i64 {
        self.n
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

        // Fast path: b is a multiple of a in Z/N
        if a_val != 0 && (b_val % gcd_raw(a_val, self.n) == 0) {
            if let Ok(q) = self.div(b_val, a_val) {
                return (a_val, 1, 0, posmod(-q, self.n), 1);
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
            posmod(r0, self.n),
            posmod(s0, self.n),
            posmod(t0, self.n),
            posmod(u, self.n),
            posmod(v, self.n),
        )
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
