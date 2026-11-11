//! Deterministic pseudo-random number generator and sampling distributions.
//!
//! Provenance: Implements SplitMix64 seeding (Steele, Lea & Flood, OOPSLA 2014) and
//! xoshiro256++ state transition (Blackman & Vigna, ACM TOMS 2021), with standard
//! Box-Muller Gaussian and Knuth/transformed-rejection Poisson sampling.

use std::f64::consts::TAU;

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next_sm = || {
            sm = sm.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        Self {
            s: [next_sm(), next_sm(), next_sm(), next_sm()],
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    /// Uniform integer in 0..upper (exclusive).
    #[inline]
    pub fn usize(&mut self, upper: usize) -> usize {
        debug_assert!(upper > 0);
        let upper_u = upper as u64;
        let zone = u64::MAX - (u64::MAX % upper_u);
        loop {
            let v = self.next_u64();
            if v < zone {
                return (v % upper_u) as usize;
            }
        }
    }

    /// Uniform integer in lo..hi (exclusive).
    #[inline]
    pub fn range_usize(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.usize(hi - lo)
    }

    /// Uniform float in [lo, hi).
    #[inline]
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }

    /// Normal(mean, sd) via Box-Muller transform.
    pub fn normal(&mut self, mean: f64, sd: f64) -> f64 {
        if let Some(z1) = self.spare_normal.take() {
            return mean + sd * z1;
        }
        let u1 = self.f64().max(1e-300);
        let u2 = self.f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = TAU * u2;
        self.spare_normal = Some(r * theta.sin());
        mean + sd * (r * theta.cos())
    }

    /// Poisson(lambda) sampling.
    pub fn poisson(&mut self, lambda: f64) -> usize {
        if lambda <= 0.0 {
            return 0;
        }
        if lambda < 30.0 {
            let l = (-lambda).exp();
            let mut k = 0usize;
            let mut p = 1.0;
            loop {
                k += 1;
                p *= self.f64();
                if p <= l {
                    return k - 1;
                }
            }
        } else {
            // Normal approximation with continuity correction for larger rates
            (self.normal(lambda, lambda.sqrt()).round().max(0.0)) as usize
        }
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.usize(i + 1);
            slice.swap(i, j);
        }
    }

    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        self.shuffle(&mut p);
        p
    }
}
