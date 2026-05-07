//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! Epsilon Memory Guards â€” Chebyshev Liveness Inheritance
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! Implements the statistical safety mechanism for Injected context data.
//!
//! # Problem (Section 3.2)
//!
//! The Chebyshev Evictor guards against false GC collections based on
//! time-alive and operation counts. Injected data has t_alive = 0,
//! making it highly susceptible to immediate eviction.
//!
//! # Solution: Signature Inheritance
//!
//! The Injected manifold block arrives with an inherited statistical
//! "anchor" from M_high. The [`ChebyshevGuard`] accepts pre-aged
//! k-standard deviation bounds via [`LivenessAnchor`], guaranteeing
//! the new context remains inside the safe boundary until the local
//! Bio-Kernel completes its rescan.
//!
//! # Mathematical Foundation
//!
//! Chebyshev's inequality guarantees that for any distribution with
//! finite mean Î¼ and standard deviation Ïƒ:
//!
//! ```text
//!   P(|X - Î¼| â‰¥ kÂ·Ïƒ) â‰¤ 1/kÂ²
//!
//!   Safe boundary: liveness > Î¼ - kÂ·Ïƒ
//!   Default k = 2 â†’ at most 25% false eviction risk
//! ```
//!
//! By inheriting (Î¼, Ïƒ, k) from the source agent, we ensure that
//! Injected data with liveness = Î¼ + Ïƒ (the anchor value) always
//! falls within the safe boundary.
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

use libm::sqrt;

/// Inherited statistical anchor from the source agent's manifold.
///
/// When a manifold payload is Injected, its liveness statistics are
/// foreign to the receiving heap. This anchor carries the source's
/// Chebyshev bounds so the receiver's evictor can protect the new data
/// during the assimilation window.
///
/// # Fields
/// - `mean`: Mean liveness from the source agent's heap (Î¼)
/// - `std_dev`: Standard deviation of liveness scores (Ïƒ)
/// - `k`: Number of standard deviations for safe boundary (default: 2.0)
/// - `anchor_liveness`: Recommended initial liveness for Injected objects
#[derive(Debug, Clone, Copy)]
pub struct LivenessAnchor {
    pub mean: f64,
    pub std_dev: f64,
    pub k: f64,
    pub anchor_liveness: f64,
}

impl LivenessAnchor {
    /// Create an anchor with explicit values.
    pub fn new(mean: f64, std_dev: f64, k: f64) -> Self {
        Self {
            mean,
            std_dev,
            k,
            // Anchor at Î¼ + Ïƒ to guarantee safety margin
            anchor_liveness: mean + std_dev,
        }
    }

    /// Compute an anchor from raw liveness samples.
    ///
    /// Uses Welford's online algorithm for numerical stability.
    pub fn from_samples(samples: &[f64], k: f64) -> Self {
        if samples.is_empty() {
            return Self::default();
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>() / n;
        let std_dev = sqrt(if variance < 0.0 { 0.0 } else { variance });

        Self {
            mean,
            std_dev,
            k,
            anchor_liveness: mean + std_dev,
        }
    }

    /// The lower boundary of the safe region: Î¼ - kÂ·Ïƒ
    pub fn safe_boundary(&self) -> f64 {
        self.mean - (self.k * self.std_dev)
    }
}

impl Default for LivenessAnchor {
    fn default() -> Self {
        Self {
            mean: 1.0,
            std_dev: 0.5,
            k: 2.0,
            anchor_liveness: 1.5,
        }
    }
}

/// Chebyshev Guard â€” statistical eviction safety protocol.
///
/// Determines whether a given liveness score is "safe" (should not be
/// evicted) based on the population statistics of the heap. Supports
/// both local computation and inherited anchors for Injected data.
///
/// # Decision Rule
/// ```text
///   safe(x) = x â‰¥ Î¼  âˆ¨  x > Î¼ - kÂ·Ïƒ
/// ```
#[derive(Debug, Clone)]
pub struct ChebyshevGuard {
    mean: f64,
    std_dev: f64,
    k: f64,
}

impl ChebyshevGuard {
    /// Create a guard from explicit statistics.
    pub fn new(mean: f64, std_dev: f64, k: f64) -> Self {
        Self { mean, std_dev, k }
    }

    /// Create a guard from a set of liveness samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self { mean: 0.0, std_dev: 0.0, k: 2.0 };
        }

        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let sum_sq: f64 = samples.iter().map(|x| x * x).sum();
        let variance = (sum_sq / n) - (mean * mean);
        let variance = if variance < 0.0 { 0.0 } else { variance };

        Self {
            mean,
            std_dev: sqrt(variance),
            k: 2.0,
        }
    }

    /// Create a guard with inherited statistics from a source agent.
    ///
    /// Instead of computing from the local heap (where Injected data
    /// has t_alive = 0), this injects pre-computed bounds from the
    /// source manifold's [`LivenessAnchor`].
    pub fn with_inherited_anchor(anchor: &LivenessAnchor) -> Self {
        Self {
            mean: anchor.mean,
            std_dev: anchor.std_dev,
            k: anchor.k,
        }
    }

    /// Check if a liveness score is within the safe boundary.
    ///
    /// Returns `true` if the object should NOT be evicted.
    pub fn is_safe(&self, liveness: f64) -> bool {
        if liveness >= self.mean { return true; }
        let boundary = self.mean - (self.k * self.std_dev);
        liveness > boundary
    }

    /// Get the computed safe boundary: Î¼ - kÂ·Ïƒ
    pub fn safe_boundary(&self) -> f64 {
        self.mean - (self.k * self.std_dev)
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Unit Tests
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_from_samples() {
        let samples = [1.0, 2.0, 3.0, 4.0, 5.0];
        let anchor = LivenessAnchor::from_samples(&samples, 2.0);

        // Mean = 3.0
        assert!((anchor.mean - 3.0).abs() < 1e-10);
        // Variance = 2.0, StdDev â‰ˆ 1.414
        assert!((anchor.std_dev - libm::sqrt(2.0)).abs() < 1e-10);
        // anchor_liveness = Î¼ + Ïƒ â‰ˆ 4.414
        assert!((anchor.anchor_liveness - (3.0 + libm::sqrt(2.0))).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev_guard_safety() {
        let guard = ChebyshevGuard::new(3.0, 1.0, 2.0);

        // Above mean â†’ always safe
        assert!(guard.is_safe(5.0));
        assert!(guard.is_safe(3.0));

        // Within kÂ·Ïƒ â†’ safe
        assert!(guard.is_safe(1.5));

        // Below Î¼ - kÂ·Ïƒ = 1.0 â†’ NOT safe
        assert!(!guard.is_safe(0.5));
        assert!(!guard.is_safe(0.0));
    }

    #[test]
    fn test_inherited_anchor_prevents_eviction() {
        // Source agent has well-established liveness distribution
        let samples = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let anchor = LivenessAnchor::from_samples(&samples, 2.0);

        let guard = ChebyshevGuard::with_inherited_anchor(&anchor);

        // Injected object at anchor_liveness must be safe
        assert!(
            guard.is_safe(anchor.anchor_liveness),
            "anchor_liveness ({:.2}) must be safe (boundary={:.2})",
            anchor.anchor_liveness, guard.safe_boundary()
        );

        // Object with zero liveness (t_alive=0 without inheritance) â†’ evictable
        assert!(
            !guard.is_safe(0.0),
            "Zero-liveness should be evictable"
        );
    }

    #[test]
    fn test_empty_samples() {
        let anchor = LivenessAnchor::from_samples(&[], 2.0);
        assert!((anchor.mean - 1.0).abs() < 1e-10); // default
    }
}
