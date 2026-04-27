//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS State Module
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements the System State Vector μ(t) ∈ ℝ^d and the Deviation Metric.
//!
//! Mathematical Foundation:
//!   - State Vector: μ(t) = [m(t), i(t), q(t), e(t)]
//!     where:
//!     m(t) = Memory Pressure (0.0 - 1.0)
//!     i(t) = IRQ Rate (interrupts per ms)
//!     q(t) = Thread Queue Depth
//!     e(t) = Entropy Pool Level
//!
//!   - Deviation: Δ(t) = ||μ(t) - μ(t_last)||₂ (Euclidean distance)
//!
//! ═══════════════════════════════════════════════════════════════════════════════

use core::ops::Sub;
use libm::sqrt;

/// System State Vector μ(t) ∈ ℝ^D
///
/// This represents a point on the state manifold. The kernel tracks
/// the trajectory of this point through time, only acting when the
/// trajectory deviates significantly from equilibrium.
///
/// # Type Parameter
/// * `D` - Dimensionality of the state manifold
///
/// # Mathematical Properties
/// * Lives in Euclidean space ℝ^D
/// * Equipped with L2 norm for deviation measurement
/// * Forms a vector space under addition
#[derive(Debug, Clone, Copy)]
pub struct SystemState<const D: usize> {
    /// The state vector components
    pub vector: [f64; D],

    /// Timestamp when this state was captured (in microseconds)
    pub timestamp: u64,
}

impl<const D: usize> SystemState<D> {
    /// Create a zero state (origin of the manifold)
    pub const fn zero() -> Self {
        Self {
            vector: [0.0; D],
            timestamp: 0,
        }
    }

    /// Create a state from components
    pub fn new(vector: [f64; D], timestamp: u64) -> Self {
        Self { vector, timestamp }
    }

    /// Calculate the Euclidean deviation (L2 norm of difference)
    ///
    /// Δ(t) = ||μ(t) - μ(t_last)||₂ = √(Σᵢ(μᵢ(t) - μᵢ(t_last))²)
    ///
    /// This is the "Action Potential" that triggers kernel wake-up.
    /// When Δ(t) ≥ ε(t), the sparse scheduler activates.
    ///
    /// # Arguments
    /// * `other` - The reference state (typically μ(t_last))
    ///
    /// # Returns
    /// The Euclidean distance between the two states
    pub fn deviation(&self, other: &Self) -> f64 {
        let mut sum_sq = 0.0;

        for i in 0..D {
            let diff = self.vector[i] - other.vector[i];
            sum_sq += diff * diff;
        }

        sqrt(sum_sq)
    }

    /// Calculate the L∞ norm (maximum component-wise deviation)
    ///
    /// Useful for detecting outliers in specific dimensions.
    pub fn max_deviation(&self, other: &Self) -> f64 {
        let mut max = 0.0;

        for i in 0..D {
            let diff = abs(self.vector[i] - other.vector[i]);
            if diff > max {
                max = diff;
            }
        }

        max
    }

    /// Calculate the L1 norm (Manhattan distance)
    ///
    /// ||μ - μ'||₁ = Σᵢ|μᵢ - μ'ᵢ|
    pub fn manhattan_deviation(&self, other: &Self) -> f64 {
        let mut sum = 0.0;

        for i in 0..D {
            sum += abs(self.vector[i] - other.vector[i]);
        }

        sum
    }

    /// Get the magnitude (L2 norm) of this state vector
    pub fn magnitude(&self) -> f64 {
        let mut sum_sq = 0.0;

        for i in 0..D {
            sum_sq += self.vector[i] * self.vector[i];
        }

        sqrt(sum_sq)
    }

    /// Elapsed time between two states (in microseconds)
    pub fn elapsed_since(&self, other: &Self) -> u64 {
        self.timestamp.saturating_sub(other.timestamp)
    }
}

impl<const D: usize> Default for SystemState<D> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const D: usize> Sub for SystemState<D> {
    type Output = [f64; D];

    fn sub(self, other: Self) -> [f64; D] {
        let mut result = [0.0; D];
        for (r, (a, b)) in result.iter_mut().zip(self.vector.iter().zip(other.vector.iter())) {
            *r = a - b;
        }
        result
    }
}

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// State Dimension Indices
// ═══════════════════════════════════════════════════════════════════════════════

/// Named indices for the 4D state vector
///
/// These provide semantic meaning to the raw dimensions.
pub mod dimensions {
    /// Memory pressure (0.0 = free, 1.0 = critical)
    pub const MEMORY_PRESSURE: usize = 0;

    /// IRQ rate (interrupts per millisecond, normalized)
    pub const IRQ_RATE: usize = 1;

    /// Thread queue depth (waiting threads, normalized)
    pub const QUEUE_DEPTH: usize = 2;

    /// Entropy pool level (0.0 = empty, 1.0 = full)
    pub const ENTROPY_LEVEL: usize = 3;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_deviation() {
        let s1 = SystemState::<4>::zero();
        let s2 = SystemState::<4>::zero();

        assert!((s1.deviation(&s2) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_deviation() {
        let s1 = SystemState::new([1.0, 0.0, 0.0, 0.0], 0);
        let s2 = SystemState::new([0.0, 0.0, 0.0, 0.0], 0);

        assert!((s1.deviation(&s2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pythagorean_deviation() {
        // 3-4-5 triangle: √(3² + 4²) = 5
        let s1 = SystemState::new([3.0, 4.0, 0.0, 0.0], 0);
        let s2 = SystemState::new([0.0, 0.0, 0.0, 0.0], 0);

        assert!((s1.deviation(&s2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_deviation() {
        let s1 = SystemState::new([3.0, 4.0, 0.0, 0.0], 0);
        let s2 = SystemState::new([0.0, 0.0, 0.0, 0.0], 0);

        assert!((s1.manhattan_deviation(&s2) - 7.0).abs() < 1e-10);
    }
}
