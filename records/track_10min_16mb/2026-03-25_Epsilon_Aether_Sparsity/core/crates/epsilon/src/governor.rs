//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! Epsilon Surgery Governor
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! Implements the Adaptive Threshold Controller with Topological Surgery
//! support. This is a self-contained PD-on-Manifold governor that extends
//! the classical control law with a surgery permit mechanism.
//!
//! # Control Law
//!
//! ```text
//!   Îµ(t+1) = Îµ(t) + Î±Â·e(t) + Î²Â·de/dt
//!
//!   where:
//!     e(t) = R_target - Î”(t)/Îµ(t)
//!     R_target = Target tick rate (Hz)
//! ```
//!
//! # Surgery Permit (Section 3.1)
//!
//! **Problem**: Instantaneous context injection causes de/dt â†’ âˆž, which
//! violently triggers the derivative gain (Î²) and forces a Quiescent Reset,
//! erasing the Injected data.
//!
//! **Solution**: Before injection, the kernel acquires a `SurgeryPermit`.
//! The governor suspends the derivative penalty for exactly one tick
//! (Î² = 0 for t_surge) to absorb the state change without oscillation panic.
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Default PD gains
const DEFAULT_ALPHA: f64 = 0.01;
const DEFAULT_BETA: f64 = 0.05;
const TARGET_TICK_RATE: f64 = 1000.0;
const EPSILON_MIN: f64 = 0.001;
const EPSILON_MAX: f64 = 10.0;
const EPSILON_INITIAL: f64 = 0.1;

/// Adaptive Threshold Controller with Surgery Permit capability.
///
/// Implements the PD-on-Manifold control law from the AEGIS specification,
/// extended with a surgery permit mechanism for safe instantaneous context
/// injection during topological Context Transfer.
///
/// # Stability Properties
/// - Bounded: Îµ âˆˆ [EPSILON_MIN, EPSILON_MAX]
/// - Asymptotically stable around R_target
/// - Damped oscillation via derivative term
/// - Surgery-safe: derivative gain can be temporarily zeroed
#[derive(Debug, Clone)]
pub struct SurgeryGovernor {
    /// Current adaptive threshold Îµ(t)
    epsilon: f64,
    /// Previous error (for derivative calculation)
    last_error: f64,
    /// Proportional gain (Î±)
    alpha: f64,
    /// Derivative gain (Î²)
    beta: f64,
    /// Number of adaptations performed
    tick_count: u64,
}

impl SurgeryGovernor {
    /// Create a governor with default parameters.
    pub fn new() -> Self {
        Self {
            epsilon: EPSILON_INITIAL,
            last_error: 0.0,
            alpha: DEFAULT_ALPHA,
            beta: DEFAULT_BETA,
            tick_count: 0,
        }
    }

    /// Create a governor with custom gains.
    pub fn with_gains(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta, ..Self::new() }
    }

    /// Current threshold Îµ(t).
    pub fn epsilon(&self) -> f64 { self.epsilon }

    /// Current derivative gain Î².
    pub fn beta(&self) -> f64 { self.beta }

    /// Previous error signal.
    pub fn last_error(&self) -> f64 { self.last_error }

    /// Total ticks processed.
    pub fn tick_count(&self) -> u64 { self.tick_count }

    /// Adapt Îµ based on observed deviation.
    ///
    /// Implements: `Îµ(t+1) = Îµ(t) + Î±Â·e(t) + Î²Â·de/dt`
    pub fn adapt(&mut self, deviation_delta: f64, dt: f64) -> f64 {
        if dt <= 0.0 || self.epsilon <= 0.0 {
            return self.epsilon;
        }

        let current_rate = deviation_delta / self.epsilon;
        let error = TARGET_TICK_RATE - current_rate;
        let d_error = (error - self.last_error) / dt;
        let adjustment = (self.alpha * error) + (self.beta * d_error);

        self.epsilon = (self.epsilon + adjustment).clamp(EPSILON_MIN, EPSILON_MAX);
        self.last_error = error;
        self.tick_count += 1;

        self.epsilon
    }

    /// Check if deviation exceeds current threshold: Î”(t) â‰¥ Îµ(t)?
    pub fn should_trigger(&self, deviation: f64) -> bool {
        deviation >= self.epsilon
    }

    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    // Topological Surgery Protocol
    // â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    /// Prepare the governor for topological surgery.
    ///
    /// Snapshots the current derivative state (`last_error`, `beta`), then
    /// zeros both to prevent the PD controller from oscillation panic when
    /// de/dt â†’ âˆž during instantaneous context injection.
    ///
    /// Returns a [`SurgeryPermit`] that MUST be passed to
    /// [`complete_surgery()`](Self::complete_surgery) after injection.
    ///
    /// # Pre-Print Reference (Section 3.1)
    /// ```text
    /// During surgery: Î² = 0, last_error = 0 for t_surge
    /// This absorbs the state change without oscillation panic.
    /// ```
    pub fn prepare_for_surgery(&mut self) -> SurgeryPermit {
        let permit = SurgeryPermit {
            saved_last_error: self.last_error,
            saved_beta: self.beta,
        };

        // Zero derivative momentum for exactly one tick
        self.last_error = 0.0;
        self.beta = 0.0;

        permit
    }

    /// Restore the governor after topological surgery completes.
    ///
    /// Re-enables the derivative gain and error history from the permit,
    /// allowing the PD controller to resume oscillation damping.
    pub fn complete_surgery(&mut self, permit: SurgeryPermit) {
        self.last_error = permit.saved_last_error;
        self.beta = permit.saved_beta;
    }

    /// Reset the governor to initial state.
    pub fn reset(&mut self) {
        self.epsilon = EPSILON_INITIAL;
        self.last_error = 0.0;
        self.tick_count = 0;
    }
}

impl Default for SurgeryGovernor {
    fn default() -> Self { Self::new() }
}

/// One-shot token representing the governor's saved derivative state
/// during topological surgery.
///
/// Intentionally non-Clone, non-Copy to enforce single-use semantics:
/// once consumed by [`SurgeryGovernor::complete_surgery()`], the token
/// cannot be reused.
///
/// # Invariants
/// - Created only by [`SurgeryGovernor::prepare_for_surgery()`]
/// - Consumed only by [`SurgeryGovernor::complete_surgery()`]
/// - While this permit exists, the governor's Î² = 0 (derivative disabled)
#[derive(Debug)]
pub struct SurgeryPermit {
    saved_last_error: f64,
    saved_beta: f64,
}

impl SurgeryPermit {
    /// Saved derivative gain Î² (for diagnostics).
    pub fn saved_beta(&self) -> f64 { self.saved_beta }

    /// Saved error signal (for diagnostics).
    pub fn saved_last_error(&self) -> f64 { self.saved_last_error }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Unit Tests
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_epsilon() {
        let gov = SurgeryGovernor::new();
        assert!((gov.epsilon() - EPSILON_INITIAL).abs() < 1e-10);
    }

    #[test]
    fn test_epsilon_bounded() {
        let mut gov = SurgeryGovernor::new();

        // Drive epsilon to extremes
        for _ in 0..10000 { gov.adapt(1000.0, 0.001); }
        assert!(gov.epsilon() >= EPSILON_MIN);

        gov.reset();
        for _ in 0..10000 { gov.adapt(0.0001, 0.001); }
        assert!(gov.epsilon() <= EPSILON_MAX);
    }

    #[test]
    fn test_trigger_threshold() {
        let mut gov = SurgeryGovernor::new();
        gov.epsilon = 0.5;
        assert!(!gov.should_trigger(0.4));
        assert!(gov.should_trigger(0.5));
        assert!(gov.should_trigger(0.6));
    }

    #[test]
    fn test_surgery_permit_zeros_derivative() {
        let mut gov = SurgeryGovernor::new();

        // Build error state
        gov.adapt(500.0, 0.01);
        gov.adapt(600.0, 0.01);
        assert!(gov.last_error().abs() > 0.0);
        assert!(gov.beta() > 0.0);

        // Acquire surgery permit
        let permit = gov.prepare_for_surgery();

        // During surgery: derivative state must be zeroed
        assert!((gov.last_error()).abs() < 1e-10);
        assert!((gov.beta()).abs() < 1e-10);

        // Permit saved original values
        assert!(permit.saved_beta() > 0.0);

        // Adapt during surgery â€” no derivative panic
        let eps = gov.adapt(100000.0, 0.001);
        assert!(eps > 0.0);

        gov.complete_surgery(permit);
    }

    #[test]
    fn test_surgery_permit_restores_state() {
        let mut gov = SurgeryGovernor::with_gains(0.02, 0.08);
        gov.adapt(1000.0, 0.01);

        let pre_error = gov.last_error();
        let pre_beta = gov.beta();

        // Surgery round-trip
        let permit = gov.prepare_for_surgery();
        gov.complete_surgery(permit);

        assert!((gov.last_error() - pre_error).abs() < 1e-10);
        assert!((gov.beta() - pre_beta).abs() < 1e-10);
    }
}
