//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Topological Convergence Detection
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! "Answers Come" - through topological structure!
//!
//! Instead of arbitrary loss thresholds, we detect convergence when:
//! 1. Betti numbers (β₀, β₁) stabilize
//! 2. Centroid drift approaches zero
//! 3. Residual manifold collapses to a point
//!
//! This gives us a mathematically principled "stop" condition.
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use heapless::Vec as HVec;
use libm::sqrt;

/// Maximum history length for Betti tracking
const MAX_HISTORY: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════════
// Betti Numbers
// ═══════════════════════════════════════════════════════════════════════════════

/// Betti numbers characterizing topological shape
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BettiNumbers {
    /// β₀: Number of connected components
    pub beta_0: u32,
    /// β₁: Number of 1-dimensional holes (loops)
    pub beta_1: u32,
}

impl BettiNumbers {
    pub fn new(beta_0: u32, beta_1: u32) -> Self {
        Self { beta_0, beta_1 }
    }

    /// "Perfect" convergence: single component, no loops
    pub fn is_singular(&self) -> bool {
        self.beta_0 == 1 && self.beta_1 == 0
    }

    /// Topological distance (simple L1 for now)
    pub fn distance(&self, other: &Self) -> u32 {
        let d0 = (self.beta_0 as i32 - other.beta_0 as i32).unsigned_abs();
        let d1 = (self.beta_1 as i32 - other.beta_1 as i32).unsigned_abs();
        d0 + d1
    }
}

impl Default for BettiNumbers {
    fn default() -> Self {
        Self {
            beta_0: 1,
            beta_1: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convergence Detector
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect convergence via topological stability
#[derive(Debug)]
pub struct ConvergenceDetector {
    /// History of Betti numbers
    betti_history: HVec<BettiNumbers, MAX_HISTORY>,
    /// History of centroid drifts
    drift_history: HVec<f64, MAX_HISTORY>,
    /// History of MSE errors
    error_history: HVec<f64, MAX_HISTORY>,
    /// Epochs required for stability
    stability_window: usize,
    /// Epsilon threshold for final convergence
    epsilon: f64,
    /// Current epoch
    epoch: u32,
}

impl ConvergenceDetector {
    pub fn new(epsilon: f64, stability_window: usize) -> Self {
        Self {
            betti_history: HVec::new(),
            drift_history: HVec::new(),
            error_history: HVec::new(),
            stability_window: stability_window.max(3),
            epsilon,
            epoch: 0,
        }
    }

    /// Record an epoch's metrics
    pub fn record_epoch(&mut self, betti: BettiNumbers, drift: f64, error: f64) {
        // Manage circular buffer
        if self.betti_history.len() >= MAX_HISTORY {
            // Shift left
            for i in 0..MAX_HISTORY - 1 {
                self.betti_history[i] = self.betti_history[i + 1];
                self.drift_history[i] = self.drift_history[i + 1];
                self.error_history[i] = self.error_history[i + 1];
            }
            self.betti_history.pop();
            self.drift_history.pop();
            self.error_history.pop();
        }

        let _ = self.betti_history.push(betti);
        let _ = self.drift_history.push(drift);
        let _ = self.error_history.push(error);
        self.epoch += 1;
    }

    /// Check if converged
    pub fn is_converged(&self) -> bool {
        // Must have enough history
        if self.betti_history.len() < self.stability_window {
            return false;
        }

        // Check error convergence
        if self.is_error_converged() {
            return true;
        }

        // Check topological convergence
        if self.is_betti_stable() && self.is_drift_stable() {
            return true;
        }

        false
    }

    /// Error below epsilon
    fn is_error_converged(&self) -> bool {
        self.error_history
            .last()
            .map(|&e| e < self.epsilon)
            .unwrap_or(false)
    }

    /// Betti numbers stable over window
    fn is_betti_stable(&self) -> bool {
        let n = self.betti_history.len();
        if n < self.stability_window {
            return false;
        }

        let window = &self.betti_history[n - self.stability_window..];
        if let Some(&first) = window.first() {
            window.iter().all(|b| *b == first)
        } else {
            false
        }
    }

    /// Drift stable and near zero
    fn is_drift_stable(&self) -> bool {
        let n = self.drift_history.len();
        if n < self.stability_window {
            return false;
        }

        let window = &self.drift_history[n - self.stability_window..];

        // Check if all drifts are small and decreasing
        let mut prev = f64::MAX;
        for &d in window {
            if d > prev * 1.5 {
                // Allow some noise, but no major increases
                return false;
            }
            prev = d;
        }

        // Final drift should be small
        window.last().map(|&d| d < 0.01).unwrap_or(false)
    }

    /// Get convergence score (0 = far, 1 = converged)
    pub fn convergence_score(&self) -> f64 {
        let n = self.betti_history.len();
        if n == 0 {
            return 0.0;
        }

        let mut score = 0.0;

        // Betti stability contribution (0.4)
        if n >= self.stability_window {
            let window = &self.betti_history[n - self.stability_window..];
            let variations = window.windows(2).filter(|w| w[0] != w[1]).count();
            let betti_score = 1.0 - (variations as f64 / self.stability_window as f64);
            score += 0.4 * betti_score;
        }

        // Drift contribution (0.3)
        if let Some(&last_drift) = self.drift_history.last() {
            let drift_score = libm::exp(-last_drift * 10.0).min(1.0);
            score += 0.3 * drift_score;
        }

        // Error contribution (0.3)
        if let Some(&last_error) = self.error_history.last() {
            let error_score = if last_error < self.epsilon {
                1.0
            } else {
                (self.epsilon / last_error).min(1.0)
            };
            score += 0.3 * error_score;
        }

        score
    }

    /// Get current epoch
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Get last Betti numbers
    pub fn last_betti(&self) -> Option<BettiNumbers> {
        self.betti_history.last().copied()
    }

    /// Get last error
    pub fn last_error(&self) -> Option<f64> {
        self.error_history.last().copied()
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.betti_history.clear();
        self.drift_history.clear();
        self.error_history.clear();
        self.epoch = 0;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Residual Manifold Analyzer
// ═══════════════════════════════════════════════════════════════════════════════

/// Analyze the topology of regression residuals
#[derive(Debug)]
pub struct ResidualAnalyzer<const D: usize> {
    /// Residual values
    residuals: HVec<f64, 256>,
    /// Connection threshold for epsilon-graph
    epsilon: f64,
}

impl<const D: usize> ResidualAnalyzer<D> {
    pub fn new(epsilon: f64) -> Self {
        Self {
            residuals: HVec::new(),
            epsilon,
        }
    }

    /// Set residuals for analysis
    pub fn set_residuals(&mut self, residuals: &[f64]) {
        self.residuals.clear();
        for &r in residuals.iter().take(256) {
            let _ = self.residuals.push(r);
        }
    }

    /// Compute Betti numbers of residual "manifold"
    ///
    /// β₀: Number of sign-change clusters
    /// β₁: Number of oscillation cycles
    pub fn compute_betti(&self) -> BettiNumbers {
        if self.residuals.is_empty() {
            return BettiNumbers::default();
        }

        // β₀: Count connected components via sign changes
        let mut beta_0 = 1u32;
        let mut prev_sign = self.residuals[0] >= 0.0;

        for &r in self.residuals.iter().skip(1) {
            let curr_sign = r >= 0.0;
            if curr_sign != prev_sign {
                beta_0 += 1;
            }
            prev_sign = curr_sign;
        }
        beta_0 = (beta_0 + 1).div_ceil(2); // Adjacent sign changes form one component

        // β₁: Count cycles via oscillation detection
        let mut beta_1 = 0u32;
        let mut increasing = true;

        for i in 1..self.residuals.len() {
            let delta = self.residuals[i] - self.residuals[i - 1];
            let curr_increasing = delta >= 0.0;

            if curr_increasing != increasing {
                beta_1 += 1;
            }
            increasing = curr_increasing;
        }
        beta_1 /= 4; // Four direction changes = one cycle

        BettiNumbers::new(beta_0, beta_1)
    }

    /// Compute centroid drift of residuals
    pub fn compute_drift(&self) -> f64 {
        if self.residuals.len() < 2 {
            return 0.0;
        }

        // Mean of absolute residuals
        let mean: f64 =
            self.residuals.iter().map(|r| abs(*r)).sum::<f64>() / self.residuals.len() as f64;

        // Variance as measure of "spread" (drift)
        let var: f64 = self
            .residuals
            .iter()
            .map(|&r| {
                let diff = abs(r) - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.residuals.len() as f64;

        sqrt(var)
    }

    /// Check if residuals form a "collapsed" manifold (converged)
    pub fn is_collapsed(&self, threshold: f64) -> bool {
        // Collapsed = all residuals near zero
        self.residuals.iter().all(|&r| abs(r) < threshold)
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
// "Answers Come" - The Philosophy
// ═══════════════════════════════════════════════════════════════════════════════

/// The Answer type - what emerges from convergence
#[derive(Debug, Clone)]
pub struct Answer {
    /// The converged coefficients
    pub coefficients: [f64; 8],
    /// Final Betti signature
    pub topology: BettiNumbers,
    /// Epochs to convergence
    pub epochs: u32,
    /// Final error
    pub error: f64,
    /// Convergence confidence (0-1)
    pub confidence: f64,
}

impl Answer {
    /// Create from convergence detector state
    pub fn from_detector(detector: &ConvergenceDetector, coefficients: [f64; 8]) -> Option<Self> {
        if !detector.is_converged() {
            return None;
        }

        Some(Self {
            coefficients,
            topology: detector.last_betti().unwrap_or_default(),
            epochs: detector.epoch(),
            error: detector.last_error().unwrap_or(0.0),
            confidence: detector.convergence_score(),
        })
    }

    /// Check if this is a "perfect" answer
    pub fn is_perfect(&self, epsilon: f64) -> bool {
        self.error < epsilon && self.topology.is_singular() && self.confidence > 0.95
    }
}
