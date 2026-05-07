//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Escalating Benchmark System
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! "Run regression benchmarks infinitely harder each until perfect"
//!
//! The benchmark system automatically escalates difficulty:
//! 1. Start with simple linear regression
//! 2. If not converged, escalate to polynomial
//! 3. If still failing, escalate to RBF
//! 4. Continue until topological convergence
//! 5. Answer emerges when topology stabilizes
//!    ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use crate::ml::convergence::{Answer, BettiNumbers, ConvergenceDetector, ResidualAnalyzer};
use crate::ml::regressor::{Coefficients, ManifoldRegressor, ModelType};
use heapless::Vec as HVec;

/// Maximum benchmark iterations
const MAX_EPOCHS: u32 = 1000;

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Configuration
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for escalating benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Target convergence epsilon
    pub epsilon: f64,
    /// Maximum epochs before giving up
    pub max_epochs: u32,
    /// Epochs before escalating model
    pub escalation_patience: u32,
    /// Stability window for topology
    pub stability_window: usize,
    /// Enable automatic escalation
    pub auto_escalate: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-6,
            max_epochs: 100,
            escalation_patience: 10,
            stability_window: 5,
            auto_escalate: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark Result
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final model type used
    pub final_model: ModelType,
    /// Final coefficients
    pub coefficients: Coefficients,
    /// Epochs run
    pub epochs: u32,
    /// Final error
    pub final_error: f64,
    /// Final topology
    pub final_betti: BettiNumbers,
    /// Number of escalations
    pub escalations: u32,
    /// The answer (if converged)
    pub answer: Option<Answer>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Escalating Benchmark Engine
// ═══════════════════════════════════════════════════════════════════════════════

/// The main escalating benchmark engine
pub struct EscalatingBenchmark<const D: usize> {
    /// Configuration
    config: BenchmarkConfig,
    /// The regressor
    regressor: ManifoldRegressor<D>,
    /// Convergence detector
    detector: ConvergenceDetector,
    /// Residual analyzer
    residuals: ResidualAnalyzer<D>,
    /// Current epoch
    epoch: u32,
    /// Epochs since last improvement
    patience_counter: u32,
    /// Number of model escalations
    escalations: u32,
    /// Best error seen
    best_error: f64,
    /// Target data
    targets: HVec<f64, 256>,
}

impl<const D: usize> EscalatingBenchmark<D> {
    pub fn new(config: BenchmarkConfig) -> Self {
        let epsilon = config.epsilon;
        let window = config.stability_window;

        Self {
            config,
            regressor: ManifoldRegressor::new(ModelType::Linear),
            detector: ConvergenceDetector::new(epsilon, window),
            residuals: ResidualAnalyzer::new(0.1),
            epoch: 0,
            patience_counter: 0,
            escalations: 0,
            best_error: f64::MAX,
            targets: HVec::new(),
        }
    }

    /// Add training data
    pub fn add_data(&mut self, point: [f64; D], target: f64) {
        self.regressor.add_point(point, target);
        let _ = self.targets.push(target);
    }

    /// Run the escalating benchmark
    pub fn run(&mut self) -> BenchmarkResult {
        self.epoch = 0;
        self.patience_counter = 0;
        self.escalations = 0;
        self.best_error = f64::MAX;
        self.detector.reset();

        while self.epoch < self.config.max_epochs {
            // Fit current model
            let error = self.regressor.fit();

            // Compute residuals and topology
            let residuals = self.compute_residuals();
            self.residuals.set_residuals(&residuals);
            let betti = self.residuals.compute_betti();
            let drift = self.residuals.compute_drift();

            // Record metrics
            self.detector.record_epoch(betti, drift, error);

            // Update best error
            if error < self.best_error {
                self.best_error = error;
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;
            }

            // Check convergence
            if self.detector.is_converged() {
                return self.create_result(true);
            }

            // Check if should escalate
            if self.config.auto_escalate && self.patience_counter >= self.config.escalation_patience
            {
                self.escalate();
            }

            self.epoch += 1;
        }

        // Max epochs reached - check if we're close enough
        let close_enough = self.detector.convergence_score() > 0.8;
        self.create_result(close_enough)
    }

    /// Compute residuals from current fit
    fn compute_residuals(&self) -> HVec<f64, 256> {
        let mut residuals = HVec::new();

        // We need to get predictions - this is a simplified version
        // In full implementation, we'd iterate over training points
        for (i, &target) in self.targets.iter().enumerate() {
            // Approximate prediction using polynomial evaluation
            let x = i as f64 * 0.1; // Simplified x
            let pred = self.regressor.coefficients().eval_polynomial(x);
            let _ = residuals.push(target - pred);
        }

        residuals
    }

    /// Escalate to more complex model
    fn escalate(&mut self) {
        self.regressor.upgrade_model();
        self.escalations += 1;
        self.patience_counter = 0;
    }

    /// Create final result
    fn create_result(&self, converged: bool) -> BenchmarkResult {
        let mut coeffs = [0.0f64; 8];
        for (i, &v) in self
            .regressor
            .coefficients()
            .values
            .iter()
            .enumerate()
            .take(8)
        {
            coeffs[i] = v;
        }

        let answer = if converged {
            Answer::from_detector(&self.detector, coeffs)
        } else {
            None
        };

        BenchmarkResult {
            converged,
            final_model: self.regressor.model(),
            coefficients: *self.regressor.coefficients(),
            epochs: self.epoch,
            final_error: self.best_error,
            final_betti: self.detector.last_betti().unwrap_or_default(),
            escalations: self.escalations,
            answer,
        }
    }

    /// Get current model type
    pub fn current_model(&self) -> ModelType {
        self.regressor.model()
    }

    /// Get current epoch
    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Get convergence score
    pub fn convergence_score(&self) -> f64 {
        self.detector.convergence_score()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Functions for Benchmarking
// ═══════════════════════════════════════════════════════════════════════════════

/// Generate test function data for benchmarking
pub fn generate_test_function(func: TestFunction, n_points: usize) -> HVec<(f64, f64), 256> {
    let mut data = HVec::new();

    for i in 0..n_points.min(256) {
        let x = (i as f64 / n_points as f64) * 2.0 * core::f64::consts::PI;
        let y = match func {
            TestFunction::Sine => libm::sin(x),
            TestFunction::Polynomial => 0.5 * x * x - 2.0 * x + 1.0,
            TestFunction::Exponential => libm::exp(-x / 2.0),
            TestFunction::Mixture => libm::sin(x) * libm::exp(-x / 4.0),
            TestFunction::Step => {
                if x < core::f64::consts::PI {
                    1.0
                } else {
                    -1.0
                }
            }
        };
        let _ = data.push((x, y));
    }

    data
}

/// Standard test functions
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestFunction {
    Sine,
    Polynomial,
    Exponential,
    Mixture,
    Step,
}
