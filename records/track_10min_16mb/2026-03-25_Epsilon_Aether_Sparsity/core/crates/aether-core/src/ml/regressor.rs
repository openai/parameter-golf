//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Manifold Regressor
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Non-linear regression on 3D manifold embeddings.
//!
//! Key insight: Instead of fitting in feature space, we fit on the
//! manifold's intrinsic geometry. This naturally handles non-linear
//! relationships because the manifold encodes the data's true shape.
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use heapless::Vec as HVec;
use libm::{pow, sqrt};

/// Maximum data points
const MAX_POINTS: usize = 256;
/// Maximum polynomial degree
const MAX_DEGREE: usize = 8;

// ═══════════════════════════════════════════════════════════════════════════════
// Regression Models
// ═══════════════════════════════════════════════════════════════════════════════

/// Model types for manifold regression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    /// Linear: y = a + bx
    Linear,
    /// Polynomial: y = Σ aᵢxⁱ
    Polynomial(u8),
    /// Radial Basis Function
    Rbf { gamma: f64 },
    /// Gaussian Process (approximate)
    GaussianProcess { length_scale: f64 },
    /// Manifold geodesic regression
    GeodesicRegression,
}

impl ModelType {
    /// Complexity level (1-10) for escalation ordering
    pub fn complexity(&self) -> u8 {
        match self {
            ModelType::Linear => 1,
            ModelType::Polynomial(d) => 1 + *d,
            ModelType::Rbf { .. } => 5,
            ModelType::GaussianProcess { .. } => 7,
            ModelType::GeodesicRegression => 9,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Coefficients
// ═══════════════════════════════════════════════════════════════════════════════

/// Regression coefficients (max 8 terms)
#[derive(Debug, Clone, Copy, Default)]
pub struct Coefficients {
    pub values: [f64; MAX_DEGREE],
    pub count: usize,
}

impl Coefficients {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_slice(vals: &[f64]) -> Self {
        let mut c = Self::new();
        for (i, &v) in vals.iter().enumerate().take(MAX_DEGREE) {
            c.values[i] = v;
            c.count = i + 1;
        }
        c
    }

    /// Evaluate polynomial at x
    pub fn eval_polynomial(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut x_pow = 1.0;

        for i in 0..self.count {
            result += self.values[i] * x_pow;
            x_pow *= x;
        }

        result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Manifold Regressor
// ═══════════════════════════════════════════════════════════════════════════════

/// Regressor operating on 3D manifold embedded data
#[derive(Debug)]
pub struct ManifoldRegressor<const D: usize> {
    /// Embedded points in D-dimensional manifold
    points: HVec<[f64; D], MAX_POINTS>,
    /// Target values
    targets: HVec<f64, MAX_POINTS>,
    /// Current model type
    model: ModelType,
    /// Fitted coefficients
    coefficients: Coefficients,
    /// Mean of targets (for centering)
    target_mean: f64,
    /// Std of targets (for scaling)
    target_std: f64,
}

impl<const D: usize> ManifoldRegressor<D> {
    pub fn new(model: ModelType) -> Self {
        Self {
            points: HVec::new(),
            targets: HVec::new(),
            model,
            coefficients: Coefficients::new(),
            target_mean: 0.0,
            target_std: 1.0,
        }
    }

    /// Add training data point
    pub fn add_point(&mut self, point: [f64; D], target: f64) {
        let _ = self.points.push(point);
        let _ = self.targets.push(target);
    }

    /// Fit the model to data
    pub fn fit(&mut self) -> f64 {
        if self.points.is_empty() {
            return f64::MAX;
        }

        // Compute mean and std for normalization
        self.compute_stats();

        // Fit appropriate model
        match self.model {
            ModelType::Linear => self.fit_linear(),
            ModelType::Polynomial(degree) => self.fit_polynomial(degree),
            ModelType::Rbf { gamma } => self.fit_rbf(gamma),
            ModelType::GaussianProcess { length_scale } => self.fit_gp(length_scale),
            ModelType::GeodesicRegression => self.fit_geodesic(),
        }
    }

    /// Compute target statistics
    fn compute_stats(&mut self) {
        let n = self.targets.len() as f64;
        if n == 0.0 {
            return;
        }

        // Mean
        let sum: f64 = self.targets.iter().sum();
        self.target_mean = sum / n;

        // Std
        let var: f64 = self
            .targets
            .iter()
            .map(|&t| {
                let diff = t - self.target_mean;
                diff * diff
            })
            .sum::<f64>()
            / n;
        self.target_std = sqrt(var).max(1e-10);
    }

    /// Fit linear model: y = a + b*x (using first manifold dimension)
    fn fit_linear(&mut self) -> f64 {
        let n = self.points.len() as f64;
        if n == 0.0 {
            return f64::MAX;
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for (p, &t) in self.points.iter().zip(self.targets.iter()) {
            let x = p[0]; // First manifold dimension
            sum_x += x;
            sum_y += t;
            sum_xy += x * t;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if abs(denom) < 1e-10 {
            self.coefficients = Coefficients::from_slice(&[self.target_mean, 0.0]);
        } else {
            let b = (n * sum_xy - sum_x * sum_y) / denom;
            let a = (sum_y - b * sum_x) / n;
            self.coefficients = Coefficients::from_slice(&[a, b]);
        }

        self.compute_mse()
    }

    /// Fit polynomial model
    fn fit_polynomial(&mut self, degree: u8) -> f64 {
        // Start with linear and add higher order corrections
        self.fit_linear();

        let degree = (degree as usize).min(MAX_DEGREE - 1);

        // Simple iterative refinement for higher orders
        // (In production, use proper least squares with Vandermonde)
        for d in 2..=degree {
            let mut correction = 0.0;
            let mut x_sum = 0.0;

            for (p, &t) in self.points.iter().zip(self.targets.iter()) {
                let x = p[0];
                let pred = self.coefficients.eval_polynomial(x);
                let residual = t - pred;
                let x_d = pow(x, d as f64);
                correction += residual * x_d;
                x_sum += x_d * x_d;
            }

            if abs(x_sum) > 1e-10 {
                self.coefficients.values[d] = correction / x_sum;
                self.coefficients.count = d + 1;
            }
        }

        self.compute_mse()
    }

    /// Fit RBF kernel model (Nadaraya-Watson estimator)
    fn fit_rbf(&mut self, gamma: f64) -> f64 {
        // For prediction, we'll use kernel regression
        // Coefficients store reference points' weights
        self.coefficients = Coefficients::new();

        // Compute weights based on kernel values
        let n = self.points.len();
        for i in 0..n.min(MAX_DEGREE) {
            self.coefficients.values[i] = 1.0 / n as f64;
        }
        self.coefficients.count = n.min(MAX_DEGREE);

        // Store gamma in a special slot
        if n > 0 {
            self.coefficients.values[MAX_DEGREE - 1] = gamma;
        }

        self.compute_mse()
    }

    /// Fit Gaussian Process (approximate)
    fn fit_gp(&mut self, length_scale: f64) -> f64 {
        // Approximate GP as RBF with appropriate gamma
        let gamma = 1.0 / (2.0 * length_scale * length_scale);
        self.fit_rbf(gamma)
    }

    /// Fit geodesic regression on manifold
    fn fit_geodesic(&mut self) -> f64 {
        // Geodesic regression: find curve on manifold minimizing distance
        // Approximate with polynomial in ambient coordinates
        self.fit_polynomial(4)
    }

    /// Compute mean squared error
    fn compute_mse(&self) -> f64 {
        if self.points.is_empty() {
            return f64::MAX;
        }

        let mut mse = 0.0;
        for (p, &t) in self.points.iter().zip(self.targets.iter()) {
            let pred = self.predict(p);
            let err = pred - t;
            mse += err * err;
        }

        sqrt(mse / self.points.len() as f64)
    }

    /// Predict target for a manifold point
    pub fn predict(&self, point: &[f64; D]) -> f64 {
        match self.model {
            ModelType::Linear | ModelType::Polynomial(_) | ModelType::GeodesicRegression => {
                self.coefficients.eval_polynomial(point[0])
            }
            ModelType::Rbf { gamma } => self.predict_rbf(point, gamma),
            ModelType::GaussianProcess { length_scale } => {
                let gamma = 1.0 / (2.0 * length_scale * length_scale);
                self.predict_rbf(point, gamma)
            }
        }
    }

    /// RBF prediction using kernel regression
    fn predict_rbf(&self, point: &[f64; D], gamma: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (p, &t) in self.points.iter().zip(self.targets.iter()) {
            let dist_sq = self.squared_distance(point, p);
            let weight = libm::exp(-gamma * dist_sq);
            weighted_sum += weight * t;
            weight_sum += weight;
        }

        if weight_sum > 1e-10 {
            weighted_sum / weight_sum
        } else {
            self.target_mean
        }
    }

    /// Squared Euclidean distance in manifold space
    fn squared_distance(&self, a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }

    /// Get current error
    pub fn error(&self) -> f64 {
        self.compute_mse()
    }

    /// Get fitted coefficients
    pub fn coefficients(&self) -> &Coefficients {
        &self.coefficients
    }

    /// Get model type
    pub fn model(&self) -> ModelType {
        self.model
    }

    /// Upgrade to more complex model
    pub fn upgrade_model(&mut self) {
        self.model = match self.model {
            ModelType::Linear => ModelType::Polynomial(2),
            ModelType::Polynomial(d) if d < 5 => ModelType::Polynomial(d + 1),
            ModelType::Polynomial(_) => ModelType::Rbf { gamma: 0.5 },
            ModelType::Rbf { gamma } if gamma < 2.0 => ModelType::Rbf { gamma: gamma * 2.0 },
            ModelType::Rbf { .. } => ModelType::GaussianProcess { length_scale: 1.0 },
            ModelType::GaussianProcess { length_scale } if length_scale > 0.1 => {
                ModelType::GaussianProcess {
                    length_scale: length_scale / 2.0,
                }
            }
            ModelType::GaussianProcess { .. } => ModelType::GeodesicRegression,
            ModelType::GeodesicRegression => ModelType::GeodesicRegression, // Max complexity
        };
    }
}

fn abs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}
