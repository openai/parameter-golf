//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Classification Algorithms
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Classification algorithms with topological features.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

// use heapless::Vec as HVec;
use libm::{exp, fabs, log, sqrt};

/// Maximum classes
const MAX_CLASSES: usize = 16;
/// Maximum data points
const MAX_POINTS: usize = 256;
/// Maximum features
const MAX_FEATURES: usize = 32;

// ═══════════════════════════════════════════════════════════════════════════════
// Logistic Regression
// ═══════════════════════════════════════════════════════════════════════════════

/// Logistic Regression classifier
#[derive(Debug)]
pub struct LogisticRegression {
    /// Weights for each feature
    pub weights: [f64; MAX_FEATURES],
    /// Bias term
    pub bias: f64,
    /// Number of features
    pub n_features: usize,
    /// Learning rate
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
}

impl LogisticRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: [0.0; MAX_FEATURES],
            bias: 0.0,
            n_features: n_features.min(MAX_FEATURES),
            learning_rate: 0.1,
            max_iter: 100,
            tol: 1e-4,
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sigmoid function
    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + exp(-z.clamp(-500.0, 500.0)))
    }

    /// Fit classifier to binary labels
    pub fn fit(&mut self, x: &[[f64; MAX_FEATURES]], y: &[f64], n: usize) -> f64 {
        let n = n.min(MAX_POINTS);
        if n == 0 {
            return 0.0;
        }

        let mut prev_loss = f64::MAX;

        for _ in 0..self.max_iter {
            let mut grad_w = [0.0; MAX_FEATURES];
            let mut grad_b = 0.0;
            let mut loss = 0.0;

            for i in 0..n {
                let z = self.linear(&x[i]);
                let p = Self::sigmoid(z);
                let error = p - y[i];

                // Gradient accumulation
                for (j, w) in grad_w.iter_mut().enumerate().take(self.n_features) {
                    *w += error * x[i][j];
                }
                grad_b += error;

                // Binary cross-entropy loss
                let p_clipped = p.clamp(1e-7, 1.0 - 1e-7);
                loss -= y[i] * log(p_clipped) + (1.0 - y[i]) * log(1.0 - p_clipped);
            }

            // Update weights
            for (j, w) in self.weights.iter_mut().enumerate().take(self.n_features) {
                *w -= self.learning_rate * grad_w[j] / n as f64;
            }
            self.bias -= self.learning_rate * grad_b / n as f64;

            loss /= n as f64;

            // Check convergence
            if fabs(prev_loss - loss) < self.tol {
                return loss;
            }
            prev_loss = loss;
        }

        prev_loss
    }

    /// Linear combination
    fn linear(&self, x: &[f64; MAX_FEATURES]) -> f64 {
        let mut z = self.bias;
        for (j, w) in self.weights.iter().enumerate().take(self.n_features) {
            z += *w * x[j];
        }
        z
    }

    /// Predict probability
    pub fn predict_proba(&self, x: &[f64; MAX_FEATURES]) -> f64 {
        Self::sigmoid(self.linear(x))
    }

    /// Predict class (0 or 1)
    pub fn predict(&self, x: &[f64; MAX_FEATURES]) -> u32 {
        if self.predict_proba(x) >= 0.5 {
            1
        } else {
            0
        }
    }

    /// Predict batch
    pub fn predict_batch(&self, x: &[[f64; MAX_FEATURES]], n: usize) -> [u32; MAX_POINTS] {
        let mut preds = [0u32; MAX_POINTS];
        for i in 0..n.min(MAX_POINTS) {
            preds[i] = self.predict(&x[i]);
        }
        preds
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// K-Nearest Neighbors
// ═══════════════════════════════════════════════════════════════════════════════

/// K-Nearest Neighbors classifier
#[derive(Debug)]
pub struct KNNClassifier<const D: usize> {
    /// Training data
    x_train: [[f64; D]; MAX_POINTS],
    /// Training labels
    y_train: [u32; MAX_POINTS],
    /// Number of training samples
    n_train: usize,
    /// Number of neighbors
    k: usize,
}

impl<const D: usize> KNNClassifier<D> {
    pub fn new(k: usize) -> Self {
        Self {
            x_train: [[0.0; D]; MAX_POINTS],
            y_train: [0; MAX_POINTS],
            n_train: 0,
            k: k.max(1),
        }
    }

    /// Fit (store) training data
    pub fn fit(&mut self, x: &[[f64; D]], y: &[u32], n: usize) {
        let n = n.min(MAX_POINTS);
        self.n_train = n;

        self.x_train[..n].copy_from_slice(&x[..n]);
        self.y_train[..n].copy_from_slice(&y[..n]);
    }

    /// Predict single sample
    pub fn predict(&self, x: &[f64; D]) -> u32 {
        if self.n_train == 0 {
            return 0;
        }

        // Find k nearest neighbors
        let mut distances = [(f64::MAX, 0u32); MAX_POINTS];
        for (i, dist) in distances.iter_mut().enumerate().take(self.n_train) {
            *dist = (self.distance(x, &self.x_train[i]), self.y_train[i]);
        }

        // Sort by distance (simple bubble sort for small k)
        for i in 0..self.k.min(self.n_train) {
            for j in (i + 1)..self.n_train {
                if distances[j].0 < distances[i].0 {
                    distances.swap(i, j);
                }
            }
        }

        // Vote among k nearest
        let mut votes = [0u32; MAX_CLASSES];
        for dist in distances.iter().take(self.k.min(self.n_train)) {
            let label = dist.1 as usize;
            if label < MAX_CLASSES {
                votes[label] += 1;
            }
        }

        // Return class with most votes
        let mut best_class = 0;
        let mut best_count = 0;
        for (c, count) in votes.iter().enumerate().take(MAX_CLASSES) {
            if *count > best_count {
                best_count = *count;
                best_class = c as u32;
            }
        }

        best_class
    }

    /// Euclidean distance
    fn distance(&self, a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sqrt(sum)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Perceptron
// ═══════════════════════════════════════════════════════════════════════════════

/// Single-layer Perceptron
#[derive(Debug)]
pub struct Perceptron {
    /// Weights
    pub weights: [f64; MAX_FEATURES],
    /// Bias
    pub bias: f64,
    /// Number of features
    pub n_features: usize,
    /// Learning rate
    learning_rate: f64,
}

impl Perceptron {
    pub fn new(n_features: usize) -> Self {
        Self {
            weights: [0.0; MAX_FEATURES],
            bias: 0.0,
            n_features: n_features.min(MAX_FEATURES),
            learning_rate: 1.0,
        }
    }

    pub fn with_lr(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Train with seal-loop style convergence
    pub fn fit(
        &mut self,
        x: &[[f64; MAX_FEATURES]],
        y: &[i32],
        n: usize,
        max_epochs: usize,
    ) -> u32 {
        let n = n.min(MAX_POINTS);

        for epoch in 0..max_epochs {
            let mut errors = 0u32;

            for i in 0..n {
                let pred = self.predict(&x[i]);
                let error = y[i] - pred;

                if error != 0 {
                    errors += 1;
                    for (j, w) in self.weights.iter_mut().enumerate().take(self.n_features) {
                        *w += self.learning_rate * error as f64 * x[i][j];
                    }
                    self.bias += self.learning_rate * error as f64;
                }
            }

            // Converged if no errors
            if errors == 0 {
                return epoch as u32;
            }
        }

        max_epochs as u32
    }

    /// Predict (-1 or +1)
    pub fn predict(&self, x: &[f64; MAX_FEATURES]) -> i32 {
        let mut z = self.bias;
        for (j, w) in self.weights.iter().enumerate().take(self.n_features) {
            z += *w * x[j];
        }
        if z >= 0.0 {
            1
        } else {
            -1
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Naive Bayes (Gaussian)
// ═══════════════════════════════════════════════════════════════════════════════

/// Gaussian Naive Bayes classifier
#[derive(Debug)]
pub struct GaussianNB {
    /// Mean per class per feature
    means: [[f64; MAX_FEATURES]; MAX_CLASSES],
    /// Variance per class per feature
    vars: [[f64; MAX_FEATURES]; MAX_CLASSES],
    /// Prior probability per class
    priors: [f64; MAX_CLASSES],
    /// Number of classes
    n_classes: usize,
    /// Number of features
    n_features: usize,
}

impl GaussianNB {
    pub fn new() -> Self {
        Self {
            means: [[0.0; MAX_FEATURES]; MAX_CLASSES],
            vars: [[1.0; MAX_FEATURES]; MAX_CLASSES],
            priors: [0.0; MAX_CLASSES],
            n_classes: 0,
            n_features: 0,
        }
    }

    /// Fit to data
    pub fn fit(&mut self, x: &[[f64; MAX_FEATURES]], y: &[u32], n: usize, n_features: usize) {
        let n = n.min(MAX_POINTS);
        self.n_features = n_features.min(MAX_FEATURES);

        if n == 0 {
            return;
        }

        // Count classes
        let mut class_counts = [0usize; MAX_CLASSES];
        for l in y.iter().take(n) {
            let c = *l as usize;
            if c < MAX_CLASSES {
                class_counts[c] += 1;
                if c >= self.n_classes {
                    self.n_classes = c + 1;
                }
            }
        }

        // Compute means
        for (c, count) in class_counts.iter().enumerate().take(self.n_classes) {
            if *count == 0 {
                continue;
            }
            self.priors[c] = (*count as f64) / (n as f64);

            for (j, u) in self.means[c].iter_mut().enumerate().take(self.n_features) {
                let mut sum = 0.0;
                for i in 0..n {
                    if y[i] as usize == c {
                        sum += x[i][j];
                    }
                }
                *u = sum / class_counts[c] as f64;
            }
        }

        // Compute variances
        for (c, count) in class_counts.iter().enumerate().take(self.n_classes) {
            if *count == 0 {
                continue;
            }

            for (j, v) in self.vars[c].iter_mut().enumerate().take(self.n_features) {
                let mut sum = 0.0;
                for i in 0..n {
                    if y[i] as usize == c {
                        let diff = x[i][j] - self.means[c][j];
                        sum += diff * diff;
                    }
                }
                *v = (sum / class_counts[c] as f64).max(1e-9);
            }
        }
    }

    /// Predict class
    pub fn predict(&self, x: &[f64; MAX_FEATURES]) -> u32 {
        let mut best_class = 0;
        let mut best_log_prob = f64::NEG_INFINITY;

        for c in 0..self.n_classes {
            if self.priors[c] <= 0.0 {
                continue;
            }

            let mut log_prob = log(self.priors[c]);

            for (j, val) in x.iter().enumerate().take(self.n_features) {
                // Log of Gaussian PDF
                let diff = val - self.means[c][j];
                log_prob -= 0.5 * log(2.0 * core::f64::consts::PI * self.vars[c][j]);
                log_prob -= 0.5 * diff * diff / self.vars[c][j];
            }

            if log_prob > best_log_prob {
                best_log_prob = log_prob;
                best_class = c as u32;
            }
        }

        best_class
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Decision Stump (for Ensemble Methods)
// ═══════════════════════════════════════════════════════════════════════════════

/// Decision Stump - single split classifier
#[derive(Debug, Clone, Copy)]
pub struct DecisionStump {
    /// Feature to split on
    pub feature: usize,
    /// Threshold value
    pub threshold: f64,
    /// Polarity (1 or -1)
    pub polarity: i32,
    /// Weighted error
    pub error: f64,
}

impl DecisionStump {
    pub fn new() -> Self {
        Self {
            feature: 0,
            threshold: 0.0,
            polarity: 1,
            error: 1.0,
        }
    }

    /// Fit stump to weighted data
    pub fn fit(
        &mut self,
        x: &[[f64; MAX_FEATURES]],
        y: &[i32],
        weights: &[f64],
        n: usize,
        n_features: usize,
    ) {
        let n = n.min(MAX_POINTS);
        let n_features = n_features.min(MAX_FEATURES);

        self.error = f64::MAX;

        #[allow(clippy::needless_range_loop)]
        for f in 0..n_features {
            // Find unique thresholds
            for i in 0..n {
                let thresh = x[i][f];

                for polarity in [-1i32, 1] {
                    let mut err = 0.0;

                    for j in 0..n {
                        let pred = if polarity as f64 * (x[j][f] - thresh) >= 0.0 {
                            1
                        } else {
                            -1
                        };
                        if pred != y[j] {
                            err += weights[j];
                        }
                    }

                    if err < self.error {
                        self.error = err;
                        self.feature = f;
                        self.threshold = thresh;
                        self.polarity = polarity;
                    }
                }
            }
        }
    }

    /// Predict
    pub fn predict(&self, x: &[f64; MAX_FEATURES]) -> i32 {
        if self.polarity as f64 * (x[self.feature] - self.threshold) >= 0.0 {
            1
        } else {
            -1
        }
    }
}

impl Default for DecisionStump {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// AdaBoost
// ═══════════════════════════════════════════════════════════════════════════════

/// AdaBoost ensemble classifier
#[derive(Debug)]
pub struct AdaBoost {
    /// Weak learners (stumps)
    stumps: [DecisionStump; 32],
    /// Learner weights (alpha)
    alphas: [f64; 32],
    /// Number of stumps
    n_stumps: usize,
}

impl AdaBoost {
    pub fn new() -> Self {
        Self {
            stumps: [DecisionStump::new(); 32],
            alphas: [0.0; 32],
            n_stumps: 0,
        }
    }

    /// Fit AdaBoost
    pub fn fit(
        &mut self,
        x: &[[f64; MAX_FEATURES]],
        y: &[i32],
        n: usize,
        n_features: usize,
        n_estimators: usize,
    ) {
        let n = n.min(MAX_POINTS);
        self.n_stumps = n_estimators.min(32);

        // Initialize weights
        let mut weights = [1.0 / n as f64; MAX_POINTS];

        for t in 0..self.n_stumps {
            // Fit weak learner
            self.stumps[t].fit(x, y, &weights, n, n_features);

            // Compute alpha
            let err = self.stumps[t].error.clamp(1e-10, 1.0 - 1e-10);
            self.alphas[t] = 0.5 * log((1.0 - err) / err);

            // Update weights
            let mut weight_sum = 0.0;
            for (i, w) in weights.iter_mut().enumerate().take(n) {
                let pred = self.stumps[t].predict(&x[i]);
                *w *= exp(-self.alphas[t] * y[i] as f64 * pred as f64);
                weight_sum += *w;
            }

            // Normalize
            for w in weights.iter_mut().take(n) {
                *w /= weight_sum;
            }
        }
    }

    /// Predict
    pub fn predict(&self, x: &[f64; MAX_FEATURES]) -> i32 {
        let mut sum = 0.0;
        for t in 0..self.n_stumps {
            sum += self.alphas[t] * self.stumps[t].predict(x) as f64;
        }
        if sum >= 0.0 {
            1
        } else {
            -1
        }
    }
}

impl Default for AdaBoost {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Centroid-Based (Geometric) Classifier
// ═══════════════════════════════════════════════════════════════════════════════

/// Nearest Centroid classifier (AEGIS-native geometric)
#[derive(Debug)]
pub struct NearestCentroid<const D: usize> {
    /// Centroids per class
    centroids: [[f64; D]; MAX_CLASSES],
    /// Number of classes
    n_classes: usize,
}

impl<const D: usize> NearestCentroid<D> {
    pub fn new() -> Self {
        Self {
            centroids: [[0.0; D]; MAX_CLASSES],
            n_classes: 0,
        }
    }

    /// Fit by computing class centroids
    pub fn fit(&mut self, x: &[[f64; D]], y: &[u32], n: usize) {
        let n = n.min(MAX_POINTS);

        // Count and sum per class
        let mut counts = [0usize; MAX_CLASSES];
        let mut sums = [[0.0f64; D]; MAX_CLASSES];

        for i in 0..n {
            let c = y[i] as usize;
            if c >= MAX_CLASSES {
                continue;
            }

            counts[c] += 1;
            for d in 0..D {
                sums[c][d] += x[i][d];
            }
            if c >= self.n_classes {
                self.n_classes = c + 1;
            }
        }

        // Compute centroids
        for c in 0..self.n_classes {
            if counts[c] > 0 {
                for (d, val) in self.centroids[c].iter_mut().enumerate().take(D) {
                    *val = sums[c][d] / counts[c] as f64;
                }
            }
        }
    }

    /// Predict by nearest centroid
    pub fn predict(&self, x: &[f64; D]) -> u32 {
        let mut best_class = 0;
        let mut best_dist = f64::MAX;

        for c in 0..self.n_classes {
            let dist = self.distance(x, &self.centroids[c]);
            if dist < best_dist {
                best_dist = dist;
                best_class = c as u32;
            }
        }

        best_class
    }

    /// Get centroid for class
    pub fn get_centroid(&self, class: usize) -> Option<&[f64; D]> {
        if class < self.n_classes {
            Some(&self.centroids[class])
        } else {
            None
        }
    }

    fn distance(&self, a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sqrt(sum)
    }
}

impl<const D: usize> Default for NearestCentroid<D> {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_regression() {
        // Simple linearly separable data
        let x = [
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        let y = [0.0, 0.0, 1.0, 1.0];

        let mut lr = LogisticRegression::new(2).with_lr(0.5).with_max_iter(100);
        lr.fit(&x, &y, 4);

        // Should classify correctly
        assert_eq!(lr.predict(&x[0]), 0);
        assert_eq!(lr.predict(&x[3]), 1);
    }

    #[test]
    fn test_perceptron() {
        let x = [
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                11.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        let y = [-1, -1, 1, 1];

        let mut p = Perceptron::new(2);
        let epochs = p.fit(&x, &y, 4, 100);

        // Should converge
        assert!(epochs < 100);
    }
}
