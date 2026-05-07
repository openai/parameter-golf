//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS ML Engine - Complete Machine Learning Library
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! A comprehensive ML library built from scratch for AEGIS:
//!
//! Core Modules:
//!   - linalg: Vector/Matrix ops, loss functions, gradients
//!   - regressor: Manifold regression (Linear, Polynomial, RBF, GP, Geodesic)
//!   - convergence: Topological convergence via Betti numbers
//!   - clustering: K-Means, DBSCAN, Hierarchical, Auto-K
//!   - classification: LogisticRegression, KNN, Perceptron, NaiveBayes, AdaBoost
//!   - neural: MLP, DenseLayer, Activations, Adam/SGD optimizers
//!   - benchmark: Escalating benchmark system
//!
//! All algorithms use seal-loop style convergence where applicable.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

// Core modules
pub mod benchmark;
pub mod convergence;
pub mod linalg;
pub mod convolution;
pub mod regressor;
pub mod tensor;
pub mod autograd;

// Extended ML library
pub mod classification;
pub mod clustering;
pub mod neural;

// Re-export key types
pub use benchmark::*;
pub use classification::{
    AdaBoost, GaussianNB, KNNClassifier, LogisticRegression, NearestCentroid, Perceptron,
};
pub use clustering::{
    AgglomerativeClustering, DBSCANResult, KMeans, KMeansResult, Linkage, DBSCAN,
};
pub use convergence::*;
// pub use linalg::{Matrix, Vector}; // Removed
pub use tensor::Tensor;
pub use neural::{Activation, DenseLayer, TrainingResult, MLP, OptimizerConfig};
pub use regressor::*;
pub mod gossip;
