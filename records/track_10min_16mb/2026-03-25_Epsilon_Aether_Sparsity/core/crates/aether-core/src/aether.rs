//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS AETHER Geometric Extensions
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements AETHER's geometric data primitives for sparse attention and
//! hierarchical data analysis.
//!
//! Reference: DOI: 10.13141/RG.2.2.14811.27684
//!
//! Core Primitives:
//!   - Block Centroids (means): Mean vector of embeddings per block
//!   - Block Radii: Max L2 distance from centroid to any point
//!   - Block Variances: Variance of distances from centroid
//!   - Block Concentrations: Average cosine alignment with centroid
//!
//! Extensions:
//!   - Hierarchical Block Trees (H-Block) for multi-granularity scoring
//!   - Geometric-Aware Compression based on variance/concentration
//!   - Semantic Drift Detection via centroid trajectory
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::sqrt;

// ═══════════════════════════════════════════════════════════════════════════════
// AETHER Constants
// ═══════════════════════════════════════════════════════════════════════════════

/// Block sizes for hierarchical tree
pub const BLOCK_LEVELS: [usize; 3] = [64, 256, 1024];

/// Maximum embedding dimension
const MAX_DIM: usize = 64;

/// Maximum blocks at finest level
const MAX_BLOCKS: usize = 128;

// ═══════════════════════════════════════════════════════════════════════════════
// Block Metadata
// ═══════════════════════════════════════════════════════════════════════════════

/// Geometric metadata for a single block
#[derive(Debug, Clone, Copy)]
pub struct BlockMetadata<const D: usize> {
    /// Centroid (mean) of the block
    pub centroid: [f64; D],

    /// Radius: max distance from centroid to any point
    pub radius: f64,

    /// Variance: variance of distances from centroid
    pub variance: f64,

    /// Concentration: average cosine alignment with centroid
    pub concentration: f64,

    /// Number of points in block
    pub count: usize,
}

impl<const D: usize> BlockMetadata<D> {
    pub const fn empty() -> Self {
        Self {
            centroid: [0.0; D],
            radius: 0.0,
            variance: 0.0,
            concentration: 0.0,
            count: 0,
        }
    }

    /// Compute metadata from a set of points
    pub fn from_points(points: &[[f64; D]]) -> Self {
        if points.is_empty() {
            return Self::empty();
        }

        let n = points.len();

        // Step 1: Compute centroid
        let mut centroid = [0.0f64; D];
        for point in points {
            for d in 0..D {
                centroid[d] += point[d];
            }
        }
        for val in centroid.iter_mut().take(D) {
            *val /= n as f64;
        }

        // Step 2: Compute distances and stats
        let mut max_dist = 0.0f64;
        let mut sum_dist = 0.0f64;
        let mut sum_dist_sq = 0.0f64;
        let mut sum_cosine = 0.0f64;

        let centroid_norm = Self::norm(&centroid);

        for point in points {
            // Distance from centroid
            let dist = Self::l2_distance(&centroid, point);
            max_dist = if dist > max_dist { dist } else { max_dist };
            sum_dist += dist;
            sum_dist_sq += dist * dist;

            // Cosine similarity with centroid
            if centroid_norm > 1e-10 {
                let point_norm = Self::norm(point);
                if point_norm > 1e-10 {
                    let dot = Self::dot(&centroid, point);
                    sum_cosine += dot / (centroid_norm * point_norm);
                }
            }
        }

        let mean_dist = sum_dist / n as f64;
        let variance = (sum_dist_sq / n as f64) - (mean_dist * mean_dist);
        let concentration = sum_cosine / n as f64;

        Self {
            centroid,
            radius: max_dist,
            variance: if variance > 0.0 { variance } else { 0.0 },
            concentration,
            count: n,
        }
    }

    /// L2 distance between two vectors
    fn l2_distance(a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0;
        for d in 0..D {
            let diff = a[d] - b[d];
            sum += diff * diff;
        }
        sqrt(sum)
    }

    /// Dot product
    fn dot(a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0;
        for d in 0..D {
            sum += a[d] * b[d];
        }
        sum
    }

    /// Vector norm
    fn norm(v: &[f64; D]) -> f64 {
        sqrt(Self::dot(v, v))
    }

    /// AETHER upper-bound score: Cauchy-Schwarz bound
    /// score ≤ ||q|| * (||centroid|| + radius)
    pub fn upper_bound_score(&self, query: &[f64; D]) -> f64 {
        let q_norm = Self::norm(query);
        let c_norm = Self::norm(&self.centroid);
        q_norm * (c_norm + self.radius)
    }

    /// Check if block can be pruned (score below threshold)
    pub fn can_prune(&self, query: &[f64; D], threshold: f64) -> bool {
        self.upper_bound_score(query) < threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hierarchical Block Tree (H-Block)
// ═══════════════════════════════════════════════════════════════════════════════

/// Hierarchical tree of geometric summaries
///
/// Level 0: 64-token blocks (finest)
/// Level 1: 256-token clusters (4 blocks)
/// Level 2: 1024-token super-clusters (16 blocks)
#[derive(Debug)]
pub struct HierarchicalBlockTree<const D: usize> {
    /// Metadata at each level
    levels: [[BlockMetadata<D>; MAX_BLOCKS]; 3],

    /// Block counts at each level
    counts: [usize; 3],
}

impl<const D: usize> HierarchicalBlockTree<D> {
    pub fn new() -> Self {
        Self {
            levels: [[BlockMetadata::empty(); MAX_BLOCKS]; 3],
            counts: [0; 3],
        }
    }

    /// Build hierarchy from fine-level blocks
    pub fn build_from_blocks(&mut self, blocks: &[BlockMetadata<D>]) {
        // Level 0: direct copy
        let n0 = blocks.len().min(MAX_BLOCKS);
        self.levels[0][..n0].copy_from_slice(&blocks[..n0]);
        self.counts[0] = n0;

        // Level 1: aggregate 4 blocks each
        let n1 = n0.div_ceil(4);
        for i in 0..n1 {
            let start = i * 4;
            let end = (start + 4).min(n0);
            self.levels[1][i] = self.aggregate_blocks(&blocks[start..end]);
        }
        self.counts[1] = n1;

        // Level 2: aggregate 4 level-1 blocks each
        let n2 = n1.div_ceil(4);
        for i in 0..n2 {
            let start = i * 4;
            let end = (start + 4).min(n1);
            let l1_slice: &[BlockMetadata<D>] = &self.levels[1][start..end];
            self.levels[2][i] = self.aggregate_metadata(l1_slice);
        }
        self.counts[2] = n2;
    }

    /// Aggregate blocks into parent block
    fn aggregate_blocks(&self, blocks: &[BlockMetadata<D>]) -> BlockMetadata<D> {
        if blocks.is_empty() {
            return BlockMetadata::empty();
        }

        // Centroid = weighted mean of child centroids
        let mut centroid = [0.0f64; D];
        let mut total_count = 0usize;
        let mut max_radius = 0.0f64;
        let mut sum_variance = 0.0f64;
        let mut sum_concentration = 0.0f64;

        for block in blocks {
            let w = block.count as f64;
            for (d, val) in centroid.iter_mut().enumerate().take(D) {
                *val += block.centroid[d] * w;
            }
            total_count += block.count;

            // Aggregate radius: max of child radii + distance between centroids
            if block.radius > max_radius {
                max_radius = block.radius;
            }

            sum_variance += block.variance * w;
            sum_concentration += block.concentration * w;
        }

        if total_count > 0 {
            for val in centroid.iter_mut().take(D) {
                *val /= total_count as f64;
            }
        }

        // Add inter-child distances to radius
        for block in blocks {
            let dist = BlockMetadata::<D>::l2_distance(&centroid, &block.centroid);
            let effective_radius = dist + block.radius;
            if effective_radius > max_radius {
                max_radius = effective_radius;
            }
        }

        BlockMetadata {
            centroid,
            radius: max_radius,
            variance: if total_count > 0 {
                sum_variance / total_count as f64
            } else {
                0.0
            },
            concentration: if total_count > 0 {
                sum_concentration / total_count as f64
            } else {
                0.0
            },
            count: total_count,
        }
    }

    /// Aggregate metadata array
    fn aggregate_metadata(&self, metas: &[BlockMetadata<D>]) -> BlockMetadata<D> {
        self.aggregate_blocks(metas)
    }

    /// Hierarchical query with early pruning
    /// Returns indices of fine-level blocks that pass threshold
    pub fn hierarchical_query(&self, query: &[f64; D], threshold: f64) -> [bool; MAX_BLOCKS] {
        let mut result = [false; MAX_BLOCKS];

        // Start from coarsest level
        let mut active_l2 = [true; MAX_BLOCKS];

        // Level 2 pruning
        for (i, active) in active_l2.iter_mut().enumerate().take(self.counts[2]) {
            if self.levels[2][i].can_prune(query, threshold) {
                *active = false;
            }
        }

        // Level 1 pruning
        let mut active_l1 = [false; MAX_BLOCKS];
        for (i, active) in active_l1.iter_mut().enumerate().take(self.counts[1]) {
            let parent = i / 4;
            if parent < self.counts[2] && active_l2[parent] 
               && !self.levels[1][i].can_prune(query, threshold) {
                *active = true;
            }
        }

        // Level 0 (finest) - final result
        for (i, res) in result.iter_mut().enumerate().take(self.counts[0]) {
            let parent = i / 4;
            if parent < self.counts[1] && active_l1[parent]
               && !self.levels[0][i].can_prune(query, threshold) {
                *res = true;
            }
        }

        result
    }

    /// Get complexity reduction factor
    pub fn pruning_ratio(&self, active_mask: &[bool; MAX_BLOCKS]) -> f64 {
        let active = active_mask.iter().filter(|&&x| x).count();
        if self.counts[0] == 0 {
            return 0.0;
        }
        1.0 - (active as f64 / self.counts[0] as f64)
    }
}

impl<const D: usize> Default for HierarchicalBlockTree<D> {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Geometric Compression
// ═══════════════════════════════════════════════════════════════════════════════

/// Compression strategy based on geometric properties
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionStrategy {
    /// Store centroid + delta encoding (low variance blocks)
    CentroidDelta,
    /// Aggressive 4-bit quantization (high concentration)
    Int4Quantize,
    /// Keep full precision (dispersed blocks)
    FullPrecision,
}

/// Determine compression strategy from block metadata
pub fn select_compression<const D: usize>(meta: &BlockMetadata<D>) -> CompressionStrategy {
    if meta.variance < 0.1 {
        CompressionStrategy::CentroidDelta
    } else if meta.concentration > 0.9 {
        CompressionStrategy::Int4Quantize
    } else {
        CompressionStrategy::FullPrecision
    }
}

/// Estimate compression ratio for a block
pub fn estimate_compression_ratio<const D: usize>(meta: &BlockMetadata<D>) -> f64 {
    match select_compression(meta) {
        CompressionStrategy::CentroidDelta => 4.0, // ~4x compression
        CompressionStrategy::Int4Quantize => 4.0,  // 16-bit to 4-bit
        CompressionStrategy::FullPrecision => 1.0, // No compression
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Semantic Drift Detection
// ═══════════════════════════════════════════════════════════════════════════════

/// Track centroid trajectory for drift detection
#[derive(Debug)]
pub struct DriftDetector<const D: usize> {
    /// History of centroids
    history: [[f64; D]; 32],
    history_len: usize,
    history_pos: usize,

    /// Velocity (rate of change)
    velocity: [f64; D],

    /// Expected next position
    expected: [f64; D],
}

impl<const D: usize> DriftDetector<D> {
    pub fn new() -> Self {
        Self {
            history: [[0.0; D]; 32],
            history_len: 0,
            history_pos: 0,
            velocity: [0.0; D],
            expected: [0.0; D],
        }
    }

    /// Update with new centroid, returns drift score
    pub fn update(&mut self, centroid: &[f64; D]) -> f64 {
        // Calculate drift from expected
        let drift = if self.history_len > 0 {
            BlockMetadata::<D>::l2_distance(&self.expected, centroid)
        } else {
            0.0
        };

        // Update velocity
        if self.history_len > 0 {
            let prev_idx = (self.history_pos + 31) % 32;
            for (d, vel) in self.velocity.iter_mut().enumerate().take(D) {
                *vel = centroid[d] - self.history[prev_idx][d];
            }
        }

        // Predict next position
        for (d, exp) in self.expected.iter_mut().enumerate().take(D) {
            *exp = centroid[d] + self.velocity[d];
        }

        // Store in history
        self.history[self.history_pos] = *centroid;
        self.history_pos = (self.history_pos + 1) % 32;
        if self.history_len < 32 {
            self.history_len += 1;
        }

        drift
    }

    /// Check if drifting (threshold-based)
    pub fn is_drifting(&self, threshold: f64) -> bool {
        // Calculate average velocity magnitude
        let vel_mag = BlockMetadata::<D>::norm(&self.velocity);
        vel_mag > threshold
    }

    /// Get current velocity magnitude
    pub fn velocity_magnitude(&self) -> f64 {
        BlockMetadata::<D>::norm(&self.velocity)
    }
}

impl<const D: usize> Default for DriftDetector<D> {
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
    fn test_block_metadata_single_point() {
        let points = [[1.0, 2.0, 3.0]];
        let meta = BlockMetadata::from_points(&points);

        assert!((meta.centroid[0] - 1.0).abs() < 1e-10);
        assert_eq!(meta.radius, 0.0);
        assert_eq!(meta.count, 1);
    }

    #[test]
    fn test_block_metadata_centroid() {
        let points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let meta = BlockMetadata::from_points(&points);

        assert!((meta.centroid[0] - 1.0).abs() < 1e-10);
        assert!((meta.radius - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compression_strategy() {
        let mut meta = BlockMetadata::<3>::empty();

        meta.variance = 0.05;
        assert_eq!(
            select_compression(&meta),
            CompressionStrategy::CentroidDelta
        );

        meta.variance = 0.5;
        meta.concentration = 0.95;
        assert_eq!(select_compression(&meta), CompressionStrategy::Int4Quantize);

        meta.concentration = 0.5;
        assert_eq!(
            select_compression(&meta),
            CompressionStrategy::FullPrecision
        );
    }

    #[test]
    fn test_drift_detector() {
        let mut detector = DriftDetector::<3>::new();

        // Steady trajectory
        detector.update(&[0.0, 0.0, 0.0]);
        detector.update(&[1.0, 0.0, 0.0]);
        detector.update(&[2.0, 0.0, 0.0]);

        assert!((detector.velocity[0] - 1.0).abs() < 1e-10);

        // Sudden drift
        let drift = detector.update(&[5.0, 0.0, 0.0]);
        assert!(drift > 1.0); // Expected 3.0, got 5.0 → drift of 2.0
    }
}
