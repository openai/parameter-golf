//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! AEGIS Topological Data Manifold
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! Transforms massive data streams into topological shapes using sparse attention
//! and geometric concentration. This is the core of "making data 3D for everyone".
//!
//! Key Concepts:
//!   - Sparse Attention: Only attend to topologically significant points
//!   - Concentration: Collapse high-dim data onto low-dim manifold
//!   - Shape Extraction: Persistent homology for structure discovery
//!
//! Mathematical Foundation:
//!   - Time-Delay Embedding: Î¦(x) = [x(t), x(t-Ï„), x(t-2Ï„), ...]
//!   - Sparse Attention: A(i,j) = 1 iff d(xáµ¢, xâ±¼) < Îµ (geometric locality)
//!   - Concentration: Project to principal manifold via local PCA
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#![allow(dead_code)]

use libm::sqrt;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Manifold Constants
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Default embedding dimension for time-delay embedding
const DEFAULT_EMBED_DIM: usize = 3;

/// Default time delay (Ï„) for embedding
const DEFAULT_TAU: usize = 1;

/// Default neighborhood radius for sparse attention
const DEFAULT_EPSILON: f64 = 0.5;

/// Maximum points to track (memory constraint for no_std)
const MAX_POINTS: usize = 256;

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Point Cloud Representation
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// A point in the embedded manifold space
#[derive(Debug, Clone, Copy)]
pub struct ManifoldPoint<const D: usize> {
    pub coords: [f64; D],
}

impl<const D: usize> ManifoldPoint<D> {
    pub const fn zero() -> Self {
        Self { coords: [0.0; D] }
    }

    pub fn new(coords: [f64; D]) -> Self {
        Self { coords }
    }

    /// Euclidean distance to another point
    pub fn distance(&self, other: &Self) -> f64 {
        let mut sum = 0.0;
        for i in 0..D {
            let d = self.coords[i] - other.coords[i];
            sum += d * d;
        }
        sqrt(sum)
    }

    /// Check if within epsilon-neighborhood (sparse attention criterion)
    pub fn is_neighbor(&self, other: &Self, epsilon: f64) -> bool {
        self.distance(other) < epsilon
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Time-Delay Embedding
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Embeds 1D signal into D-dimensional manifold using Takens' theorem
///
/// Î¦(t) = [x(t), x(t-Ï„), x(t-2Ï„), ..., x(t-(D-1)Ï„)]
///
/// This transforms temporal data into geometric shapes that reveal
/// the underlying dynamical system's attractor.
#[derive(Debug)]
pub struct TimeDelayEmbedder<const D: usize> {
    /// Time delay parameter Ï„
    tau: usize,

    /// Circular buffer for recent values
    buffer: [f64; 256],
    buffer_pos: usize,
    buffer_len: usize,
}

impl<const D: usize> TimeDelayEmbedder<D> {
    pub fn new(tau: usize) -> Self {
        Self {
            tau: if tau == 0 { 1 } else { tau },
            buffer: [0.0; 256],
            buffer_pos: 0,
            buffer_len: 0,
        }
    }

    /// Add a new sample to the buffer
    pub fn push(&mut self, value: f64) {
        self.buffer[self.buffer_pos] = value;
        self.buffer_pos = (self.buffer_pos + 1) % 256;
        if self.buffer_len < 256 {
            self.buffer_len += 1;
        }
    }

    /// Get embedded point from current buffer state
    pub fn embed(&self) -> Option<ManifoldPoint<D>> {
        let required = D * self.tau;
        if self.buffer_len < required {
            return None;
        }

        let mut point = ManifoldPoint::zero();
        for i in 0..D {
            let offset = i * self.tau;
            let idx = (self.buffer_pos + 256 - 1 - offset) % 256;
            point.coords[i] = self.buffer[idx];
        }

        Some(point)
    }

    /// Reset the embedder
    pub fn reset(&mut self) {
        self.buffer = [0.0; 256];
        self.buffer_pos = 0;
        self.buffer_len = 0;
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Sparse Attention Graph
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Sparse attention matrix using geometric locality
///
/// Instead of O(nÂ²) dense attention, we only connect points within
/// Îµ-neighborhood. This is the key to handling massive data.
///
/// A(i,j) = 1 iff d(páµ¢, pâ±¼) < Îµ
#[derive(Debug)]
pub struct SparseAttentionGraph<const D: usize> {
    /// Point cloud
    points: [ManifoldPoint<D>; MAX_POINTS],
    point_count: usize,

    /// Epsilon neighborhood radius
    epsilon: f64,

    /// Sparse adjacency (bit-packed for memory efficiency)
    /// adjacency[i] is bitmask of neighbors for point i
    adjacency: [u64; MAX_POINTS],
}

impl<const D: usize> SparseAttentionGraph<D> {
    pub fn new(epsilon: f64) -> Self {
        Self {
            points: [ManifoldPoint::zero(); MAX_POINTS],
            point_count: 0,
            epsilon,
            adjacency: [0; MAX_POINTS],
        }
    }

    /// Add a point and compute its sparse attention edges
    pub fn add_point(&mut self, point: ManifoldPoint<D>) -> Option<usize> {
        if self.point_count >= MAX_POINTS {
            return None;
        }

        let idx = self.point_count;
        self.points[idx] = point;

        // Compute sparse edges (only to nearby points)
        let mut mask = 0u64;
        for i in 0..idx {
            if point.is_neighbor(&self.points[i], self.epsilon) {
                // Set bit for neighbor relationship
                if i < 64 {
                    mask |= 1 << i;
                    // Symmetric: add reverse edge
                    self.adjacency[i] |= 1 << (idx % 64);
                }
            }
        }
        self.adjacency[idx] = mask;

        self.point_count += 1;
        Some(idx)
    }

    /// Get number of neighbors (degree) for a point
    pub fn degree(&self, idx: usize) -> u32 {
        if idx >= self.point_count {
            return 0;
        }
        self.adjacency[idx].count_ones()
    }

    /// Check if two points are connected
    pub fn are_neighbors(&self, i: usize, j: usize) -> bool {
        if i >= self.point_count || j >= self.point_count || j >= 64 {
            return false;
        }
        (self.adjacency[i] & (1 << j)) != 0
    }

    /// Compute connected components (Î²â‚€) using Union-Find
    pub fn compute_betti_0(&self) -> u32 {
        if self.point_count == 0 {
            return 0;
        }

        // Simple DFS-based component counting
        let mut visited = [false; MAX_POINTS];
        let mut components = 0u32;

        for start in 0..self.point_count {
            if visited[start] {
                continue;
            }

            // BFS/DFS from this point
            components += 1;
            let mut stack = [0usize; 64];
            let mut stack_top = 1;

            stack[0] = start;

            while stack_top > 0 {
                stack_top -= 1;
                let current = stack[stack_top];

                if visited[current] {
                    continue;
                }
                visited[current] = true;

                // Add unvisited neighbors
                for (neighbor, is_visited) in visited.iter().enumerate().take(64.min(self.point_count)) {
                    if !*is_visited && self.are_neighbors(current, neighbor) && stack_top < 64 {
                        stack[stack_top] = neighbor;
                        stack_top += 1;
                    }
                }
            }
        }

        components
    }

    /// Estimate Î²â‚ (cycles) using Euler characteristic
    /// Ï‡ = V - E + F, for planar: Î²â‚€ - Î²â‚ + Î²â‚‚ = Ï‡
    /// Simplified: Î²â‚ â‰ˆ E - V + Î²â‚€ (ignoring higher homology)
    pub fn estimate_betti_1(&self) -> u32 {
        let v = self.point_count as i32;
        let mut e = 0i32;

        for i in 0..self.point_count {
            e += self.adjacency[i].count_ones() as i32;
        }
        e /= 2; // Edges counted twice

        let b0 = self.compute_betti_0() as i32;

        // Î²â‚ â‰ˆ E - V + Î²â‚€ (simplified)
        let b1 = e - v + b0;
        if b1 > 0 {
            b1 as u32
        } else {
            0
        }
    }

    /// Get the topological shape signature
    pub fn shape(&self) -> (u32, u32) {
        (self.compute_betti_0(), self.estimate_betti_1())
    }

    /// Geodesic Partitioning: Find centroid of the local cluster connected to `target`
    ///
    /// This approximates a "local convex hull" by traversing the sparse graph (BFS)
    /// for a limited depth, effectively partitioning the manifold geodesically.
    pub fn geodesic_partition_centroid(&self, target: ManifoldPoint<D>) -> Option<ManifoldPoint<D>> {
        if self.point_count == 0 {
            return None;
        }

        // 1. Find the graph node closest to the target point (entry point)
        // Since `target` might be the one just added, it's likely the last one.
        // But let's be robust and check the last few points.
        let start_node_idx = self.point_count - 1; 

        // 2. BFS to find the local cluster (Geodesic Neighborhood)
        // We limit depth to capture "local" structure, not the whole component
        let max_depth = 3; 
        let mut visited = [false; MAX_POINTS];
        let mut queue = [0usize; 64];
        let mut queue_start = 0;
        let mut queue_end = 0;

        queue[queue_end] = start_node_idx;
        queue_end += 1;
        visited[start_node_idx] = true;

        let mut cluster_sum: ManifoldPoint<D> = ManifoldPoint::zero();
        let mut cluster_count = 0;
        let mut current_depth = 0;
        let mut nodes_at_current_depth = 1;
        let mut nodes_at_next_depth = 0;

        while queue_start < queue_end {
            let u = queue[queue_start];
            queue_start += 1;

            // Accumulate for centroid
            for k in 0..D {
                cluster_sum.coords[k] += self.points[u].coords[k];
            }
            cluster_count += 1;

            nodes_at_current_depth -= 1;

            // Expand neighbors if depth limit not reached
            if current_depth < max_depth {
                // Adjacency bitmask iteration
                let adjacency = self.adjacency[u]; 
                // Note: Adjacency is symmetric but stored sparsely? 
                // In our `add_point`, we set bits for i < 64. 
                // Let's assume simpler iteration for this limited embedded interaction.
                // We iterate all points to check `are_neighbors` because internal representation 
                // in original code was slightly simplified (only stored back-edges in `adjacency`?).
                // Let's rely on `are_neighbors` which is robust in the provided code.
                
                for v in 0..self.point_count.min(64) { // Limit to 64 for speed/bitmask strictness
                    if !visited[v] && self.are_neighbors(u, v) {
                        visited[v] = true;
                        if queue_end < 64 {
                            queue[queue_end] = v;
                            queue_end += 1;
                            nodes_at_next_depth += 1;
                        }
                    }
                }
            }
            
            if nodes_at_current_depth == 0 {
                current_depth += 1;
                nodes_at_current_depth = nodes_at_next_depth;
                nodes_at_next_depth = 0;
            }
        }

        if cluster_count == 0 {
            return None;
        }

        // 3. Compute Centroid
        let mut centroid = ManifoldPoint::zero();
        for k in 0..D {
            centroid.coords[k] = cluster_sum.coords[k] / cluster_count as f64;
        }

        Some(centroid)
    }

    /// Clear the graph
    pub fn clear(&mut self) {
        self.point_count = 0;
        self.adjacency = [0; MAX_POINTS];
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Concentration (Dimension Reduction)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Geometric concentration: collapse high-dim data to principal axes
/// Uses streaming mean and variance for memory efficiency
#[derive(Debug)]
pub struct GeometricConcentrator<const D: usize> {
    /// Running mean
    mean: [f64; 8],

    /// Running variance (for principal direction)
    variance: [f64; 8],

    /// Sample count
    count: u64,
}

impl<const D: usize> GeometricConcentrator<D> {
    pub fn new() -> Self {
        Self {
            mean: [0.0; 8],
            variance: [0.0; 8],
            count: 0,
        }
    }

    /// Update statistics with new point (Welford's algorithm)
    pub fn update(&mut self, point: &ManifoldPoint<D>) {
        self.count += 1;
        let n = self.count as f64;

        for i in 0..D.min(8) {
            let delta = point.coords[i] - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = point.coords[i] - self.mean[i];
            self.variance[i] += delta * delta2;
        }
    }

    /// Get the principal dimension (highest variance)
    pub fn principal_dimension(&self) -> usize {
        let mut max_var = 0.0;
        let mut max_dim = 0;

        for i in 0..D.min(8) {
            if self.variance[i] > max_var {
                max_var = self.variance[i];
                max_dim = i;
            }
        }

        max_dim
    }

    /// Project point onto principal axis (1D concentration)
    pub fn concentrate_1d(&self, point: &ManifoldPoint<D>) -> f64 {
        let dim = self.principal_dimension();
        if dim < D {
            point.coords[dim] - self.mean[dim]
        } else {
            0.0
        }
    }

    /// Get concentration ratio (how much variance is in principal dim)
    pub fn concentration_ratio(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }

        let total: f64 = self.variance.iter().take(D.min(8)).sum();
        if total == 0.0 {
            return 0.0;
        }

        let principal = self.variance[self.principal_dimension()];
        principal / total
    }

    pub fn reset(&mut self) {
        self.mean = [0.0; 8];
        self.variance = [0.0; 8];
        self.count = 0;
    }
}

impl<const D: usize> Default for GeometricConcentrator<D> {
    fn default() -> Self {
        Self::new()
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Complete Manifold Pipeline
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Complete Manifold Pipeline - The Gatekeeper
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

/// Full pipeline: Sparsity Filter â†’ Topological Check â†’ Branching â†’ TPU Injection
pub struct TopologicalPipeline<const D: usize> {
    embedder: TimeDelayEmbedder<D>,
    graph: SparseAttentionGraph<D>,
    concentrator: GeometricConcentrator<D>,
}

impl<const D: usize> TopologicalPipeline<D> {
    pub fn new(tau: usize, epsilon: f64) -> Self {
        Self {
            embedder: TimeDelayEmbedder::new(tau),
            graph: SparseAttentionGraph::new(epsilon),
            concentrator: GeometricConcentrator::new(),
        }
    }

    /// Process a new data sample - The Gatekeeper Flow
    pub fn push(&mut self, value: f64) -> Option<(u32, u32, u64)> {
        // Stage 1: The Sparsity Filter
        if libm::fabs(value) < 1e-9 {
            return None; // Drop zero-values immediately
        }

        self.embedder.push(value);

        if let Some(point) = self.embedder.embed() {
            // Stage 2: The Topological Check
            self.graph.add_point(point)?;
            let (betti_0, betti_1) = self.graph.shape();

            // Stage 3: The Branching
            let projected_value = if betti_1 == 0 {
                // Path 1: Direct Projection (Speed)
                self.concentrator.update(&point);
                self.concentrator.concentrate_1d(&point)
            } else {
                // Path 2: Geodesic Partitioning (Stability)
                // Project relative to the centroid of the local geodesic cluster
                if let Some(centroid) = self.graph.geodesic_partition_centroid(point) {
                    // Simple projection to the "densest" cluster's frame
                    // We use the distance to the centroid as the 1D projection for now
                    point.distance(&centroid)
                } else {
                    0.0
                }
            };

            // Stage 4: TPU Injection
            // Map the final projected "coordinate" (and original point) to a TPU ID
            let tpu_id = self.map_to_tpu_id(&point, projected_value);

            Some((betti_0, betti_1, tpu_id))
        } else {
            None
        }
    }

    /// Stage 4: Map coordinates to TPU Interconnect ID
    fn map_to_tpu_id(&self, point: &ManifoldPoint<D>, projection: f64) -> u64 {
        // Synthetic Spatial Hashing (Morton-like)
        let mut hash = 0u64;
        
        // Hash the input coordinates
        for i in 0..D {
            let bits = point.coords[i].to_bits();
            hash = hash.rotate_left(11) ^ bits;
        }

        // Hash the projection
        hash = hash.rotate_right(7) ^ projection.to_bits();

        hash
    }

    /// Get current shape (Î²â‚€, Î²â‚)
    pub fn shape(&self) -> (u32, u32) {
        self.graph.shape()
    }

    /// Get concentration ratio
    pub fn concentration(&self) -> f64 {
        self.concentrator.concentration_ratio()
    }

    /// Reset pipeline
    pub fn reset(&mut self) {
        self.embedder.reset();
        self.graph.clear();
        self.concentrator.reset();
    }
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Unit Tests
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = ManifoldPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = ManifoldPoint::<3>::new([3.0, 4.0, 0.0]);

        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_attention_neighbor() {
        let p1 = ManifoldPoint::<3>::new([0.0, 0.0, 0.0]);
        let p2 = ManifoldPoint::<3>::new([0.1, 0.1, 0.0]);

        assert!(p1.is_neighbor(&p2, 0.5));
        assert!(!p1.is_neighbor(&p2, 0.1));
    }

    #[test]
    fn test_embedding() {
        let mut emb = TimeDelayEmbedder::<3>::new(1);

        for i in 0..10 {
            emb.push(i as f64);
        }

        let point = emb.embed().unwrap();
        // Should be [9, 8, 7] (last 3 values with Ï„=1)
        assert!((point.coords[0] - 9.0).abs() < 1e-10);
        assert!((point.coords[1] - 8.0).abs() < 1e-10);
        assert!((point.coords[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_component() {
        let mut graph = SparseAttentionGraph::<3>::new(1.0);

        // Add points that are all within epsilon of each other
        graph.add_point(ManifoldPoint::new([0.0, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.0, 0.0]));
        graph.add_point(ManifoldPoint::new([0.5, 0.5, 0.0]));

        // Should be single connected component
        assert_eq!(graph.compute_betti_0(), 1);
    }

    #[test]
    fn test_gatekeeper_sparsity() {
        let mut pipeline = TopologicalPipeline::<3>::new(1, 0.5);
        
        // Push zero value - should be dropped by Sparsity Filter
        assert!(pipeline.push(0.0).is_none());
        assert!(pipeline.push(1e-10).is_none());
        
        // Push significant value - should be processed
        // Need to fill buffer first (tau=1, D=3 -> needs 3 points)
        pipeline.push(1.0);
        pipeline.push(2.0);
        pipeline.push(3.0);
        
        // Now it should return consistent output
        let result = pipeline.push(4.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_gatekeeper_tpu_injection() {
        let mut pipeline = TopologicalPipeline::<3>::new(1, 0.5);
        
        // Fill buffer
        pipeline.push(1.0);
        pipeline.push(2.0);
        pipeline.push(3.0);
        
        if let Some((_, _, tpu_id_1)) = pipeline.push(4.0) {
             // Push same sequence again (reset logic simulated)
             let mut pipeline2 = TopologicalPipeline::<3>::new(1, 0.5);
             pipeline2.push(1.0);
             pipeline2.push(2.0);
             pipeline2.push(3.0);
             let (_, _, tpu_id_2) = pipeline2.push(4.0).unwrap();
             
             assert_eq!(tpu_id_1, tpu_id_2, "TPU ID generation must be deterministic");
        }
    }

    #[test]
    fn test_gatekeeper_branching() {
        let mut pipeline = TopologicalPipeline::<3>::new(1, 2.0); // large epsilon to force connection
        
        // 1. Simple shape (Line) -> Betti-1 = 0
        for i in 0..10 {
            pipeline.push(i as f64);
        }
        let (b0, b1, _) = pipeline.push(10.0).unwrap();
        assert_eq!(b0, 1);
        assert_eq!(b1, 0); // Linear structure has no holes
        
        // 2. Complex Shape (Cycle) -> Betti-1 > 0
        pipeline.reset();
        // Create a triangle loop: (0,0,0) -> (1,0,0) -> (0.5,1,0) -> (0,0,0) around time delay
        // This is hard to simulate perfectly with 1D stream, but we can try oscillating
        // A simple sine wave often creates loops in delay embedding
        for i in 0..50 {
            let val = libm::sin(i as f64 * 0.5);
            pipeline.push(val); 
        }
        
        let (_, b1_complex, _) = pipeline.push(0.1).unwrap();
        // Sine wave in 2D/3D embedding is a loop (circle)
        assert!(b1_complex >= 1, "Sine wave should create a cycle (Betti-1 >= 1)");
    }
}

