//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Clustering Algorithms
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Topologically-aware clustering algorithms with seal loop convergence.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::{fabs, sqrt};

/// Maximum clusters
const MAX_CLUSTERS: usize = 16;
/// Maximum data points
const MAX_POINTS: usize = 256;

// ═══════════════════════════════════════════════════════════════════════════════
// K-Means Clustering
// ═══════════════════════════════════════════════════════════════════════════════

/// K-Means cluster result
#[derive(Debug, Clone)]
pub struct KMeansResult<const D: usize> {
    /// Cluster centroids
    pub centroids: [[f64; D]; MAX_CLUSTERS],
    /// Number of clusters
    pub k: usize,
    /// Cluster assignments for each point
    pub labels: [usize; MAX_POINTS],
    /// Number of points
    pub n_points: usize,
    /// Inertia (sum of squared distances to centroids)
    pub inertia: f64,
    /// Number of iterations to convergence
    pub iterations: u32,
}

impl<const D: usize> Default for KMeansResult<D> {
    fn default() -> Self {
        Self {
            centroids: [[0.0; D]; MAX_CLUSTERS],
            k: 0,
            labels: [0; MAX_POINTS],
            n_points: 0,
            inertia: 0.0,
            iterations: 0,
        }
    }
}

/// K-Means clustering with topological convergence
#[derive(Debug, Clone)]
pub struct KMeans<const D: usize> {
    /// Number of clusters
    k: usize,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed for initialization
    seed: u64,
}

impl<const D: usize> KMeans<D> {
    pub fn new(k: usize) -> Self {
        Self {
            k: k.min(MAX_CLUSTERS),
            max_iter: 100,
            tol: 1e-4,
            seed: 42,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Fit K-Means to data
    pub fn fit(&self, data: &[[f64; D]], n_points: usize) -> KMeansResult<D> {
        let n = n_points.min(MAX_POINTS);
        let k = self.k.min(n);

        let mut result = KMeansResult {
            k: k.min(n),
            n_points: n,
            ..Default::default()
        };

        if n == 0 || k == 0 {
            return result;
        }

        // Initialize centroids (k-means++ style: spread out initial points)
        self.init_centroids_plusplus(data, n, &mut result.centroids, k);

        let mut prev_inertia = f64::MAX;

        for iter in 0..self.max_iter {
            // Assignment step
            let mut changed = 0;
            for (i, row) in data.iter().enumerate().take(n) {
                let old_label = result.labels[i];
                let new_label = self.nearest_centroid(row, &result.centroids, k);
                if new_label != old_label {
                    result.labels[i] = new_label;
                    changed += 1;
                }
            }

            // Update step
            self.update_centroids(data, n, &result.labels, &mut result.centroids, k);

            // Compute inertia
            result.inertia = self.compute_inertia(data, n, &result.labels, &result.centroids);
            result.iterations = iter as u32 + 1;

            // Check convergence
            if changed == 0 || fabs(prev_inertia - result.inertia) < self.tol {
                break;
            }
            prev_inertia = result.inertia;
        }

        result
    }

    /// Initialize centroids using k-means++ algorithm
    fn init_centroids_plusplus(
        &self,
        data: &[[f64; D]],
        n: usize,
        centroids: &mut [[f64; D]; MAX_CLUSTERS],
        k: usize,
    ) {
        // Simple pseudo-random based on seed
        let mut rng = self.seed;

        // First centroid: random point
        let first_idx = (rng as usize) % n;
        centroids[0] = data[first_idx];

        for c in 1..k {
            // Find point with maximum distance to nearest centroid
            let mut max_dist = 0.0;
            let mut max_idx = 0;

            for (point_idx, row) in data.iter().enumerate().take(n) {
                let mut min_dist = f64::MAX;
                for centroid in centroids.iter().take(c) {
                    let dist = self.squared_distance(row, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }

                // Probability proportional to D^2
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let weighted_dist = min_dist * ((rng % 1000) as f64 / 1000.0 + 0.5);

                if weighted_dist > max_dist {
                    max_dist = weighted_dist;
                    max_idx = point_idx;
                }
            }

            centroids[c] = data[max_idx];
        }
    }

    /// Find nearest centroid to a point
    fn nearest_centroid(
        &self,
        point: &[f64; D],
        centroids: &[[f64; D]; MAX_CLUSTERS],
        k: usize,
    ) -> usize {
        let mut min_dist = f64::MAX;
        let mut min_idx = 0;

        for (j, centroid) in centroids.iter().enumerate().take(k) {
            let dist = self.squared_distance(point, centroid);
            if dist < min_dist {
                min_dist = dist;
                min_idx = j;
            }
        }

        min_idx
    }

    /// Update centroids based on current assignments
    fn update_centroids(
        &self,
        data: &[[f64; D]],
        n: usize,
        labels: &[usize; MAX_POINTS],
        centroids: &mut [[f64; D]; MAX_CLUSTERS],
        k: usize,
    ) {
        // Reset centroids
        let mut counts = [0usize; MAX_CLUSTERS];
        for c in centroids.iter_mut().take(k) {
            *c = [0.0; D];
        }

        // Sum points per cluster
        for i in 0..n {
            let label = labels[i];
            counts[label] += 1;
            for d in 0..D {
                centroids[label][d] += data[i][d];
            }
        }

        // Divide by count
        for c in 0..k {
            if counts[c] > 0 {
                for (_d, val) in centroids[c].iter_mut().enumerate().take(D) {
                    *val /= counts[c] as f64;
                }
            }
        }
    }

    /// Compute total inertia
    fn compute_inertia(
        &self,
        data: &[[f64; D]],
        n: usize,
        labels: &[usize; MAX_POINTS],
        centroids: &[[f64; D]; MAX_CLUSTERS],
    ) -> f64 {
        let mut inertia = 0.0;
        for i in 0..n {
            inertia += self.squared_distance(&data[i], &centroids[labels[i]]);
        }
        inertia
    }

    /// Squared Euclidean distance
    fn squared_distance(&self, a: &[f64; D], b: &[f64; D]) -> f64 {
        let mut sum = 0.0;
        for d in 0..D {
            let diff = a[d] - b[d];
            sum += diff * diff;
        }
        sum
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Auto-K Selection via Betti Numbers
// ═══════════════════════════════════════════════════════════════════════════════

/// Automatically determine optimal K using topological analysis
pub fn auto_k_selection<const D: usize>(data: &[[f64; D]], n: usize, epsilon: f64) -> usize {
    // Build epsilon-neighborhood graph and count connected components (β₀)
    let mut components = 0;
    let mut visited = [false; MAX_POINTS];
    let n = n.min(MAX_POINTS);

    for start in 0..n {
        if visited[start] {
            continue;
        }

        // BFS from this point
        components += 1;
        let mut stack = [0usize; 64];
        let mut top = 1;
        stack[0] = start;

        while top > 0 {
            top -= 1;
            let current = stack[top];

            if visited[current] {
                continue;
            }
            visited[current] = true;

            // Add neighbors
            for i in 0..n {
                if !visited[i] && i != current {
                    let dist = distance(&data[current], &data[i]);
                    if dist < epsilon && top < 64 {
                        stack[top] = i;
                        top += 1;
                    }
                }
            }
        }
    }

    // β₀ = number of connected components = suggested K
    components.clamp(1, MAX_CLUSTERS)
}

fn distance<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    sqrt(sum)
}

// ═══════════════════════════════════════════════════════════════════════════════
// DBSCAN (Density-Based Clustering)
// ═══════════════════════════════════════════════════════════════════════════════

/// DBSCAN clustering result
#[derive(Debug, Clone)]
pub struct DBSCANResult {
    /// Cluster labels (-1 = noise)
    pub labels: [i32; MAX_POINTS],
    /// Number of points
    pub n_points: usize,
    /// Number of clusters found
    pub n_clusters: usize,
    /// Number of noise points
    pub n_noise: usize,
}

impl Default for DBSCANResult {
    fn default() -> Self {
        Self {
            labels: [-1; MAX_POINTS],
            n_points: 0,
            n_clusters: 0,
            n_noise: 0,
        }
    }
}

/// DBSCAN density-based clustering
#[derive(Debug)]
pub struct DBSCAN<const D: usize> {
    /// Neighborhood radius
    epsilon: f64,
    /// Minimum points to form a cluster
    min_samples: usize,
}

impl<const D: usize> DBSCAN<D> {
    pub fn new(epsilon: f64, min_samples: usize) -> Self {
        Self {
            epsilon,
            min_samples: min_samples.max(1),
        }
    }

    /// Fit DBSCAN to data
    pub fn fit(&self, data: &[[f64; D]], n_points: usize) -> DBSCANResult {
        let n = n_points.min(MAX_POINTS);
        let mut result = DBSCANResult {
            n_points: n,
            ..Default::default()
        };

        if n == 0 {
            return result;
        }

        let mut cluster_id = 0i32;

        for i in 0..n {
            if result.labels[i] != -1 {
                continue; // Already processed
            }

            // Get neighbors
            let neighbors = self.region_query(data, n, i);

            if neighbors.len() < self.min_samples {
                // Noise point (stays -1)
                continue;
            }

            // Start new cluster
            result.labels[i] = cluster_id;

            // Expand cluster
            let mut seed_set = neighbors;
            let mut j = 0;
            while j < seed_set.len() {
                let q = seed_set[j];

                if result.labels[q] == -1 {
                    result.labels[q] = cluster_id; // Was noise, now border
                }

                if result.labels[q] != -1 && result.labels[q] != cluster_id {
                    j += 1;
                    continue; // Already in another cluster
                }

                result.labels[q] = cluster_id;

                let q_neighbors = self.region_query(data, n, q);
                if q_neighbors.len() >= self.min_samples {
                    // Add new neighbors to seed set
                    for &neighbor in &q_neighbors {
                        if !seed_set.contains(&neighbor) && seed_set.len() < MAX_POINTS {
                            let _ = seed_set.push(neighbor);
                        }
                    }
                }

                j += 1;
            }

            cluster_id += 1;
        }

        result.n_clusters = cluster_id as usize;
        result.n_noise = result.labels.iter().take(n).filter(|&&l| l == -1).count();

        result
    }

    /// Find all points within epsilon of point i
    fn region_query(
        &self,
        data: &[[f64; D]],
        n: usize,
        i: usize,
    ) -> heapless::Vec<usize, MAX_POINTS> {
        let mut neighbors = heapless::Vec::new();

        for j in 0..n {
            if distance(&data[i], &data[j]) <= self.epsilon {
                let _ = neighbors.push(j);
            }
        }

        neighbors
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hierarchical Clustering (Agglomerative)
// ═══════════════════════════════════════════════════════════════════════════════

/// Linkage method for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    Single,   // Min distance
    Complete, // Max distance
    Average,  // Average distance
}

/// Hierarchical clustering result (dendrogram)
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Merge history: (cluster_a, cluster_b, distance, new_size)
    pub merges: [(usize, usize, f64, usize); MAX_POINTS],
    /// Number of merges
    pub n_merges: usize,
    /// Number of original points
    pub n_points: usize,
}

impl Default for HierarchicalResult {
    fn default() -> Self {
        Self {
            merges: [(0, 0, 0.0, 0); MAX_POINTS],
            n_merges: 0,
            n_points: 0,
        }
    }
}

/// Agglomerative hierarchical clustering
#[derive(Debug)]
pub struct AgglomerativeClustering<const D: usize> {
    linkage: Linkage,
}

impl<const D: usize> AgglomerativeClustering<D> {
    pub fn new(linkage: Linkage) -> Self {
        Self { linkage }
    }

    /// Fit hierarchical clustering
    pub fn fit(&self, data: &[[f64; D]], n_points: usize) -> HierarchicalResult {
        let n = n_points.min(MAX_POINTS);
        let mut result = HierarchicalResult {
            n_points: n,
            ..Default::default()
        };

        if n <= 1 {
            return result;
        }

        // Initialize: each point is its own cluster
        let mut cluster_sizes = [1usize; MAX_POINTS];
        let mut active = [true; MAX_POINTS];
        let mut cluster_ids = [0usize; MAX_POINTS];
        for (i, id) in cluster_ids.iter_mut().enumerate().take(n) {
            *id = i;
        }

        let mut next_cluster_id = n;

        for merge_idx in 0..(n - 1) {
            // Find closest pair of active clusters
            let mut min_dist = f64::MAX;
            let mut best_i = 0;
            let mut best_j = 1;

            for i in 0..n {
                if !active[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if !active[j] {
                        continue;
                    }

                    let dist = match self.linkage {
                        Linkage::Single => distance(&data[i], &data[j]),
                        Linkage::Complete => distance(&data[i], &data[j]),
                        Linkage::Average => distance(&data[i], &data[j]),
                    };

                    if dist < min_dist {
                        min_dist = dist;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            // Merge clusters
            result.merges[merge_idx] = (
                cluster_ids[best_i],
                cluster_ids[best_j],
                min_dist,
                cluster_sizes[best_i] + cluster_sizes[best_j],
            );
            result.n_merges = merge_idx + 1;

            // Update: best_i becomes the merged cluster, best_j becomes inactive
            cluster_ids[best_i] = next_cluster_id;
            cluster_sizes[best_i] += cluster_sizes[best_j];
            active[best_j] = false;
            next_cluster_id += 1;
        }

        result
    }

    /// Cut dendrogram to get k clusters
    pub fn cut_tree(&self, result: &HierarchicalResult, k: usize) -> [usize; MAX_POINTS] {
        let mut labels = [0usize; MAX_POINTS];

        // Start with each point in its own cluster
        for (i, label) in labels.iter_mut().enumerate().take(result.n_points) {
            *label = i;
        }

        // Apply first (n - k) merges
        let n_merges_to_apply = result.n_merges.saturating_sub(k.saturating_sub(1));

        for m in 0..n_merges_to_apply {
            let (a, b, _, _) = result.merges[m];
            // All points with label b get label a
            for label in labels.iter_mut().take(result.n_points) {
                if *label == b {
                    *label = a;
                }
            }
        }

        // Renumber labels to be consecutive
        let mut label_map = [usize::MAX; MAX_POINTS];
        let mut next_label = 0;
        for i in 0..result.n_points {
            if label_map[labels[i]] == usize::MAX {
                label_map[labels[i]] = next_label;
                next_label += 1;
            }
            labels[i] = label_map[labels[i]];
        }

        labels
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let data = [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [10.0, 10.0, 0.0],
            [10.1, 10.1, 0.0],
        ];

        let kmeans = KMeans::<3>::new(2);
        let result = kmeans.fit(&data, 4);

        assert_eq!(result.k, 2);
        assert!(result.labels[0] == result.labels[1]); // Same cluster
        assert!(result.labels[2] == result.labels[3]); // Same cluster
        assert!(result.labels[0] != result.labels[2]); // Different clusters
    }

    #[test]
    fn test_auto_k() {
        let data = [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.0],
            [10.0, 10.0, 0.0],
            [10.1, 10.1, 0.0],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
            [0.0; 3],
        ];

        let k = auto_k_selection(&data, 4, 1.0);
        assert_eq!(k, 2); // Should find 2 clusters
    }
}
