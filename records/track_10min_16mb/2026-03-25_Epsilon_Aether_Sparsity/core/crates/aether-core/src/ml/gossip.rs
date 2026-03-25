//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Gossip Protocol (Distributed TDA)
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Implements a TPU-style "Gossip Protocol" for asynchronous consistency of
//! topological centroids across distributed cores.
//!
//! Mechanism:
//! 1. Local Betti-Centroid: Each core computes the centroid of its local manifold.
//! 2. Ring-Pass ICL: Centroids are passed to neighbors via Inter-Chip Link (ICL).
//! 3. Running Average: Nodes update their global estimate using a weighted average.
//!
//! Convergence:
//! Asynchronous consistency allows the solver to proceed without "stop-the-world"
//! synchronization, converging mathematically before the backward pass completes.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::sqrt;
use crate::ml::clustering::{KMeans, KMeansResult}; // Reuse existing clustering logic if needed

/// Maximum dimension for the manifold points
const MAX_DIM: usize = 3; 

// ═══════════════════════════════════════════════════════════════════════════════
// Core Structures
// ═══════════════════════════════════════════════════════════════════════════════

/// A node representing a TPU core or distributed worker
#[derive(Debug, Clone)]
pub struct GossipNode {
    /// Unique ID of the core
    pub id: usize,
    
    /// Local data shard (simulated)
    local_data: heapless::Vec<[f64; MAX_DIM], 256>,

    /// The locally computed Betti-centroid (from local data)
    pub local_centroid: [f64; MAX_DIM],

    /// The node's current estimate of the global centroid
    pub global_estimate: [f64; MAX_DIM],

    /// Weight for running average (confidence/count)
    weight: f64,
}

impl GossipNode {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            local_data: heapless::Vec::new(),
            local_centroid: [0.0; MAX_DIM],
            global_estimate: [0.0; MAX_DIM],
            weight: 1.0,
        }
    }

    /// Add data point to local shard
    pub fn push_data(&mut self, point: [f64; MAX_DIM]) {
        if self.local_data.len() < 256 {
            let _ = self.local_data.push(point);
        }
    }

    /// Compute the local Betti-centroid based on current data
    /// For simplicity, we use the geometric mean, but this could be topologically weighted.
    pub fn compute_local_centroid(&mut self) {
        if self.local_data.is_empty() {
            return;
        }

        let mut sum = [0.0; MAX_DIM];
        for p in &self.local_data {
            for i in 0..MAX_DIM {
                sum[i] += p[i];
            }
        }

        let n = self.local_data.len() as f64;
        for i in 0..MAX_DIM {
            self.local_centroid[i] = sum[i] / n;
        }

        // Initially, global estimate is just the local centroid
        if self.weight <= 1.0 { 
             self.global_estimate = self.local_centroid;
        }
    }

    /// Receive a gossip message (neighbor's estimate) and update local state
    /// Uses exponential moving average or weighted average for consensus.
    /// 
    /// Formula: NewEstimate = (OldEstimate * Weight + NeighborEstimate) / (Weight + 1)
    pub fn update_consensus(&mut self, neighbor_estimate: [f64; MAX_DIM]) {
        let alpha = 0.5; // Mixing rate. 0.5 means equal weight to self and neighbor (fast mixing)

        for i in 0..MAX_DIM {
            self.global_estimate[i] = (self.global_estimate[i] * (1.0 - alpha)) + (neighbor_estimate[i] * alpha);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Gossip Ring (ICL Simulation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Simulates the Ring-Pass ICL network
pub struct GossipRing {
    pub nodes: heapless::Vec<GossipNode, 16>, // Max 16 cores for simulation
}

impl GossipRing {
    pub fn new() -> Self {
        Self {
            nodes: heapless::Vec::new(),
        }
    }

    pub fn add_node(&mut self, node: GossipNode) {
        let _ = self.nodes.push(node);
    }

    /// Perform one "tick" of the gossip protocol (one ring pass)
    /// Node i sends to Node i+1 (wrapping around)
    pub fn tick(&mut self) {
        let n = self.nodes.len();
        if n < 2 {
            return;
        }

        // Snapshot current estimates to simulate simultaneous transmission
        let mut estimates = heapless::Vec::<[f64; MAX_DIM], 16>::new();
        for node in &self.nodes {
            let _ = estimates.push(node.global_estimate);
        }

        // Update each node with its predecessor's estimate
        for i in 0..n {
            let neighbor_idx = if i == 0 { n - 1 } else { i - 1 };
            let neighbor_est = estimates[neighbor_idx];
            self.nodes[i].update_consensus(neighbor_est);
        }
    }

    /// Run until convergence or max iterations
    pub fn converge(&mut self, tolerance: f64, max_iters: usize) -> usize {
        for iter in 0..max_iters {
            self.tick();

            if self.is_converged(tolerance) {
                return iter + 1;
            }
        }
        max_iters
    }

    /// Check if all nodes agreed (variance < tolerance)
    pub fn is_converged(&self, tolerance: f64) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        // Calculate global mean of estimates
        let mut sum = [0.0; MAX_DIM];
        for node in &self.nodes {
            for i in 0..MAX_DIM {
                sum[i] += node.global_estimate[i];
            }
        }
        let n = self.nodes.len() as f64;
        let mut mean = [0.0; MAX_DIM];
        for i in 0..MAX_DIM {
            mean[i] = sum[i] / n;
        }

        // Check max deviation from mean
        for node in &self.nodes {
            let mut dist_sq = 0.0;
            for i in 0..MAX_DIM {
                let d = node.global_estimate[i] - mean[i];
                dist_sq += d * d;
            }
            if sqrt(dist_sq) > tolerance {
                return false;
            }
        }

        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gossip_convergence() {
        let mut ring = GossipRing::new();

        // Node 0: Cluster at [0, 0, 0]
        let mut n0 = GossipNode::new(0);
        n0.push_data([0.0, 0.0, 0.0]);
        n0.compute_local_centroid();
        ring.add_node(n0);

        // Node 1: Cluster at [10, 10, 0]
        let mut n1 = GossipNode::new(1);
        n1.push_data([10.0, 10.0, 0.0]);
        n1.compute_local_centroid();
        ring.add_node(n1);

        // Expected global average: [5, 5, 0]
        
        // Run gossip
        let iters = ring.converge(0.1, 100);
        
        println!("Converged in {} iterations", iters);

        // Verify
        for node in &ring.nodes {
            assert!((node.global_estimate[0] - 5.0).abs() < 0.2);
            assert!((node.global_estimate[1] - 5.0).abs() < 0.2);
        }
    }
}
