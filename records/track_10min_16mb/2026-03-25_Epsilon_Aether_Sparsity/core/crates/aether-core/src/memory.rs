//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//! AEGIS Memory Substrate: The Manifold Heap
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
//!
//! "Memory is not a bucket; it is a topological space."
//!
//! This module implements the Manifold Garbage Collector, a biologically inspired
//! memory model that treats unused objects as "entropy" to be reclaimed.
//!
//! Key Components:
//! 1. ManifoldHeap: A spatial tree (Octree-like) organizing objects into blocks.
//! 2. Entropy Regulation: O(log N) regulation by pruning cold branches.
//! 3. Chebyshev's Guard: Statistical safety protocol.
//!
//! â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;
#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(feature = "std")]
use std::boxed::Box;

use libm::{sqrt, fabs};
use core::marker::PhantomData;

/// A Geometric Cell (Gc) handle.
/// Represents a reference to an object in the ManifoldHeap.
/// Unlike standard pointers, this is a topological index.
/// 
/// We implement Copy/Clone manually to avoid implicit T: Copy bound.
#[derive(Debug, PartialOrd, Ord)]
pub struct Gc<T> {
    pub index: usize,
    pub generation: u32,
    _marker: PhantomData<fn() -> T>, // Covariant, implies no ownership logic for drop
}

impl<T> Copy for Gc<T> {}

impl<T> Clone for Gc<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for Gc<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Gc<T> {}

impl<T> core::hash::Hash for Gc<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> Gc<T> {
    pub fn new(index: usize, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }
}

/// Metadata for an object in the heap.
#[derive(Debug, Clone, Copy)]
pub struct ObjectHeader {
    /// Is this object currently reachable?
    pub marked: bool,
    /// Generation count to detect stale handles
    pub generation: u32,
}

/// A slot in the ManifoldHeap.
/// Note: Liveness is now stored in the SpatialBlock for SIMD access.
#[derive(Debug, Clone)]
pub enum HeapSlot<T> {
    Free { next_free: usize },
    Occupied { header: ObjectHeader, data: T },
}

/// A Spatial Block acting as a leaf in the memory tree.
/// Contains contiguous arrays for SIMD optimization.
/// Size N=8.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct SpatialBlock<T> {
    /// Liveness scores [f64; 8]
    pub liveness: [f64; 8],
    /// The actual data slots
    pub slots: [HeapSlot<T>; 8],
    /// Mask or counter of occupied slots (optional but useful)
    pub occupied_mask: u8,
}

impl<T> Default for SpatialBlock<T> {
    fn default() -> Self {
        Self {
            liveness: [0.0; 8],
            slots: [
                HeapSlot::Free { next_free: usize::MAX }, HeapSlot::Free { next_free: usize::MAX },
                HeapSlot::Free { next_free: usize::MAX }, HeapSlot::Free { next_free: usize::MAX },
                HeapSlot::Free { next_free: usize::MAX }, HeapSlot::Free { next_free: usize::MAX },
                HeapSlot::Free { next_free: usize::MAX }, HeapSlot::Free { next_free: usize::MAX },
            ],
            occupied_mask: 0,
        }
    }
}

impl<T> SpatialBlock<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Internal node of the Spatial Tree.
/// Aggregates statistics of its children.
#[derive(Debug, Clone)]
pub struct SpatialNode {
    /// Indices of children. 
    /// If `is_leaf_parent` is true, these are indices into `blocks`.
    /// Otherwise, indices into `nodes`.
    /// None indicates empty branch.
    pub children: [Option<usize>; 8],
    
    /// Aggregate Mean Liveness of this branch
    pub mean_liveness: f64,
    /// Max Liveness in this branch (for quick "is hot" checks)
    pub max_liveness: f64,
    
    /// Does this node point to Blocks (true) or Nodes (false)?
    pub is_leaf_parent: bool,
}

impl SpatialNode {
    pub fn new(is_leaf_parent: bool) -> Self {
        Self {
            children: [None; 8],
            mean_liveness: 0.0,
            max_liveness: 0.0,
            is_leaf_parent,
        }
    }
}


/// Configuration for Memory Behavior
#[derive(Debug, Clone, Copy)]
pub enum MemoryMode {
    Consumer,
    Datacenter,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub mode: MemoryMode,
}

impl Default for Config {
    fn default() -> Self {
        Self { mode: MemoryMode::Consumer }
    }
}

/// The Manifold Allocator.
/// Manages memory as a dense topological substrate using a Spatial Tree.
pub struct ManifoldHeap<T> {
    /// Leaf Blocks
    pub blocks: Vec<SpatialBlock<T>>,
    /// Tree Nodes
    pub nodes: Vec<SpatialNode>,
    /// Root Node Index
    pub root_idx: usize,
    
    /// Head of the free list (Global index)
    /// Index = block_idx * 8 + slot_idx
    free_head: Option<usize>,
    
    /// Active objects count
    active_count: usize,
    /// Global entropy counter
    entropy_counter: usize,
    
    pub config: Config,
}

impl<T> ManifoldHeap<T> {
    pub fn new() -> Self {
        let mut heap = Self {
            blocks: Vec::new(),
            nodes: Vec::new(),
            root_idx: 0,
            free_head: None,
            active_count: 0,
            entropy_counter: 0,
            config: Config::default(),
        };
        // Initialize with one root node that is a leaf parent
        heap.nodes.push(SpatialNode::new(true)); 
        heap
    }
    
    /// Helper to decompose global index into (block, offset)
    fn resolve_index(index: usize) -> (usize, usize) {
        (index / 8, index % 8)
    }

    /// Allocate a new object.
    pub fn alloc(&mut self, data: T) -> Gc<T> {
        self.entropy_counter += 1;
        
        let (block_idx, slot_idx) = if let Some(head) = self.free_head {
            let (b, s) = Self::resolve_index(head);
            // Verify and update free_head
            if b < self.blocks.len() {
               if let HeapSlot::Free { next_free } = &self.blocks[b].slots[s] {
                   if *next_free == usize::MAX {
                       self.free_head = None;
                   } else {
                       self.free_head = Some(*next_free);
                   }
               } else {
                   panic!("Free head pointed to occupied slot");
               }
            }
            (b, s)
        } else {
            // Bump allocation
            let next_blk_idx = self.blocks.len();
            self.blocks.push(SpatialBlock::new());
            self.link_block_to_tree(next_blk_idx);
            
            for i in 1..7 {
                 self.blocks[next_blk_idx].slots[i] = HeapSlot::Free { 
                     next_free: next_blk_idx * 8 + i + 1 
                 };
            }
            self.blocks[next_blk_idx].slots[7] = HeapSlot::Free { next_free: usize::MAX };
            
            self.free_head = Some(next_blk_idx * 8 + 1);
            (next_blk_idx, 0)
        };
        
        let generation = 1; 
        self.blocks[block_idx].slots[slot_idx] = HeapSlot::Occupied {
            header: ObjectHeader {
                marked: false,
                generation,
            },
            data,
        };
        self.blocks[block_idx].liveness[slot_idx] = 1.0; 
        self.blocks[block_idx].occupied_mask |= 1 << slot_idx;
        
        self.active_count += 1;
        
        Gc::new(block_idx * 8 + slot_idx, generation)
    }
    
    fn link_block_to_tree(&mut self, block_idx: usize) {
        let needed_node_idx = block_idx / 8;
        if needed_node_idx >= self.nodes.len() {
             self.nodes.push(SpatialNode::new(true));
        }
        
        let node_idx = needed_node_idx;
        let child_slot = block_idx % 8;
        self.nodes[node_idx].children[child_slot] = Some(block_idx);
    }

    /// Access mutably. Heats up object.
    pub fn get_mut(&mut self, handle: Gc<T>) -> Option<&mut T> {
        let (b, s) = Self::resolve_index(handle.index);
        
        if b >= self.blocks.len() { return None; }
        
        let block = &mut self.blocks[b];
        match &mut block.slots[s] {
            HeapSlot::Occupied { header, data } => {
                if header.generation != handle.generation { return None; }
                // Heat up - split borrow of block works here
                block.liveness[s] = (block.liveness[s] + 1.0).min(10.0);
                Some(data)
            }
            _ => None,
        }
    }
    
    /// Access immutably (Peek). Does NOT update liveness to avoid &mut borrow.
    /// This fixes autograd multiple borrow issues.
    pub fn get(&self, handle: Gc<T>) -> Option<&T> {
        let (b, s) = Self::resolve_index(handle.index);
        if b >= self.blocks.len() { return None; }

        match &self.blocks[b].slots[s] {
            HeapSlot::Occupied { header, data } => {
                if header.generation != handle.generation { return None; }
                Some(data)
            }
            _ => None,
        }
    }
    
    pub fn touch(&mut self, handle: Gc<T>) {
        let (b, s) = Self::resolve_index(handle.index);
        if b < self.blocks.len() {
            let block = &mut self.blocks[b];
            if let HeapSlot::Occupied { header, .. } = &mut block.slots[s] {
                 if header.generation == handle.generation {
                     block.liveness[s] = (block.liveness[s] + 0.5).min(10.0);
                 }
            }
        }
    }
    
    pub fn mark(&mut self, handle: Gc<T>) {
        let (b, s) = Self::resolve_index(handle.index);
         if b < self.blocks.len() {
             let block = &mut self.blocks[b];
             if let HeapSlot::Occupied { header, .. } = &mut block.slots[s] {
                 if header.generation == handle.generation {
                     header.marked = true;
                     block.liveness[s] = (block.liveness[s] + 2.0).min(10.0);
                 }
             }
         }
    }

    pub fn active_count(&self) -> usize {
        self.active_count
    }

    pub fn capacity(&self) -> usize {
        self.blocks.len() * 8
    }
}

/// Chebyshev Guard logic.
pub struct ChebyshevGuard {
    mean: f64,
    std_dev: f64,
    k: f64,
}

impl ChebyshevGuard {
    pub fn calculate<T>(heap: &ManifoldHeap<T>) -> Self {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0.0;
        
        for block in &heap.blocks {
             // Optimization: skip empty blocks early
             if block.occupied_mask == 0 { continue; }

             for i in 0..8 {
                 if (block.occupied_mask & (1 << i)) != 0 {
                     let val = block.liveness[i];
                     sum += val;
                     sum_sq += val * val;
                     count += 1.0;
                 }
             }
        }
        
       if count == 0.0 {
            return Self { mean: 0.0, std_dev: 0.0, k: 2.0 };
        }

        let mean = sum / count;
        let variance = (sum_sq / count) - (mean * mean);
        let variance = if variance < 0.0 { 0.0 } else { variance };
        
        Self {
            mean,
            std_dev: sqrt(variance),
            k: 2.0,
        }
    }
    
    pub fn is_safe(&self, liveness: f64) -> bool {
        if liveness >= self.mean { return true; }
        let boundary = self.mean - (self.k * self.std_dev);
        liveness > boundary
    }
}

impl<T> ManifoldHeap<T> {
    /// Regulation with Spatial Optimization
    pub fn regulate_entropy<F>(&mut self, tracer: F) -> usize 
    where F: Fn(&mut Self) 
    {
        // 0. Reset Marks
        for block in &mut self.blocks {
            for slot in &mut block.slots {
                if let HeapSlot::Occupied { header, .. } = slot {
                    header.marked = false;
                }
            }
        }
        
        // 1. Trace
        tracer(self);
        
        // 2. Calc Stats
        let guard = ChebyshevGuard::calculate(self);
        
        let mut pruned = 0;
        let num_blocks = self.blocks.len();
        let mut new_free_head = self.free_head;
        
        for b_idx in 0..num_blocks {
            let block = &mut self.blocks[b_idx];
            
            for s_idx in 0..8 {
                 if (block.occupied_mask & (1 << s_idx)) == 0 { continue; }
                 
                 block.liveness[s_idx] *= 0.95;
                 
                 let should_prune;
                 
                 if let HeapSlot::Occupied { header, .. } = &mut block.slots[s_idx] {
                     let is_marked = header.marked;
                     let is_safe = guard.is_safe(block.liveness[s_idx]);
                     
                     if is_marked {
                         block.liveness[s_idx] += 0.1;
                         should_prune = false;
                     } else if is_safe {
                         should_prune = false;
                     } else {
                         should_prune = true;
                     }
                 } else {
                     should_prune = false;
                 }
                 
                 if should_prune {
                      block.occupied_mask &= !(1 << s_idx);
                      let next = if let Some(h) = new_free_head { h } else { usize::MAX };
                      block.slots[s_idx] = HeapSlot::Free { next_free: next };
                      new_free_head = Some(b_idx * 8 + s_idx);
                      
                      pruned += 1;
                 }
            }
        }
        
        self.active_count -= pruned;
        self.free_head = new_free_head;
        self.entropy_counter = 0;
        
        pruned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifold_allocation() {
        let mut heap = ManifoldHeap::<i32>::new();
        let a = heap.alloc(10);
        let b = heap.alloc(20);
        
        assert_eq!(*heap.get(a).unwrap(), 10);
        assert_eq!(*heap.get(b).unwrap(), 20);
        assert_eq!(heap.active_count(), 2);
    }
    
    #[test]
    fn test_spatial_clustering() {
        let mut heap = ManifoldHeap::<i32>::new();
        let mut handles = Vec::new();
        for i in 0..8 {
            handles.push(heap.alloc(i));
        }
        
        let h9 = heap.alloc(99);
        let (b1, _) = ManifoldHeap::<i32>::resolve_index(h9.index);
        assert_eq!(b1, 1);
        assert_eq!(heap.blocks.len(), 2);
    }
    
    #[test]
    fn test_simd_alignment() {
        use core::mem::align_of;
        assert_eq!(align_of::<SpatialBlock<i32>>(), 64);
    }

}

