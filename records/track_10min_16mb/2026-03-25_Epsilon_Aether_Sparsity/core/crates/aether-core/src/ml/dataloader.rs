//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Data Loaders
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! Efficient data loading with batching and shuffling.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec as StdVec;

use super::tensor::Tensor;

/// Data Loader
#[derive(Debug, Clone)]
pub struct DataLoader {
    pub features: Vec<Tensor>,
    pub targets: Vec<Tensor>,
    pub batch_size: usize,
    pub shuffle: bool,
}

impl DataLoader {
    /// Create new DataLoader
    pub fn new(features: Vec<Tensor>, targets: Vec<Tensor>, batch_size: usize, shuffle: bool) -> Self {
        assert_eq!(features.len(), targets.len(), "Features and targets must have same length");
        Self {
            features,
            targets,
            batch_size,
            shuffle,
        }
    }

    /// Convert raw slices to DataLoader
    pub fn from_slice(x: &[Tensor], y: &[Tensor], batch_size: usize, shuffle: bool) -> Self {
        Self {
            features: x.to_vec(),
            targets: y.to_vec(),
            batch_size,
            shuffle,
        }
    }

    /// Iterate over batches
    pub fn iter(&self) -> BatchIterator {
        let n = self.features.len();
        let mut indices: Vec<usize> = (0..n).collect();
        
        if self.shuffle {
            // Simple Linear Congruential Generator for no_std compatibility
            // X_{n+1} = (aX_n + c) % m
            let mut rng = 42u64; // Should ideally take a seed
             for i in (1..n).rev() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let j = (rng as usize) % (i + 1);
                indices.swap(i, j);
            }
        }
        
        BatchIterator {
            loader: self,
            indices,
            current_idx: 0,
        }
    }
}

/// Iterator over batches
pub struct BatchIterator<'a> {
    loader: &'a DataLoader,
    indices: Vec<usize>,
    current_idx: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Vec<Tensor>, Vec<Tensor>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.loader.features.len() {
            return None;
        }

        let start = self.current_idx;
        let end = (start + self.loader.batch_size).min(self.loader.features.len());
        self.current_idx = end;

        let mut batch_x = Vec::with_capacity(end - start);
        let mut batch_y = Vec::with_capacity(end - start);

        for i in start..end {
            let idx = self.indices[i];
            batch_x.push(self.loader.features[idx].clone());
            batch_y.push(self.loader.targets[idx].clone());
        }

        Some((batch_x, batch_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader() {
        let x = vec![
            Tensor::zeros(&[1]), Tensor::zeros(&[1]),
            Tensor::zeros(&[1]), Tensor::zeros(&[1]),
            Tensor::zeros(&[1])
        ];
        let y = x.clone();
        
        let loader = DataLoader::new(x, y, 2, false);
        let mut iter = loader.iter();
        
        let batch1 = iter.next();
        assert!(batch1.is_some());
        assert_eq!(batch1.unwrap().0.len(), 2);
        
        let batch2 = iter.next();
        assert!(batch2.is_some());
        assert_eq!(batch2.unwrap().0.len(), 2);
        
        let batch3 = iter.next(); // Last batch of 1
        assert!(batch3.is_some());
        assert_eq!(batch3.unwrap().0.len(), 1);
        
        assert!(iter.next().is_none());
    }
}
