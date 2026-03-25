//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Tensor Engine
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! N-dimensional tensor implementation with shared storage and strided access.
//! Optimized for CPU, with future hooks for wgpu.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "alloc")]
use alloc::rc::Rc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(not(feature = "alloc"))]
use std::rc::Rc;
#[cfg(not(feature = "alloc"))]
use std::vec::Vec;
#[cfg(not(feature = "alloc"))]
use std::vec;

use core::cell::RefCell;
use libm::{exp, sqrt};

/// AEGIS Tensor: N-dimensional array
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// Shared data storage (flat generic buffer)
    pub data: Rc<RefCell<Vec<f64>>>,
    /// Shape of the tensor dimensions
    pub shape: Vec<usize>,
    /// Strides for traversing the data
    pub strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor from a raw vector (consuming it) and shape
    pub fn from_vec(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape product");

        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self {
            data: Rc::new(RefCell::new(data)),
            shape,
            strides,
        }
    }

    /// Create a new tensor from a slice and shape
    pub fn new(data: &[f64], shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        assert_eq!(data.len(), total_size, "Data length must match shape product");

        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        Self {
            data: Rc::new(RefCell::new(data.to_vec())),
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![0.0; total_size];
        Self::new(&data, shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        let data = vec![1.0; total_size];
        Self::new(&data, shape)
    }

    /// Create a tensor with Xavier initialization
    pub fn kaiming_uniform(shape: &[usize]) -> Self {
        let total_size: usize = shape.iter().product();
        let fan_in = if shape.len() > 1 { shape[1] } else { 1 };
        let bound = sqrt(3.0 / fan_in as f64);
        
        // Simple LCG for deterministic randomness in no_std
        let mut rng = 42u64;
        let mut data: Vec<f64> = Vec::with_capacity(total_size);
        
        for _ in 0..total_size {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
            data.push(r * bound);
        }

        Self::new(&data, shape)
    }

    /// Get value at index (handles strides)
    pub fn get(&self, indices: &[usize]) -> f64 {
        let offset = self.compute_offset(indices);
        self.data.borrow()[offset]
    }

    /// Set value at index
    pub fn set(&self, indices: &[usize], value: f64) {
        let offset = self.compute_offset(indices);
        self.data.borrow_mut()[offset] = value;
    }

    fn compute_offset(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.shape.len(), "Index rank mismatch");
        let mut offset = 0;
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < self.shape[i], "Index out of bounds");
            offset += idx * self.strides[i];
        }
        offset
    }

    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Self {
        let total_size: usize = self.shape.iter().product();
        let mut new_data = Vec::with_capacity(total_size);
        let data = self.data.borrow();
        
        new_data.extend_from_slice(&*data);
        
        Self {
            data: Rc::new(RefCell::new(new_data)),
            shape: vec![total_size],
            strides: vec![1],
        }
    }

    /// Matrix multiplication (2D only for now)
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Matmul requires 2D tensors");
        assert_eq!(other.shape.len(), 2, "Matmul requires 2D tensors");
        assert_eq!(self.shape[1], other.shape[0], "Dimension mismatch for matmul");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = Tensor::zeros(&[m, n]);
        let data_a = self.data.borrow();
        let data_b = other.data.borrow();
        let mut data_c = result.data.borrow_mut();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let val_a = data_a[i * self.strides[0] + l * self.strides[1]];
                    let val_b = data_b[l * other.strides[0] + j * other.strides[1]];
                    sum += val_a * val_b;
                }
                data_c[i * result.strides[0] + j * result.strides[1]] = sum;
            }
        }
        
        drop(data_c);
        result
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for add");
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        
        let data_a = self.data.borrow();
        let data_b = other.data.borrow();

        for i in 0..total_size {
            result_data.push(data_a[i] + data_b[i]);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for mul");
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        
        let data_a = self.data.borrow();
        let data_b = other.data.borrow();

        for i in 0..total_size {
            result_data.push(data_a[i] * data_b[i]);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Scalar multiplication
    pub fn scale(&self, s: f64) -> Tensor {
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        let data = self.data.borrow();

        for i in 0..total_size {
            result_data.push(data[i] * s);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Transpose (2D)
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose support 2D only for now");
        let rows = self.shape[0];
        let cols = self.shape[1];
        
        let mut result = Tensor::zeros(&[cols, rows]);
        let data = self.data.borrow();
        let mut res_data = result.data.borrow_mut();

        for i in 0..rows {
            for j in 0..cols {
                res_data[j * result.strides[0] + i * result.strides[1]] = 
                    data[i * self.strides[0] + j * self.strides[1]];
            }
        }
        
        drop(res_data);
        result
    }

    /// Sum all elements
    pub fn sum(&self) -> f64 {
        self.data.borrow().iter().sum()
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch for sub");
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        
        let data_a = self.data.borrow();
        let data_b = other.data.borrow();

        for i in 0..total_size {
            result_data.push(data_a[i] - data_b[i]);
        }

        Self::new(&result_data, &self.shape)
    }

    /// Element-wise mapping
    pub fn map<F>(&self, f: F) -> Self 
    where F: Fn(f64) -> f64 {
        let total_size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(total_size);
        let data = self.data.borrow();
        
        for i in 0..total_size {
            result_data.push(f(data[i]));
        }
        
        Self::new(&result_data, &self.shape)
    }
}
