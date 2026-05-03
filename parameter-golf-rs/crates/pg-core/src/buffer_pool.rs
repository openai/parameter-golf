use std::collections::HashMap;

use crate::dtype::DType;
use crate::tensor::GpuTensor;

/// Pre-allocated arena for all training buffers.
///
/// Eliminates runtime CUDA malloc during training. All buffers are allocated
/// at init based on the model config and batch size.
pub struct BufferPool {
    named_buffers: HashMap<String, GpuTensor>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            named_buffers: HashMap::new(),
        }
    }

    /// Pre-allocate a named CPU buffer.
    pub fn alloc_cpu(&mut self, name: &str, shape: &[usize], dtype: DType) -> &GpuTensor {
        let tensor = GpuTensor::zeros_cpu(shape, dtype);
        self.named_buffers.insert(name.to_string(), tensor);
        &self.named_buffers[name]
    }

    pub fn get(&self, name: &str) -> Option<&GpuTensor> {
        self.named_buffers.get(name)
    }

    pub fn get_mut(&mut self, name: &str) -> Option<&mut GpuTensor> {
        self.named_buffers.get_mut(name)
    }

    pub fn total_bytes(&self) -> usize {
        self.named_buffers.values().map(|t| t.nbytes()).sum()
    }

    pub fn count(&self) -> usize {
        self.named_buffers.len()
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}
