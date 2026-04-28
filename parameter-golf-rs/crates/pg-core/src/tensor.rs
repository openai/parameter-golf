use smallvec::SmallVec;

use crate::dtype::DType;
use crate::error::{PgError, PgResult};

/// A tensor that can be backed by either GPU memory (CUDA) or CPU memory.
///
/// On macOS (no CUDA), all tensors use CPU storage.
/// On Linux with CUDA, tensors can be on device.
///
/// Supports zero-copy slicing along the first dimension via `offset`.
/// This is critical for the parameter banking pattern where 3D banks
/// like `qo_bank [22, 512, 512]` are sliced per-layer without copying.
#[derive(Clone)]
pub struct GpuTensor {
    /// Raw bytes storage.
    data: TensorStorage,
    /// Shape of this tensor view.
    shape: SmallVec<[usize; 4]>,
    /// Strides in elements (not bytes).
    strides: SmallVec<[usize; 4]>,
    /// Element data type.
    dtype: DType,
    /// Byte offset into `data` for this view.
    offset: usize,
}

#[derive(Clone)]
enum TensorStorage {
    Cpu(std::sync::Arc<Vec<u8>>),
    #[cfg(feature = "cuda")]
    Gpu {
        data: std::sync::Arc<cudarc::driver::CudaSlice<u8>>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    },
}

impl GpuTensor {
    /// Allocate a new zeroed tensor on CPU.
    pub fn zeros_cpu(shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let nbytes = numel * dtype.size_bytes();
        let data = vec![0u8; nbytes];

        Self {
            data: TensorStorage::Cpu(std::sync::Arc::new(data)),
            shape: SmallVec::from_slice(shape),
            strides: Self::contiguous_strides(shape),
            dtype,
            offset: 0,
        }
    }

    /// Create a tensor from host data (CPU).
    pub fn from_host_data(host_data: Vec<u8>, shape: &[usize], dtype: DType) -> PgResult<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.size_bytes();
        if host_data.len() != expected_bytes {
            return Err(PgError::InvalidOp(format!(
                "host data length {} != expected {} for shape {:?} dtype {}",
                host_data.len(),
                expected_bytes,
                shape,
                dtype
            )));
        }
        Ok(Self {
            data: TensorStorage::Cpu(std::sync::Arc::new(host_data)),
            shape: SmallVec::from_slice(shape),
            strides: Self::contiguous_strides(shape),
            dtype,
            offset: 0,
        })
    }

    #[cfg(feature = "cuda")]
    /// Allocate a zeroed tensor on GPU.
    pub fn from_host_data_gpu(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        host_data: &[u8],
        shape: &[usize],
        dtype: DType,
    ) -> PgResult<Self> {
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * dtype.size_bytes();
        if host_data.len() != expected_bytes {
            return Err(PgError::InvalidOp(format!(
                "host data length {} != expected {} for shape {:?}",
                host_data.len(),
                expected_bytes,
                shape
            )));
        }

        stream
            .context()
            .bind_to_thread()
            .map_err(|e| PgError::InvalidOp(format!("CUDA context bind failed: {:?}", e)))?;
        let mut dev_data = stream
            .alloc_zeros::<u8>(expected_bytes)
            .map_err(|e| PgError::InvalidOp(format!("GPU alloc failed: {:?}", e)))?;

        stream.memcpy_htod(host_data, &mut dev_data).map_err(|e| {
            PgError::InvalidOp(format!(
                "GPU memcpy failed for from_host_data_gpu shape {:?} dtype {:?} bytes {}: {:?}",
                shape, dtype, expected_bytes, e
            ))
        })?;

        Ok(Self {
            data: TensorStorage::Gpu {
                data: std::sync::Arc::new(dev_data),
                stream,
            },
            shape: SmallVec::from_slice(shape),
            strides: Self::contiguous_strides(shape),
            dtype,
            offset: 0,
        })
    }
    #[cfg(feature = "cuda")]
    pub fn zeros_gpu(
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        shape: &[usize],
        dtype: DType,
    ) -> PgResult<Self> {
        let numel: usize = shape.iter().product();
        let nbytes = numel * dtype.size_bytes();
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| PgError::InvalidOp(format!("CUDA context bind failed: {:?}", e)))?;
        let data = stream
            .alloc_zeros::<u8>(nbytes)
            .map_err(|e| PgError::InvalidOp(format!("GPU alloc failed: {:?}", e)))?;
        Ok(Self {
            data: TensorStorage::Gpu {
                data: std::sync::Arc::new(data),
                stream,
            },
            shape: SmallVec::from_slice(shape),
            strides: Self::contiguous_strides(shape),
            dtype,
            offset: 0,
        })
    }

    /// Download tensor data to host bytes.
    pub fn to_host_bytes(&self) -> PgResult<Vec<u8>> {
        let nbytes = self.nbytes();
        match &self.data {
            TensorStorage::Cpu(data) => Ok(data[self.offset..self.offset + nbytes].to_vec()),
            #[cfg(feature = "cuda")]
            TensorStorage::Gpu { data, stream } => {
                stream.context().bind_to_thread().map_err(|e| {
                    PgError::InvalidOp(format!("CUDA context bind failed: {:?}", e))
                })?;
                let all_data = stream
                    .memcpy_dtov(data.as_ref())
                    .map_err(|e| PgError::InvalidOp(format!("GPU memcpy failed: {:?}", e)))?;
                Ok(all_data[self.offset..self.offset + nbytes].to_vec())
            }
        }
    }

    /// Overwrite the full contents of a base tensor from host bytes.
    ///
    /// This intentionally rejects sliced views for now. The CUDA training
    /// runtime uses it to refresh persistent device-resident weights between
    /// steps without rebuilding the entire `GpuModel`.
    pub fn copy_from_host_bytes(&mut self, host_data: &[u8]) -> PgResult<()> {
        let expected_bytes = self.nbytes();
        if host_data.len() != expected_bytes {
            return Err(PgError::InvalidOp(format!(
                "host data length {} != expected {} for shape {:?} dtype {}",
                host_data.len(),
                expected_bytes,
                self.shape,
                self.dtype
            )));
        }
        if self.offset != 0 {
            return Err(PgError::InvalidOp(
                "copy_from_host_bytes only supports base tensors with offset=0".into(),
            ));
        }
        match &mut self.data {
            TensorStorage::Cpu(data) => {
                let buf = std::sync::Arc::make_mut(data);
                buf[..expected_bytes].copy_from_slice(host_data);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            TensorStorage::Gpu { data, stream } => {
                stream.context().bind_to_thread().map_err(|e| {
                    PgError::InvalidOp(format!("CUDA context bind failed: {:?}", e))
                })?;
                let dev = std::sync::Arc::get_mut(data).ok_or_else(|| {
                    PgError::InvalidOp(
                        "cannot copy into shared GPU storage; tensor has outstanding aliases"
                            .into(),
                    )
                })?;
                stream
                    .memcpy_htod(host_data, dev)
                    .map_err(|e| {
                        PgError::InvalidOp(format!(
                            "GPU memcpy failed for copy_from_host_bytes shape {:?} dtype {:?} bytes {}: {:?}",
                            self.shape, self.dtype, expected_bytes, e
                        ))
                    })
            }
        }
    }

    /// Set the full base tensor contents to zero.
    ///
    /// CUDA record paths use this for persistent gradient buffers. Sliced views
    /// are intentionally rejected because zeroing a view through the underlying
    /// allocation would otherwise corrupt neighboring slices.
    pub fn zero_bytes(&mut self) -> PgResult<()> {
        if self.offset != 0 {
            return Err(PgError::InvalidOp(
                "zero_bytes only supports base tensors with offset=0".into(),
            ));
        }
        match &mut self.data {
            TensorStorage::Cpu(data) => {
                let buf = std::sync::Arc::make_mut(data);
                buf.fill(0);
                Ok(())
            }
            #[cfg(feature = "cuda")]
            TensorStorage::Gpu { data, stream } => {
                stream.context().bind_to_thread().map_err(|e| {
                    PgError::InvalidOp(format!("CUDA context bind failed: {:?}", e))
                })?;
                let dev = std::sync::Arc::get_mut(data).ok_or_else(|| {
                    PgError::InvalidOp(
                        "cannot zero shared GPU storage; tensor has outstanding aliases".into(),
                    )
                })?;
                stream
                    .memset_zeros(dev)
                    .map_err(|e| PgError::InvalidOp(format!("GPU memset zero failed: {:?}", e)))
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn cu_ptr(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> PgResult<u64> {
        match &self.data {
            TensorStorage::Gpu { data, .. } => {
                use cudarc::driver::DevicePtr;
                let (base, _) = data.device_ptr(stream);
                Ok(base as u64 + self.offset as u64)
            }
            _ => Err(PgError::InvalidOp("Not a GPU tensor".into())),
        }
    }

    /// Zero-copy slice along the first dimension.
    pub fn slice_first(&self, index: usize) -> PgResult<Self> {
        if self.shape.is_empty() {
            return Err(PgError::InvalidOp("cannot slice scalar".into()));
        }
        if index >= self.shape[0] {
            return Err(PgError::InvalidOp(format!(
                "slice index {} >= dim 0 size {}",
                index, self.shape[0]
            )));
        }

        let new_shape: SmallVec<[usize; 4]> = self.shape[1..].into();
        let new_strides: SmallVec<[usize; 4]> = self.strides[1..].into();
        let elem_offset = index * self.strides[0];
        let byte_offset = self.offset + elem_offset * self.dtype.size_bytes();

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
            dtype: self.dtype,
            offset: byte_offset,
        })
    }

    /// Slice a range along the first dimension.
    pub fn slice_range(&self, start: usize, end: usize) -> PgResult<Self> {
        if self.shape.is_empty() {
            return Err(PgError::InvalidOp("cannot slice scalar".into()));
        }
        if end > self.shape[0] || start >= end {
            return Err(PgError::InvalidOp(format!(
                "invalid range [{}..{}) for dim 0 size {}",
                start, end, self.shape[0]
            )));
        }

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;
        let byte_offset = self.offset + start * self.strides[0] * self.dtype.size_bytes();

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            dtype: self.dtype,
            offset: byte_offset,
        })
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total bytes of data for this view.
    pub fn nbytes(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns true when two tensor views overlap the same backing allocation.
    ///
    /// This is intentionally byte-range based instead of shape based. CUDA
    /// backward code uses it as a guard against accidentally using the same
    /// scratch allocation as both a read input and write output in ping-pong
    /// schedules.
    pub fn overlaps_storage_region(&self, other: &Self) -> bool {
        let same_storage = match (&self.data, &other.data) {
            (TensorStorage::Cpu(lhs), TensorStorage::Cpu(rhs)) => std::sync::Arc::ptr_eq(lhs, rhs),
            #[cfg(feature = "cuda")]
            (TensorStorage::Gpu { data: lhs, .. }, TensorStorage::Gpu { data: rhs, .. }) => {
                std::sync::Arc::ptr_eq(lhs, rhs)
            }
            #[cfg(feature = "cuda")]
            _ => false,
        };
        if !same_storage {
            return false;
        }
        let lhs_start = self.offset;
        let lhs_end = lhs_start + self.nbytes();
        let rhs_start = other.offset;
        let rhs_end = rhs_start + other.nbytes();
        lhs_start < rhs_end && rhs_start < lhs_end
    }

    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride = 1;
        for i in (0..self.ndim()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i];
        }
        true
    }

    /// Reshape (only valid for contiguous tensors with matching numel).
    pub fn reshape(&self, new_shape: &[usize]) -> PgResult<Self> {
        if !self.is_contiguous() {
            return Err(PgError::InvalidOp(
                "cannot reshape non-contiguous tensor".into(),
            ));
        }
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(PgError::ShapeMismatch {
                expected: new_shape.to_vec(),
                got: self.shape.to_vec(),
            });
        }
        Ok(Self {
            data: self.data.clone(),
            shape: SmallVec::from_slice(new_shape),
            strides: Self::contiguous_strides(new_shape),
            dtype: self.dtype,
            offset: self.offset,
        })
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self.data, TensorStorage::Cpu(_))
    }

    fn contiguous_strides(shape: &[usize]) -> SmallVec<[usize; 4]> {
        let ndim = shape.len();
        let mut strides = SmallVec::from_elem(0, ndim);
        if ndim > 0 {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }
}

impl std::fmt::Debug for GpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape.as_slice())
            .field("dtype", &self.dtype)
            .field("offset", &self.offset)
            .field("contiguous", &self.is_contiguous())
            .field("cpu", &self.is_cpu())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_cpu() {
        let t = GpuTensor::zeros_cpu(&[3, 4, 5], DType::BF16);
        assert_eq!(t.shape(), &[3, 4, 5]);
        assert_eq!(t.numel(), 60);
        assert_eq!(t.nbytes(), 120); // 60 * 2 bytes
        assert!(t.is_contiguous());
        assert!(t.is_cpu());
    }

    #[test]
    fn test_slice_first() {
        // Simulate a parameter bank [3, 4, 5] in bf16
        let t = GpuTensor::zeros_cpu(&[3, 4, 5], DType::BF16);

        let s = t.slice_first(1).unwrap();
        assert_eq!(s.shape(), &[4, 5]);
        assert_eq!(s.offset(), 1 * 20 * 2); // stride[0]=20, bf16=2 bytes
        assert_eq!(s.numel(), 20);
    }

    #[test]
    fn test_slice_range() {
        let t = GpuTensor::zeros_cpu(&[10, 5], DType::F32);
        let s = t.slice_range(2, 7).unwrap();
        assert_eq!(s.shape(), &[5, 5]);
        assert_eq!(s.offset(), 2 * 5 * 4); // stride[0]=5, f32=4 bytes
    }

    #[test]
    fn test_storage_overlap_detection() {
        let t = GpuTensor::zeros_cpu(&[10, 5], DType::F32);
        let lhs = t.slice_range(2, 7).unwrap();
        let rhs = t.slice_range(6, 9).unwrap();
        let disjoint = t.slice_range(7, 10).unwrap();
        let other = GpuTensor::zeros_cpu(&[5, 5], DType::F32);

        assert!(lhs.overlaps_storage_region(&rhs));
        assert!(!lhs.overlaps_storage_region(&disjoint));
        assert!(!lhs.overlaps_storage_region(&other));
    }

    #[test]
    fn test_reshape() {
        let t = GpuTensor::zeros_cpu(&[3, 4, 5], DType::BF16);
        let r = t.reshape(&[12, 5]).unwrap();
        assert_eq!(r.shape(), &[12, 5]);
        assert_eq!(r.numel(), 60);
    }

    #[test]
    fn test_bank_slicing() {
        // Test the exact parameter banking pattern:
        // qo_bank [22, 512, 512] → slice to get layer i's Q weight [512, 512]
        let bank = GpuTensor::zeros_cpu(&[22, 512, 512], DType::BF16);
        assert_eq!(bank.numel(), 22 * 512 * 512);

        // Get Q weight for layer 5
        let q_weight = bank.slice_first(5).unwrap();
        assert_eq!(q_weight.shape(), &[512, 512]);

        // Get O weight for layer 5 (stored at index 11 + 5 = 16)
        let o_weight = bank.slice_first(16).unwrap();
        assert_eq!(o_weight.shape(), &[512, 512]);
    }

    #[test]
    fn test_copy_from_host_bytes_cpu() {
        let mut t = GpuTensor::zeros_cpu(&[4], DType::I8);
        t.copy_from_host_bytes(&[4, 3, 2, 1]).unwrap();
        assert_eq!(t.to_host_bytes().unwrap(), vec![4, 3, 2, 1]);
    }
}
