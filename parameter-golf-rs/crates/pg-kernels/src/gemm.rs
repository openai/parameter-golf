use std::sync::Arc;

use cudarc::cublas::{Cublas, CublasError};
use cudarc::driver::CudaDevice;

use pg_core::error::{PgError, PgResult};
use pg_core::tensor::GpuTensor;

/// cuBLAS GEMM wrapper for the competition's critical matrix shapes.
///
/// All GEMM in the model:
/// - Attention Q/K/V projection: [B*T, 512] × [512, 512] or [512, 256]
/// - Attention output:           [B*T, 512] × [512, 512]
/// - MLP up:                     [B*T, 512] × [1536, 512]^T
/// - MLP down:                   [B*T, 1536] × [512, 1536]^T
///
/// Newton-Schulz uses strided batched GEMM:
/// - X @ X^T:  [B, M, N] × [B, N, M] → [B, M, M]
/// - B @ X:    [B, M, M] × [B, M, N] → [B, M, N]
pub struct GemmEngine {
    cublas: Arc<Cublas>,
    device: Arc<CudaDevice>,
}

impl GemmEngine {
    pub fn new(device: Arc<CudaDevice>) -> PgResult<Self> {
        let cublas = Cublas::new(device.clone()).map_err(|e| PgError::CuBlas(e.to_string()))?;
        Ok(Self {
            cublas: Arc::new(cublas),
            device,
        })
    }

    /// C = alpha * A @ B + beta * C
    /// A: [M, K], B: [K, N], C: [M, N]
    /// All in bf16 with f32 accumulation.
    ///
    /// Note: cuBLAS uses column-major, so we compute C^T = B^T @ A^T
    /// to stay in row-major without explicit transposes.
    pub fn matmul(
        &self,
        _a: &GpuTensor,
        _b: &GpuTensor,
        _c: &mut GpuTensor,
    ) -> PgResult<()> {
        // TODO: Implement when CUDA device is available
        // Will use cublas.gemm() with bf16 inputs and f32 compute
        Ok(())
    }

    /// Strided batched GEMM for Newton-Schulz.
    /// A: [B, M, K], B: [B, K, N], C: [B, M, N]
    pub fn batched_matmul(
        &self,
        _a: &GpuTensor,
        _b: &GpuTensor,
        _c: &mut GpuTensor,
    ) -> PgResult<()> {
        // TODO: Implement with cublas.gemm_strided_batched()
        Ok(())
    }
}
