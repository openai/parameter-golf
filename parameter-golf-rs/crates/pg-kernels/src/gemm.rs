use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig, Gemm, StridedBatchedConfig};
use cudarc::cublas::sys::{cublasOperation_t, cudaDataType_t, cublasComputeType_t, cublasGemmAlgo_t};
use cudarc::driver::{CudaStream, CudaSlice, DevicePtr, DevicePtrMut};
use half::bf16;

use pg_core::error::{PgError, PgResult};

pub struct GemmEngine {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
}

impl GemmEngine {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| PgError::CuBlas(e.to_string()))?;
        Ok(Self { blas, stream })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub unsafe fn matmul_f32(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| PgError::CuBlas(format!("gemm_ex failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn batched_matmul_f32(
        &self,
        a: u64,
        b: u64,
        c: u64,
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        let stride_a = (m * k) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_b,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                stride_a,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| PgError::CuBlas(format!("gemm_strided_batched_ex failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_f32_bt(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            ).map_err(|e| PgError::CuBlas(format!("gemm_ex failed: {:?}", e)))?;
        }
        Ok(())
    }
}
