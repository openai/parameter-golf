use std::sync::{Arc, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
};
use cudarc::driver::CudaStream;

use pg_core::error::{PgError, PgResult};

pub struct GemmEngine {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
}

pub fn fast_tf32_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PG_CUBLAS_FAST_TF32")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub fn f32_compute_mode_label() -> &'static str {
    if fast_tf32_enabled() {
        if force_tensor_op_algo_enabled() {
            "fast_tf32_tensor_op"
        } else {
            "fast_tf32"
        }
    } else {
        "pedantic_f32"
    }
}

fn f32_compute_type() -> cublasComputeType_t {
    if fast_tf32_enabled() {
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        cublasComputeType_t::CUBLAS_COMPUTE_32F
    }
}

fn f32_gemm_algo() -> cublasGemmAlgo_t {
    if fast_tf32_enabled() && force_tensor_op_algo_enabled() {
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
    } else {
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT
    }
}

fn bf16_compute_type() -> cublasComputeType_t {
    if bf16_pedantic_enabled() {
        cublasComputeType_t::CUBLAS_COMPUTE_32F
    } else {
        cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
    }
}

pub fn bf16_compute_mode_label() -> &'static str {
    if bf16_pedantic_enabled() {
        "bf16_pedantic_32f"
    } else if let Ok(algo) = std::env::var("PG_CUBLAS_BF16_ALGO") {
        match algo.to_ascii_lowercase().as_str() {
            "0" | "algo0" => "fast_16bf_tensor_op_algo0",
            "1" | "algo1" => "fast_16bf_tensor_op_algo1",
            "2" | "algo2" => "fast_16bf_tensor_op_algo2",
            "3" | "algo3" => "fast_16bf_tensor_op_algo3",
            "4" | "algo4" => "fast_16bf_tensor_op_algo4",
            "5" | "algo5" => "fast_16bf_tensor_op_algo5",
            "6" | "algo6" => "fast_16bf_tensor_op_algo6",
            "7" | "algo7" => "fast_16bf_tensor_op_algo7",
            "8" | "algo8" => "fast_16bf_tensor_op_algo8",
            "9" | "algo9" => "fast_16bf_tensor_op_algo9",
            "10" | "algo10" => "fast_16bf_tensor_op_algo10",
            "11" | "algo11" => "fast_16bf_tensor_op_algo11",
            "12" | "algo12" => "fast_16bf_tensor_op_algo12",
            "13" | "algo13" => "fast_16bf_tensor_op_algo13",
            "14" | "algo14" => "fast_16bf_tensor_op_algo14",
            "15" | "algo15" => "fast_16bf_tensor_op_algo15",
            _ => "fast_16bf_tensor_op",
        }
    } else {
        "fast_16bf_tensor_op"
    }
}

fn bf16_gemm_algo() -> cublasGemmAlgo_t {
    static ALGO: OnceLock<cublasGemmAlgo_t> = OnceLock::new();
    *ALGO.get_or_init(|| {
        match std::env::var("PG_CUBLAS_BF16_ALGO")
            .unwrap_or_else(|_| "default".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "0" | "algo0" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO0_TENSOR_OP,
            "1" | "algo1" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO1_TENSOR_OP,
            "2" | "algo2" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO2_TENSOR_OP,
            "3" | "algo3" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO3_TENSOR_OP,
            "4" | "algo4" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO4_TENSOR_OP,
            "5" | "algo5" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO5_TENSOR_OP,
            "6" | "algo6" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO6_TENSOR_OP,
            "7" | "algo7" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO7_TENSOR_OP,
            "8" | "algo8" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO8_TENSOR_OP,
            "9" | "algo9" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO9_TENSOR_OP,
            "10" | "algo10" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO10_TENSOR_OP,
            "11" | "algo11" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO11_TENSOR_OP,
            "12" | "algo12" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO12_TENSOR_OP,
            "13" | "algo13" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO13_TENSOR_OP,
            "14" | "algo14" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO14_TENSOR_OP,
            "15" | "algo15" => cublasGemmAlgo_t::CUBLAS_GEMM_ALGO15_TENSOR_OP,
            _ => cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        }
    })
}

fn bf16_pedantic_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PG_CUBLAS_BF16_PEDANTIC")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn force_tensor_op_algo_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PG_CUBLAS_FORCE_TENSOR_OP_ALGO")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

impl GemmEngine {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        let blas = CudaBlas::new(stream.clone()).map_err(|e| PgError::CuBlas(e.to_string()))?;
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
        // Row-major linear projection:
        //   a = X [m, k]
        //   b = W [n, k]
        //   c = Y [m, n]
        // with CPU semantics Y = X @ W^T.
        //
        // cuBLAS is column-major, so we compute:
        //   Y^T [n, m] = W [n, k] @ X^T [k, m]
        // by interpreting the row-major buffers as transposed column-major views.
        unsafe { self.matmul_f32_bt(a, b, c, m, n, k, alpha, beta) }
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
                f32_compute_type(),
                f32_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_strided_batched_ex failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn batched_matmul_f32_bt(
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
        // Row-major strided batched:
        //   A [batch, m, k], B [batch, n, k], C [batch, m, n]
        //   C = A @ B^T.
        let stride_a = (m * k) as i64;
        let stride_b = (n * k) as i64;
        let stride_c = (m * n) as i64;

        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
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
                f32_compute_type(),
                f32_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_strided_batched_ex bt failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn batched_matmul_f32_nn(
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
        unsafe { self.batched_matmul_f32(a, b, c, batch, m, n, k, alpha, beta) }
    }

    pub unsafe fn batched_matmul_f32_tn(
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
        // Row-major strided batched:
        //   A [batch, k, m], B [batch, k, n], C [batch, m, n]
        //   C = A^T @ B.
        let stride_a = (k * m) as i64;
        let stride_b = (k * n) as i64;
        let stride_c = (m * n) as i64;

        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_T,
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
                m as i32,
                stride_a,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                stride_c,
                batch as i32,
                f32_compute_type(),
                f32_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_strided_batched_ex tn failed: {:?}", e)))?;
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
                f32_compute_type(),
                f32_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_ex failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_f32_nn(
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
        // Row-major:
        //   A [m, k], B [k, n], C [m, n]
        //   C = A @ B
        //
        // cuBLAS column-major view:
        //   C^T [n, m] = B^T [n, k] @ A^T [k, m]
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
                f32_compute_type(),
                f32_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_ex nn failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_f32_tn(
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
        // Row-major:
        //   A [k, m], B [k, n], C [m, n]
        //   C = A^T @ B
        //
        // cuBLAS column-major view:
        //   C^T [n, m] = B^T [n, k] @ A [k, m]
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_T,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_32F,
                m as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                f32_compute_type(),
                f32_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("gemm_ex tn failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn linear_backward_input_f32(
        &self,
        dy: u64,
        w: u64,
        dx: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Forward uses y[t,out] = x[t,in] @ w[out,in]^T.
        // Backward input is dx[t,in] += dy[t,out] @ w[out,in].
        unsafe { self.matmul_f32_nn(dy, w, dx, tokens, in_dim, out_dim, alpha, beta) }
    }

    pub unsafe fn linear_backward_weight_f32(
        &self,
        dy: u64,
        x: u64,
        dw: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        // Forward uses y[t,out] = x[t,in] @ w[out,in]^T.
        // Backward weight is dw[out,in] += dy[t,out]^T @ x[t,in].
        unsafe { self.matmul_f32_tn(dy, x, dw, out_dim, in_dim, tokens, alpha, beta) }
    }

    pub unsafe fn matmul_bf16_bt(
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
        // Row-major BF16 forward:
        //   A [m, k], B [n, k], C [m, n], C = A @ B^T.
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
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_16BF,
                n as i32,
                bf16_compute_type(),
                bf16_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("bf16 gemm_ex bt failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_bf16_bt_to_f32(
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
        // Row-major BF16 inputs, F32 output:
        //   A [m, k], B [n, k], C [m, n], C = A @ B^T.
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
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                bf16_compute_type(),
                bf16_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("bf16 gemm_ex bt->f32 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_bf16_nn(
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
        // Row-major BF16: C [m, n] = A [m, k] @ B [k, n].
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
                cudaDataType_t::CUDA_R_16BF,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_16BF,
                n as i32,
                bf16_compute_type(),
                bf16_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("bf16 gemm_ex nn failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_bf16_nn_to_f32(
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
        // Row-major BF16 inputs, F32 output:
        //   C [m, n] = A [m, k] @ B [k, n].
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
                cudaDataType_t::CUDA_R_16BF,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_16BF,
                k as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                bf16_compute_type(),
                bf16_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("bf16 gemm_ex nn->f32 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn matmul_bf16_tn_to_f32(
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
        // Row-major BF16 inputs, F32 accumulation/output:
        //   A [k, m], B [k, n], C [m, n], C += A^T @ B.
        unsafe {
            cudarc::cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_T,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b as *const _,
                cudaDataType_t::CUDA_R_16BF,
                n as i32,
                a as *const _,
                cudaDataType_t::CUDA_R_16BF,
                m as i32,
                &beta as *const f32 as *const _,
                c as *mut _,
                cudaDataType_t::CUDA_R_32F,
                n as i32,
                bf16_compute_type(),
                bf16_gemm_algo(),
            )
            .map_err(|e| PgError::CuBlas(format!("bf16 gemm_ex tn->f32 failed: {:?}", e)))?;
        }
        Ok(())
    }

    pub unsafe fn linear_backward_input_bf16(
        &self,
        dy: u64,
        w: u64,
        dx: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe { self.matmul_bf16_nn(dy, w, dx, tokens, in_dim, out_dim, alpha, beta) }
    }

    pub unsafe fn linear_backward_input_bf16_to_bf16(
        &self,
        dy: u64,
        w: u64,
        dx: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe { self.matmul_bf16_nn(dy, w, dx, tokens, in_dim, out_dim, alpha, beta) }
    }

    pub unsafe fn linear_backward_input_bf16_to_f32(
        &self,
        dy: u64,
        w: u64,
        dx: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe { self.matmul_bf16_nn_to_f32(dy, w, dx, tokens, in_dim, out_dim, alpha, beta) }
    }

    pub unsafe fn linear_backward_weight_bf16_to_f32(
        &self,
        dy: u64,
        x: u64,
        dw: u64,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        alpha: f32,
        beta: f32,
    ) -> PgResult<()> {
        unsafe { self.matmul_bf16_tn_to_f32(dy, x, dw, out_dim, in_dim, tokens, alpha, beta) }
    }
}
