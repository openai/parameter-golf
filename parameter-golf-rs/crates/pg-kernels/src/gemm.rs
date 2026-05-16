use std::collections::HashMap;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType_t,
};
use cudarc::cublaslt::{result as cublaslt_result, sys as cublaslt_sys};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use pg_core::error::{PgError, PgResult};

struct CudaBlasLtHandle(cublaslt_sys::cublasLtHandle_t);

unsafe impl Send for CudaBlasLtHandle {}
unsafe impl Sync for CudaBlasLtHandle {}

impl Drop for CudaBlasLtHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let handle = std::mem::replace(&mut self.0, ptr::null_mut());
            unsafe {
                let _ = cublaslt_result::destroy_handle(handle);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CublasLtPlanKey {
    m: usize,
    n: usize,
    k: usize,
    transa: bool,
    transb: bool,
    lda: usize,
    ldb: usize,
    ldc: usize,
    f32_output: bool,
    workspace_size: usize,
}

struct CublasLtMatmulPlan {
    desc: cublaslt_sys::cublasLtMatmulDesc_t,
    a_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    b_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    c_layout: cublaslt_sys::cublasLtMatrixLayout_t,
    algo: cublaslt_sys::cublasLtMatmulAlgo_t,
}

unsafe impl Send for CublasLtMatmulPlan {}
unsafe impl Sync for CublasLtMatmulPlan {}

impl Drop for CublasLtMatmulPlan {
    fn drop(&mut self) {
        unsafe {
            let _ = cublaslt_result::destroy_matmul_desc(self.desc);
            let _ = cublaslt_result::destroy_matrix_layout(self.a_layout);
            let _ = cublaslt_result::destroy_matrix_layout(self.b_layout);
            let _ = cublaslt_result::destroy_matrix_layout(self.c_layout);
        }
    }
}

pub struct GemmEngine {
    blas: CudaBlas,
    lt: Option<CudaBlasLtHandle>,
    lt_workspace: Option<CudaSlice<u8>>,
    lt_plan_cache: Mutex<HashMap<CublasLtPlanKey, CublasLtMatmulPlan>>,
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
    if cublaslt_bf16_enabled() {
        "cublaslt_fast_16bf_heuristic"
    } else if bf16_pedantic_enabled() {
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

fn cublaslt_bf16_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PG_CUBLASLT_BF16_GEMM")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn cublaslt_bf16_strict_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("PG_CUBLASLT_BF16_STRICT")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn cublaslt_workspace_bytes() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let mib = std::env::var("PG_CUBLASLT_WORKSPACE_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(32);
        mib.saturating_mul(1024 * 1024)
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
        let (lt, lt_workspace) = if cublaslt_bf16_enabled() {
            let handle = CudaBlasLtHandle(
                cublaslt_result::create_handle()
                    .map_err(|e| PgError::CuBlas(format!("cublasLtCreate failed: {e:?}")))?,
            );
            let workspace_bytes = cublaslt_workspace_bytes();
            let workspace = if workspace_bytes > 0 {
                Some(stream.alloc_zeros::<u8>(workspace_bytes).map_err(|e| {
                    PgError::CuBlas(format!("cublasLt workspace allocation failed: {e:?}"))
                })?)
            } else {
                None
            };
            (Some(handle), workspace)
        } else {
            (None, None)
        };
        Ok(Self {
            blas,
            lt,
            lt_workspace,
            lt_plan_cache: Mutex::new(HashMap::new()),
            stream,
        })
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
        if self.lt.is_some()
            && unsafe { self.try_matmul_bf16_bt_cublaslt(a, b, c, m, n, k, alpha, beta, false) }
                .map_or_else(
                    |err| {
                        if cublaslt_bf16_strict_enabled() {
                            Some(Err(err))
                        } else {
                            None
                        }
                    },
                    |_| Some(Ok(())),
                )
                .transpose()?
                .is_some()
        {
            return Ok(());
        }
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
        if self.lt.is_some()
            && unsafe { self.try_matmul_bf16_bt_cublaslt(a, b, c, m, n, k, alpha, beta, true) }
                .map_or_else(
                    |err| {
                        if cublaslt_bf16_strict_enabled() {
                            Some(Err(err))
                        } else {
                            None
                        }
                    },
                    |_| Some(Ok(())),
                )
                .transpose()?
                .is_some()
        {
            return Ok(());
        }
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
        if self.lt.is_some()
            && unsafe { self.try_matmul_bf16_nn_cublaslt(a, b, c, m, n, k, alpha, beta, false) }
                .map_or_else(
                    |err| {
                        if cublaslt_bf16_strict_enabled() {
                            Some(Err(err))
                        } else {
                            None
                        }
                    },
                    |_| Some(Ok(())),
                )
                .transpose()?
                .is_some()
        {
            return Ok(());
        }
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
        if self.lt.is_some()
            && unsafe { self.try_matmul_bf16_nn_cublaslt(a, b, c, m, n, k, alpha, beta, true) }
                .map_or_else(
                    |err| {
                        if cublaslt_bf16_strict_enabled() {
                            Some(Err(err))
                        } else {
                            None
                        }
                    },
                    |_| Some(Ok(())),
                )
                .transpose()?
                .is_some()
        {
            return Ok(());
        }
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
        if self.lt.is_some()
            && unsafe { self.try_matmul_bf16_tn_to_f32_cublaslt(a, b, c, m, n, k, alpha, beta) }
                .map_or_else(
                    |err| {
                        if cublaslt_bf16_strict_enabled() {
                            Some(Err(err))
                        } else {
                            None
                        }
                    },
                    |_| Some(Ok(())),
                )
                .transpose()?
                .is_some()
        {
            return Ok(());
        }
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

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_matmul_bf16_bt_cublaslt(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        f32_output: bool,
    ) -> PgResult<()> {
        // Legacy cuBLAS call:
        //   op(A)=B^T, op(B)=A, output is C^T with dimensions [n, m].
        unsafe {
            self.matmul_bf16_cublaslt_raw(
                b, a, c, n, m, k, true, false, k, k, n, alpha, beta, f32_output,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_matmul_bf16_nn_cublaslt(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        beta: f32,
        f32_output: bool,
    ) -> PgResult<()> {
        // Legacy cuBLAS call:
        //   op(A)=B, op(B)=A, output is C^T with dimensions [n, m].
        unsafe {
            self.matmul_bf16_cublaslt_raw(
                b, a, c, n, m, k, false, false, n, k, n, alpha, beta, f32_output,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn try_matmul_bf16_tn_to_f32_cublaslt(
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
        // Legacy cuBLAS call:
        //   op(A)=B, op(B)=A^T, output is C^T with dimensions [n, m].
        unsafe {
            self.matmul_bf16_cublaslt_raw(b, a, c, n, m, k, false, true, n, m, n, alpha, beta, true)
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn matmul_bf16_cublaslt_raw(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        n: usize,
        k: usize,
        transa: bool,
        transb: bool,
        lda: usize,
        ldb: usize,
        ldc: usize,
        alpha: f32,
        beta: f32,
        f32_output: bool,
    ) -> PgResult<()> {
        let lt = self.lt.as_ref().ok_or_else(|| {
            PgError::CuBlas("cuBLASLt BF16 GEMM requested but handle is not initialized".into())
        })?;
        let workspace_size = self
            .lt_workspace
            .as_ref()
            .map(|workspace| workspace.num_bytes())
            .unwrap_or(0);
        let key = CublasLtPlanKey {
            m,
            n,
            k,
            transa,
            transb,
            lda,
            ldb,
            ldc,
            f32_output,
            workspace_size,
        };
        let mut cache = self
            .lt_plan_cache
            .lock()
            .map_err(|_| PgError::CuBlas("cublasLt plan cache lock poisoned".into()))?;
        if !cache.contains_key(&key) {
            let plan = unsafe { self.create_cublaslt_plan(lt.0, key)? };
            cache.insert(key, plan);
        }
        let plan = cache
            .get(&key)
            .ok_or_else(|| PgError::CuBlas("cublasLt plan cache insert failed".into()))?;

        let (workspace_ptr, _workspace_sync) = self
            .lt_workspace
            .as_ref()
            .map(|workspace| workspace.device_ptr(self.stream.as_ref()))
            .unwrap_or((0, cudarc::driver::SyncOnDrop::Record(None)));
        let launch_result = unsafe {
            cublaslt_result::matmul(
                lt.0,
                plan.desc,
                (&alpha) as *const _ as *const _,
                (&beta) as *const _ as *const _,
                a as *const _,
                plan.a_layout,
                b as *const _,
                plan.b_layout,
                c as *const _,
                plan.c_layout,
                c as *mut _,
                plan.c_layout,
                (&plan.algo) as *const _,
                workspace_ptr as *mut _,
                workspace_size,
                self.stream.cu_stream() as *mut _,
            )
        };

        launch_result.map_err(|e| PgError::CuBlas(format!("cublasLtMatmul failed: {e:?}")))
    }

    unsafe fn create_cublaslt_plan(
        &self,
        lt: cublaslt_sys::cublasLtHandle_t,
        key: CublasLtPlanKey,
    ) -> PgResult<CublasLtMatmulPlan> {
        let output_type = if key.f32_output {
            cublaslt_sys::cudaDataType_t::CUDA_R_32F
        } else {
            cublaslt_sys::cudaDataType_t::CUDA_R_16BF
        };
        let a_rows = if key.transa { key.k } else { key.m };
        let a_cols = if key.transa { key.m } else { key.k };
        let b_rows = if key.transb { key.n } else { key.k };
        let b_cols = if key.transb { key.k } else { key.n };

        let a_layout = cublaslt_result::create_matrix_layout(
            cublaslt_sys::cudaDataType_t::CUDA_R_16BF,
            a_rows as u64,
            a_cols as u64,
            key.lda as i64,
        )
        .map_err(|e| PgError::CuBlas(format!("cublasLt A layout failed: {e:?}")))?;
        let b_layout = cublaslt_result::create_matrix_layout(
            cublaslt_sys::cudaDataType_t::CUDA_R_16BF,
            b_rows as u64,
            b_cols as u64,
            key.ldb as i64,
        )
        .map_err(|e| {
            unsafe {
                let _ = cublaslt_result::destroy_matrix_layout(a_layout);
            }
            PgError::CuBlas(format!("cublasLt B layout failed: {e:?}"))
        })?;
        let c_layout = cublaslt_result::create_matrix_layout(
            output_type,
            key.m as u64,
            key.n as u64,
            key.ldc as i64,
        )
        .map_err(|e| {
            unsafe {
                let _ = cublaslt_result::destroy_matrix_layout(a_layout);
                let _ = cublaslt_result::destroy_matrix_layout(b_layout);
            }
            PgError::CuBlas(format!("cublasLt C layout failed: {e:?}"))
        })?;
        let desc = cublaslt_result::create_matmul_desc(
            cublaslt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            cublaslt_sys::cudaDataType_t::CUDA_R_32F,
        )
        .map_err(|e| {
            unsafe {
                let _ = cublaslt_result::destroy_matrix_layout(a_layout);
                let _ = cublaslt_result::destroy_matrix_layout(b_layout);
                let _ = cublaslt_result::destroy_matrix_layout(c_layout);
            }
            PgError::CuBlas(format!("cublasLt matmul desc failed: {e:?}"))
        })?;
        let pref = cublaslt_result::create_matmul_pref().map_err(|e| {
            unsafe {
                let _ = cublaslt_result::destroy_matmul_desc(desc);
                let _ = cublaslt_result::destroy_matrix_layout(a_layout);
                let _ = cublaslt_result::destroy_matrix_layout(b_layout);
                let _ = cublaslt_result::destroy_matrix_layout(c_layout);
            }
            PgError::CuBlas(format!("cublasLt pref failed: {e:?}"))
        })?;

        let transa_value = if key.transa { 1i32 } else { 0i32 };
        let transb_value = if key.transb { 1i32 } else { 0i32 };
        let setup_result = unsafe {
            cublaslt_result::set_matmul_desc_attribute(
                desc,
                cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&transa_value) as *const _ as *const _,
                std::mem::size_of::<i32>(),
            )
            .and_then(|_| {
                cublaslt_result::set_matmul_desc_attribute(
                    desc,
                    cublaslt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                    (&transb_value) as *const _ as *const _,
                    std::mem::size_of::<i32>(),
                )
            })
            .and_then(|_| {
                cublaslt_result::set_matmul_pref_attribute(
                    pref,
                    cublaslt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    (&key.workspace_size) as *const _ as *const _,
                    std::mem::size_of::<usize>(),
                )
            })
        };
        if let Err(err) = setup_result {
            unsafe {
                let _ = cublaslt_result::destroy_matmul_pref(pref);
                let _ = cublaslt_result::destroy_matmul_desc(desc);
                let _ = cublaslt_result::destroy_matrix_layout(a_layout);
                let _ = cublaslt_result::destroy_matrix_layout(b_layout);
                let _ = cublaslt_result::destroy_matrix_layout(c_layout);
            }
            return Err(PgError::CuBlas(format!(
                "cublasLt plan setup failed: {err:?}"
            )));
        }

        let heuristic = match unsafe {
            cublaslt_result::get_matmul_algo_heuristic(
                lt, desc, a_layout, b_layout, c_layout, c_layout, pref,
            )
        } {
            Ok(heuristic) => heuristic,
            Err(err) => {
                unsafe {
                    let _ = cublaslt_result::destroy_matmul_pref(pref);
                    let _ = cublaslt_result::destroy_matmul_desc(desc);
                    let _ = cublaslt_result::destroy_matrix_layout(a_layout);
                    let _ = cublaslt_result::destroy_matrix_layout(b_layout);
                    let _ = cublaslt_result::destroy_matrix_layout(c_layout);
                }
                return Err(PgError::CuBlas(format!(
                    "cublasLt heuristic failed: {err:?}"
                )));
            }
        };
        unsafe {
            let _ = cublaslt_result::destroy_matmul_pref(pref);
        }

        Ok(CublasLtMatmulPlan {
            desc,
            a_layout,
            b_layout,
            c_layout,
            algo: heuristic.algo,
        })
    }
}
