use cudarc::driver::CudaStream;
use pg_core::error::{PgError, PgResult};
use std::sync::Arc;

#[cfg(has_cuda_cpp)]
use std::ffi::c_void;

#[cfg(has_cuda_cpp)]
unsafe extern "C" {
    fn run_naive_sdpa_f32(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch_tokens: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;
}

#[cfg(has_cudnn_frontend_sdpa)]
unsafe extern "C" {
    fn run_cudnn_sdpa_bf16_f32_forward(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_f32_forward_with_stats(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        stats: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_f32_forward_with_stats_saved_bf16(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        stats: u64,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out_bf16: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_f32_forward_with_stats_prepacked_bf16(
        stream: *mut c_void,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out: u64,
        stats: u64,
        out_bf16: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_forward_with_stats_prepacked_bf16_only(
        stream: *mut c_void,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        stats: u64,
        out_bf16: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_f32_backward(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_f32_backward_with_stats(
        stream: *mut c_void,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        stats: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;

    fn run_cudnn_sdpa_bf16_f32_backward_with_saved_bf16_stats(
        stream: *mut c_void,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out_bf16: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        stats: u64,
        batch_tokens: i32,
        seq_len: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> i32;
}

/// CUDA C++ F32 SDPA backend compiled from `cpp/sdpa.cu`.
///
/// This is intentionally named as a C++ attention backend, not
/// FlashAttention: the current kernel is a correctness-first F32 causal SDPA
/// implementation. Record mode must use a separate production BF16 fused
/// SDPA/FlashAttention backend.
pub struct CudaCppAttention {
    #[cfg_attr(not(has_cuda_cpp), allow(dead_code))]
    stream: Arc<CudaStream>,
}

/// cuDNN frontend SDPA backend.
///
/// The FFI wrapper accepts the model's F32 `[B, S, H, D]` physical layout,
/// converts Q/K/V and gradients to BF16 scratch buffers, runs cuDNN frontend
/// SDPA forward/backward with causal masking, then converts outputs back to
/// F32. The conversion shim keeps the rest of the Rust stack correct while the
/// broader model is still F32; moving the whole train path to BF16 remains the
/// next throughput milestone.
pub struct CudnnFrontendAttention {
    #[cfg_attr(not(has_cudnn_frontend_sdpa), allow(dead_code))]
    stream: Arc<CudaStream>,
}

impl CudaCppAttention {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        #[cfg(has_cuda_cpp)]
        {
            Ok(Self { stream })
        }
        #[cfg(not(has_cuda_cpp))]
        {
            let _ = stream;
            Err(PgError::InvalidOp(
                "CUDA C++ attention backend was not compiled for this build".into(),
            ))
        }
    }

    pub fn is_available() -> bool {
        cfg!(has_cuda_cpp)
    }

    pub fn backend_name(&self) -> &'static str {
        "naive_f32_cpp_sdpa"
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cuda_cpp)]
        unsafe {
            let status = run_naive_sdpa_f32(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                batch_tokens as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "CUDA attention backend failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cuda_cpp))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                batch_tokens,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "CUDA C++ attention backend was not compiled for this build".into(),
            ))
        }
    }
}

impl CudnnFrontendAttention {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        #[cfg(has_cudnn_frontend_sdpa)]
        {
            Ok(Self { stream })
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = stream;
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    pub fn is_available() -> bool {
        cfg!(has_cudnn_frontend_sdpa)
    }

    pub fn backend_name(&self) -> &'static str {
        "cudnn_frontend_sdpa_bf16_f32_bridge"
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_forward(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA forward failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_stats(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        stats: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_forward_with_stats(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                stats,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA forward_with_stats failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                stats,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_stats_saved_bf16(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        stats: u64,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out_bf16: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_forward_with_stats_saved_bf16(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                stats,
                q_bf16,
                k_bf16,
                v_bf16,
                out_bf16,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA forward_with_stats_saved_bf16 failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                stats,
                q_bf16,
                k_bf16,
                v_bf16,
                out_bf16,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_stats_prepacked_bf16(
        &self,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out: u64,
        stats: u64,
        out_bf16: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_forward_with_stats_prepacked_bf16(
                self.stream.cu_stream() as *mut c_void,
                q_bf16,
                k_bf16,
                v_bf16,
                out,
                stats,
                out_bf16,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA forward_with_stats_prepacked_bf16 failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q_bf16,
                k_bf16,
                v_bf16,
                out,
                stats,
                out_bf16,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_stats_prepacked_bf16_only(
        &self,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        stats: u64,
        out_bf16: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_forward_with_stats_prepacked_bf16_only(
                self.stream.cu_stream() as *mut c_void,
                q_bf16,
                k_bf16,
                v_bf16,
                stats,
                out_bf16,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA forward_with_stats_prepacked_bf16_only failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q_bf16,
                k_bf16,
                v_bf16,
                stats,
                out_bf16,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_backward(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA backward failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_with_stats(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        stats: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_backward_with_stats(
                self.stream.cu_stream() as *mut c_void,
                q,
                k,
                v,
                out,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                stats,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA backward_with_stats failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q,
                k,
                v,
                out,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                stats,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_with_saved_bf16_stats(
        &self,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out_bf16: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        stats: u64,
        batch_tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        #[cfg(has_cudnn_frontend_sdpa)]
        unsafe {
            let status = run_cudnn_sdpa_bf16_f32_backward_with_saved_bf16_stats(
                self.stream.cu_stream() as *mut c_void,
                q_bf16,
                k_bf16,
                v_bf16,
                out_bf16,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                stats,
                batch_tokens as i32,
                seq_len as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "cuDNN frontend SDPA backward_with_saved_bf16_stats failed with status code {}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(has_cudnn_frontend_sdpa))]
        {
            let _ = (
                q_bf16,
                k_bf16,
                v_bf16,
                out_bf16,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                stats,
                batch_tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
            Err(PgError::InvalidOp(
                "cuDNN frontend SDPA backend was not compiled for this build".into(),
            ))
        }
    }
}

#[deprecated(
    since = "0.1.0",
    note = "use CudaCppAttention; this backend is not FlashAttention"
)]
pub type FlashAttention = CudaCppAttention;
