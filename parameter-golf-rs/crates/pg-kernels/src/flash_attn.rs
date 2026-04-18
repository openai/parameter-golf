use std::sync::Arc;
use cudarc::driver::CudaStream;
use pg_core::error::{PgError, PgResult};
use std::ffi::c_void;

unsafe extern "C" {
    fn run_cudnn_sdpa_f32(
        stream: *mut c_void,
        q: u64, k: u64, v: u64, out: u64,
        batch_tokens: i32, num_heads: i32, num_kv_heads: i32, head_dim: i32
    ) -> i32;
}

/// Static C++ wrapper for cuDNN Flash Attention.
pub struct FlashAttention {
    _stream: Arc<CudaStream>,
}

impl FlashAttention {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        Ok(Self { _stream: stream })
    }

    /// Fused forward pass: Q, K, V -> O
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        q: u64, // [B*T, h, hd]
        k: u64, // [B*T, hkv, hd]
        v: u64, // [B*T, hkv, hd]
        out: u64, // [B*T, h, hd]
        batch_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        unsafe {
            let status = run_cudnn_sdpa_f32(
                std::ptr::null_mut(), // CudaStream ptr extraction if needed
                q, k, v, out,
                batch_tokens as i32,
                num_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
            );
            
            if status != 0 {
                return Err(PgError::InvalidOp(format!("cuDNN SDPA failed with status code {}", status)));
            }
        }
        Ok(())
    }
}
