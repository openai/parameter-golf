use cudarc::driver::CudaStream;
use pg_core::error::{PgError, PgResult};
use std::sync::Arc;

#[cfg(has_fused_output_ce)]
use std::ffi::c_void;

#[cfg(has_fused_output_ce)]
unsafe extern "C" {
    fn run_fused_output_ce_stats_bf16(
        stream: *mut c_void,
        hidden_bf16: u64,
        weight_bf16: u64,
        targets: u64,
        row_max: u64,
        row_sum: u64,
        target_logit: u64,
        losses: u64,
        logits_scratch_f32: u64,
        m: i32,
        v: i32,
        d: i32,
        tile_vocab: i32,
        softcap: f32,
    ) -> i32;

    fn run_fused_output_ce_backward_bf16(
        stream: *mut c_void,
        hidden_bf16: u64,
        weight_bf16: u64,
        targets: u64,
        row_max: u64,
        row_sum: u64,
        d_hidden_f32: u64,
        d_weight_f32: u64,
        logits_scratch_f32: u64,
        grad_scratch_bf16: u64,
        m: i32,
        v: i32,
        d: i32,
        tile_vocab: i32,
        softcap: f32,
        loss_scale: f32,
    ) -> i32;
}

/// Exact output projection + softcapped CE extension boundary.
///
/// The v0 implementation lives in C++/CUDA and uses tensor-core cuBLAS BF16
/// projection tiles plus CUDA kernels for exact streaming softmax stats. It
/// intentionally remains opt-in until H100 A/B proves it beats the chunked
/// BF16 cache bridge.
pub struct FusedOutputCe {
    #[cfg_attr(not(has_fused_output_ce), allow(dead_code))]
    stream: Arc<CudaStream>,
}

impl FusedOutputCe {
    pub fn new(stream: Arc<CudaStream>) -> PgResult<Self> {
        #[cfg(has_fused_output_ce)]
        {
            Ok(Self { stream })
        }
        #[cfg(not(has_fused_output_ce))]
        {
            let _ = stream;
            Err(PgError::InvalidOp(
                "fused output CE backend was not compiled for this build".into(),
            ))
        }
    }

    pub fn is_available() -> bool {
        cfg!(has_fused_output_ce)
    }

    pub fn implementation_name(&self) -> &'static str {
        "fused_exact_cublas_tile_warp_rows_v1"
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stats_bf16(
        &self,
        hidden_bf16: u64,
        weight_bf16: u64,
        targets: u64,
        row_max: u64,
        row_sum: u64,
        target_logit: u64,
        losses: u64,
        logits_scratch_f32: u64,
        m: usize,
        v: usize,
        d: usize,
        tile_vocab: usize,
        softcap: f32,
    ) -> PgResult<()> {
        #[cfg(has_fused_output_ce)]
        unsafe {
            let status = run_fused_output_ce_stats_bf16(
                self.stream.cu_stream() as *mut c_void,
                hidden_bf16,
                weight_bf16,
                targets,
                row_max,
                row_sum,
                target_logit,
                losses,
                logits_scratch_f32,
                m as i32,
                v as i32,
                d as i32,
                tile_vocab as i32,
                softcap,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "fused output CE stats failed with status code {status}"
                )));
            }
            Ok(())
        }
        #[cfg(not(has_fused_output_ce))]
        {
            let _ = (
                hidden_bf16,
                weight_bf16,
                targets,
                row_max,
                row_sum,
                target_logit,
                losses,
                logits_scratch_f32,
                m,
                v,
                d,
                tile_vocab,
                softcap,
            );
            Err(PgError::InvalidOp(
                "fused output CE backend was not compiled for this build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn backward_bf16(
        &self,
        hidden_bf16: u64,
        weight_bf16: u64,
        targets: u64,
        row_max: u64,
        row_sum: u64,
        d_hidden_f32: u64,
        d_weight_f32: u64,
        logits_scratch_f32: u64,
        grad_scratch_bf16: u64,
        m: usize,
        v: usize,
        d: usize,
        tile_vocab: usize,
        softcap: f32,
        loss_scale: f32,
    ) -> PgResult<()> {
        #[cfg(has_fused_output_ce)]
        unsafe {
            let status = run_fused_output_ce_backward_bf16(
                self.stream.cu_stream() as *mut c_void,
                hidden_bf16,
                weight_bf16,
                targets,
                row_max,
                row_sum,
                d_hidden_f32,
                d_weight_f32,
                logits_scratch_f32,
                grad_scratch_bf16,
                m as i32,
                v as i32,
                d as i32,
                tile_vocab as i32,
                softcap,
                loss_scale,
            );
            if status != 0 {
                return Err(PgError::InvalidOp(format!(
                    "fused output CE backward failed with status code {status}"
                )));
            }
            Ok(())
        }
        #[cfg(not(has_fused_output_ce))]
        {
            let _ = (
                hidden_bf16,
                weight_bf16,
                targets,
                row_max,
                row_sum,
                d_hidden_f32,
                d_weight_f32,
                logits_scratch_f32,
                grad_scratch_bf16,
                m,
                v,
                d,
                tile_vocab,
                softcap,
                loss_scale,
            );
            Err(PgError::InvalidOp(
                "fused output CE backend was not compiled for this build".into(),
            ))
        }
    }
}
