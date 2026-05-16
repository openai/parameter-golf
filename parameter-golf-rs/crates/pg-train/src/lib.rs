use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pg_model::backward::GradBuffers;
use pg_model::{
    AttentionBackend, BackwardChainProfile, CudaGraphProfile, DistributedOptimizerBackend,
    EvalAdaptationBackend, ExecutionPlan, ForwardBuffer, GptModel, ModelComputePrecision,
    NcclOverlapMode, OutputCeBackend, RecordProfile, RecurrentBackwardProfile, RunMode, RunSpec,
    TrainBackend, TttMask,
};
use pg_optim::adamw::{AdamW, AdamWState};
use pg_optim::ema::{Ema, Swa};
#[cfg(feature = "cuda")]
use pg_optim::gpu::{AdamWHyper, GpuAdamWState, GpuMuon, GpuOptimizer};
use pg_optim::muon::Muon;
use pg_optim::scheduler;

use pg_core::PgResult;
use pg_data::bpb::{BpbLuts, compute_bpb};
use pg_data::token_stream::DistributedTokenLoader;

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[derive(Clone)]
struct Sha256State {
    h: [u32; 8],
    len: u64,
    buf: [u8; 64],
    buf_len: usize,
}

impl Sha256State {
    fn new() -> Self {
        Self {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            len: 0,
            buf: [0; 64],
            buf_len: 0,
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        self.len = self.len.wrapping_add(input.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while input.len() >= 64 {
            self.compress(&input[..64]);
            input = &input[64..];
        }
        if !input.is_empty() {
            self.buf[..input.len()].copy_from_slice(input);
            self.buf_len = input.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.len.wrapping_mul(8);
        self.buf[self.buf_len] = 0x80;
        self.buf_len += 1;
        if self.buf_len > 56 {
            for b in &mut self.buf[self.buf_len..] {
                *b = 0;
            }
            let block = self.buf;
            self.compress(&block);
            self.buf_len = 0;
        }
        for b in &mut self.buf[self.buf_len..56] {
            *b = 0;
        }
        self.buf[56..64].copy_from_slice(&bit_len.to_be_bytes());
        let block = self.buf;
        self.compress(&block);
        let mut out = [0u8; 32];
        for (chunk, word) in out.chunks_exact_mut(4).zip(self.h) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    fn compress(&mut self, block: &[u8]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).take(16).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in self.h.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = (*slot).wrapping_add(value);
        }
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

fn sha256_file(path: &Path) -> PgResult<String> {
    let mut file = File::open(path)?;
    let mut state = Sha256State::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        state.update(&buf[..n]);
    }
    Ok(hex_digest(state.finalize()))
}

fn simple_glob_paths(pattern: &str) -> PgResult<Vec<PathBuf>> {
    let Some(star) = pattern.find('*') else {
        return Ok(vec![PathBuf::from(pattern)]);
    };
    let prefix = &pattern[..star];
    let suffix = &pattern[star + 1..];
    let prefix_path = Path::new(prefix);
    let dir = prefix_path.parent().unwrap_or_else(|| Path::new("."));
    let stem_prefix = prefix_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with(stem_prefix) && path.to_string_lossy().ends_with(suffix) {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

fn sha256_file_set(pattern: &str) -> PgResult<Option<String>> {
    let paths = simple_glob_paths(pattern)?;
    if paths.is_empty() {
        return Ok(None);
    }
    let mut state = Sha256State::new();
    for path in paths {
        state.update(path.to_string_lossy().as_bytes());
        state.update(b"\0");
        let digest = sha256_file(&path)?;
        state.update(digest.as_bytes());
        state.update(b"\0");
    }
    Ok(Some(hex_digest(state.finalize())))
}

#[derive(Debug, Clone)]
pub struct VariantResult {
    pub run_name: String,
    pub mode: RunMode,
    pub train_backend: TrainBackend,
    pub variant_fingerprint: String,
    pub steps_completed: usize,
    pub train_loss: f32,
    pub train_loss_source: String,
    pub proxy_bpb: Option<f64>,
    pub eval_loss: Option<f64>,
    pub final_bpb: Option<f64>,
    pub eval_tokens: Option<usize>,
    pub artifact_bytes: Option<usize>,
    pub submission_code_bytes: Option<usize>,
    pub submission_total_bytes: Option<usize>,
    pub artifact_budget_ok: Option<bool>,
    pub artifact_model_sha256: Option<String>,
    pub artifact_code_sha256: Option<String>,
    pub caseops_byte_sidecar_sha256: Option<String>,
    pub attention_backend: String,
    pub distributed_optimizer_backend: String,
    pub eval_adaptation_backend: String,
    pub frontier_record_ready: bool,
    pub leaderboard_algorithm_ready: bool,
    pub record_shape: bool,
    pub record_attention_grade: bool,
    pub microbatch_serial_loop: bool,
    pub bank_update_backend: String,
    pub train_data_source: String,
    pub bpb_byte_source: String,
    pub proxy_metric_source: Option<String>,
    pub timing_backend: String,
    pub ms_per_step: f64,
    pub wallclock_seconds: f64,
    pub timing_steps: usize,
    pub timing_measured_ms_per_step: f64,
    pub timing_recurrent_active_steps: usize,
    pub timing_recurrent_inactive_steps: usize,
    pub timing_recurrent_active_ms: f64,
    pub timing_recurrent_inactive_ms: f64,
    pub host_batch_flatten_calls: usize,
    pub host_to_device_batch_bytes: usize,
    pub device_batch_ready_steps: usize,
    pub device_batch_missing_steps: usize,
    pub device_to_host_scalar_reads: usize,
    pub timing_data_sampling_ms: f64,
    pub timing_train_step_ms: f64,
    pub timing_cuda_zero_grads_ms: f64,
    pub timing_cuda_h2d_ms: f64,
    pub timing_cuda_backward_ms: f64,
    pub timing_cuda_backward_forward_ms: f64,
    pub timing_cuda_backward_forward_embed_ms: f64,
    pub timing_cuda_backward_forward_encoder_ms: f64,
    pub timing_cuda_backward_forward_encoder_layer_max_ms: f64,
    pub timing_cuda_backward_forward_decoder_ms: f64,
    pub timing_cuda_backward_forward_decoder_layer_max_ms: f64,
    pub timing_cuda_backward_forward_logits_ms: f64,
    pub timing_cuda_backward_forward_block_pre_attn_ms: f64,
    pub timing_cuda_backward_forward_block_attention_ms: f64,
    pub timing_cuda_backward_forward_block_post_attn_ms: f64,
    pub timing_cuda_backward_forward_block_mlp_ms: f64,
    pub timing_cuda_backward_block_recompute_ms: f64,
    pub timing_cuda_backward_block_mlp_ms: f64,
    pub timing_cuda_backward_block_mlp_residual_ms: f64,
    pub timing_cuda_backward_block_mlp_down_ms: f64,
    pub timing_cuda_backward_block_mlp_act_ms: f64,
    pub timing_cuda_backward_block_mlp_up_ms: f64,
    pub timing_cuda_backward_block_mlp_norm_ms: f64,
    pub timing_cuda_backward_block_attn_out_ms: f64,
    pub timing_cuda_backward_block_attn_out_residual_ms: f64,
    pub timing_cuda_backward_block_attn_out_proj_ms: f64,
    pub timing_cuda_backward_block_attn_out_gate_xsa_ms: f64,
    pub timing_cuda_backward_block_attention_ms: f64,
    pub timing_cuda_backward_block_attention_sdpa_ms: f64,
    pub timing_cuda_backward_block_attention_xsa_accum_ms: f64,
    pub timing_cuda_backward_block_qkv_ms: f64,
    pub timing_cuda_backward_block_qkv_rope_ms: f64,
    pub timing_cuda_backward_block_qkv_proj_ms: f64,
    pub timing_cuda_backward_block_qkv_ve_ms: f64,
    pub timing_cuda_backward_block_qkv_norm_resid_ms: f64,
    pub timing_cuda_backward_output_ms: f64,
    pub timing_cuda_backward_decoder_ms: f64,
    pub timing_cuda_backward_encoder_ms: f64,
    pub timing_cuda_backward_tail_ms: f64,
    pub timing_cuda_non_bank_sync_ms: f64,
    pub timing_cuda_bank_update_ms: f64,
    pub timing_cuda_bank_update_stage_pack_ms: f64,
    pub timing_cuda_bank_update_stage_grad_collectives_ms: f64,
    pub timing_cuda_bank_update_stage_scale_ms: f64,
    pub timing_cuda_bank_update_stage_norm_ms: f64,
    pub timing_cuda_bank_update_stage_clip_ms: f64,
    pub timing_cuda_bank_update_stage_muon_ms: f64,
    pub timing_cuda_bank_update_stage_param_gather_ms: f64,
    pub timing_cuda_bank_update_stage_copy_back_ms: f64,
    pub timing_cuda_non_bank_update_ms: f64,
    pub timing_post_train_sync_ms: f64,
    pub timing_artifact_export_ms: f64,
    pub timing_eval_ms: f64,
    pub rank: usize,
    pub world_size: usize,
    pub distributed_sync: bool,
    pub seq_len: usize,
    pub global_batch_tokens: usize,
    pub local_microbatches_per_step: usize,
    pub tokens_seen_global: usize,
}

pub struct VariantRunner {
    pub run_spec: RunSpec,
    pub plan: ExecutionPlan,
}

#[derive(Debug, Clone, Copy)]
struct StepBatchPlan {
    microbatch_tokens: usize,
    global_batch_tokens: usize,
    local_microbatches_per_step: usize,
}

#[derive(Debug, Clone, Default)]
struct RunTiming {
    data_sampling_ms: f64,
    train_step_ms: f64,
    cuda_zero_grads_ms: f64,
    cuda_h2d_ms: f64,
    cuda_backward_ms: f64,
    cuda_backward_forward_ms: f64,
    cuda_backward_forward_embed_ms: f64,
    cuda_backward_forward_encoder_ms: f64,
    cuda_backward_forward_encoder_layer_max_ms: f64,
    cuda_backward_forward_decoder_ms: f64,
    cuda_backward_forward_decoder_layer_max_ms: f64,
    cuda_backward_forward_logits_ms: f64,
    cuda_backward_forward_block_pre_attn_ms: f64,
    cuda_backward_forward_block_attention_ms: f64,
    cuda_backward_forward_block_post_attn_ms: f64,
    cuda_backward_forward_block_mlp_ms: f64,
    cuda_backward_block_recompute_ms: f64,
    cuda_backward_block_mlp_ms: f64,
    cuda_backward_block_mlp_residual_ms: f64,
    cuda_backward_block_mlp_down_ms: f64,
    cuda_backward_block_mlp_act_ms: f64,
    cuda_backward_block_mlp_up_ms: f64,
    cuda_backward_block_mlp_norm_ms: f64,
    cuda_backward_block_attn_out_ms: f64,
    cuda_backward_block_attn_out_residual_ms: f64,
    cuda_backward_block_attn_out_proj_ms: f64,
    cuda_backward_block_attn_out_gate_xsa_ms: f64,
    cuda_backward_block_attention_ms: f64,
    cuda_backward_block_attention_sdpa_ms: f64,
    cuda_backward_block_attention_xsa_accum_ms: f64,
    cuda_backward_block_qkv_ms: f64,
    cuda_backward_block_qkv_rope_ms: f64,
    cuda_backward_block_qkv_proj_ms: f64,
    cuda_backward_block_qkv_ve_ms: f64,
    cuda_backward_block_qkv_norm_resid_ms: f64,
    cuda_backward_output_ms: f64,
    cuda_backward_decoder_ms: f64,
    cuda_backward_encoder_ms: f64,
    cuda_backward_tail_ms: f64,
    cuda_non_bank_sync_ms: f64,
    cuda_bank_update_ms: f64,
    cuda_bank_update_stage_pack_ms: f64,
    cuda_bank_update_stage_grad_collectives_ms: f64,
    cuda_bank_update_stage_scale_ms: f64,
    cuda_bank_update_stage_norm_ms: f64,
    cuda_bank_update_stage_clip_ms: f64,
    cuda_bank_update_stage_muon_ms: f64,
    cuda_bank_update_stage_param_gather_ms: f64,
    cuda_bank_update_stage_copy_back_ms: f64,
    cuda_non_bank_update_ms: f64,
    post_train_sync_ms: f64,
    artifact_export_ms: f64,
    eval_ms: f64,
    recurrent_active_steps: usize,
    recurrent_inactive_steps: usize,
    recurrent_active_ms: f64,
    recurrent_inactive_ms: f64,
    host_batch_flatten_calls: usize,
    host_to_device_batch_bytes: usize,
    device_batch_ready_steps: usize,
    device_batch_missing_steps: usize,
    device_to_host_scalar_reads: usize,
}

#[cfg(feature = "cuda")]
fn cuda_event_timing_enabled() -> bool {
    matches!(
        std::env::var("PG_CUDA_EVENT_TIMING")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_overwrite_bank_grads_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_OVERWRITE_BANK_GRADS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn cuda_stage_timing_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_BACKWARD_STAGE_TIMING")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn cuda_backward_graph_enabled() -> bool {
    matches!(
        std::env::var("PG_CUDA_BACKWARD_GRAPH")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) && matches!(
        std::env::var("PG_CUDA_BACKWARD_GRAPH_STRICT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) && !cuda_stage_timing_enabled()
}

#[cfg(not(feature = "cuda"))]
fn cuda_backward_graph_enabled() -> bool {
    false
}

#[cfg(feature = "cuda")]
fn cuda_graph_capture_mode() -> cudarc::driver::sys::CUstreamCaptureMode {
    match std::env::var("PG_CUDA_GRAPH_CAPTURE_MODE")
        .unwrap_or_else(|_| "thread_local".to_string())
        .to_ascii_lowercase()
        .as_str()
    {
        "relaxed" => cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
        "global" => cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
        _ => cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    }
}

#[cfg(feature = "cuda")]
fn cuda_backward_graph_warmup_steps() -> usize {
    std::env::var("PG_CUDA_BACKWARD_GRAPH_WARMUP_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1)
}

#[cfg(feature = "cuda")]
fn sharded_parallel_muon_local_graph_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SHARDED_MUON_LOCAL_GRAPH")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) && cuda_backward_graph_enabled()
        && !cuda_stage_timing_enabled()
}

#[cfg(feature = "cuda")]
fn sharded_parallel_muon_pre_norm_graph_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SHARDED_MUON_PRE_NORM_GRAPH")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) && cuda_backward_graph_enabled()
        && !cuda_stage_timing_enabled()
}

#[cfg(feature = "cuda")]
fn sharded_parallel_muon_local_graph_warmup_steps() -> usize {
    std::env::var("PG_GPU_SHARDED_MUON_LOCAL_GRAPH_WARMUP_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1)
}

#[cfg(feature = "cuda")]
fn sharded_parallel_muon_pre_norm_graph_warmup_steps() -> usize {
    std::env::var("PG_GPU_SHARDED_MUON_PRE_NORM_GRAPH_WARMUP_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(2)
        .max(1)
}

#[cfg(feature = "cuda")]
fn cuda_timing_backend_label() -> &'static str {
    if cuda_backward_graph_enabled() && cuda_event_timing_enabled() {
        "cuda_event_max_per_replica_backward_graph"
    } else if cuda_backward_graph_enabled() {
        "host_wallclock_cuda_boundary_backward_graph"
    } else if cuda_event_timing_enabled() && cuda_stage_timing_enabled() {
        "cuda_event_max_per_replica_stage_instrumented"
    } else if cuda_event_timing_enabled() {
        "cuda_event_max_per_replica"
    } else {
        "host_wallclock_cuda_boundary"
    }
}

fn env_flag_raw(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn set_env_var(name: &str, value: &str) {
    unsafe {
        std::env::set_var(name, value);
    }
}

fn apply_bool_runtime_env(name: &str, expected: bool, record_authoritative: bool) -> PgResult<()> {
    if record_authoritative {
        if let Some(actual) = env_flag_raw(name) {
            if actual != expected {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "record mode runtime profile owns {name}; env requested {actual}, spec requires {expected}"
                )));
            }
        }
        set_env_var(name, if expected { "1" } else { "0" });
    } else if std::env::var_os(name).is_none() {
        set_env_var(name, if expected { "1" } else { "0" });
    }
    Ok(())
}

fn apply_string_runtime_env(
    name: &str,
    expected: &str,
    record_authoritative: bool,
) -> PgResult<()> {
    if record_authoritative {
        if let Ok(actual) = std::env::var(name) {
            if actual != expected {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "record mode runtime profile owns {name}; env requested {actual:?}, spec requires {expected:?}"
                )));
            }
        }
        set_env_var(name, expected);
    } else if std::env::var_os(name).is_none() {
        set_env_var(name, expected);
    }
    Ok(())
}

fn apply_usize_runtime_env(
    name: &str,
    expected: usize,
    record_authoritative: bool,
) -> PgResult<()> {
    if record_authoritative {
        if let Ok(actual) = std::env::var(name) {
            let actual_value = actual.parse::<usize>().map_err(|_| {
                pg_core::PgError::InvalidOp(format!(
                    "record mode runtime profile owns {name}; env requested non-usize {actual:?}, spec requires {expected}"
                ))
            })?;
            if actual_value != expected {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "record mode runtime profile owns {name}; env requested {actual_value}, spec requires {expected}"
                )));
            }
        }
        set_env_var(name, &expected.to_string());
    } else if std::env::var_os(name).is_none() {
        set_env_var(name, &expected.to_string());
    }
    Ok(())
}

fn apply_runtime_profile_env(run_spec: &RunSpec, mode: RunMode) -> PgResult<()> {
    if !is_record_shaped_mode(mode) {
        return Ok(());
    }
    let record_authoritative = mode == RunMode::Record;
    let bf16_chain = run_spec.runtime.backward_chain_profile != BackwardChainProfile::Off;
    let direct_compact =
        run_spec.runtime.backward_chain_profile == BackwardChainProfile::Bf16DirectCompact;

    apply_bool_runtime_env(
        "PG_GPU_BF16_BACKWARD_CHAIN",
        bf16_chain,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_BF16_BACKWARD_CHAIN_STRICT",
        bf16_chain,
        record_authoritative,
    )?;
    if bf16_chain {
        apply_string_runtime_env("PG_GPU_SAVE_LAYER_ACTS", "all", record_authoritative)?;
        apply_bool_runtime_env("PG_GPU_DIRECT_SAVED_ACTS", true, record_authoritative)?;
        apply_bool_runtime_env(
            "PG_GPU_CUDNN_PREPACKED_BF16_ATTN",
            true,
            record_authoritative,
        )?;
        apply_bool_runtime_env("PG_GPU_FUSED_QKV_PROJ", true, record_authoritative)?;
        apply_bool_runtime_env(
            "PG_GPU_FUSED_QKV_PROJ_RECORD_OK",
            true,
            record_authoritative,
        )?;
        apply_bool_runtime_env("PG_GPU_BF16_ATTN_BACKWARD_TAIL", true, record_authoritative)?;
        apply_bool_runtime_env("PG_GPU_BF16_ATTN_TAIL_QKV_PACK", true, record_authoritative)?;
        apply_bool_runtime_env(
            "PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK",
            direct_compact,
            record_authoritative,
        )?;
        apply_bool_runtime_env(
            "PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO",
            direct_compact,
            record_authoritative,
        )?;
        apply_bool_runtime_env("PG_GPU_BF16_QKV_DX_OUTPUT", true, record_authoritative)?;
        apply_bool_runtime_env("PG_GPU_BF16_MLP_DOWN_DX", true, record_authoritative)?;
        apply_string_runtime_env(
            "PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER",
            if direct_compact {
                "direct_compact"
            } else {
                "split_compact"
            },
            record_authoritative,
        )?;
        apply_bool_runtime_env(
            "PG_GPU_SPLIT_QKV_NORM_RESID_BWD",
            !direct_compact,
            record_authoritative,
        )?;
        apply_bool_runtime_env(
            "PG_GPU_CHUNKED_QKV_NORM_RESID_BWD",
            false,
            record_authoritative,
        )?;
        apply_bool_runtime_env(
            "PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT",
            true,
            record_authoritative,
        )?;
        apply_bool_runtime_env(
            "PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS",
            false,
            record_authoritative,
        )?;
        apply_bool_runtime_env(
            "PG_GPU_SPLIT_RESIDUAL_MIX_GRAD",
            false,
            record_authoritative,
        )?;
    }

    apply_bool_runtime_env(
        "PG_RECORD_REQUIRE_DEVICE_BATCH",
        run_spec.runtime.require_device_batch,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_TOKEN_RING_FULL_SCHEDULE",
        run_spec.runtime.token_ring_full_schedule,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_BIGRAM_EMBED_MERGE",
        run_spec.runtime.bigram_embedding_merge,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_COMBINED_QKV_ROPE_TAIL_BWD",
        run_spec.runtime.combined_qkv_rope_tail_backward,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_GRAPH_SIDE_GEMM_CAPTURE",
        run_spec.runtime.graph_side_gemm_capture,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_SHARDED_MUON_LOCAL_GRAPH",
        run_spec.runtime.sharded_muon_local_graph,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_SHARDED_MUON_PRE_NORM_GRAPH",
        run_spec.runtime.sharded_muon_pre_norm_graph,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_SHARDED_MUON_FUSED_GLOBAL_CLIP",
        run_spec.runtime.sharded_muon_fused_global_clip,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER",
        run_spec.runtime.sharded_muon_bf16_shadow_all_gather,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_RECURRENT_PASS1_STRAIGHT_THROUGH",
        matches!(
            run_spec.runtime.recurrent_backward_profile,
            RecurrentBackwardProfile::Pass1StraightThrough
        ),
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_RECURRENT_ALL_STRAIGHT_THROUGH",
        matches!(
            run_spec.runtime.recurrent_backward_profile,
            RecurrentBackwardProfile::AllStraightThrough
        ),
        record_authoritative,
    )?;
    apply_usize_runtime_env(
        "PG_GPU_RECURRENT_STRAIGHT_THROUGH_LAYERS",
        run_spec.runtime.recurrent_straight_through_layers,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_SKIP_RECURRENT_BANK_GRADS",
        run_spec.runtime.skip_recurrent_bank_grads,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_GPU_SKIP_RECURRENT_PASS1_BANK_GRADS",
        run_spec.runtime.skip_recurrent_pass1_bank_grads,
        record_authoritative,
    )?;
    apply_usize_runtime_env(
        "PG_GPU_MUON_NS_STEPS",
        run_spec.train.muon_newton_schulz_steps.max(1),
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_NCCL_BACKWARD_BUCKET_OVERLAP",
        run_spec.runtime.nccl_overlap_mode == NcclOverlapMode::BucketedMeasured,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_CUDA_BACKWARD_GRAPH",
        run_spec.runtime.cuda_graph_profile != CudaGraphProfile::Off,
        record_authoritative,
    )?;
    apply_bool_runtime_env(
        "PG_CUDA_BACKWARD_GRAPH_STRICT",
        run_spec.runtime.cuda_graph_profile != CudaGraphProfile::Off,
        record_authoritative,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_saved_layer_activations_enabled() -> bool {
    !matches!(
        std::env::var("PG_GPU_SAVE_LAYER_ACTS")
            .unwrap_or_else(|_| "off".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "" | "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "cuda")]
fn gemm_compute_mode_label() -> &'static str {
    pg_kernels::gemm::f32_compute_mode_label()
}

#[cfg(feature = "cuda")]
fn bf16_gemm_compute_mode_label() -> &'static str {
    pg_kernels::gemm::bf16_compute_mode_label()
}

#[cfg(not(feature = "cuda"))]
fn gemm_compute_mode_label() -> &'static str {
    "pedantic_f32"
}

#[cfg(not(feature = "cuda"))]
fn bf16_gemm_compute_mode_label() -> &'static str {
    "unavailable"
}

#[cfg(not(feature = "cuda"))]
fn gpu_saved_layer_activations_enabled() -> bool {
    false
}

#[cfg(not(feature = "cuda"))]
fn cuda_timing_backend_label() -> &'static str {
    "host_wallclock_cuda_boundary"
}

#[cfg(feature = "cuda")]
struct CudaBackwardGraph(cudarc::driver::CudaGraph);

// The graph is owned by one GPU replica. Scoped worker threads may move that
// replica across steps, but a replica's graph is never launched concurrently.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaBackwardGraph {}

#[cfg(feature = "cuda")]
struct CudaMuonLocalGraph(cudarc::driver::CudaGraph);

// One local Muon graph is owned by one sharded optimizer replica and is never
// launched concurrently for that replica.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaMuonLocalGraph {}

#[cfg(feature = "cuda")]
struct CudaSingleHybridRuntime {
    gpu_model: pg_model::gpu::GpuModel,
    backward_state: pg_model::gpu::GpuBackwardState,
    input_ids: pg_core::GpuTensor,
    targets: pg_core::GpuTensor,
}

#[cfg(feature = "cuda")]
impl CudaSingleHybridRuntime {
    fn new(model: &GptModel, plan: &ExecutionPlan, tokens: usize) -> PgResult<Self> {
        let ctx = cudarc::driver::CudaContext::new(0).map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda context init failed: {:?}", e))
        })?;
        let stream = ctx.new_stream().map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda stream init failed: {:?}", e))
        })?;
        let gpu_model =
            pg_model::gpu::GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())?;
        Ok(Self {
            backward_state: pg_model::gpu::GpuBackwardState::new_for_plan(
                plan,
                tokens,
                stream.clone(),
            )?,
            gpu_model,
            input_ids: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens],
                pg_core::DType::U32,
            )?,
            targets: pg_core::GpuTensor::zeros_gpu(stream, &[tokens], pg_core::DType::U32)?,
        })
    }
}

#[cfg(feature = "cuda")]
struct CudaSingleFastRuntime {
    gpu_model: pg_model::gpu::GpuModel,
    backward_state: pg_model::gpu::GpuBackwardState,
    backward_graph: Option<CudaBackwardGraph>,
    backward_graph_seq_len: usize,
    backward_graph_warmup_steps_done: usize,
    input_ids: pg_core::GpuTensor,
    targets: pg_core::GpuTensor,
    token_span_u16: pg_core::GpuTensor,
    token_ring_u16: pg_core::GpuTensor,
    host_input_ids: Vec<u32>,
    host_targets: Vec<u32>,
    host_token_span_u16: Vec<u16>,
    host_token_ring_u16: Vec<u16>,
    token_ring_base_step: Option<usize>,
    token_ring_steps_loaded: usize,
    device_batch_ready: bool,
    gpu_buf: pg_model::gpu::GpuActivations,
    gpu_grads: pg_model::gpu::GpuGradBuffers,
    gpu_optimizer: GpuOptimizer,
    state_tok_emb: GpuAdamWState,
    state_bigram_embed: GpuAdamWState,
    state_bigram_proj: GpuAdamWState,
    state_smear_gate: GpuAdamWState,
    state_skip_weights: GpuAdamWState,
    state_ve_embed: GpuAdamWState,
    state_ve_proj: GpuAdamWState,
    state_ve_layer_scales: GpuAdamWState,
    state_attn_scale: Vec<GpuAdamWState>,
    state_mlp_scale: Vec<GpuAdamWState>,
    state_resid_mix: Vec<GpuAdamWState>,
    state_q_gain: Vec<GpuAdamWState>,
    state_attn_gate_weight: Vec<GpuAdamWState>,
    state_attn_gate_bias: Vec<GpuAdamWState>,
    state_sparse_attn_gate_weight: Vec<GpuAdamWState>,
    state_bigram_scale: GpuAdamWState,
    state_ve_scale: GpuAdamWState,
    grad_norm_scratch: pg_core::GpuTensor,
    gpu_muon: GpuMuon,
}

#[cfg(feature = "cuda")]
impl CudaSingleFastRuntime {
    fn new(
        model: &GptModel,
        plan: &ExecutionPlan,
        tokens: usize,
        train_config: &pg_model::TrainConfig,
    ) -> PgResult<Self> {
        Self::new_on_device(model, plan, tokens, train_config, 0)
    }

    fn new_on_device(
        model: &GptModel,
        plan: &ExecutionPlan,
        tokens: usize,
        train_config: &pg_model::TrainConfig,
        device_ordinal: usize,
    ) -> PgResult<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_ordinal).map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda context init failed: {:?}", e))
        })?;
        if cuda_backward_graph_enabled() {
            // cudarc's allocation event tracking injects stream waits in
            // `DevicePtr::device_ptr`. Those waits are not part of the capture
            // and CUDA rejects the first captured kernel with
            // STREAM_CAPTURE_ISOLATION. The record graph path already uses
            // explicit stream ordering, so allocate graph-mode tensors without
            // cudarc's implicit per-pointer event bookkeeping.
            unsafe {
                ctx.disable_event_tracking();
            }
        }
        let stream = ctx.new_stream().map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda stream init failed: {:?}", e))
        })?;
        let gpu_model =
            pg_model::gpu::GpuModel::from_cpu_reference(model, plan, ctx, stream.clone())?;
        let gpu_buf = pg_model::gpu::GpuActivations::new_for_plan(plan, tokens, stream.clone())?;
        let gpu_grads = pg_model::gpu::GpuGradBuffers::new(&model.config, stream.clone())?;
        let n = model.config.num_layers;
        let d = model.config.model_dim;
        let kv = model.config.kv_dim();
        let mlp = model.config.mlp_dim;
        let bank_shapes = vec![[2 * n, d, d], [2 * n, kv, d], [n, mlp, d], [n, d, mlp]];

        Ok(Self {
            backward_state: pg_model::gpu::GpuBackwardState::new_for_plan(
                plan,
                tokens,
                stream.clone(),
            )?,
            backward_graph: None,
            backward_graph_seq_len: 0,
            backward_graph_warmup_steps_done: 0,
            input_ids: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens],
                pg_core::DType::U32,
            )?,
            targets: pg_core::GpuTensor::zeros_gpu(stream.clone(), &[tokens], pg_core::DType::U32)?,
            token_span_u16: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens + 1],
                pg_core::DType::U16,
            )?,
            token_ring_u16: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[(tokens + 1) * gpu_token_ring_sampler_alloc_steps_for_runtime()],
                pg_core::DType::U16,
            )?,
            host_input_ids: Vec::with_capacity(tokens),
            host_targets: Vec::with_capacity(tokens),
            host_token_span_u16: Vec::with_capacity(tokens + 1),
            host_token_ring_u16: Vec::with_capacity(
                (tokens + 1) * gpu_token_ring_sampler_alloc_steps_for_runtime(),
            ),
            token_ring_base_step: None,
            token_ring_steps_loaded: 0,
            device_batch_ready: false,
            state_tok_emb: GpuAdamWState::new_like(&gpu_model.weights.tok_emb, stream.clone())?,
            state_bigram_embed: GpuAdamWState::new_like(
                &gpu_model.weights.bigram_embed,
                stream.clone(),
            )?,
            state_bigram_proj: GpuAdamWState::new_like(
                &gpu_model.weights.bigram_proj,
                stream.clone(),
            )?,
            state_smear_gate: GpuAdamWState::new_like(
                &gpu_model.weights.smear_gate,
                stream.clone(),
            )?,
            state_skip_weights: GpuAdamWState::new_like(
                &gpu_model.weights.skip_weights,
                stream.clone(),
            )?,
            state_ve_embed: GpuAdamWState::new_like(&gpu_model.weights.ve_embed, stream.clone())?,
            state_ve_proj: GpuAdamWState::new_like(&gpu_model.weights.ve_proj, stream.clone())?,
            state_ve_layer_scales: GpuAdamWState::new_like(
                &gpu_model.weights.ve_layer_scales,
                stream.clone(),
            )?,
            state_attn_scale: gpu_model
                .weights
                .attn_scales
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_mlp_scale: gpu_model
                .weights
                .mlp_scales
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_resid_mix: gpu_model
                .weights
                .resid_mix
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_q_gain: gpu_model
                .weights
                .q_gains
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_attn_gate_weight: gpu_model
                .weights
                .attn_gate_weights
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_attn_gate_bias: gpu_model
                .weights
                .attn_gate_biases
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_sparse_attn_gate_weight: gpu_model
                .weights
                .sparse_attn_gate_weights
                .iter()
                .map(|t| GpuAdamWState::new_like(t, stream.clone()))
                .collect::<PgResult<_>>()?,
            state_bigram_scale: GpuAdamWState::new_like(
                &gpu_model.weights.bigram_scale_param,
                stream.clone(),
            )?,
            state_ve_scale: GpuAdamWState::new_like(
                &gpu_model.weights.ve_scale_param,
                stream.clone(),
            )?,
            grad_norm_scratch: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[1],
                pg_core::DType::F32,
            )?,
            gpu_muon: GpuMuon::new(
                stream,
                train_config.matrix_lr,
                train_config.muon_momentum,
                train_config.newton_schulz_steps,
                true,
                train_config.muon_wd,
                &bank_shapes,
            )?,
            gpu_model,
            gpu_buf,
            gpu_grads,
            gpu_optimizer: GpuOptimizer::new(),
        })
    }
}

#[cfg(feature = "cuda")]
fn set_cuda_runtime_recurrence_active(runtime: &mut CudaSingleFastRuntime, active: bool) {
    if runtime.gpu_model.recurrence_active() == active {
        return;
    }
    runtime.gpu_model.set_recurrence_active(active);
    runtime.backward_graph = None;
    runtime.backward_graph_seq_len = 0;
    runtime.backward_graph_warmup_steps_done = 0;
}

#[cfg(feature = "cuda")]
struct CudaDistributedRuntime {
    replicas: Vec<CudaSingleFastRuntime>,
    comms: Vec<pg_core::nccl::NcclComm>,
    /// One side stream per replica, used to issue post-backward NCCL
    /// collectives away from the main GEMM stream when side-stream collectives
    /// are enabled. Always allocated so future bucket-fire-during-backward
    /// callers can record events on these streams without per-step setup.
    comm_side_streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>>,
    /// Reusable cross-stream events for main↔side NCCL handoffs. These are
    /// created once with timing disabled; recording them per boundary avoids
    /// allocating CUDA events inside the training step.
    comm_side_events: Vec<NcclSideStreamEvents>,
    /// Second NCCL communicator pool, one per replica, bound to `comm_side_streams`.
    /// Populated only when side-stream collectives are enabled. Built via a
    /// second `Comm::from_devices(...)` call — NCCL supports multiple
    /// concurrent communicators per device, so the main-stream comms in
    /// `comms` and these side-stream comms operate as independent channels on
    /// the same NVLink
    /// fabric.
    comm_side_comms: Option<Vec<pg_core::nccl::NcclComm>>,
    parallel_muon: Option<ShardedParallelMuonRuntime>,
    all_grad_sync: Vec<AllGradSyncBuffers>,
    non_bank_sync: Vec<NonBankSyncBuffers>,
    distributed_sync: bool,
}

#[cfg(feature = "cuda")]
struct NcclSideStreamEvents {
    main_to_side: cudarc::driver::CudaEvent,
    side_to_main: cudarc::driver::CudaEvent,
}

#[cfg(feature = "cuda")]
struct ShardedParallelMuonRuntime {
    replicas: Vec<ShardedParallelMuonReplica>,
}

#[cfg(feature = "cuda")]
struct ShardedParallelMuonReplica {
    banks: Vec<ShardedBankBuffers>,
    muon: GpuMuon,
    pre_norm_graph: Option<CudaMuonLocalGraph>,
    pre_norm_graph_warmup_steps_done: usize,
    local_update_graph: Option<CudaMuonLocalGraph>,
    local_update_graph_warmup_steps_done: usize,
}

#[cfg(feature = "cuda")]
struct ShardedBankBuffers {
    padded_grad: pg_core::GpuTensor,
    padded_grad_bf16: pg_core::GpuTensor,
    shard_grad: pg_core::GpuTensor,
    shard_grad_bf16: pg_core::GpuTensor,
    shard_param: pg_core::GpuTensor,
    shard_param_bf16: pg_core::GpuTensor,
    padded_param: pg_core::GpuTensor,
    padded_param_bf16: pg_core::GpuTensor,
    real_batch: usize,
    chunk_batch: usize,
}

#[cfg(feature = "cuda")]
struct AllGradSyncBuffers {
    packed_grad: pg_core::GpuTensor,
}

#[cfg(feature = "cuda")]
struct NonBankSyncBuffers {
    packed_grad: pg_core::GpuTensor,
}

#[cfg(feature = "cuda")]
fn sharded_bank_real_batch(buffers: &ShardedBankBuffers, rank: usize) -> usize {
    let shard_start = rank * buffers.chunk_batch;
    let shard_end = (shard_start + buffers.chunk_batch).min(buffers.real_batch);
    shard_end.saturating_sub(shard_start)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn sharded_parallel_muon_local_rank_step(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    grad_norm_scratch: &pg_core::GpuTensor,
    sharded_replica: &mut ShardedParallelMuonReplica,
    rank: usize,
    bank_count: usize,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
    apply_global_clip: bool,
    fused_global_clip: bool,
    bf16_shadow_all_gather: bool,
    use_device_hyper: bool,
) -> PgResult<()> {
    for bank_idx in 0..bank_count {
        let buffers = &mut sharded_replica.banks[bank_idx];
        let active_batch = sharded_bank_real_batch(buffers, rank);
        if active_batch == 0 {
            continue;
        }
        let shard_param = buffers.shard_param.slice_range(0, active_batch)?;
        let shard_param_bf16 = if bf16_shadow_all_gather {
            Some(buffers.shard_param_bf16.slice_range(0, active_batch)?)
        } else {
            None
        };
        let shard_grad = buffers.shard_grad.slice_range(0, active_batch)?;
        sharded_replica.muon.lr = train_config.matrix_lr * lr_scale;
        sharded_replica.muon.momentum = train_config.muon_momentum_at(step);
        sharded_replica.muon.weight_decay = train_config.muon_wd;
        if apply_global_clip && fused_global_clip {
            if let Some(shard_param_bf16) = shard_param_bf16.as_ref() {
                if use_device_hyper {
                    sharded_replica
                        .muon
                        .step_bank_with_global_clip_bf16_shadow_hyper(
                            kernels,
                            bank_idx,
                            &shard_param,
                            shard_param_bf16,
                            &shard_grad,
                            Some(grad_norm_scratch),
                            train_config.grad_clip_norm,
                        )?;
                } else {
                    sharded_replica
                        .muon
                        .step_bank_with_global_clip_bf16_shadow(
                            kernels,
                            bank_idx,
                            &shard_param,
                            shard_param_bf16,
                            &shard_grad,
                            Some(grad_norm_scratch),
                            train_config.grad_clip_norm,
                        )?;
                }
            } else {
                if use_device_hyper {
                    sharded_replica.muon.step_bank_with_global_clip_hyper(
                        kernels,
                        bank_idx,
                        &shard_param,
                        &shard_grad,
                        Some(grad_norm_scratch),
                        train_config.grad_clip_norm,
                    )?;
                } else {
                    sharded_replica.muon.step_bank_with_global_clip(
                        kernels,
                        bank_idx,
                        &shard_param,
                        &shard_grad,
                        Some(grad_norm_scratch),
                        train_config.grad_clip_norm,
                    )?;
                }
            }
        } else if let Some(shard_param_bf16) = shard_param_bf16.as_ref() {
            if use_device_hyper {
                sharded_replica
                    .muon
                    .step_bank_with_global_clip_bf16_shadow_hyper(
                        kernels,
                        bank_idx,
                        &shard_param,
                        shard_param_bf16,
                        &shard_grad,
                        None,
                        0.0,
                    )?;
            } else {
                sharded_replica
                    .muon
                    .step_bank_with_global_clip_bf16_shadow(
                        kernels,
                        bank_idx,
                        &shard_param,
                        shard_param_bf16,
                        &shard_grad,
                        None,
                        0.0,
                    )?;
            }
        } else {
            if use_device_hyper {
                sharded_replica.muon.step_bank_with_global_clip_hyper(
                    kernels,
                    bank_idx,
                    &shard_param,
                    &shard_grad,
                    None,
                    0.0,
                )?;
            } else {
                sharded_replica
                    .muon
                    .step_bank(kernels, bank_idx, &shard_param, &shard_grad)?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn sharded_parallel_muon_local_rank_step_graph_or_eager(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    grad_norm_scratch: &pg_core::GpuTensor,
    sharded_replica: &mut ShardedParallelMuonReplica,
    rank: usize,
    bank_count: usize,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
    apply_global_clip: bool,
    fused_global_clip: bool,
    bf16_shadow_all_gather: bool,
) -> PgResult<()> {
    let graph_enabled = sharded_parallel_muon_local_graph_enabled();
    if !graph_enabled {
        return sharded_parallel_muon_local_rank_step(
            kernels,
            grad_norm_scratch,
            sharded_replica,
            rank,
            bank_count,
            train_config,
            step,
            lr_scale,
            apply_global_clip,
            fused_global_clip,
            bf16_shadow_all_gather,
            false,
        );
    }

    let lr = train_config.matrix_lr * lr_scale;
    let momentum = train_config.muon_momentum_at(step);
    sharded_replica
        .muon
        .update_hyper_device(lr, momentum, train_config.muon_wd)?;

    if let Some(graph) = sharded_replica.local_update_graph.as_ref() {
        graph.0.launch().map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon local graph launch failed: {e:?}"
            ))
        })?;
        return Ok(());
    }

    let stream = kernels.stream().clone();
    let required_warmup = sharded_parallel_muon_local_graph_warmup_steps();
    if sharded_replica.local_update_graph_warmup_steps_done < required_warmup {
        sharded_parallel_muon_local_rank_step(
            kernels,
            grad_norm_scratch,
            sharded_replica,
            rank,
            bank_count,
            train_config,
            step,
            lr_scale,
            apply_global_clip,
            fused_global_clip,
            bf16_shadow_all_gather,
            true,
        )?;
        stream.synchronize().map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon local graph warmup sync failed: {e:?}"
            ))
        })?;
        sharded_replica.local_update_graph_warmup_steps_done += 1;
        return Ok(());
    }

    stream.synchronize().map_err(|e| {
        pg_core::PgError::InvalidOp(format!(
            "sharded Parallel Muon local graph pre-capture sync failed: {e:?}"
        ))
    })?;
    stream
        .begin_capture(cuda_graph_capture_mode())
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon local graph capture begin failed: {e:?}"
            ))
        })?;
    let capture_result = sharded_parallel_muon_local_rank_step(
        kernels,
        grad_norm_scratch,
        sharded_replica,
        rank,
        bank_count,
        train_config,
        step,
        lr_scale,
        apply_global_clip,
        fused_global_clip,
        bf16_shadow_all_gather,
        true,
    );
    if let Err(err) = capture_result {
        let _ = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        return Err(err);
    }
    let graph = stream
        .end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon local graph capture end failed: {e:?}"
            ))
        })?;
    sharded_replica.local_update_graph = graph.map(CudaMuonLocalGraph);
    sharded_replica
        .local_update_graph
        .as_ref()
        .ok_or_else(|| {
            pg_core::PgError::InvalidOp(
                "sharded Parallel Muon local graph capture produced no graph".into(),
            )
        })?
        .0
        .launch()
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon local graph launch failed: {e:?}"
            ))
        })?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn sharded_parallel_muon_pre_norm_rank_step(
    replica: &mut CudaSingleFastRuntime,
    non_bank_packed: &pg_core::GpuTensor,
    sharded_replica: &ShardedParallelMuonReplica,
    rank: usize,
    world_size: usize,
    inv_total: f32,
    apply_global_clip: bool,
) -> PgResult<()> {
    let kernels = &replica.gpu_model.kernels;
    if apply_global_clip {
        scale_gpu_tensor(kernels, &replica.grad_norm_scratch, 0.0)?;
        let scratch =
            pg_kernels::gpu_kernels::CudaPtr(replica.grad_norm_scratch.cu_ptr(kernels.stream())?);
        let non_bank_alpha = 1.0f32 / world_size as f32;
        kernels.scale_square_accumulate(
            pg_kernels::gpu_kernels::CudaPtr(non_bank_packed.cu_ptr(kernels.stream())?),
            scratch,
            inv_total,
            non_bank_alpha,
            non_bank_packed.numel() as u32,
        )?;
        for buffers in &sharded_replica.banks {
            let active_batch = sharded_bank_real_batch(buffers, rank);
            if active_batch == 0 {
                continue;
            }
            let shard_grad = buffers.shard_grad.slice_range(0, active_batch)?;
            if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                let shard_grad_bf16 = buffers.shard_grad_bf16.slice_range(0, active_batch)?;
                kernels.bf16_to_f32_scale_square_accumulate(
                    pg_kernels::gpu_kernels::CudaPtr(shard_grad_bf16.cu_ptr(kernels.stream())?),
                    pg_kernels::gpu_kernels::CudaPtr(shard_grad.cu_ptr(kernels.stream())?),
                    scratch,
                    inv_total,
                    1.0,
                    shard_grad.numel() as u32,
                )?;
            } else {
                kernels.scale_square_accumulate(
                    pg_kernels::gpu_kernels::CudaPtr(shard_grad.cu_ptr(kernels.stream())?),
                    scratch,
                    inv_total,
                    1.0,
                    shard_grad.numel() as u32,
                )?;
            }
        }
    } else {
        scale_gpu_tensor(kernels, non_bank_packed, inv_total)?;
        for buffers in &sharded_replica.banks {
            let active_batch = sharded_bank_real_batch(buffers, rank);
            if active_batch == 0 {
                continue;
            }
            let shard_grad = buffers.shard_grad.slice_range(0, active_batch)?;
            if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                let shard_grad_bf16 = buffers.shard_grad_bf16.slice_range(0, active_batch)?;
                kernels.bf16_to_f32(
                    pg_kernels::gpu_kernels::CudaPtr(shard_grad_bf16.cu_ptr(kernels.stream())?),
                    pg_kernels::gpu_kernels::CudaPtr(shard_grad.cu_ptr(kernels.stream())?),
                    shard_grad.numel() as u32,
                )?;
            }
            scale_gpu_tensor(kernels, &shard_grad, inv_total)?;
        }
    }

    Ok(())
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn sharded_parallel_muon_pre_norm_rank_step_graph_or_eager(
    replica: &mut CudaSingleFastRuntime,
    non_bank_packed: &pg_core::GpuTensor,
    sharded_replica: &mut ShardedParallelMuonReplica,
    rank: usize,
    world_size: usize,
    inv_total: f32,
    apply_global_clip: bool,
) -> PgResult<()> {
    let graph_enabled = sharded_parallel_muon_pre_norm_graph_enabled();
    if !graph_enabled {
        return sharded_parallel_muon_pre_norm_rank_step(
            replica,
            non_bank_packed,
            sharded_replica,
            rank,
            world_size,
            inv_total,
            apply_global_clip,
        );
    }

    if let Some(graph) = sharded_replica.pre_norm_graph.as_ref() {
        graph.0.launch().map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon pre-norm graph launch failed: {e:?}"
            ))
        })?;
        return Ok(());
    }

    let stream = replica.gpu_model.kernels.stream().clone();
    let required_warmup = sharded_parallel_muon_pre_norm_graph_warmup_steps();
    if sharded_replica.pre_norm_graph_warmup_steps_done < required_warmup {
        sharded_parallel_muon_pre_norm_rank_step(
            replica,
            non_bank_packed,
            sharded_replica,
            rank,
            world_size,
            inv_total,
            apply_global_clip,
        )?;
        stream.synchronize().map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon pre-norm graph warmup sync failed: {e:?}"
            ))
        })?;
        sharded_replica.pre_norm_graph_warmup_steps_done += 1;
        return Ok(());
    }

    stream.synchronize().map_err(|e| {
        pg_core::PgError::InvalidOp(format!(
            "sharded Parallel Muon pre-norm graph pre-capture sync failed: {e:?}"
        ))
    })?;
    stream
        .begin_capture(cuda_graph_capture_mode())
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon pre-norm graph capture begin failed: {e:?}"
            ))
        })?;
    let capture_result = sharded_parallel_muon_pre_norm_rank_step(
        replica,
        non_bank_packed,
        sharded_replica,
        rank,
        world_size,
        inv_total,
        apply_global_clip,
    );
    if let Err(err) = capture_result {
        let _ = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        return Err(err);
    }
    let graph = stream
        .end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon pre-norm graph capture end failed: {e:?}"
            ))
        })?;
    sharded_replica.pre_norm_graph = graph.map(CudaMuonLocalGraph);
    sharded_replica
        .pre_norm_graph
        .as_ref()
        .ok_or_else(|| {
            pg_core::PgError::InvalidOp(
                "sharded Parallel Muon pre-norm graph capture produced no graph".into(),
            )
        })?
        .0
        .launch()
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "sharded Parallel Muon pre-norm graph launch failed: {e:?}"
            ))
        })?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn sharded_bank_owner(
    buffers: &ShardedBankBuffers,
    matrix_index: usize,
) -> PgResult<(usize, usize)> {
    if matrix_index >= buffers.real_batch {
        return Err(pg_core::PgError::InvalidOp(format!(
            "bank matrix index {matrix_index} >= real batch {}",
            buffers.real_batch
        )));
    }
    Ok((
        matrix_index / buffers.chunk_batch.max(1),
        matrix_index % buffers.chunk_batch.max(1),
    ))
}

#[cfg(feature = "cuda")]
impl CudaDistributedRuntime {
    fn new(
        model: &GptModel,
        plan: &ExecutionPlan,
        tokens: usize,
        train_config: &pg_model::TrainConfig,
        world_size: usize,
    ) -> PgResult<Self> {
        let device_count = cudarc::driver::CudaContext::device_count().map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda device_count failed: {:?}", e))
        })? as usize;
        if world_size > device_count {
            return Err(pg_core::PgError::InvalidOp(format!(
                "cuda-distributed requested world_size {} but only {} local CUDA devices are visible",
                world_size, device_count
            )));
        }
        let replicas = (0..world_size)
            .map(|ordinal| {
                CudaSingleFastRuntime::new_on_device(model, plan, tokens, train_config, ordinal)
            })
            .collect::<PgResult<Vec<_>>>()?;
        let n = model.config.num_layers;
        let d = model.config.model_dim;
        let kv = model.config.kv_dim();
        let mlp = model.config.mlp_dim;
        let bank_shapes = vec![[2 * n, d, d], [2 * n, kv, d], [n, mlp, d], [n, d, mlp]];
        let streams = replicas
            .iter()
            .map(|runtime| runtime.gpu_model.gemm.stream().clone())
            .collect::<Vec<_>>();
        let comms = pg_core::nccl::NcclComm::from_local_devices(streams)?;

        // Allocate one side stream per replica. Cheap (a side stream is just a
        // CUDA stream handle); used for cross-stream sync once side-stream
        // collectives are enabled. We allocate unconditionally so the field is always available
        // for future per-bucket scheduling without per-step setup overhead.
        let comm_side_streams = replicas
            .iter()
            .map(|runtime| {
                runtime
                    .gpu_model
                    .gemm
                    .stream()
                    .context()
                    .new_stream()
                    .map_err(|e| {
                        pg_core::PgError::InvalidOp(format!("comm side stream init failed: {e:?}"))
                    })
            })
            .collect::<PgResult<Vec<_>>>()?;
        let comm_side_events = comm_side_streams
            .iter()
            .map(|stream| {
                let context = stream.context();
                Ok(NcclSideStreamEvents {
                    main_to_side: context.new_event(None).map_err(|e| {
                        pg_core::PgError::InvalidOp(format!(
                            "comm side main->side event init failed: {e:?}"
                        ))
                    })?,
                    side_to_main: context.new_event(None).map_err(|e| {
                        pg_core::PgError::InvalidOp(format!(
                            "comm side side->main event init failed: {e:?}"
                        ))
                    })?,
                })
            })
            .collect::<PgResult<Vec<_>>>()?;
        // Only build the side NCCL comm pool when side-stream collectives are
        // requested.
        // Building two NCCL pools per device costs initialization time and a
        // second set of NCCL channel buffers; we don't pay that unless enabled.
        let comm_side_comms = if nccl_side_stream_collectives_enabled() {
            Some(pg_core::nccl::NcclComm::from_local_devices(
                comm_side_streams.clone(),
            )?)
        } else {
            None
        };
        let all_grad_sync = if plan.run_spec.train.distributed_optimizer_backend
            == DistributedOptimizerBackend::AllReduceReplicatedMuon
        {
            replicas
                .iter()
                .map(|replica| {
                    let len = all_grad_numel(&replica.gpu_grads);
                    Ok(AllGradSyncBuffers {
                        packed_grad: pg_core::GpuTensor::zeros_gpu(
                            replica.gpu_model.gemm.stream().clone(),
                            &[len],
                            pg_core::DType::F32,
                        )?,
                    })
                })
                .collect::<PgResult<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let non_bank_sync = replicas
            .iter()
            .map(|replica| {
                let len = non_bank_grad_numel(&replica.gpu_grads);
                Ok(NonBankSyncBuffers {
                    packed_grad: pg_core::GpuTensor::zeros_gpu(
                        replica.gpu_model.gemm.stream().clone(),
                        &[len],
                        pg_core::DType::F32,
                    )?,
                })
            })
            .collect::<PgResult<Vec<_>>>()?;
        let parallel_muon = if plan.run_spec.train.distributed_optimizer_backend
            == DistributedOptimizerBackend::ShardedParallelMuon
        {
            Some(ShardedParallelMuonRuntime::new(
                &replicas,
                train_config,
                &bank_shapes,
                world_size,
            )?)
        } else {
            None
        };
        Ok(Self {
            replicas,
            comms,
            comm_side_streams,
            comm_side_events,
            comm_side_comms,
            parallel_muon,
            all_grad_sync,
            non_bank_sync,
            distributed_sync: false,
        })
    }
}

#[cfg(feature = "cuda")]
impl ShardedParallelMuonRuntime {
    fn new(
        replicas: &[CudaSingleFastRuntime],
        train_config: &pg_model::TrainConfig,
        bank_shapes: &[[usize; 3]],
        world_size: usize,
    ) -> PgResult<Self> {
        let replicas = replicas
            .iter()
            .map(|replica| {
                let stream = replica.gpu_model.gemm.stream().clone();
                let mut shard_shapes = Vec::with_capacity(bank_shapes.len());
                let mut banks = Vec::with_capacity(bank_shapes.len());
                for &[batch, rows, cols] in bank_shapes {
                    let chunk_batch = batch.div_ceil(world_size);
                    let padded_batch = chunk_batch * world_size;
                    let padded_shape = [padded_batch, rows, cols];
                    let shard_shape = [chunk_batch, rows, cols];
                    shard_shapes.push(shard_shape);
                    banks.push(ShardedBankBuffers {
                        padded_grad: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &padded_shape,
                            pg_core::DType::F32,
                        )?,
                        padded_grad_bf16: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &padded_shape,
                            pg_core::DType::BF16,
                        )?,
                        shard_grad: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &shard_shape,
                            pg_core::DType::F32,
                        )?,
                        shard_grad_bf16: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &shard_shape,
                            pg_core::DType::BF16,
                        )?,
                        shard_param: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &shard_shape,
                            pg_core::DType::F32,
                        )?,
                        shard_param_bf16: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &shard_shape,
                            pg_core::DType::BF16,
                        )?,
                        padded_param: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &padded_shape,
                            pg_core::DType::F32,
                        )?,
                        padded_param_bf16: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &padded_shape,
                            pg_core::DType::BF16,
                        )?,
                        real_batch: batch,
                        chunk_batch,
                    });
                }
                Ok(ShardedParallelMuonReplica {
                    banks,
                    muon: GpuMuon::new(
                        stream,
                        train_config.matrix_lr,
                        train_config.muon_momentum,
                        train_config.newton_schulz_steps,
                        true,
                        train_config.muon_wd,
                        &shard_shapes,
                    )?,
                    pre_norm_graph: None,
                    pre_norm_graph_warmup_steps_done: 0,
                    local_update_graph: None,
                    local_update_graph_warmup_steps_done: 0,
                })
            })
            .collect::<PgResult<Vec<_>>>()?;
        Ok(Self { replicas })
    }
}

#[cfg(feature = "cuda")]
fn record_replica_events(
    runtime: &CudaDistributedRuntime,
) -> PgResult<Vec<cudarc::driver::CudaEvent>> {
    record_replica_events_for(&runtime.replicas)
}

#[cfg(feature = "cuda")]
fn record_replica_events_for(
    replicas: &[CudaSingleFastRuntime],
) -> PgResult<Vec<cudarc::driver::CudaEvent>> {
    replicas
        .iter()
        .map(|replica| {
            replica
                .gpu_model
                .gemm
                .stream()
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| {
                    pg_core::PgError::InvalidOp(format!("cuda event record failed: {e:?}"))
                })
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn max_elapsed_replica_events_ms(
    starts: Vec<cudarc::driver::CudaEvent>,
    ends: Vec<cudarc::driver::CudaEvent>,
) -> PgResult<f64> {
    if starts.len() != ends.len() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "cuda event timing mismatch: {} starts, {} ends",
            starts.len(),
            ends.len()
        )));
    }
    let mut max_ms = 0.0f64;
    for (start, end) in starts.iter().zip(ends.iter()) {
        let elapsed = start
            .elapsed_ms(end)
            .map_err(|e| pg_core::PgError::InvalidOp(format!("cuda event elapsed failed: {e:?}")))?
            as f64;
        max_ms = max_ms.max(elapsed);
    }
    Ok(max_ms)
}

#[cfg(feature = "cuda")]
fn record_sharded_muon_phase_ms(
    replicas: &[CudaSingleFastRuntime],
    phase_start_events: &mut Option<Vec<cudarc::driver::CudaEvent>>,
    phase_start_host: &mut Option<Instant>,
    timing: &mut RunTiming,
    field: fn(&mut RunTiming) -> &mut f64,
) -> PgResult<()> {
    if let Some(starts) = phase_start_events.take() {
        let ends = record_replica_events_for(replicas)?;
        *field(timing) += max_elapsed_replica_events_ms(starts, ends)?;
        *phase_start_events = Some(record_replica_events_for(replicas)?);
    } else if let Some(start) = phase_start_host.as_mut() {
        let now = Instant::now();
        *field(timing) += now.duration_since(*start).as_secs_f64() * 1000.0;
        *start = now;
    }
    Ok(())
}

impl VariantRunner {
    pub fn new(run_spec: RunSpec) -> PgResult<Self> {
        let plan = ExecutionPlan::from_run_spec(&run_spec)?;
        Ok(Self { run_spec, plan })
    }

    pub fn run(&self, mode: RunMode) -> PgResult<VariantResult> {
        let model_config = self.run_spec.model.to_model_config();
        let train_config = self.run_spec.train.to_train_config();
        apply_runtime_profile_env(&self.run_spec, mode)?;
        validate_backend_request(&self.run_spec, mode)?;
        validate_executable_variant(&self.run_spec, mode)?;
        let mut model = GptModel::new(model_config.clone());
        model.fill_deterministic();
        let world_size = self.run_spec.train.world_size.max(1);
        let rank = self.run_spec.train.rank;
        if rank >= world_size {
            return Err(pg_core::PgError::InvalidOp(format!(
                "rank {} must be < world_size {}",
                rank, world_size
            )));
        }
        if world_size > 1
            && !matches!(mode, RunMode::Smoke)
            && self.run_spec.train.backend != TrainBackend::CudaDistributed
        {
            return Err(pg_core::PgError::InvalidOp(
                "multi-rank proxy/record training requires backend=cuda-distributed; CPU only supports distributed smoke/data-shard preflight".into(),
            ));
        }
        let batch_plan = step_batch_plan(&self.run_spec, mode, &model_config, world_size)?;
        let active_tokens = match self.run_spec.train.backend {
            TrainBackend::CudaSingle | TrainBackend::CudaDistributed => {
                batch_plan.microbatch_tokens * batch_plan.local_microbatches_per_step
            }
            TrainBackend::Cpu | TrainBackend::CudaSingleParity => batch_plan.microbatch_tokens,
        };
        let mut buf = ForwardBuffer::new(&model_config, active_tokens);
        let mut grads = GradBuffers::new(&model_config);
        let mut data_loader = if self.run_spec.train.backend == TrainBackend::CudaDistributed {
            None
        } else if let Some(pattern) = self.run_spec.train.train_data_pattern.as_deref() {
            Some(DistributedTokenLoader::new(pattern, rank, world_size)?)
        } else {
            None
        };
        let mut distributed_data_loaders = if self.run_spec.train.backend
            == TrainBackend::CudaDistributed
        {
            if let Some(pattern) = self.run_spec.train.train_data_pattern.as_deref() {
                Some(
                    (0..world_size)
                        .map(|rank_idx| DistributedTokenLoader::new(pattern, rank_idx, world_size))
                        .collect::<PgResult<Vec<_>>>()?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let train_data_source = if data_loader.is_some() {
            "shards"
        } else if self.run_spec.train.backend == TrainBackend::CudaDistributed
            && self.run_spec.train.train_data_pattern.is_some()
        {
            "shards"
        } else {
            "synthetic_sequence"
        };

        let n = model_config.num_layers;
        let d = model_config.model_dim;
        let kv = model_config.kv_dim();
        let mlp = model_config.mlp_dim;
        let bank_shapes: Vec<[usize; 3]> =
            vec![[2 * n, d, d], [2 * n, kv, d], [n, mlp, d], [n, d, mlp]];

        let mut muon = Muon::new(
            train_config.matrix_lr,
            train_config.muon_momentum,
            train_config.newton_schulz_steps,
            true,
            train_config.muon_wd,
            &bank_shapes,
        );
        let mut adamw_embed = AdamW::new(
            train_config.embed_lr,
            train_config.adam_beta1,
            train_config.adam_beta2,
            train_config.adam_eps,
            train_config.adam_wd,
        );
        let mut adamw_scalar = AdamW::new(
            train_config.scalar_lr,
            train_config.adam_beta1,
            train_config.adam_beta2,
            train_config.adam_eps,
            train_config.adam_wd,
        );

        let mut state_tok_emb = AdamWState::new(model.tok_emb.len());
        let mut state_bigram_embed = AdamWState::new(model.bigram_embed.len());
        let mut state_bigram_proj = AdamWState::new(model.bigram_proj.len());
        let mut state_smear_gate = AdamWState::new(model.smear_gate.len());
        let mut state_skip_weights = AdamWState::new(model.skip_weights.len());
        let mut state_ve_embed = AdamWState::new(model.ve_embed.len());
        let mut state_ve_proj = AdamWState::new(model.ve_proj.len());
        let mut state_ve_scale = AdamWState::new(1);
        let mut state_ve_layer_scales = AdamWState::new(model.ve_layer_scales.len());
        let mut state_bigram_scale = AdamWState::new(1);
        let mut state_attn_scale: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(d)).collect();
        let mut state_mlp_scale: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(d)).collect();
        let mut state_resid_mix: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(2 * d)).collect();
        let mut state_q_gain: Vec<AdamWState> = (0..n)
            .map(|_| AdamWState::new(model_config.num_heads))
            .collect();
        let mut state_attn_gate_weight: Vec<AdamWState> = (0..n)
            .map(|_| {
                AdamWState::new(model_config.num_heads * model_config.attn_out_gate_width.max(1))
            })
            .collect();
        let mut state_attn_gate_bias: Vec<AdamWState> = (0..n)
            .map(|_| AdamWState::new(model_config.num_heads))
            .collect();
        let mut state_sparse_attn_gate_weight: Vec<AdamWState> = (0..n)
            .map(|_| {
                AdamWState::new(model_config.num_heads * model_config.sparse_attn_gate_width.max(1))
            })
            .collect();

        let total_params = count_params(&model);
        let mut ema = Ema::new(train_config.ema_decay, total_params);
        let mut swa = Swa::new(total_params);
        let mut flat_buf = vec![0.0f32; total_params];

        let requested_steps = match mode {
            RunMode::Smoke => 4usize,
            RunMode::Proxy => 32usize,
            RunMode::RecordShapedProxy => {
                let proxy_cap = std::env::var("PG_RECORD_SHAPED_PROXY_MAX_STEPS")
                    .ok()
                    .and_then(|raw| raw.parse::<usize>().ok())
                    .filter(|&value| value > 0)
                    .unwrap_or(8);
                train_config.total_iterations.min(proxy_cap)
            }
            RunMode::Record => train_config.total_iterations,
        };
        let max_steps = requested_steps.min(train_config.total_iterations);
        let fast_bank_updates =
            matches!(mode, RunMode::Smoke) || self.run_spec.train.fast_bank_updates;
        let bank_update_backend = match self.run_spec.train.backend {
            TrainBackend::Cpu => {
                if fast_bank_updates {
                    "fast_sgd"
                } else {
                    "muon_ns5"
                }
            }
            TrainBackend::CudaSingle | TrainBackend::CudaSingleParity => {
                if self.run_spec.train.backend == TrainBackend::CudaSingleParity {
                    if fast_bank_updates {
                        "cpu_fast_sgd_mirror"
                    } else {
                        "cpu_muon_ns5_mirror"
                    }
                } else {
                    "gpu_muon_ns5"
                }
            }
            TrainBackend::CudaDistributed => {
                match self.run_spec.train.distributed_optimizer_backend {
                    DistributedOptimizerBackend::AllReduceReplicatedMuon => {
                        "nccl_allreduce_replicated_gpu_muon_ns5"
                    }
                    DistributedOptimizerBackend::ShardedParallelMuon => {
                        "nccl_reduce_scatter_all_gather_parallel_muon_ns5"
                    }
                }
            }
        };
        let frontier_record_ready = frontier_record_gaps(&self.run_spec).is_empty();
        let leaderboard_algorithm_gaps = leaderboard_algorithm_gaps(&self.run_spec);
        let leaderboard_algorithm_ready = leaderboard_algorithm_gaps.is_empty();
        let record_shape = is_record_shaped_mode(mode);
        let record_attention_grade = attention_backend_record_grade(&self.run_spec);
        let microbatch_serial_loop = matches!(
            self.run_spec.train.backend,
            TrainBackend::Cpu | TrainBackend::CudaSingleParity
        ) && batch_plan.local_microbatches_per_step > 1;
        println!(
            "record_audit_json={}",
            record_path_audit_json(
                &self.run_spec,
                mode,
                &batch_plan,
                world_size,
                frontier_record_ready,
                leaderboard_algorithm_ready,
                &leaderboard_algorithm_gaps,
                record_attention_grade,
                microbatch_serial_loop,
                bank_update_backend,
            )
        );
        if mode == RunMode::Record
            && (record_require_gpu_data_sampler_for_audit()
                || self.run_spec.runtime.require_device_batch)
            && !gpu_record_data_sampler_final_for_audit(&self.run_spec, mode)
        {
            return Err(pg_core::PgError::InvalidOp(
                "record mode requires a final GPU data sampler, but the current data path would use host-sampled per-step batches or host ring refills; enable PG_GPU_TOKEN_RING_SAMPLER=1, PG_GPU_TOKEN_RING_FULL_SCHEDULE=1, and a PG_GPU_TOKEN_RING_STEPS value covering the full record run".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        let mut cuda_single_parity_runtime =
            if self.run_spec.train.backend == TrainBackend::CudaSingleParity {
                Some(CudaSingleHybridRuntime::new(
                    &model, &self.plan, buf.tokens,
                )?)
            } else {
                None
            };
        #[cfg(feature = "cuda")]
        let mut cuda_single_fast_runtime =
            if self.run_spec.train.backend == TrainBackend::CudaSingle {
                Some(CudaSingleFastRuntime::new(
                    &model,
                    &self.plan,
                    buf.tokens,
                    &train_config,
                )?)
            } else {
                None
            };
        #[cfg(feature = "cuda")]
        let mut cuda_distributed_runtime =
            if self.run_spec.train.backend == TrainBackend::CudaDistributed {
                Some(CudaDistributedRuntime::new(
                    &model,
                    &self.plan,
                    buf.tokens,
                    &train_config,
                    world_size,
                )?)
            } else {
                None
            };

        let start = Instant::now();
        let mut timing = RunTiming::default();
        let timing_skip_steps = record_timing_skip_steps();
        let mut timing_steps = 0usize;
        let mut timing_measured_wallclock_ms = 0.0f64;
        let mut final_loss = 0.0f32;
        let mut steps_completed = 0usize;
        let mut last_input_ids = Vec::new();
        let mut last_targets = Vec::new();
        for step in 0..max_steps {
            let elapsed = start.elapsed().as_secs_f32();
            if elapsed > train_config.max_wallclock_seconds {
                break;
            }
            let active_batch_plan = step_batch_plan_for_train_step(
                &self.run_spec,
                mode,
                &model_config,
                world_size,
                step,
                max_steps,
            )?;
            let recurrence_active = active_recurrence_for_train_step(
                &self.run_spec,
                step,
                train_config.total_iterations,
                elapsed,
                train_config.max_wallclock_seconds,
            );
            #[cfg(feature = "cuda")]
            {
                if let Some(runtime) = cuda_single_fast_runtime.as_mut() {
                    set_cuda_runtime_recurrence_active(runtime, recurrence_active);
                }
                if let Some(runtime) = cuda_distributed_runtime.as_mut() {
                    for replica in runtime.replicas.iter_mut() {
                        set_cuda_runtime_recurrence_active(replica, recurrence_active);
                    }
                }
            }
            let lr_scale = scheduler::lr_scale_with_floor(
                step,
                train_config.warmup_steps,
                train_config.total_iterations,
                train_config.warmdown_iters,
                train_config.min_lr_scale,
            );
            muon.lr = train_config.matrix_lr * lr_scale;
            muon.momentum = train_config.muon_momentum_at(step);
            adamw_embed.lr = train_config.embed_lr * lr_scale;
            adamw_scalar.lr = train_config.scalar_lr * lr_scale;

            let measure_step = step >= timing_skip_steps;
            let step_wall_t0 = Instant::now();
            let data_t0 = Instant::now();
            #[cfg_attr(not(feature = "cuda"), allow(unused_mut, unused_variables))]
            let mut distributed_batches_preloaded = false;
            let synthetic_distributed_batches = |step_plan: StepBatchPlan| {
                (0..world_size)
                    .map(|rank_idx| {
                        let local_tokens =
                            step_plan.microbatch_tokens * step_plan.local_microbatches_per_step;
                        let mut x = Vec::with_capacity(local_tokens);
                        let mut y = Vec::with_capacity(local_tokens);
                        for micro_idx in 0..step_plan.local_microbatches_per_step {
                            let offset = step * step_plan.global_batch_tokens
                                + micro_idx * step_plan.microbatch_tokens * world_size
                                + rank_idx * step_plan.microbatch_tokens;
                            x.extend(
                                (0..step_plan.microbatch_tokens)
                                    .map(|i| ((offset + i) % model_config.vocab_size) as u32),
                            );
                            y.extend(
                                (1..=step_plan.microbatch_tokens)
                                    .map(|i| ((offset + i) % model_config.vocab_size) as u32),
                            );
                        }
                        vec![(x, y)]
                    })
                    .collect::<Vec<_>>()
            };
            let distributed_batches: Option<Vec<Vec<(Vec<u32>, Vec<u32>)>>> = if self
                .run_spec
                .train
                .backend
                == TrainBackend::CudaDistributed
            {
                if is_record_shaped_mode(mode) {
                    #[cfg(feature = "cuda")]
                    if self.run_spec.train.train_data_pattern.is_none()
                        && gpu_resident_synthetic_sampler_enabled_for_audit()
                    {
                        let runtime = cuda_distributed_runtime.as_mut().ok_or_else(|| {
                                pg_core::PgError::InvalidOp(
                                    "GPU-resident synthetic sampler requires initialized cuda-distributed runtime".into(),
                                )
                            })?;
                        cuda_distributed_generate_synthetic_record_batches(
                            runtime,
                            step,
                            &active_batch_plan,
                            model_config.vocab_size,
                        )?;
                        distributed_batches_preloaded = true;
                        None
                    } else if gpu_token_ring_sampler_enabled_for_audit()
                        && self.run_spec.train.train_data_pattern.is_some()
                    {
                        let loaders = distributed_data_loaders.as_mut().ok_or_else(|| {
                            pg_core::PgError::InvalidOp(
                                "GPU token-ring sampler requires distributed train-data loaders"
                                    .into(),
                            )
                        })?;
                        let runtime = cuda_distributed_runtime.as_mut().ok_or_else(|| {
                            pg_core::PgError::InvalidOp(
                                "GPU token-ring sampler requires initialized cuda-distributed runtime"
                                    .into(),
                            )
                        })?;
                        cuda_distributed_prepare_token_ring_batches(
                            runtime,
                            loaders,
                            step,
                            &active_batch_plan,
                        )?;
                        distributed_batches_preloaded = true;
                        None
                    } else if let (Some(loaders), Some(runtime)) = (
                        distributed_data_loaders.as_mut(),
                        cuda_distributed_runtime.as_mut(),
                    ) {
                        for (loader, replica) in loaders.iter_mut().zip(runtime.replicas.iter_mut())
                        {
                            if gpu_shifted_u16_batch_upload_enabled_for_audit() {
                                loader.next_batch_shifted_span_u16_into(
                                    active_batch_plan.global_batch_tokens,
                                    &mut replica.host_token_span_u16,
                                )?;
                                replica.host_input_ids.clear();
                                replica.host_targets.clear();
                            } else {
                                loader.next_batch_u32_into(
                                    active_batch_plan.global_batch_tokens,
                                    &mut replica.host_input_ids,
                                    &mut replica.host_targets,
                                )?;
                                replica.host_token_span_u16.clear();
                            }
                        }
                        distributed_batches_preloaded = true;
                        None
                    } else {
                        Some(synthetic_distributed_batches(active_batch_plan))
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        Some(synthetic_distributed_batches(active_batch_plan))
                    }
                } else if let Some(loaders) = distributed_data_loaders.as_mut() {
                    Some(
                        loaders
                            .iter_mut()
                            .map(|loader| {
                                let (x, y) = loader.next_batch(
                                    active_batch_plan.global_batch_tokens,
                                    active_batch_plan.microbatch_tokens,
                                )?;
                                Ok::<_, pg_core::PgError>(vec![(
                                    x.into_iter().map(|v| v as u32).collect(),
                                    y.into_iter().map(|v| v as u32).collect(),
                                )])
                            })
                            .collect::<PgResult<Vec<_>>>()?,
                    )
                } else {
                    Some(synthetic_distributed_batches(active_batch_plan))
                }
            } else {
                None
            };
            let local_batches: Vec<(Vec<u32>, Vec<u32>)> = if distributed_batches_preloaded {
                Vec::new()
            } else if let Some(batches) = distributed_batches.as_ref() {
                batches[0].clone()
            } else if let Some(loader) = data_loader.as_mut() {
                (0..active_batch_plan.local_microbatches_per_step)
                    .map(|_| {
                        let global_tokens = active_batch_plan.microbatch_tokens * world_size;
                        let (x, y) = loader
                            .next_batch(global_tokens, active_batch_plan.microbatch_tokens)?;
                        Ok::<_, pg_core::PgError>((
                            x.into_iter().map(|v| v as u32).collect(),
                            y.into_iter().map(|v| v as u32).collect(),
                        ))
                    })
                    .collect::<PgResult<Vec<_>>>()?
            } else {
                (0..active_batch_plan.local_microbatches_per_step)
                    .map(|micro_idx| {
                        let offset = step * active_batch_plan.global_batch_tokens
                            + micro_idx * active_batch_plan.microbatch_tokens * world_size
                            + rank * active_batch_plan.microbatch_tokens;
                        (
                            (0..active_batch_plan.microbatch_tokens)
                                .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                                .collect(),
                            (1..=active_batch_plan.microbatch_tokens)
                                .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                                .collect(),
                        )
                    })
                    .collect()
            };
            if measure_step {
                timing.data_sampling_ms += data_t0.elapsed().as_secs_f64() * 1000.0;
            }
            if matches!(mode, RunMode::Proxy) {
                if let Some(batches) = distributed_batches.as_ref() {
                    last_input_ids = batches
                        .iter()
                        .flat_map(|rank_batches| rank_batches.iter())
                        .flat_map(|(x, _)| x.iter().copied())
                        .collect();
                    last_targets = batches
                        .iter()
                        .flat_map(|rank_batches| rank_batches.iter())
                        .flat_map(|(_, y)| y.iter().copied())
                        .collect();
                } else {
                    last_input_ids = local_batches
                        .iter()
                        .flat_map(|(x, _)| x.iter().copied())
                        .collect();
                    last_targets = local_batches
                        .iter()
                        .flat_map(|(_, y)| y.iter().copied())
                        .collect();
                }
            }

            let train_t0 = Instant::now();
            #[cfg(feature = "cuda")]
            let cuda_fast_grad_buffers = matches!(
                self.run_spec.train.backend,
                TrainBackend::CudaSingle | TrainBackend::CudaDistributed
            );
            #[cfg(not(feature = "cuda"))]
            let cuda_fast_grad_buffers = false;
            if !cuda_fast_grad_buffers {
                grads.zero();
            }
            final_loss = match self.run_spec.train.backend {
                TrainBackend::Cpu => {
                    let mut loss_sum = 0.0f32;
                    for (input_ids, targets) in &local_batches {
                        loss_sum += model.backward(input_ids, targets, &mut buf, &mut grads);
                    }
                    if local_batches.len() > 1 {
                        scale_cpu_grads(&mut grads, 1.0 / local_batches.len() as f32);
                    }
                    loss_sum / local_batches.len().max(1) as f32
                }
                #[cfg(feature = "cuda")]
                TrainBackend::CudaSingleParity => {
                    let (input_ids, targets) = local_batches.first().ok_or_else(|| {
                        pg_core::PgError::InvalidOp("missing cuda-single-parity microbatch".into())
                    })?;
                    let runtime = cuda_single_parity_runtime
                        .as_mut()
                        .expect("cuda single parity runtime must be initialized");
                    cuda_single_hybrid_step(
                        runtime, &model, &self.plan, &input_ids, &targets, &mut grads,
                    )?
                }
                #[cfg(feature = "cuda")]
                TrainBackend::CudaSingle => {
                    let runtime = cuda_single_fast_runtime
                        .as_mut()
                        .expect("cuda single fast runtime must be initialized");
                    cuda_single_fast_step(
                        runtime,
                        &mut model,
                        &self.plan,
                        &local_batches,
                        &train_config,
                        step,
                        lr_scale,
                    )?
                }
                #[cfg(feature = "cuda")]
                TrainBackend::CudaDistributed => {
                    let runtime = cuda_distributed_runtime
                        .as_mut()
                        .expect("cuda distributed runtime must be initialized");
                    let mut skipped_step_timing = RunTiming::default();
                    let step_timing = if measure_step {
                        &mut timing
                    } else {
                        &mut skipped_step_timing
                    };
                    cuda_distributed_step(
                        runtime,
                        if distributed_batches_preloaded {
                            None
                        } else {
                            Some(
                                distributed_batches.as_ref().expect(
                                    "distributed backend must materialize per-rank batches",
                                ),
                            )
                        },
                        &train_config,
                        step,
                        lr_scale,
                        self.run_spec.train.distributed_optimizer_backend,
                        !is_record_shaped_mode(mode),
                        active_batch_plan.microbatch_tokens,
                        step_timing,
                    )?
                }
                #[cfg(not(feature = "cuda"))]
                _ => unreachable!("backend should have been rejected before run"),
            };
            if !matches!(
                self.run_spec.train.backend,
                TrainBackend::CudaSingle | TrainBackend::CudaDistributed
            ) {
                if train_config.grad_clip_norm > 0.0 {
                    grads.clip_grad_norm(train_config.grad_clip_norm);
                }

                if fast_bank_updates {
                    // Smoke mode is a correctness/liveness gate. Full CPU NS5 over 26M+
                    // parameters is too slow for that path, so use the same gradients
                    // with a cheap bank update and reserve Muon for proxy/record runs.
                    smoke_bank_step(
                        &mut model.qo_bank,
                        &grads.qo_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                    smoke_bank_step(
                        &mut model.kv_bank,
                        &grads.kv_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                    smoke_bank_step(
                        &mut model.mlp_up_bank,
                        &grads.mlp_up_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                    smoke_bank_step(
                        &mut model.mlp_down_bank,
                        &grads.mlp_down_bank,
                        muon.lr,
                        train_config.muon_wd,
                    );
                } else {
                    muon.step_bank(0, &mut model.qo_bank, &grads.qo_bank, &bank_shapes[0]);
                    muon.step_bank(1, &mut model.kv_bank, &grads.kv_bank, &bank_shapes[1]);
                    muon.step_bank(
                        2,
                        &mut model.mlp_up_bank,
                        &grads.mlp_up_bank,
                        &bank_shapes[2],
                    );
                    muon.step_bank(
                        3,
                        &mut model.mlp_down_bank,
                        &grads.mlp_down_bank,
                        &bank_shapes[3],
                    );
                }

                adamw_embed.step(&mut model.tok_emb, &grads.tok_emb, &mut state_tok_emb);
                adamw_embed.step(
                    &mut model.bigram_embed,
                    &grads.bigram_embed,
                    &mut state_bigram_embed,
                );
                adamw_embed.step(
                    &mut model.bigram_proj,
                    &grads.bigram_proj,
                    &mut state_bigram_proj,
                );
                adamw_embed.step(&mut model.ve_embed, &grads.ve_embed, &mut state_ve_embed);

                adamw_scalar.step(
                    &mut model.smear_gate,
                    &grads.smear_gate,
                    &mut state_smear_gate,
                );
                adamw_scalar.step(
                    &mut model.skip_weights,
                    &grads.skip_weights,
                    &mut state_skip_weights,
                );
                adamw_scalar.step(&mut model.ve_proj, &grads.ve_proj, &mut state_ve_proj);
                {
                    let mut ve_scale_slice = [model.ve_scale];
                    let grad_ve_scale_slice = [grads.ve_scale];
                    adamw_scalar.step(
                        &mut ve_scale_slice,
                        &grad_ve_scale_slice,
                        &mut state_ve_scale,
                    );
                    model.ve_scale = ve_scale_slice[0];
                }
                adamw_scalar.step(
                    &mut model.ve_layer_scales,
                    &grads.ve_layer_scales,
                    &mut state_ve_layer_scales,
                );
                {
                    let mut bigram_scale_slice = [model.bigram_scale];
                    let grad_bigram_scale_slice = [grads.bigram_scale];
                    adamw_scalar.step(
                        &mut bigram_scale_slice,
                        &grad_bigram_scale_slice,
                        &mut state_bigram_scale,
                    );
                    model.bigram_scale = bigram_scale_slice[0];
                }
                for i in 0..n {
                    adamw_scalar.step(
                        &mut model.blocks[i].attn_scale,
                        &grads.block_attn_scale[i],
                        &mut state_attn_scale[i],
                    );
                    adamw_scalar.step(
                        &mut model.blocks[i].mlp_scale,
                        &grads.block_mlp_scale[i],
                        &mut state_mlp_scale[i],
                    );
                    adamw_scalar.step(
                        &mut model.blocks[i].resid_mix,
                        &grads.block_resid_mix[i],
                        &mut state_resid_mix[i],
                    );
                    adamw_scalar.step(
                        &mut model.blocks[i].q_gain,
                        &grads.block_q_gain[i],
                        &mut state_q_gain[i],
                    );
                    if model_config.attn_out_gate_enabled {
                        adamw_scalar.step(
                            &mut model.blocks[i].attn_gate_weight,
                            &grads.block_attn_gate_weight[i],
                            &mut state_attn_gate_weight[i],
                        );
                        adamw_scalar.step(
                            &mut model.blocks[i].attn_gate_bias,
                            &grads.block_attn_gate_bias[i],
                            &mut state_attn_gate_bias[i],
                        );
                    }
                    if model_config.sparse_attn_gate_enabled {
                        adamw_scalar.step(
                            &mut model.blocks[i].sparse_attn_gate_weight,
                            &grads.block_sparse_attn_gate_weight[i],
                            &mut state_sparse_attn_gate_weight[i],
                        );
                    }
                }

                flatten_params_into(&model, &mut flat_buf);
                ema.update(&flat_buf);
                if train_config.should_swa(step) {
                    swa.accumulate(&flat_buf);
                }
            }
            if measure_step {
                timing.train_step_ms += train_t0.elapsed().as_secs_f64() * 1000.0;
                let step_wall_ms = step_wall_t0.elapsed().as_secs_f64() * 1000.0;
                timing_measured_wallclock_ms += step_wall_ms;
                timing_steps += 1;
                if recurrence_active {
                    timing.recurrent_active_steps += 1;
                    timing.recurrent_active_ms += step_wall_ms;
                } else {
                    timing.recurrent_inactive_steps += 1;
                    timing.recurrent_inactive_ms += step_wall_ms;
                }
            }
            steps_completed = step + 1;
        }

        let wallclock_seconds = start.elapsed().as_secs_f64();
        let ms_per_step = if steps_completed > 0 {
            (wallclock_seconds * 1000.0) / steps_completed as f64
        } else {
            0.0
        };
        let timing_measured_ms_per_step = if timing_steps > 0 {
            timing_measured_wallclock_ms / timing_steps as f64
        } else {
            0.0
        };
        if mode == RunMode::Record && timing_steps > 0 {
            let max_ms = record_max_ms_per_step_for_submission(&self.run_spec);
            if max_ms > 0.0 && timing_measured_ms_per_step > max_ms {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "record run is too slow for a leaderboard submission: measured_ms_per_step={timing_measured_ms_per_step:.3} max_ms_per_step={max_ms:.3}. Set PG_RECORD_MAX_MS_PER_STEP=0 only for non-submission debugging."
                )));
            }
            if self.run_spec.runtime.require_device_batch
                && (timing.host_batch_flatten_calls > 0
                    || timing.host_to_device_batch_bytes > 0
                    || timing.device_batch_missing_steps > 0)
            {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "record runtime requires zero hot-path host batch work after warmup: host_batch_flatten_calls={} host_to_device_batch_bytes={} device_batch_missing_steps={}",
                    timing.host_batch_flatten_calls,
                    timing.host_to_device_batch_bytes,
                    timing.device_batch_missing_steps
                )));
            }
        }

        let export_record_shaped_artifact =
            mode == RunMode::RecordShapedProxy && record_shaped_artifact_export_enabled();
        let needs_cpu_model_after_training =
            !matches!(mode, RunMode::Smoke | RunMode::RecordShapedProxy)
                || export_record_shaped_artifact;
        let sync_t0 = Instant::now();
        if needs_cpu_model_after_training {
            #[cfg(feature = "cuda")]
            if let Some(runtime) = cuda_distributed_runtime.as_mut() {
                if sharded_parallel_muon_bf16_shadow_all_gather_enabled_for_audit() {
                    cuda_distributed_sync_f32_banks_from_sharded_muon(runtime)?;
                }
            }
            #[cfg(feature = "cuda")]
            if let Some(runtime) = cuda_single_fast_runtime.as_ref() {
                runtime
                    .gpu_model
                    .sync_to_cpu_reference(&mut model, &self.plan)?;
            }
            #[cfg(feature = "cuda")]
            if let Some(runtime) = cuda_distributed_runtime.as_ref() {
                runtime.replicas[0]
                    .gpu_model
                    .sync_to_cpu_reference(&mut model, &self.plan)?;
            }
        }
        timing.post_train_sync_ms += sync_t0.elapsed().as_secs_f64() * 1000.0;
        #[cfg(feature = "cuda")]
        let distributed_sync = cuda_distributed_runtime
            .as_ref()
            .map(|runtime| runtime.distributed_sync)
            .unwrap_or(false);
        #[cfg(not(feature = "cuda"))]
        let distributed_sync = false;

        let artifact_t0 = Instant::now();
        let artifact_bytes = match mode {
            RunMode::Smoke => None,
            RunMode::RecordShapedProxy if !export_record_shaped_artifact => None,
            _ => {
                let artifact_path = std::path::Path::new(&self.run_spec.train.artifact_path);
                let exported = pg_quant::export::export_model_with_spec(
                    &model,
                    &self.run_spec.quant,
                    &self.plan.variant_fingerprint,
                    artifact_path,
                );
                if mode == RunMode::Record {
                    Some(exported?)
                } else {
                    exported.ok()
                }
            }
        };
        timing.artifact_export_ms += artifact_t0.elapsed().as_secs_f64() * 1000.0;
        let submission_code_bytes = artifact_bytes.and_then(|_| current_executable_bytes());
        let artifact_model_sha256 = artifact_bytes.and_then(|_| {
            sha256_file(std::path::Path::new(&self.run_spec.train.artifact_path)).ok()
        });
        let artifact_code_sha256 = artifact_bytes.and_then(|_| current_executable_sha256());
        let caseops_byte_sidecar_sha256 = self
            .run_spec
            .eval
            .caseops_byte_sidecar_pattern
            .as_deref()
            .and_then(|pattern| sha256_file_set(pattern).ok().flatten());
        let submission_total_bytes = artifact_bytes
            .zip(submission_code_bytes)
            .map(|(a, c)| a + c);
        let artifact_budget_ok =
            artifact_bytes
                .zip(submission_code_bytes)
                .map(|(model_bytes, code_bytes)| {
                    self.plan.submission_budget_ok(code_bytes, model_bytes)
                });
        if mode == RunMode::Record && artifact_budget_ok != Some(true) {
            return Err(pg_core::PgError::InvalidOp(format!(
                "record artifact budget failed: artifact_bytes={artifact_bytes:?} submission_code_bytes={submission_code_bytes:?} submission_total_bytes={submission_total_bytes:?} limit={}",
                self.plan.quant_layout.target_artifact_bytes
            )));
        }
        if mode == RunMode::Record
            && (artifact_model_sha256.is_none()
                || artifact_code_sha256.is_none()
                || (self.run_spec.model.caseops.enabled
                    && self.run_spec.model.caseops.byte_sidecar
                    && caseops_byte_sidecar_sha256.is_none()))
        {
            return Err(pg_core::PgError::InvalidOp(format!(
                "record artifact audit failed: artifact_model_sha256_known={} artifact_code_sha256_known={} caseops_byte_sidecar_sha256_known={}",
                artifact_model_sha256.is_some(),
                artifact_code_sha256.is_some(),
                caseops_byte_sidecar_sha256.is_some()
            )));
        }
        if let (Some(model_bytes), Some(code_bytes), Some(total_bytes), Some(ok)) = (
            artifact_bytes,
            submission_code_bytes,
            submission_total_bytes,
            artifact_budget_ok,
        ) {
            let budget = self.plan.submission_budget(code_bytes, model_bytes);
            println!(
                "submission_budget_json={{\"event\":\"submission_byte_budget\",\"limit\":{},\"model_bytes\":{},\"code_bytes\":{},\"total_bytes\":{},\"under_budget\":{},\"strict_decimal_bytes\":true}}",
                budget.limit_bytes,
                budget.compressed_model_bytes,
                budget.code_bytes,
                total_bytes,
                ok
            );
        }
        if artifact_bytes.is_some() {
            println!(
                "record_artifact_audit_json={}",
                record_artifact_audit_json(
                    &self.run_spec,
                    artifact_bytes,
                    submission_code_bytes,
                    submission_total_bytes,
                    artifact_budget_ok,
                    artifact_model_sha256.as_deref(),
                    artifact_code_sha256.as_deref(),
                    caseops_byte_sidecar_sha256.as_deref(),
                )
            );
        }
        let (bpb_luts, bpb_byte_source) =
            load_bpb_luts(&self.run_spec, model_config.vocab_size, mode)?;
        let proxy_bpb = if matches!(mode, RunMode::Proxy) {
            let prev: Vec<u16> = last_input_ids.iter().map(|&v| v as u16).collect();
            let tgt: Vec<u16> = last_targets.iter().map(|&v| v as u16).collect();
            let byte_count = bpb_luts.count_bytes(&prev, &tgt);
            Some(compute_bpb(
                final_loss as f64,
                last_targets.len() as f64,
                byte_count,
            ))
        } else {
            None
        };
        let proxy_metric_source = proxy_bpb.map(|_| "last_batch_train_loss".to_string());
        let train_loss_source = if is_record_shaped_mode(mode) {
            "disabled_for_record_shaped_timing".to_string()
        } else {
            "mean_step_loss".to_string()
        };
        let eval_t0 = Instant::now();
        let (eval_loss, final_bpb, eval_tokens) =
            if !matches!(mode, RunMode::Smoke | RunMode::RecordShapedProxy) {
                if let Some(pattern) = self.run_spec.train.validation_data_pattern.as_deref() {
                    let mut eval_model = GptModel::new(model_config.clone());
                    eval_model.fill_deterministic();
                    if !matches!(mode, RunMode::Smoke) && artifact_bytes.is_some() {
                        pg_quant::export::load_artifact(
                            std::path::Path::new(&self.run_spec.train.artifact_path),
                            &mut eval_model,
                        )?;
                    } else {
                        eval_model = model;
                    }
                    let max_eval_tokens = self.run_spec.eval.max_tokens.map(|limit| limit.max(2));
                    let tokens = pg_data::token_stream::load_validation_tokens_limited(
                        pattern,
                        max_eval_tokens,
                    )?
                    .into_iter()
                    .map(|v| v as u32)
                    .collect::<Vec<_>>();
                    let token_bytes = eval_target_byte_counts(
                        &self.run_spec,
                        &tokens,
                        &bpb_luts,
                        max_eval_tokens,
                    )?;
                    let seq_len = self
                        .run_spec
                        .model
                        .eval_seq_len
                        .min(tokens.len().saturating_sub(1))
                        .max(1);
                    let (loss, bpb) = if self.run_spec.eval.adaptation_backend
                        == EvalAdaptationBackend::GpuLoraPhased
                    {
                        eval_gpu_lora_phased_from_train(
                            &eval_model,
                            &self.plan,
                            &tokens,
                            &token_bytes,
                        )?
                    } else if self.run_spec.eval.qttt {
                        let mut cfg = pg_eval::qttt::QttTConfig::paper_default(seq_len);
                        cfg.stride = self.run_spec.eval.stride;
                        cfg.seq_len = seq_len;
                        cfg.chunk_tokens = self.run_spec.eval.chunk_tokens;
                        pg_eval::qttt::eval_qttt(&mut eval_model, &tokens, &token_bytes, &cfg)
                    } else {
                        pg_eval::sliding::eval_sliding(
                            &eval_model,
                            &tokens,
                            &token_bytes,
                            self.run_spec.eval.stride,
                            seq_len,
                        )
                    };
                    (Some(loss), Some(bpb), Some(tokens.len()))
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };
        timing.eval_ms += eval_t0.elapsed().as_secs_f64() * 1000.0;

        Ok(VariantResult {
            run_name: self.run_spec.name.clone(),
            mode,
            train_backend: self.run_spec.train.backend,
            variant_fingerprint: self.plan.variant_fingerprint.clone(),
            steps_completed,
            train_loss: final_loss,
            train_loss_source,
            proxy_bpb,
            eval_loss,
            final_bpb,
            eval_tokens,
            artifact_bytes,
            submission_code_bytes,
            submission_total_bytes,
            artifact_budget_ok,
            artifact_model_sha256,
            artifact_code_sha256,
            caseops_byte_sidecar_sha256,
            attention_backend: format!("{:?}", self.run_spec.model.attention_backend),
            distributed_optimizer_backend: format!(
                "{:?}",
                self.run_spec.train.distributed_optimizer_backend
            ),
            eval_adaptation_backend: format!("{:?}", self.run_spec.eval.adaptation_backend),
            frontier_record_ready,
            leaderboard_algorithm_ready,
            record_shape,
            record_attention_grade,
            microbatch_serial_loop,
            bank_update_backend: bank_update_backend.to_string(),
            train_data_source: train_data_source.to_string(),
            bpb_byte_source: bpb_byte_source.to_string(),
            proxy_metric_source,
            timing_backend: cuda_timing_backend_label().to_string(),
            ms_per_step,
            wallclock_seconds,
            timing_steps,
            timing_measured_ms_per_step,
            timing_recurrent_active_steps: timing.recurrent_active_steps,
            timing_recurrent_inactive_steps: timing.recurrent_inactive_steps,
            timing_recurrent_active_ms: timing.recurrent_active_ms,
            timing_recurrent_inactive_ms: timing.recurrent_inactive_ms,
            host_batch_flatten_calls: timing.host_batch_flatten_calls,
            host_to_device_batch_bytes: timing.host_to_device_batch_bytes,
            device_batch_ready_steps: timing.device_batch_ready_steps,
            device_batch_missing_steps: timing.device_batch_missing_steps,
            device_to_host_scalar_reads: timing.device_to_host_scalar_reads,
            timing_data_sampling_ms: timing.data_sampling_ms,
            timing_train_step_ms: timing.train_step_ms,
            timing_cuda_zero_grads_ms: timing.cuda_zero_grads_ms,
            timing_cuda_h2d_ms: timing.cuda_h2d_ms,
            timing_cuda_backward_ms: timing.cuda_backward_ms,
            timing_cuda_backward_forward_ms: timing.cuda_backward_forward_ms,
            timing_cuda_backward_forward_embed_ms: timing.cuda_backward_forward_embed_ms,
            timing_cuda_backward_forward_encoder_ms: timing.cuda_backward_forward_encoder_ms,
            timing_cuda_backward_forward_encoder_layer_max_ms: timing
                .cuda_backward_forward_encoder_layer_max_ms,
            timing_cuda_backward_forward_decoder_ms: timing.cuda_backward_forward_decoder_ms,
            timing_cuda_backward_forward_decoder_layer_max_ms: timing
                .cuda_backward_forward_decoder_layer_max_ms,
            timing_cuda_backward_forward_logits_ms: timing.cuda_backward_forward_logits_ms,
            timing_cuda_backward_forward_block_pre_attn_ms: timing
                .cuda_backward_forward_block_pre_attn_ms,
            timing_cuda_backward_forward_block_attention_ms: timing
                .cuda_backward_forward_block_attention_ms,
            timing_cuda_backward_forward_block_post_attn_ms: timing
                .cuda_backward_forward_block_post_attn_ms,
            timing_cuda_backward_forward_block_mlp_ms: timing.cuda_backward_forward_block_mlp_ms,
            timing_cuda_backward_block_recompute_ms: timing.cuda_backward_block_recompute_ms,
            timing_cuda_backward_block_mlp_ms: timing.cuda_backward_block_mlp_ms,
            timing_cuda_backward_block_mlp_residual_ms: timing.cuda_backward_block_mlp_residual_ms,
            timing_cuda_backward_block_mlp_down_ms: timing.cuda_backward_block_mlp_down_ms,
            timing_cuda_backward_block_mlp_act_ms: timing.cuda_backward_block_mlp_act_ms,
            timing_cuda_backward_block_mlp_up_ms: timing.cuda_backward_block_mlp_up_ms,
            timing_cuda_backward_block_mlp_norm_ms: timing.cuda_backward_block_mlp_norm_ms,
            timing_cuda_backward_block_attn_out_ms: timing.cuda_backward_block_attn_out_ms,
            timing_cuda_backward_block_attn_out_residual_ms: timing
                .cuda_backward_block_attn_out_residual_ms,
            timing_cuda_backward_block_attn_out_proj_ms: timing
                .cuda_backward_block_attn_out_proj_ms,
            timing_cuda_backward_block_attn_out_gate_xsa_ms: timing
                .cuda_backward_block_attn_out_gate_xsa_ms,
            timing_cuda_backward_block_attention_ms: timing.cuda_backward_block_attention_ms,
            timing_cuda_backward_block_attention_sdpa_ms: timing
                .cuda_backward_block_attention_sdpa_ms,
            timing_cuda_backward_block_attention_xsa_accum_ms: timing
                .cuda_backward_block_attention_xsa_accum_ms,
            timing_cuda_backward_block_qkv_ms: timing.cuda_backward_block_qkv_ms,
            timing_cuda_backward_block_qkv_rope_ms: timing.cuda_backward_block_qkv_rope_ms,
            timing_cuda_backward_block_qkv_proj_ms: timing.cuda_backward_block_qkv_proj_ms,
            timing_cuda_backward_block_qkv_ve_ms: timing.cuda_backward_block_qkv_ve_ms,
            timing_cuda_backward_block_qkv_norm_resid_ms: timing
                .cuda_backward_block_qkv_norm_resid_ms,
            timing_cuda_backward_output_ms: timing.cuda_backward_output_ms,
            timing_cuda_backward_decoder_ms: timing.cuda_backward_decoder_ms,
            timing_cuda_backward_encoder_ms: timing.cuda_backward_encoder_ms,
            timing_cuda_backward_tail_ms: timing.cuda_backward_tail_ms,
            timing_cuda_non_bank_sync_ms: timing.cuda_non_bank_sync_ms,
            timing_cuda_bank_update_ms: timing.cuda_bank_update_ms,
            timing_cuda_bank_update_stage_pack_ms: timing.cuda_bank_update_stage_pack_ms,
            timing_cuda_bank_update_stage_grad_collectives_ms: timing
                .cuda_bank_update_stage_grad_collectives_ms,
            timing_cuda_bank_update_stage_scale_ms: timing.cuda_bank_update_stage_scale_ms,
            timing_cuda_bank_update_stage_norm_ms: timing.cuda_bank_update_stage_norm_ms,
            timing_cuda_bank_update_stage_clip_ms: timing.cuda_bank_update_stage_clip_ms,
            timing_cuda_bank_update_stage_muon_ms: timing.cuda_bank_update_stage_muon_ms,
            timing_cuda_bank_update_stage_param_gather_ms: timing
                .cuda_bank_update_stage_param_gather_ms,
            timing_cuda_bank_update_stage_copy_back_ms: timing.cuda_bank_update_stage_copy_back_ms,
            timing_cuda_non_bank_update_ms: timing.cuda_non_bank_update_ms,
            timing_post_train_sync_ms: timing.post_train_sync_ms,
            timing_artifact_export_ms: timing.artifact_export_ms,
            timing_eval_ms: timing.eval_ms,
            rank,
            world_size,
            distributed_sync,
            seq_len: batch_plan.microbatch_tokens,
            global_batch_tokens: batch_plan.global_batch_tokens,
            local_microbatches_per_step: batch_plan.local_microbatches_per_step,
            tokens_seen_global: steps_completed * batch_plan.global_batch_tokens,
        })
    }
}

#[cfg(feature = "cuda")]
fn cuda_single_hybrid_step(
    runtime: &mut CudaSingleHybridRuntime,
    model: &GptModel,
    plan: &ExecutionPlan,
    input_ids: &[u32],
    targets: &[u32],
    grads: &mut GradBuffers,
) -> PgResult<f32> {
    use pg_model::gpu::{GpuActivations, GpuGradBuffers};

    runtime.gpu_model.sync_from_cpu_reference(model, plan)?;
    let stream = runtime.gpu_model.gemm.stream().clone();
    runtime
        .input_ids
        .copy_from_host_bytes(bytemuck::cast_slice(input_ids))?;
    runtime
        .targets
        .copy_from_host_bytes(bytemuck::cast_slice(targets))?;
    let mut gpu_buf = GpuActivations::new_for_plan(plan, input_ids.len(), stream.clone())?;
    let mut gpu_grads = GpuGradBuffers::new(&model.config, stream.clone())?;
    let loss = runtime.gpu_model.backward_with_state(
        &runtime.input_ids,
        &runtime.targets,
        &mut gpu_buf,
        &mut runtime.backward_state,
        &mut gpu_grads,
    )?;
    stream
        .synchronize()
        .map_err(|e| pg_core::PgError::InvalidOp(format!("stream sync failed: {:?}", e)))?;

    grads
        .tok_emb
        .copy_from_slice(&download_gpu_f32(&gpu_grads.tok_emb)?);
    grads
        .bigram_embed
        .copy_from_slice(&download_gpu_f32(&gpu_grads.bigram_embed)?);
    grads
        .bigram_proj
        .copy_from_slice(&download_gpu_f32(&gpu_grads.bigram_proj)?);
    grads.bigram_scale = download_gpu_f32(&gpu_grads.bigram_scale)?[0];
    grads
        .smear_gate
        .copy_from_slice(&download_gpu_f32(&gpu_grads.smear_gate)?);
    grads
        .skip_weights
        .copy_from_slice(&download_gpu_f32(&gpu_grads.skip_weights)?);
    grads
        .qo_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.qo_bank)?);
    grads
        .kv_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.kv_bank)?);
    grads
        .mlp_up_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.mlp_up_bank)?);
    grads
        .mlp_down_bank
        .copy_from_slice(&download_gpu_f32(&gpu_grads.mlp_down_bank)?);
    for (dst, src) in grads
        .block_attn_scale
        .iter_mut()
        .zip(gpu_grads.block_attn_scale.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_mlp_scale
        .iter_mut()
        .zip(gpu_grads.block_mlp_scale.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_resid_mix
        .iter_mut()
        .zip(gpu_grads.block_resid_mix.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_q_gain
        .iter_mut()
        .zip(gpu_grads.block_q_gain.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_attn_gate_weight
        .iter_mut()
        .zip(gpu_grads.block_attn_gate_weight.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_attn_gate_bias
        .iter_mut()
        .zip(gpu_grads.block_attn_gate_bias.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    for (dst, src) in grads
        .block_sparse_attn_gate_weight
        .iter_mut()
        .zip(gpu_grads.block_sparse_attn_gate_weight.iter())
    {
        dst.copy_from_slice(&download_gpu_f32(src)?);
    }
    if model.config.ve_enabled {
        grads
            .ve_embed
            .copy_from_slice(&download_gpu_f32(&gpu_grads.ve_embed)?);
        grads
            .ve_proj
            .copy_from_slice(&download_gpu_f32(&gpu_grads.ve_proj)?);
        grads.ve_scale = download_gpu_f32(&gpu_grads.ve_scale)?[0];
        grads
            .ve_layer_scales
            .copy_from_slice(&download_gpu_f32(&gpu_grads.ve_layer_scales)?);
    } else {
        grads.ve_embed.fill(0.0);
        grads.ve_proj.fill(0.0);
        grads.ve_scale = 0.0;
        grads.ve_layer_scales.fill(0.0);
    }

    Ok(loss)
}

#[cfg(feature = "cuda")]
fn zero_gpu_grads(
    _kernels: &pg_kernels::gpu_kernels::GpuKernels,
    grads: &mut pg_model::gpu::GpuGradBuffers,
    config: &pg_model::ModelConfig,
) -> PgResult<()> {
    let zero = |tensor: &mut pg_core::GpuTensor| tensor.zero_bytes();

    zero(&mut grads.tok_emb)?;
    if config.bigram_vocab_size > 0 && config.bigram_dim > 0 {
        zero(&mut grads.bigram_embed)?;
        zero(&mut grads.bigram_proj)?;
        zero(&mut grads.bigram_scale)?;
    }
    zero(&mut grads.smear_gate)?;
    zero(&mut grads.skip_weights)?;
    if !gpu_overwrite_bank_grads_enabled() {
        zero(&mut grads.qo_bank)?;
        zero(&mut grads.kv_bank)?;
        zero(&mut grads.mlp_up_bank)?;
        zero(&mut grads.mlp_down_bank)?;
    }
    for tensor in &mut grads.block_attn_scale {
        zero(tensor)?;
    }
    for tensor in &mut grads.block_mlp_scale {
        zero(tensor)?;
    }
    for tensor in &mut grads.block_resid_mix {
        zero(tensor)?;
    }
    for tensor in &mut grads.block_q_gain {
        zero(tensor)?;
    }
    if config.attn_out_gate_enabled {
        for tensor in &mut grads.block_attn_gate_weight {
            zero(tensor)?;
        }
        for tensor in &mut grads.block_attn_gate_bias {
            zero(tensor)?;
        }
    }
    for tensor in &mut grads.block_sparse_attn_gate_weight {
        zero(tensor)?;
    }
    if config.ve_enabled {
        zero(&mut grads.ve_embed)?;
        zero(&mut grads.ve_proj)?;
        zero(&mut grads.ve_scale)?;
        zero(&mut grads.ve_layer_scales)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_single_fast_step(
    runtime: &mut CudaSingleFastRuntime,
    _model: &mut GptModel,
    plan: &ExecutionPlan,
    microbatches: &[(Vec<u32>, Vec<u32>)],
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<f32> {
    zero_gpu_grads(
        &runtime.gpu_model.kernels,
        &mut runtime.gpu_grads,
        &runtime.gpu_model.config,
    )?;
    let strict_device_batch = record_strict_device_batch_enabled();
    let seq_len = if strict_device_batch && runtime.device_batch_ready {
        plan.run_spec.train.seq_len.max(1)
    } else {
        microbatches
            .first()
            .map(|(input_ids, _)| input_ids.len())
            .unwrap_or(1)
            .max(1)
    };
    if strict_device_batch && !runtime.device_batch_ready {
        return Err(pg_core::PgError::InvalidOp(
            "record runtime requires device_batch_ready before cuda_single_fast_step; refusing host microbatch flatten".into(),
        ));
    }
    if !strict_device_batch {
        flatten_microbatches_into(
            microbatches,
            &mut runtime.host_input_ids,
            &mut runtime.host_targets,
        );
    }
    let loss_sum = cuda_fast_accumulate_runtime_grads(runtime, true, seq_len, None, None)?;
    cuda_fast_apply_updates(runtime, train_config, step, lr_scale)?;
    Ok(loss_sum)
}

#[cfg(feature = "cuda")]
fn cuda_fast_accumulate_runtime_grads(
    runtime: &mut CudaSingleFastRuntime,
    compute_loss: bool,
    runtime_seq_len: usize,
    mut timing: Option<&mut RunTiming>,
    observer: Option<&mut dyn pg_model::gpu::GpuBackwardLayerObserver>,
) -> PgResult<f32> {
    let h2d_t0 = Instant::now();
    if runtime.device_batch_ready {
        runtime.device_batch_ready = false;
        if let Some(timing) = timing.as_deref_mut() {
            timing.device_batch_ready_steps += 1;
        }
    } else if gpu_shifted_u16_batch_upload_enabled_for_audit()
        && runtime.host_token_span_u16.len() == runtime.input_ids.numel() + 1
    {
        runtime
            .token_span_u16
            .copy_from_host_bytes(bytemuck::cast_slice(&runtime.host_token_span_u16))?;
        let stream = runtime.gpu_model.kernels.stream();
        runtime.gpu_model.kernels.shifted_u16_to_u32(
            pg_kernels::gpu_kernels::CudaPtr(runtime.token_span_u16.cu_ptr(stream)?),
            pg_kernels::gpu_kernels::CudaPtr(runtime.input_ids.cu_ptr(stream)?),
            pg_kernels::gpu_kernels::CudaPtr(runtime.targets.cu_ptr(stream)?),
            runtime.input_ids.numel() as u32,
        )?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.device_batch_missing_steps += 1;
            timing.host_to_device_batch_bytes += runtime.host_token_span_u16.len() * 2;
        }
    } else {
        if record_strict_device_batch_enabled() {
            return Err(pg_core::PgError::InvalidOp(
                "PG_RECORD_REQUIRE_DEVICE_BATCH=1 but a step fell back to host_input_ids \
                 H2D copy (device_batch_ready was false and shifted-u16 upload was not active); \
                 wire the device-resident token-ring sampler to keep host copies off the record hot path"
                    .into(),
            ));
        }
        runtime
            .input_ids
            .copy_from_host_bytes(bytemuck::cast_slice(&runtime.host_input_ids))?;
        runtime
            .targets
            .copy_from_host_bytes(bytemuck::cast_slice(&runtime.host_targets))?;
        if let Some(timing) = timing.as_deref_mut() {
            timing.device_batch_missing_steps += 1;
            timing.host_to_device_batch_bytes += (runtime.host_input_ids.len()
                + runtime.host_targets.len())
                * std::mem::size_of::<u32>();
        }
    }
    if let Some(timing) = timing.as_deref_mut() {
        timing.cuda_h2d_ms += h2d_t0.elapsed().as_secs_f64() * 1000.0;
    }

    if let Some(observer) = observer {
        if compute_loss {
            return Err(pg_core::PgError::InvalidOp(
                "observed CUDA backward does not support loss-return mode".into(),
            ));
        }
        runtime
            .gpu_model
            .backward_with_state_seq_len_no_loss_observed(
                &runtime.input_ids,
                &runtime.targets,
                &mut runtime.gpu_buf,
                &mut runtime.backward_state,
                &mut runtime.gpu_grads,
                runtime_seq_len,
                observer,
            )?;
        Ok(0.0)
    } else if compute_loss {
        runtime.gpu_model.backward_with_state_seq_len(
            &runtime.input_ids,
            &runtime.targets,
            &mut runtime.gpu_buf,
            &mut runtime.backward_state,
            &mut runtime.gpu_grads,
            runtime_seq_len,
        )
    } else if cuda_backward_graph_enabled() {
        cuda_fast_accumulate_runtime_grads_no_loss_graph(runtime, runtime_seq_len)?;
        Ok(0.0)
    } else {
        runtime.gpu_model.backward_with_state_seq_len_no_loss(
            &runtime.input_ids,
            &runtime.targets,
            &mut runtime.gpu_buf,
            &mut runtime.backward_state,
            &mut runtime.gpu_grads,
            runtime_seq_len,
        )?;
        Ok(0.0)
    }
}

#[cfg(feature = "cuda")]
fn cuda_fast_accumulate_runtime_grads_no_loss_graph(
    runtime: &mut CudaSingleFastRuntime,
    runtime_seq_len: usize,
) -> PgResult<()> {
    let stream = runtime.gpu_model.gemm.stream().clone();
    if runtime.backward_graph.is_some() && runtime.backward_graph_seq_len == runtime_seq_len {
        runtime
            .backward_graph
            .as_ref()
            .expect("checked graph exists")
            .0
            .launch()
            .map_err(|e| pg_core::PgError::InvalidOp(format!("cuda graph launch failed: {e:?}")))?;
        return Ok(());
    }

    if runtime.backward_graph.is_some() {
        runtime.backward_graph = None;
        runtime.backward_graph_seq_len = 0;
        runtime.backward_graph_warmup_steps_done = 0;
    }

    let required_warmup_steps = cuda_backward_graph_warmup_steps();
    if runtime.backward_graph_warmup_steps_done < required_warmup_steps {
        zero_gpu_grads(
            &runtime.gpu_model.kernels,
            &mut runtime.gpu_grads,
            &runtime.gpu_model.config,
        )?;
        runtime.gpu_model.backward_with_state_seq_len_no_loss(
            &runtime.input_ids,
            &runtime.targets,
            &mut runtime.gpu_buf,
            &mut runtime.backward_state,
            &mut runtime.gpu_grads,
            runtime_seq_len,
        )?;
        stream.synchronize().map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda graph warmup sync failed: {e:?}"))
        })?;
        runtime.backward_graph_warmup_steps_done += 1;
        return Ok(());
    }

    // The device batch is materialized immediately before this function. If the
    // capture begins while that uncaptured kernel is still in flight, CUDA marks
    // the first captured forward kernel as depending on uncaptured work and
    // invalidates the graph with STREAM_CAPTURE_ISOLATION. Pay this sync once at
    // instantiation time; steady-state graph launches still run without it.
    stream.synchronize().map_err(|e| {
        pg_core::PgError::InvalidOp(format!("cuda graph pre-capture sync failed: {e:?}"))
    })?;

    stream
        .begin_capture(cuda_graph_capture_mode())
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda backward graph capture begin failed: {e:?}"))
        })?;
    let capture_result = (|| -> PgResult<()> {
        zero_gpu_grads(
            &runtime.gpu_model.kernels,
            &mut runtime.gpu_grads,
            &runtime.gpu_model.config,
        )?;
        runtime.gpu_model.backward_with_state_seq_len_no_loss(
            &runtime.input_ids,
            &runtime.targets,
            &mut runtime.gpu_buf,
            &mut runtime.backward_state,
            &mut runtime.gpu_grads,
            runtime_seq_len,
        )?;
        Ok(())
    })();
    if let Err(err) = capture_result {
        let _ = stream.end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        return Err(err);
    }
    let graph = stream
        .end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda backward graph capture end failed: {e:?}"))
        })?;
    runtime.backward_graph = graph.map(CudaBackwardGraph);
    runtime.backward_graph_seq_len = runtime_seq_len;
    runtime
        .backward_graph
        .as_ref()
        .ok_or_else(|| {
            pg_core::PgError::InvalidOp("cuda backward graph capture produced no graph".into())
        })?
        .0
        .launch()
        .map_err(|e| pg_core::PgError::InvalidOp(format!("cuda graph launch failed: {e:?}")))?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn flatten_microbatches_into(
    microbatches: &[(Vec<u32>, Vec<u32>)],
    input: &mut Vec<u32>,
    target: &mut Vec<u32>,
) {
    let total_tokens: usize = microbatches
        .iter()
        .map(|(input_ids, _)| input_ids.len())
        .sum();
    input.clear();
    target.clear();
    input.reserve(total_tokens.saturating_sub(input.capacity()));
    target.reserve(total_tokens.saturating_sub(target.capacity()));
    for (input_ids, targets) in microbatches {
        input.extend_from_slice(input_ids);
        target.extend_from_slice(targets);
    }
}

#[cfg(feature = "cuda")]
fn cuda_distributed_generate_synthetic_record_batches(
    runtime: &mut CudaDistributedRuntime,
    step: usize,
    batch_plan: &StepBatchPlan,
    vocab_size: usize,
) -> PgResult<()> {
    let world_size = runtime.replicas.len();
    for (rank_idx, replica) in runtime.replicas.iter_mut().enumerate() {
        let stream = replica.gpu_model.kernels.stream();
        replica.gpu_model.kernels.synthetic_record_batch_u32(
            pg_kernels::gpu_kernels::CudaPtr(replica.input_ids.cu_ptr(stream)?),
            pg_kernels::gpu_kernels::CudaPtr(replica.targets.cu_ptr(stream)?),
            step as u64,
            batch_plan.global_batch_tokens as u64,
            rank_idx as u32,
            world_size as u32,
            batch_plan.microbatch_tokens as u32,
            batch_plan.local_microbatches_per_step as u32,
            vocab_size as u32,
        )?;
        replica.device_batch_ready = true;
        replica.host_input_ids.clear();
        replica.host_targets.clear();
        replica.host_token_span_u16.clear();
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_prepare_token_ring_batches(
    runtime: &mut CudaDistributedRuntime,
    loaders: &mut [DistributedTokenLoader],
    step: usize,
    batch_plan: &StepBatchPlan,
) -> PgResult<()> {
    let ring_steps = gpu_token_ring_sampler_steps_for_runtime().max(1);
    let full_schedule = gpu_token_ring_full_schedule_enabled_for_audit();
    let ring_base_step = if full_schedule {
        0
    } else {
        (step / ring_steps) * ring_steps
    };
    if full_schedule && step >= ring_steps {
        return Err(pg_core::PgError::InvalidOp(format!(
            "GPU token full-schedule sampler configured for {ring_steps} steps, but step {step} was requested; set PG_GPU_TOKEN_RING_STEPS to cover the full record schedule"
        )));
    }
    let local_tokens = batch_plan.microbatch_tokens * batch_plan.local_microbatches_per_step;
    let slot_tokens = local_tokens + 1;
    let expected_ring_tokens = slot_tokens * ring_steps;
    if expected_ring_tokens > u32::MAX as usize {
        return Err(pg_core::PgError::InvalidOp(format!(
            "GPU token-ring sampler ring has {expected_ring_tokens} tokens, exceeding u32 kernel offset limit"
        )));
    }

    for (loader, replica) in loaders.iter_mut().zip(runtime.replicas.iter_mut()) {
        if replica.token_ring_base_step != Some(ring_base_step)
            || replica.token_ring_steps_loaded != ring_steps
        {
            replica.host_token_ring_u16.clear();
            replica.host_token_ring_u16.reserve(
                expected_ring_tokens.saturating_sub(replica.host_token_ring_u16.capacity()),
            );
            for _ in 0..ring_steps {
                loader.next_batch_shifted_span_u16_into(
                    batch_plan.global_batch_tokens,
                    &mut replica.host_token_span_u16,
                )?;
                if replica.host_token_span_u16.len() != slot_tokens {
                    return Err(pg_core::PgError::InvalidOp(format!(
                        "GPU token-ring sampler expected shifted span length {slot_tokens}, got {}",
                        replica.host_token_span_u16.len()
                    )));
                }
                replica
                    .host_token_ring_u16
                    .extend_from_slice(&replica.host_token_span_u16);
            }
            if replica.host_token_ring_u16.len() != expected_ring_tokens {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "GPU token-ring sampler expected ring length {expected_ring_tokens}, got {}",
                    replica.host_token_ring_u16.len()
                )));
            }
            replica
                .token_ring_u16
                .copy_from_host_bytes(bytemuck::cast_slice(&replica.host_token_ring_u16))?;
            replica.token_ring_base_step = Some(ring_base_step);
            replica.token_ring_steps_loaded = ring_steps;
        }

        let slot = step - ring_base_step;
        if slot >= ring_steps {
            return Err(pg_core::PgError::InvalidOp(format!(
                "GPU token-ring sampler slot {slot} exceeds ring_steps {ring_steps}"
            )));
        }
        let ring_offset = slot * slot_tokens;
        if ring_offset + slot_tokens > expected_ring_tokens {
            return Err(pg_core::PgError::InvalidOp(format!(
                "GPU token-ring sampler offset {} + slot {} exceeds ring {}",
                ring_offset, slot_tokens, expected_ring_tokens
            )));
        }
        let stream = replica.gpu_model.kernels.stream();
        replica.gpu_model.kernels.shifted_u16_ring_to_u32(
            pg_kernels::gpu_kernels::CudaPtr(replica.token_ring_u16.cu_ptr(stream)?),
            pg_kernels::gpu_kernels::CudaPtr(replica.input_ids.cu_ptr(stream)?),
            pg_kernels::gpu_kernels::CudaPtr(replica.targets.cu_ptr(stream)?),
            local_tokens as u32,
            ring_offset as u32,
        )?;
        replica.device_batch_ready = true;
        replica.host_input_ids.clear();
        replica.host_targets.clear();
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn accumulate_gpu_backward_stage_timing(
    replicas: &[CudaSingleFastRuntime],
    timing: &mut RunTiming,
) {
    let mut max_stage = pg_model::gpu::GpuBackwardStageTiming::default();
    for replica in replicas {
        let t = replica.backward_state.stage_timing;
        max_stage.forward_ms = max_stage.forward_ms.max(t.forward_ms);
        max_stage.forward_embed_ms = max_stage.forward_embed_ms.max(t.forward_embed_ms);
        max_stage.forward_encoder_ms = max_stage.forward_encoder_ms.max(t.forward_encoder_ms);
        max_stage.forward_encoder_layer_max_ms = max_stage
            .forward_encoder_layer_max_ms
            .max(t.forward_encoder_layer_max_ms);
        max_stage.forward_decoder_ms = max_stage.forward_decoder_ms.max(t.forward_decoder_ms);
        max_stage.forward_decoder_layer_max_ms = max_stage
            .forward_decoder_layer_max_ms
            .max(t.forward_decoder_layer_max_ms);
        max_stage.forward_logits_ms = max_stage.forward_logits_ms.max(t.forward_logits_ms);
        max_stage.forward_block_pre_attn_ms = max_stage
            .forward_block_pre_attn_ms
            .max(t.forward_block_pre_attn_ms);
        max_stage.forward_block_attention_ms = max_stage
            .forward_block_attention_ms
            .max(t.forward_block_attention_ms);
        max_stage.forward_block_post_attn_ms = max_stage
            .forward_block_post_attn_ms
            .max(t.forward_block_post_attn_ms);
        max_stage.forward_block_mlp_ms = max_stage.forward_block_mlp_ms.max(t.forward_block_mlp_ms);
        max_stage.backward_block_recompute_ms = max_stage
            .backward_block_recompute_ms
            .max(t.backward_block_recompute_ms);
        max_stage.backward_block_mlp_ms =
            max_stage.backward_block_mlp_ms.max(t.backward_block_mlp_ms);
        max_stage.backward_block_mlp_residual_ms = max_stage
            .backward_block_mlp_residual_ms
            .max(t.backward_block_mlp_residual_ms);
        max_stage.backward_block_mlp_down_ms = max_stage
            .backward_block_mlp_down_ms
            .max(t.backward_block_mlp_down_ms);
        max_stage.backward_block_mlp_act_ms = max_stage
            .backward_block_mlp_act_ms
            .max(t.backward_block_mlp_act_ms);
        max_stage.backward_block_mlp_up_ms = max_stage
            .backward_block_mlp_up_ms
            .max(t.backward_block_mlp_up_ms);
        max_stage.backward_block_mlp_norm_ms = max_stage
            .backward_block_mlp_norm_ms
            .max(t.backward_block_mlp_norm_ms);
        max_stage.backward_block_attn_out_ms = max_stage
            .backward_block_attn_out_ms
            .max(t.backward_block_attn_out_ms);
        max_stage.backward_block_attn_out_residual_ms = max_stage
            .backward_block_attn_out_residual_ms
            .max(t.backward_block_attn_out_residual_ms);
        max_stage.backward_block_attn_out_proj_ms = max_stage
            .backward_block_attn_out_proj_ms
            .max(t.backward_block_attn_out_proj_ms);
        max_stage.backward_block_attn_out_gate_xsa_ms = max_stage
            .backward_block_attn_out_gate_xsa_ms
            .max(t.backward_block_attn_out_gate_xsa_ms);
        max_stage.backward_block_attention_ms = max_stage
            .backward_block_attention_ms
            .max(t.backward_block_attention_ms);
        max_stage.backward_block_attention_sdpa_ms = max_stage
            .backward_block_attention_sdpa_ms
            .max(t.backward_block_attention_sdpa_ms);
        max_stage.backward_block_attention_xsa_accum_ms = max_stage
            .backward_block_attention_xsa_accum_ms
            .max(t.backward_block_attention_xsa_accum_ms);
        max_stage.backward_block_qkv_ms =
            max_stage.backward_block_qkv_ms.max(t.backward_block_qkv_ms);
        max_stage.backward_block_qkv_rope_ms = max_stage
            .backward_block_qkv_rope_ms
            .max(t.backward_block_qkv_rope_ms);
        max_stage.backward_block_qkv_proj_ms = max_stage
            .backward_block_qkv_proj_ms
            .max(t.backward_block_qkv_proj_ms);
        max_stage.backward_block_qkv_ve_ms = max_stage
            .backward_block_qkv_ve_ms
            .max(t.backward_block_qkv_ve_ms);
        max_stage.backward_block_qkv_norm_resid_ms = max_stage
            .backward_block_qkv_norm_resid_ms
            .max(t.backward_block_qkv_norm_resid_ms);
        max_stage.output_ms = max_stage.output_ms.max(t.output_ms);
        max_stage.decoder_ms = max_stage.decoder_ms.max(t.decoder_ms);
        max_stage.encoder_ms = max_stage.encoder_ms.max(t.encoder_ms);
        max_stage.tail_ms = max_stage.tail_ms.max(t.tail_ms);
    }
    timing.cuda_backward_forward_ms += max_stage.forward_ms;
    timing.cuda_backward_forward_embed_ms += max_stage.forward_embed_ms;
    timing.cuda_backward_forward_encoder_ms += max_stage.forward_encoder_ms;
    timing.cuda_backward_forward_encoder_layer_max_ms += max_stage.forward_encoder_layer_max_ms;
    timing.cuda_backward_forward_decoder_ms += max_stage.forward_decoder_ms;
    timing.cuda_backward_forward_decoder_layer_max_ms += max_stage.forward_decoder_layer_max_ms;
    timing.cuda_backward_forward_logits_ms += max_stage.forward_logits_ms;
    timing.cuda_backward_forward_block_pre_attn_ms += max_stage.forward_block_pre_attn_ms;
    timing.cuda_backward_forward_block_attention_ms += max_stage.forward_block_attention_ms;
    timing.cuda_backward_forward_block_post_attn_ms += max_stage.forward_block_post_attn_ms;
    timing.cuda_backward_forward_block_mlp_ms += max_stage.forward_block_mlp_ms;
    timing.cuda_backward_block_recompute_ms += max_stage.backward_block_recompute_ms;
    timing.cuda_backward_block_mlp_ms += max_stage.backward_block_mlp_ms;
    timing.cuda_backward_block_mlp_residual_ms += max_stage.backward_block_mlp_residual_ms;
    timing.cuda_backward_block_mlp_down_ms += max_stage.backward_block_mlp_down_ms;
    timing.cuda_backward_block_mlp_act_ms += max_stage.backward_block_mlp_act_ms;
    timing.cuda_backward_block_mlp_up_ms += max_stage.backward_block_mlp_up_ms;
    timing.cuda_backward_block_mlp_norm_ms += max_stage.backward_block_mlp_norm_ms;
    timing.cuda_backward_block_attn_out_ms += max_stage.backward_block_attn_out_ms;
    timing.cuda_backward_block_attn_out_residual_ms +=
        max_stage.backward_block_attn_out_residual_ms;
    timing.cuda_backward_block_attn_out_proj_ms += max_stage.backward_block_attn_out_proj_ms;
    timing.cuda_backward_block_attn_out_gate_xsa_ms +=
        max_stage.backward_block_attn_out_gate_xsa_ms;
    timing.cuda_backward_block_attention_ms += max_stage.backward_block_attention_ms;
    timing.cuda_backward_block_attention_sdpa_ms += max_stage.backward_block_attention_sdpa_ms;
    timing.cuda_backward_block_attention_xsa_accum_ms +=
        max_stage.backward_block_attention_xsa_accum_ms;
    timing.cuda_backward_block_qkv_ms += max_stage.backward_block_qkv_ms;
    timing.cuda_backward_block_qkv_rope_ms += max_stage.backward_block_qkv_rope_ms;
    timing.cuda_backward_block_qkv_proj_ms += max_stage.backward_block_qkv_proj_ms;
    timing.cuda_backward_block_qkv_ve_ms += max_stage.backward_block_qkv_ve_ms;
    timing.cuda_backward_block_qkv_norm_resid_ms += max_stage.backward_block_qkv_norm_resid_ms;
    timing.cuda_backward_output_ms += max_stage.output_ms;
    timing.cuda_backward_decoder_ms += max_stage.decoder_ms;
    timing.cuda_backward_encoder_ms += max_stage.encoder_ms;
    timing.cuda_backward_tail_ms += max_stage.tail_ms;
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default)]
struct ReplicaBackwardLaunchResult {
    loss: f32,
    loss_count: usize,
    cuda_zero_grads_ms: f64,
    cuda_h2d_ms: f64,
    cuda_backward_ms: f64,
    host_batch_flatten_calls: usize,
    host_to_device_batch_bytes: usize,
    device_batch_ready_steps: usize,
    device_batch_missing_steps: usize,
    backward_bucket_flushes: usize,
    backward_bucket_layers: usize,
    backward_bucket_matrix_reductions: usize,
}

#[cfg(feature = "cuda")]
fn collect_gpu_grad_refs(grads: &pg_model::gpu::GpuGradBuffers) -> Vec<&pg_core::GpuTensor> {
    let mut grad_refs: Vec<&pg_core::GpuTensor> = vec![
        &grads.tok_emb,
        &grads.bigram_embed,
        &grads.bigram_proj,
        &grads.bigram_scale,
        &grads.smear_gate,
        &grads.skip_weights,
        &grads.qo_bank,
        &grads.kv_bank,
        &grads.mlp_up_bank,
        &grads.mlp_down_bank,
        &grads.ve_embed,
        &grads.ve_proj,
        &grads.ve_scale,
        &grads.ve_layer_scales,
    ];
    grad_refs.extend(grads.block_attn_scale.iter());
    grad_refs.extend(grads.block_mlp_scale.iter());
    grad_refs.extend(grads.block_resid_mix.iter());
    grad_refs.extend(grads.block_q_gain.iter());
    grad_refs.extend(grads.block_attn_gate_weight.iter());
    grad_refs.extend(grads.block_attn_gate_bias.iter());
    grad_refs.extend(grads.block_sparse_attn_gate_weight.iter());
    grad_refs
}

#[cfg(feature = "cuda")]
fn cuda_fast_apply_updates(
    runtime: &mut CudaSingleFastRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<()> {
    cuda_fast_apply_updates_inner(runtime, train_config, step, lr_scale, true, true, true)
}

#[cfg(feature = "cuda")]
fn cuda_fast_apply_non_bank_updates_unclipped_no_shadow_refresh(
    runtime: &mut CudaSingleFastRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<()> {
    cuda_fast_apply_updates_inner(runtime, train_config, step, lr_scale, false, false, false)
}

#[cfg(feature = "cuda")]
fn cuda_fast_apply_updates_inner(
    runtime: &mut CudaSingleFastRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
    update_banks: bool,
    clip_grads: bool,
    refresh_bf16_shadows: bool,
) -> PgResult<()> {
    let (gpu_optimizer, grad_norm_scratch, gpu_model, gpu_grads) = (
        &mut runtime.gpu_optimizer,
        &mut runtime.grad_norm_scratch,
        &runtime.gpu_model,
        &runtime.gpu_grads,
    );
    if clip_grads && train_config.grad_clip_norm > 0.0 {
        let grad_refs = collect_gpu_grad_refs(gpu_grads);
        gpu_optimizer.clip_grad_norm(
            &gpu_model.kernels,
            &grad_refs,
            train_config.grad_clip_norm,
            grad_norm_scratch,
        )?;
    }

    let embed_hyper = AdamWHyper {
        lr: train_config.embed_lr * lr_scale,
        beta1: train_config.adam_beta1,
        beta2: train_config.adam_beta2,
        eps: train_config.adam_eps,
        weight_decay: train_config.adam_wd,
    };
    let scalar_hyper = AdamWHyper {
        lr: train_config.scalar_lr * lr_scale,
        beta1: train_config.adam_beta1,
        beta2: train_config.adam_beta2,
        eps: train_config.adam_eps,
        weight_decay: train_config.adam_wd,
    };
    if update_banks {
        let matrix_lr = train_config.matrix_lr * lr_scale;
        runtime.gpu_muon.lr = matrix_lr;
        runtime.gpu_muon.momentum = train_config.muon_momentum_at(step);
        runtime.gpu_muon.weight_decay = train_config.muon_wd;
        runtime.gpu_muon.step_bank(
            &runtime.gpu_model.kernels,
            0,
            &runtime.gpu_model.weights.qo_bank,
            &runtime.gpu_grads.qo_bank,
        )?;
        runtime.gpu_muon.step_bank(
            &runtime.gpu_model.kernels,
            1,
            &runtime.gpu_model.weights.kv_bank,
            &runtime.gpu_grads.kv_bank,
        )?;
        runtime.gpu_muon.step_bank(
            &runtime.gpu_model.kernels,
            2,
            &runtime.gpu_model.weights.mlp_up_bank,
            &runtime.gpu_grads.mlp_up_bank,
        )?;
        runtime.gpu_muon.step_bank(
            &runtime.gpu_model.kernels,
            3,
            &runtime.gpu_model.weights.mlp_down_bank,
            &runtime.gpu_grads.mlp_down_bank,
        )?;
    }

    if gpu_adamw_bf16_shadow_update_enabled() {
        runtime.gpu_optimizer.adamw_step_bf16_shadow(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.tok_emb,
            &runtime.gpu_model.weights.tok_emb_bf16,
            &runtime.gpu_grads.tok_emb,
            &mut runtime.state_tok_emb,
            embed_hyper,
        )?;
    } else {
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.tok_emb,
            &runtime.gpu_grads.tok_emb,
            &mut runtime.state_tok_emb,
            embed_hyper,
        )?;
    }
    if runtime.gpu_model.config.bigram_vocab_size > 0 && runtime.gpu_model.config.bigram_dim > 0 {
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.bigram_embed,
            &runtime.gpu_grads.bigram_embed,
            &mut runtime.state_bigram_embed,
            embed_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.bigram_proj,
            &runtime.gpu_grads.bigram_proj,
            &mut runtime.state_bigram_proj,
            embed_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.bigram_scale_param,
            &runtime.gpu_grads.bigram_scale,
            &mut runtime.state_bigram_scale,
            scalar_hyper,
        )?;
    }
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.smear_gate,
        &runtime.gpu_grads.smear_gate,
        &mut runtime.state_smear_gate,
        scalar_hyper,
    )?;
    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.skip_weights,
        &runtime.gpu_grads.skip_weights,
        &mut runtime.state_skip_weights,
        scalar_hyper,
    )?;
    if runtime.gpu_model.config.ve_enabled {
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.ve_embed,
            &runtime.gpu_grads.ve_embed,
            &mut runtime.state_ve_embed,
            embed_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.ve_proj,
            &runtime.gpu_grads.ve_proj,
            &mut runtime.state_ve_proj,
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.ve_scale_param,
            &runtime.gpu_grads.ve_scale,
            &mut runtime.state_ve_scale,
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.ve_layer_scales,
            &runtime.gpu_grads.ve_layer_scales,
            &mut runtime.state_ve_layer_scales,
            scalar_hyper,
        )?;
    }
    for i in 0..runtime.state_attn_scale.len() {
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.attn_scales[i],
            &runtime.gpu_grads.block_attn_scale[i],
            &mut runtime.state_attn_scale[i],
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.mlp_scales[i],
            &runtime.gpu_grads.block_mlp_scale[i],
            &mut runtime.state_mlp_scale[i],
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.resid_mix[i],
            &runtime.gpu_grads.block_resid_mix[i],
            &mut runtime.state_resid_mix[i],
            scalar_hyper,
        )?;
        runtime.gpu_optimizer.adamw_step(
            &runtime.gpu_model.kernels,
            &runtime.gpu_model.weights.q_gains[i],
            &runtime.gpu_grads.block_q_gain[i],
            &mut runtime.state_q_gain[i],
            scalar_hyper,
        )?;
        if runtime.gpu_model.config.attn_out_gate_enabled {
            runtime.gpu_optimizer.adamw_step(
                &runtime.gpu_model.kernels,
                &runtime.gpu_model.weights.attn_gate_weights[i],
                &runtime.gpu_grads.block_attn_gate_weight[i],
                &mut runtime.state_attn_gate_weight[i],
                scalar_hyper,
            )?;
            runtime.gpu_optimizer.adamw_step(
                &runtime.gpu_model.kernels,
                &runtime.gpu_model.weights.attn_gate_biases[i],
                &runtime.gpu_grads.block_attn_gate_bias[i],
                &mut runtime.state_attn_gate_bias[i],
                scalar_hyper,
            )?;
        }
        if runtime.gpu_model.config.sparse_attn_gate_enabled {
            runtime.gpu_optimizer.adamw_step(
                &runtime.gpu_model.kernels,
                &runtime.gpu_model.weights.sparse_attn_gate_weights[i],
                &runtime.gpu_grads.block_sparse_attn_gate_weight[i],
                &mut runtime.state_sparse_attn_gate_weight[i],
                scalar_hyper,
            )?;
        }
    }

    if gpu_host_scalar_updates_enabled() {
        let stream = runtime.gpu_model.gemm.stream().clone();
        stream
            .synchronize()
            .map_err(|e| pg_core::PgError::InvalidOp(format!("stream sync failed: {:?}", e)))?;
        runtime.gpu_model.weights.bigram_scale =
            download_gpu_f32(&runtime.gpu_model.weights.bigram_scale_param)?[0];
        if runtime.gpu_model.config.ve_enabled {
            runtime.gpu_model.weights.ve_scale =
                download_gpu_f32(&runtime.gpu_model.weights.ve_scale_param)?[0];
            runtime.gpu_model.weights.ve_layer_scales_host =
                download_gpu_f32(&runtime.gpu_model.weights.ve_layer_scales)?;
        }
    }
    if refresh_bf16_shadows {
        runtime.gpu_model.refresh_bf16_shadows()?;
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_host_scalar_updates_enabled() -> bool {
    !matches!(
        std::env::var("PG_GPU_HOST_SCALAR_UPDATES")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "cuda")]
fn gpu_adamw_bf16_shadow_update_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_ADAMW_BF16_SHADOW_UPDATE")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn scale_gpu_tensor(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    tensor: &pg_core::GpuTensor,
    scale: f32,
) -> PgResult<()> {
    kernels.scale_inplace(
        pg_kernels::gpu_kernels::CudaPtr(tensor.cu_ptr(kernels.stream())?),
        scale,
        tensor.numel() as u32,
    )
}

#[cfg(feature = "cuda")]
fn copy_gpu_tensor(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    src: &pg_core::GpuTensor,
    dst: &pg_core::GpuTensor,
) -> PgResult<()> {
    if src.dtype() != pg_core::DType::F32 || dst.dtype() != pg_core::DType::F32 {
        return Err(pg_core::PgError::InvalidOp(format!(
            "copy_gpu_tensor requires F32 tensors, got {:?} -> {:?}",
            src.dtype(),
            dst.dtype()
        )));
    }
    if src.numel() != dst.numel() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "copy_gpu_tensor numel mismatch: {} -> {}",
            src.numel(),
            dst.numel()
        )));
    }
    kernels.copy_fwd(
        pg_kernels::gpu_kernels::CudaPtr(src.cu_ptr(kernels.stream())?),
        pg_kernels::gpu_kernels::CudaPtr(dst.cu_ptr(kernels.stream())?),
        src.numel() as u32,
    )
}

#[cfg(feature = "cuda")]
fn all_grad_numel(grads: &pg_model::gpu::GpuGradBuffers) -> usize {
    collect_gpu_grad_refs(grads)
        .into_iter()
        .map(pg_core::GpuTensor::numel)
        .sum()
}

#[cfg(feature = "cuda")]
fn non_bank_grad_numel(grads: &pg_model::gpu::GpuGradBuffers) -> usize {
    let mut total = 0usize;
    total += grads.tok_emb.numel();
    total += grads.bigram_embed.numel();
    total += grads.bigram_proj.numel();
    total += grads.bigram_scale.numel();
    total += grads.smear_gate.numel();
    total += grads.skip_weights.numel();
    total += grads.ve_embed.numel();
    total += grads.ve_proj.numel();
    total += grads.ve_scale.numel();
    total += grads.ve_layer_scales.numel();
    total += grads
        .block_attn_scale
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total += grads
        .block_mlp_scale
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total += grads
        .block_resid_mix
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total += grads
        .block_q_gain
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total += grads
        .block_attn_gate_weight
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total += grads
        .block_attn_gate_bias
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total += grads
        .block_sparse_attn_gate_weight
        .iter()
        .map(pg_core::GpuTensor::numel)
        .sum::<usize>();
    total
}

#[cfg(feature = "cuda")]
fn pack_all_gpu_grads(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    grads: &pg_model::gpu::GpuGradBuffers,
    packed: &pg_core::GpuTensor,
) -> PgResult<()> {
    let mut offset = 0usize;
    macro_rules! pack_one {
        ($tensor:expr) => {{
            let tensor = $tensor;
            let len = tensor.numel();
            if len > 0 {
                let end = offset + len;
                let dst = packed.slice_range(offset, end)?;
                copy_gpu_tensor(kernels, tensor, &dst)?;
                offset = end;
            }
        }};
    }

    pack_one!(&grads.tok_emb);
    pack_one!(&grads.bigram_embed);
    pack_one!(&grads.bigram_proj);
    pack_one!(&grads.bigram_scale);
    pack_one!(&grads.smear_gate);
    pack_one!(&grads.skip_weights);
    pack_one!(&grads.qo_bank);
    pack_one!(&grads.kv_bank);
    pack_one!(&grads.mlp_up_bank);
    pack_one!(&grads.mlp_down_bank);
    pack_one!(&grads.ve_embed);
    pack_one!(&grads.ve_proj);
    pack_one!(&grads.ve_scale);
    pack_one!(&grads.ve_layer_scales);
    for tensor in &grads.block_attn_scale {
        pack_one!(tensor);
    }
    for tensor in &grads.block_mlp_scale {
        pack_one!(tensor);
    }
    for tensor in &grads.block_resid_mix {
        pack_one!(tensor);
    }
    for tensor in &grads.block_q_gain {
        pack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_weight {
        pack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_bias {
        pack_one!(tensor);
    }
    for tensor in &grads.block_sparse_attn_gate_weight {
        pack_one!(tensor);
    }
    if offset != packed.numel() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "packed all-grad length mismatch: wrote {}, buffer has {}",
            offset,
            packed.numel()
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn unpack_all_gpu_grads(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    packed: &pg_core::GpuTensor,
    grads: &mut pg_model::gpu::GpuGradBuffers,
) -> PgResult<()> {
    let mut offset = 0usize;
    macro_rules! unpack_one {
        ($tensor:expr) => {{
            let tensor = $tensor;
            let len = tensor.numel();
            if len > 0 {
                let end = offset + len;
                let src = packed.slice_range(offset, end)?;
                copy_gpu_tensor(kernels, &src, tensor)?;
                offset = end;
            }
        }};
    }

    unpack_one!(&grads.tok_emb);
    unpack_one!(&grads.bigram_embed);
    unpack_one!(&grads.bigram_proj);
    unpack_one!(&grads.bigram_scale);
    unpack_one!(&grads.smear_gate);
    unpack_one!(&grads.skip_weights);
    unpack_one!(&grads.qo_bank);
    unpack_one!(&grads.kv_bank);
    unpack_one!(&grads.mlp_up_bank);
    unpack_one!(&grads.mlp_down_bank);
    unpack_one!(&grads.ve_embed);
    unpack_one!(&grads.ve_proj);
    unpack_one!(&grads.ve_scale);
    unpack_one!(&grads.ve_layer_scales);
    for tensor in &grads.block_attn_scale {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_mlp_scale {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_resid_mix {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_q_gain {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_weight {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_bias {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_sparse_attn_gate_weight {
        unpack_one!(tensor);
    }
    if offset != packed.numel() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "unpacked all-grad length mismatch: read {}, buffer has {}",
            offset,
            packed.numel()
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn pack_non_bank_gpu_grads(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    grads: &pg_model::gpu::GpuGradBuffers,
    packed: &pg_core::GpuTensor,
) -> PgResult<()> {
    let mut offset = 0usize;
    macro_rules! pack_one {
        ($tensor:expr) => {{
            let tensor = $tensor;
            let len = tensor.numel();
            if len > 0 {
                let end = offset + len;
                let dst = packed.slice_range(offset, end)?;
                copy_gpu_tensor(kernels, tensor, &dst)?;
                offset = end;
            }
        }};
    }

    pack_one!(&grads.tok_emb);
    pack_one!(&grads.bigram_embed);
    pack_one!(&grads.bigram_proj);
    pack_one!(&grads.bigram_scale);
    pack_one!(&grads.smear_gate);
    pack_one!(&grads.skip_weights);
    pack_one!(&grads.ve_embed);
    pack_one!(&grads.ve_proj);
    pack_one!(&grads.ve_scale);
    pack_one!(&grads.ve_layer_scales);
    for tensor in &grads.block_attn_scale {
        pack_one!(tensor);
    }
    for tensor in &grads.block_mlp_scale {
        pack_one!(tensor);
    }
    for tensor in &grads.block_resid_mix {
        pack_one!(tensor);
    }
    for tensor in &grads.block_q_gain {
        pack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_weight {
        pack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_bias {
        pack_one!(tensor);
    }
    for tensor in &grads.block_sparse_attn_gate_weight {
        pack_one!(tensor);
    }
    if offset != packed.numel() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "packed non-bank grad length mismatch: wrote {}, buffer has {}",
            offset,
            packed.numel()
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn unpack_non_bank_gpu_grads(
    kernels: &pg_kernels::gpu_kernels::GpuKernels,
    packed: &pg_core::GpuTensor,
    grads: &mut pg_model::gpu::GpuGradBuffers,
) -> PgResult<()> {
    let mut offset = 0usize;
    macro_rules! unpack_one {
        ($tensor:expr) => {{
            let tensor = $tensor;
            let len = tensor.numel();
            if len > 0 {
                let end = offset + len;
                let src = packed.slice_range(offset, end)?;
                copy_gpu_tensor(kernels, &src, tensor)?;
                offset = end;
            }
        }};
    }

    unpack_one!(&grads.tok_emb);
    unpack_one!(&grads.bigram_embed);
    unpack_one!(&grads.bigram_proj);
    unpack_one!(&grads.bigram_scale);
    unpack_one!(&grads.smear_gate);
    unpack_one!(&grads.skip_weights);
    unpack_one!(&grads.ve_embed);
    unpack_one!(&grads.ve_proj);
    unpack_one!(&grads.ve_scale);
    unpack_one!(&grads.ve_layer_scales);
    for tensor in &grads.block_attn_scale {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_mlp_scale {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_resid_mix {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_q_gain {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_weight {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_attn_gate_bias {
        unpack_one!(tensor);
    }
    for tensor in &grads.block_sparse_attn_gate_weight {
        unpack_one!(tensor);
    }
    if offset != packed.numel() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "unpacked non-bank grad length mismatch: read {}, buffer has {}",
            offset,
            packed.numel()
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_all_reduce_average(
    runtime: &mut CudaDistributedRuntime,
    local_microbatches: usize,
) -> PgResult<()> {
    if runtime.all_grad_sync.len() != runtime.replicas.len() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "all-grad sync buffer count {} does not match replica count {}",
            runtime.all_grad_sync.len(),
            runtime.replicas.len()
        )));
    }
    for rank in 0..runtime.replicas.len() {
        let replica = &runtime.replicas[rank];
        let packed = &runtime.all_grad_sync[rank].packed_grad;
        if packed.numel() != all_grad_numel(&replica.gpu_grads) {
            return Err(pg_core::PgError::InvalidOp(format!(
                "rank {rank} packed all-grad buffer has {} elements, expected {}",
                packed.numel(),
                all_grad_numel(&replica.gpu_grads)
            )));
        }
        pack_all_gpu_grads(&replica.gpu_model.kernels, &replica.gpu_grads, packed)?;
    }

    // Same side-stream pattern as the sharded-parallel-muon path: when
    // side-stream collectives are enabled, route the all-reduce through the
    // side comm pool and bracket it with reusable cross-stream events.
    let use_side_comms_all = runtime.comm_side_comms.is_some();
    if use_side_comms_all {
        nccl_side_stream_main_to_side(
            &runtime.replicas,
            &runtime.comm_side_streams,
            &runtime.comm_side_events,
            "all-grad all-reduce",
        )?;
    }
    let active_comms_all = if use_side_comms_all {
        runtime
            .comm_side_comms
            .as_ref()
            .expect("checked use_side_comms_all")
    } else {
        &runtime.comms
    };
    cudarc::nccl::group_start()
        .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
    for (buffers, comm) in runtime
        .all_grad_sync
        .iter_mut()
        .zip(active_comms_all.iter())
    {
        comm.all_reduce_sum_tensor_f32_in_place(&mut buffers.packed_grad)?;
    }
    cudarc::nccl::group_end()
        .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
    if use_side_comms_all {
        nccl_side_stream_side_to_main(
            &runtime.replicas,
            &runtime.comm_side_streams,
            &runtime.comm_side_events,
            "all-grad all-reduce",
        )?;
    }

    let inv_world = 1.0f32 / (runtime.replicas.len() * local_microbatches.max(1)) as f32;
    for rank in 0..runtime.replicas.len() {
        let replica = &mut runtime.replicas[rank];
        let packed = &runtime.all_grad_sync[rank].packed_grad;
        scale_gpu_tensor(&replica.gpu_model.kernels, packed, inv_world)?;
        unpack_all_gpu_grads(&replica.gpu_model.kernels, packed, &mut replica.gpu_grads)?;
    }
    runtime.distributed_sync = true;
    Ok(())
}

#[cfg(feature = "cuda")]
fn pack_non_bank_sync_buffers(runtime: &mut CudaDistributedRuntime) -> PgResult<()> {
    if runtime.non_bank_sync.len() != runtime.replicas.len() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "non-bank sync buffer count {} does not match replica count {}",
            runtime.non_bank_sync.len(),
            runtime.replicas.len()
        )));
    }
    for rank in 0..runtime.replicas.len() {
        let replica = &runtime.replicas[rank];
        let packed = &runtime.non_bank_sync[rank].packed_grad;
        if packed.numel() != non_bank_grad_numel(&replica.gpu_grads) {
            return Err(pg_core::PgError::InvalidOp(format!(
                "rank {rank} packed non-bank grad has {} elements, expected {}",
                packed.numel(),
                non_bank_grad_numel(&replica.gpu_grads)
            )));
        }
        pack_non_bank_gpu_grads(&replica.gpu_model.kernels, &replica.gpu_grads, packed)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn bank_param(replica: &CudaSingleFastRuntime, bank_idx: usize) -> PgResult<&pg_core::GpuTensor> {
    match bank_idx {
        0 => Ok(&replica.gpu_model.weights.qo_bank),
        1 => Ok(&replica.gpu_model.weights.kv_bank),
        2 => Ok(&replica.gpu_model.weights.mlp_up_bank),
        3 => Ok(&replica.gpu_model.weights.mlp_down_bank),
        _ => Err(pg_core::PgError::InvalidOp(format!(
            "invalid bank index {bank_idx}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn bank_param_bf16(
    replica: &CudaSingleFastRuntime,
    bank_idx: usize,
) -> PgResult<&pg_core::GpuTensor> {
    match bank_idx {
        0 => Ok(&replica.gpu_model.weights.qo_bank_bf16),
        1 => Ok(&replica.gpu_model.weights.kv_bank_bf16),
        2 => Ok(&replica.gpu_model.weights.mlp_up_bank_bf16),
        3 => Ok(&replica.gpu_model.weights.mlp_down_bank_bf16),
        _ => Err(pg_core::PgError::InvalidOp(format!(
            "invalid bank index {bank_idx}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn nccl_side_stream_main_to_side(
    replicas: &[CudaSingleFastRuntime],
    side_streams: &[std::sync::Arc<cudarc::driver::CudaStream>],
    side_events: &[NcclSideStreamEvents],
    label: &str,
) -> PgResult<()> {
    if side_streams.len() != replicas.len() || side_events.len() != replicas.len() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "nccl side-stream handoff {label}: replica/stream/event count mismatch (replicas={}, side_streams={}, events={})",
            replicas.len(),
            side_streams.len(),
            side_events.len()
        )));
    }
    for rank in 0..replicas.len() {
        let main_stream = replicas[rank].gpu_model.gemm.stream();
        let event = &side_events[rank].main_to_side;
        event.record(main_stream).map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "nccl side-stream collectives: {label} rank {rank} main->side event record failed: {e:?}"
            ))
        })?;
        side_streams[rank].wait(event).map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "nccl side-stream collectives: {label} rank {rank} side wait failed: {e:?}"
            ))
        })?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn nccl_side_stream_side_to_main(
    replicas: &[CudaSingleFastRuntime],
    side_streams: &[std::sync::Arc<cudarc::driver::CudaStream>],
    side_events: &[NcclSideStreamEvents],
    label: &str,
) -> PgResult<()> {
    if side_streams.len() != replicas.len() || side_events.len() != replicas.len() {
        return Err(pg_core::PgError::InvalidOp(format!(
            "nccl side-stream handoff {label}: replica/stream/event count mismatch (replicas={}, side_streams={}, events={})",
            replicas.len(),
            side_streams.len(),
            side_events.len()
        )));
    }
    for rank in 0..replicas.len() {
        let main_stream = replicas[rank].gpu_model.gemm.stream();
        let event = &side_events[rank].side_to_main;
        event.record(&side_streams[rank]).map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "nccl side-stream collectives: {label} rank {rank} side->main event record failed: {e:?}"
            ))
        })?;
        main_stream.wait(event).map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "nccl side-stream collectives: {label} rank {rank} main wait failed: {e:?}"
            ))
        })?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
struct BackwardBucketOverlapObserver<'a> {
    rank: usize,
    world_size: usize,
    num_layers: usize,
    bucket_layers: usize,
    pending_layers: Vec<usize>,
    bucket_flushes: usize,
    bucket_layers_reduced: usize,
    bucket_matrix_reductions: usize,
    comm: &'a pg_core::nccl::NcclComm,
    side_stream: &'a std::sync::Arc<cudarc::driver::CudaStream>,
    side_events: &'a NcclSideStreamEvents,
    sharded: &'a mut ShardedParallelMuonReplica,
}

#[cfg(feature = "cuda")]
impl<'a> BackwardBucketOverlapObserver<'a> {
    fn prepare_bank_matrix_bf16(
        &mut self,
        model: &pg_model::gpu::GpuModel,
        bank_idx: usize,
        matrix_index: usize,
        grad_bank: &pg_core::GpuTensor,
    ) -> PgResult<()> {
        if !sharded_bank_grad_bf16_wire_enabled_for_audit() {
            return Ok(());
        }
        let kernels = &model.kernels;
        let buffers = &mut self.sharded.banks[bank_idx];
        let (owner_rank, _) = sharded_bank_owner(buffers, matrix_index)?;
        if owner_rank >= self.world_size {
            return Ok(());
        }

        let grad_matrix = grad_bank.slice_first(matrix_index)?;
        let dst = buffers.padded_grad_bf16.slice_first(matrix_index)?;
        kernels.f32_to_bf16(
            pg_kernels::gpu_kernels::CudaPtr(grad_matrix.cu_ptr(kernels.stream())?),
            pg_kernels::gpu_kernels::CudaPtr(dst.cu_ptr(kernels.stream())?),
            grad_matrix.numel() as u32,
        )?;
        Ok(())
    }

    fn record_layer_ready(&self, model: &pg_model::gpu::GpuModel, layer: usize) -> PgResult<()> {
        let kernels = &model.kernels;
        let event = &self.side_events.main_to_side;
        event.record(kernels.stream()).map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "backward NCCL bucket overlap: layer {layer} rank {} main->side record failed: {e:?}",
                self.rank
            ))
        })?;
        self.side_stream.wait(event).map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "backward NCCL bucket overlap: layer {layer} rank {} side wait failed: {e:?}",
                self.rank
            ))
        })
    }

    fn reduce_prepared_bank_matrix(
        &mut self,
        bank_idx: usize,
        matrix_index: usize,
        grad_bank: &pg_core::GpuTensor,
    ) -> PgResult<()> {
        let buffers = &mut self.sharded.banks[bank_idx];
        let (owner_rank, local_idx) = sharded_bank_owner(buffers, matrix_index)?;
        if owner_rank >= self.world_size {
            return Ok(());
        }

        // Fire a per-matrix reduce directly from the stable gradient bank slice
        // into the owner rank's shard slot. This avoids the unsafe staging
        // buffer pattern where the main stream could overwrite a send bucket
        // while the side stream was still reading it. The first production
        // bucket scheduler now supports the same BF16-on-wire switch as the
        // post-backward reduce-scatter path, but still writes the final Muon
        // input through `shard_grad` after the side stream completes.
        if sharded_bank_grad_bf16_wire_enabled_for_audit() {
            let send_bf16 = buffers.padded_grad_bf16.slice_first(matrix_index)?;
            if owner_rank == self.rank {
                let mut recv_bf16 = buffers.shard_grad_bf16.slice_first(local_idx)?;
                self.comm.reduce_sum_tensor_bf16_to_rank(
                    &send_bf16,
                    Some(&mut recv_bf16),
                    owner_rank,
                )?;
            } else {
                self.comm
                    .reduce_sum_tensor_bf16_to_rank(&send_bf16, None, owner_rank)?;
            }
        } else if owner_rank == self.rank {
            let grad_matrix = grad_bank.slice_first(matrix_index)?;
            let mut recv = buffers.shard_grad.slice_first(local_idx)?;
            self.comm
                .reduce_sum_tensor_f32_to_rank(&grad_matrix, Some(&mut recv), owner_rank)?;
        } else {
            let grad_matrix = grad_bank.slice_first(matrix_index)?;
            self.comm
                .reduce_sum_tensor_f32_to_rank(&grad_matrix, None, owner_rank)?;
        }
        self.bucket_matrix_reductions += 1;
        Ok(())
    }

    fn prepare_layer_bf16(
        &mut self,
        model: &pg_model::gpu::GpuModel,
        layer: usize,
        grads: &pg_model::gpu::GpuGradBuffers,
    ) -> PgResult<()> {
        let n = self.num_layers;
        self.prepare_bank_matrix_bf16(model, 0, layer, &grads.qo_bank)?;
        self.prepare_bank_matrix_bf16(model, 0, n + layer, &grads.qo_bank)?;
        self.prepare_bank_matrix_bf16(model, 1, layer, &grads.kv_bank)?;
        self.prepare_bank_matrix_bf16(model, 1, n + layer, &grads.kv_bank)?;
        self.prepare_bank_matrix_bf16(model, 2, layer, &grads.mlp_up_bank)?;
        self.prepare_bank_matrix_bf16(model, 3, layer, &grads.mlp_down_bank)?;
        Ok(())
    }

    fn reduce_prepared_layer(
        &mut self,
        layer: usize,
        grads: &pg_model::gpu::GpuGradBuffers,
    ) -> PgResult<()> {
        let n = self.num_layers;
        self.reduce_prepared_bank_matrix(0, layer, &grads.qo_bank)?;
        self.reduce_prepared_bank_matrix(0, n + layer, &grads.qo_bank)?;
        self.reduce_prepared_bank_matrix(1, layer, &grads.kv_bank)?;
        self.reduce_prepared_bank_matrix(1, n + layer, &grads.kv_bank)?;
        self.reduce_prepared_bank_matrix(2, layer, &grads.mlp_up_bank)?;
        self.reduce_prepared_bank_matrix(3, layer, &grads.mlp_down_bank)?;
        Ok(())
    }

    fn flush_pending_layers(
        &mut self,
        model: &pg_model::gpu::GpuModel,
        grads: &pg_model::gpu::GpuGradBuffers,
    ) -> PgResult<()> {
        if self.pending_layers.is_empty() {
            return Ok(());
        }
        let first = *self.pending_layers.first().unwrap_or(&0);
        let last = *self.pending_layers.last().unwrap_or(&first);
        let pending_len = self.pending_layers.len();
        self.record_layer_ready(model, first)?;
        cudarc::nccl::group_start()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
        let pending_layers = std::mem::take(&mut self.pending_layers);
        let mut result = Ok(());
        for layer in pending_layers {
            if let Err(err) = self.reduce_prepared_layer(layer, grads) {
                result = Err(err);
                break;
            }
        }
        let group_result = cudarc::nccl::group_end()
            .map(|_| ())
            .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")));
        result?;
        group_result.map_err(|e| {
            pg_core::PgError::Nccl(format!(
                "backward NCCL bucket overlap: layers {first}..{last} group_end failed: {e}"
            ))
        })?;
        self.bucket_flushes += 1;
        self.bucket_layers_reduced += pending_len;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl pg_model::gpu::GpuBackwardLayerObserver for BackwardBucketOverlapObserver<'_> {
    fn after_layer(
        &mut self,
        model: &pg_model::gpu::GpuModel,
        layer: usize,
        grads: &pg_model::gpu::GpuGradBuffers,
    ) -> PgResult<()> {
        self.prepare_layer_bf16(model, layer, grads)?;
        self.pending_layers.push(layer);
        if self.pending_layers.len() >= self.bucket_layers.max(1) || layer == 0 {
            self.flush_pending_layers(model, grads)?;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn bank_grad(replica: &CudaSingleFastRuntime, bank_idx: usize) -> PgResult<&pg_core::GpuTensor> {
    match bank_idx {
        0 => Ok(&replica.gpu_grads.qo_bank),
        1 => Ok(&replica.gpu_grads.kv_bank),
        2 => Ok(&replica.gpu_grads.mlp_up_bank),
        3 => Ok(&replica.gpu_grads.mlp_down_bank),
        _ => Err(pg_core::PgError::InvalidOp(format!(
            "invalid bank index {bank_idx}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn cuda_distributed_sharded_parallel_muon_step(
    runtime: &mut CudaDistributedRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
    local_microbatches: usize,
    timing: &mut RunTiming,
) -> PgResult<()> {
    let event_timing = cuda_event_timing_enabled();
    let bank_start_events = if event_timing {
        Some(record_replica_events(runtime)?)
    } else {
        None
    };
    let bank_t0 = Instant::now();
    pack_non_bank_sync_buffers(runtime)?;
    let world_size = runtime.replicas.len();
    let inv_total = 1.0f32 / (world_size * local_microbatches.max(1)) as f32;
    let bank_count = runtime
        .parallel_muon
        .as_ref()
        .ok_or_else(|| {
            pg_core::PgError::InvalidOp("sharded Parallel Muon runtime was not initialized".into())
        })?
        .replicas
        .first()
        .map(|replica| replica.banks.len())
        .unwrap_or(0);
    let bf16_shadow_all_gather = sharded_parallel_muon_bf16_shadow_all_gather_enabled_for_audit();
    let bank_grads_bucketed_during_backward =
        backward_nccl_bucket_overlap_enabled() && runtime.comm_side_comms.is_some();
    let apply_global_clip = train_config.grad_clip_norm > 0.0;
    let bank_phase_timing = sharded_parallel_muon_phase_timing_enabled();
    let mut bank_phase_events = if bank_phase_timing && event_timing {
        Some(record_replica_events(runtime)?)
    } else {
        None
    };
    let mut bank_phase_host_start = if bank_phase_timing && !event_timing {
        Some(Instant::now())
    } else {
        None
    };
    if runtime
        .parallel_muon
        .as_ref()
        .map(|parallel_muon| parallel_muon.replicas.len())
        != Some(world_size)
    {
        return Err(pg_core::PgError::InvalidOp(
            "sharded Parallel Muon replica count mismatch".into(),
        ));
    }

    {
        let parallel_muon = runtime
            .parallel_muon
            .as_mut()
            .expect("validated sharded Parallel Muon runtime");
        for bank_idx in 0..bank_count {
            // Stage all reduced bank shards first. Global clipping must see
            // averaged non-bank grads plus every reduce-scattered bank shard
            // before Muon consumes any bank gradient.
            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let kernels = &replica.gpu_model.kernels;
                let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                if !bank_grads_bucketed_during_backward {
                    let grad = bank_grad(replica, bank_idx)?;
                    if buffers.real_batch < buffers.padded_grad.shape()[0] {
                        let tail = buffers
                            .padded_grad
                            .slice_range(buffers.real_batch, buffers.padded_grad.shape()[0])?;
                        scale_gpu_tensor(kernels, &tail, 0.0)?;
                    }
                    let grad_dst = buffers.padded_grad.slice_range(0, buffers.real_batch)?;
                    kernels.copy_fwd(
                        pg_kernels::gpu_kernels::CudaPtr(grad.cu_ptr(kernels.stream())?),
                        pg_kernels::gpu_kernels::CudaPtr(grad_dst.cu_ptr(kernels.stream())?),
                        grad.numel() as u32,
                    )?;
                    if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                        kernels.f32_to_bf16(
                            pg_kernels::gpu_kernels::CudaPtr(
                                buffers.padded_grad.cu_ptr(kernels.stream())?,
                            ),
                            pg_kernels::gpu_kernels::CudaPtr(
                                buffers.padded_grad_bf16.cu_ptr(kernels.stream())?,
                            ),
                            buffers.padded_grad.numel() as u32,
                        )?;
                    }
                }

                let shard_start = rank * buffers.chunk_batch;
                let shard_end = (shard_start + buffers.chunk_batch).min(buffers.real_batch);
                if shard_start < shard_end {
                    if !bf16_shadow_all_gather || step == 0 {
                        let real_count = shard_end - shard_start;
                        let param = bank_param(replica, bank_idx)?;
                        let param_src = param.slice_range(shard_start, shard_end)?;
                        let param_dst = buffers.shard_param.slice_range(0, real_count)?;
                        kernels.copy_fwd(
                            pg_kernels::gpu_kernels::CudaPtr(param_src.cu_ptr(kernels.stream())?),
                            pg_kernels::gpu_kernels::CudaPtr(param_dst.cu_ptr(kernels.stream())?),
                            param_src.numel() as u32,
                        )?;
                    }
                }
            }
        }
        record_sharded_muon_phase_ms(
            &runtime.replicas,
            &mut bank_phase_events,
            &mut bank_phase_host_start,
            timing,
            |timing| &mut timing.cuda_bank_update_stage_pack_ms,
        )?;

        // Choose which NCCL communicator pool to use. Side-stream collectives
        // run on `comm_side_streams[rank]`. In bucketed mode the side stream
        // already contains per-layer bank reductions issued by the backward
        // observer, so the non-bank all-reduce is queued after those buckets
        // and a single side→main wait below makes every reduced shard visible
        // before clipping and Muon consume it.
        let use_side_comms = runtime.comm_side_comms.is_some();
        if use_side_comms {
            nccl_side_stream_main_to_side(
                &runtime.replicas,
                &runtime.comm_side_streams,
                &runtime.comm_side_events,
                if bank_grads_bucketed_during_backward {
                    "non-bank all-reduce after backward bank buckets"
                } else {
                    "bank grad reduce-scatter"
                },
            )?;
        }
        let active_comms = if use_side_comms {
            runtime
                .comm_side_comms
                .as_ref()
                .expect("checked use_side_comms")
        } else {
            &runtime.comms
        };
        if bank_grads_bucketed_during_backward {
            cudarc::nccl::group_start()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
            for (buffers, comm) in runtime.non_bank_sync.iter_mut().zip(active_comms.iter()) {
                comm.all_reduce_sum_tensor_f32_in_place(&mut buffers.packed_grad)?;
            }
            cudarc::nccl::group_end()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
        } else if sharded_grouped_grad_collectives_enabled_for_audit() {
            cudarc::nccl::group_start()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
            for (buffers, comm) in runtime.non_bank_sync.iter_mut().zip(active_comms.iter()) {
                comm.all_reduce_sum_tensor_f32_in_place(&mut buffers.packed_grad)?;
            }
            for bank_idx in 0..bank_count {
                for rank in 0..world_size {
                    let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                    if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                        active_comms[rank].reduce_scatter_sum_tensor_bf16(
                            &buffers.padded_grad_bf16,
                            &mut buffers.shard_grad_bf16,
                        )?;
                    } else {
                        active_comms[rank].reduce_scatter_sum_tensor_f32(
                            &buffers.padded_grad,
                            &mut buffers.shard_grad,
                        )?;
                    }
                }
            }
            cudarc::nccl::group_end()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
        } else {
            cudarc::nccl::group_start()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
            for (buffers, comm) in runtime.non_bank_sync.iter_mut().zip(active_comms.iter()) {
                comm.all_reduce_sum_tensor_f32_in_place(&mut buffers.packed_grad)?;
            }
            cudarc::nccl::group_end()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;

            for bank_idx in 0..bank_count {
                cudarc::nccl::group_start()
                    .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
                for rank in 0..world_size {
                    let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                    if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                        active_comms[rank].reduce_scatter_sum_tensor_bf16(
                            &buffers.padded_grad_bf16,
                            &mut buffers.shard_grad_bf16,
                        )?;
                    } else {
                        active_comms[rank].reduce_scatter_sum_tensor_f32(
                            &buffers.padded_grad,
                            &mut buffers.shard_grad,
                        )?;
                    }
                }
                cudarc::nccl::group_end()
                    .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
            }
        }
        if use_side_comms {
            nccl_side_stream_side_to_main(
                &runtime.replicas,
                &runtime.comm_side_streams,
                &runtime.comm_side_events,
                if bank_grads_bucketed_during_backward {
                    "backward bank buckets and non-bank all-reduce"
                } else {
                    "bank grad reduce-scatter"
                },
            )?;
        }
        record_sharded_muon_phase_ms(
            &runtime.replicas,
            &mut bank_phase_events,
            &mut bank_phase_host_start,
            timing,
            |timing| &mut timing.cuda_bank_update_stage_grad_collectives_ms,
        )?;
    }

    {
        let parallel_muon = runtime
            .parallel_muon
            .as_mut()
            .expect("validated sharded Parallel Muon runtime");
        for rank in 0..world_size {
            let replica = &mut runtime.replicas[rank];
            let non_bank_packed = &runtime.non_bank_sync[rank].packed_grad;
            let sharded_replica = &mut parallel_muon.replicas[rank];
            sharded_parallel_muon_pre_norm_rank_step_graph_or_eager(
                replica,
                non_bank_packed,
                sharded_replica,
                rank,
                world_size,
                inv_total,
                apply_global_clip,
            )?;
        }
    }
    record_sharded_muon_phase_ms(
        &runtime.replicas,
        &mut bank_phase_events,
        &mut bank_phase_host_start,
        timing,
        |timing| &mut timing.cuda_bank_update_stage_scale_ms,
    )?;
    if apply_global_clip {
        // Exact global-norm clipping for sharded Parallel Muon:
        // - non-bank grads are replicated after all-reduce, so each rank contributes
        //   1/world_size of their squared norm before the scalar all-reduce.
        // - bank grads are reduce-scattered, so each rank contributes only its shard.
        // Grad-norm scalar all-reduce. Same side-stream collective pattern as the
        // non-bank packed reduce above: main stream records "norm scratch ready"
        // after the dot_accumulate kernels, side stream waits + issues the
        // all-reduce, then main stream waits before clip kernels read the reduced
        // scalar.
        let use_side_comms_norm = runtime.comm_side_comms.is_some();
        if use_side_comms_norm {
            nccl_side_stream_main_to_side(
                &runtime.replicas,
                &runtime.comm_side_streams,
                &runtime.comm_side_events,
                "grad-norm all-reduce",
            )?;
        }
        let active_comms_norm = if use_side_comms_norm {
            runtime
                .comm_side_comms
                .as_ref()
                .expect("checked use_side_comms_norm")
        } else {
            &runtime.comms
        };
        cudarc::nccl::group_start()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
        for rank in 0..world_size {
            active_comms_norm[rank].all_reduce_sum_tensor_f32_in_place(
                &mut runtime.replicas[rank].grad_norm_scratch,
            )?;
        }
        cudarc::nccl::group_end()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
        if use_side_comms_norm {
            nccl_side_stream_side_to_main(
                &runtime.replicas,
                &runtime.comm_side_streams,
                &runtime.comm_side_events,
                "grad-norm all-reduce",
            )?;
        }
        record_sharded_muon_phase_ms(
            &runtime.replicas,
            &mut bank_phase_events,
            &mut bank_phase_host_start,
            timing,
            |timing| &mut timing.cuda_bank_update_stage_norm_ms,
        )?;

        for rank in 0..world_size {
            let replica = &mut runtime.replicas[rank];
            let kernels = &replica.gpu_model.kernels;
            let packed = &runtime.non_bank_sync[rank].packed_grad;
            let scratch = pg_kernels::gpu_kernels::CudaPtr(
                replica.grad_norm_scratch.cu_ptr(kernels.stream())?,
            );
            kernels.clip_by_global_norm(
                pg_kernels::gpu_kernels::CudaPtr(packed.cu_ptr(kernels.stream())?),
                scratch,
                train_config.grad_clip_norm,
                packed.numel() as u32,
            )?;
            unpack_non_bank_gpu_grads(kernels, packed, &mut replica.gpu_grads)?;
        }
    } else {
        for rank in 0..world_size {
            let replica = &mut runtime.replicas[rank];
            let kernels = &replica.gpu_model.kernels;
            let packed = &runtime.non_bank_sync[rank].packed_grad;
            unpack_non_bank_gpu_grads(kernels, packed, &mut replica.gpu_grads)?;
        }
    }
    if apply_global_clip && !sharded_parallel_muon_fused_global_clip_enabled_for_audit() {
        let parallel_muon = runtime
            .parallel_muon
            .as_ref()
            .expect("validated sharded Parallel Muon runtime");
        for rank in 0..world_size {
            let replica = &runtime.replicas[rank];
            let kernels = &replica.gpu_model.kernels;
            let scratch = pg_kernels::gpu_kernels::CudaPtr(
                replica.grad_norm_scratch.cu_ptr(kernels.stream())?,
            );
            for buffers in &parallel_muon.replicas[rank].banks {
                let active_batch = sharded_bank_real_batch(buffers, rank);
                if active_batch == 0 {
                    continue;
                }
                let shard_grad = buffers.shard_grad.slice_range(0, active_batch)?;
                kernels.clip_by_global_norm(
                    pg_kernels::gpu_kernels::CudaPtr(shard_grad.cu_ptr(kernels.stream())?),
                    scratch,
                    train_config.grad_clip_norm,
                    shard_grad.numel() as u32,
                )?;
            }
        }
    }
    record_sharded_muon_phase_ms(
        &runtime.replicas,
        &mut bank_phase_events,
        &mut bank_phase_host_start,
        timing,
        |timing| &mut timing.cuda_bank_update_stage_clip_ms,
    )?;

    {
        let parallel_muon = runtime
            .parallel_muon
            .as_mut()
            .expect("validated sharded Parallel Muon runtime");
        let fused_global_clip = sharded_parallel_muon_fused_global_clip_enabled_for_audit();
        if sharded_parallel_muon_parallel_local_enabled() {
            std::thread::scope(|scope| -> PgResult<()> {
                let mut handles = Vec::with_capacity(world_size);
                for (rank, sharded_replica) in parallel_muon.replicas.iter_mut().enumerate() {
                    let replica = &runtime.replicas[rank];
                    let kernels = &replica.gpu_model.kernels;
                    let grad_norm_scratch = &replica.grad_norm_scratch;
                    handles.push(scope.spawn(move || {
                        sharded_parallel_muon_local_rank_step_graph_or_eager(
                            kernels,
                            grad_norm_scratch,
                            sharded_replica,
                            rank,
                            bank_count,
                            train_config,
                            step,
                            lr_scale,
                            apply_global_clip,
                            fused_global_clip,
                            bf16_shadow_all_gather,
                        )
                    }));
                }
                for handle in handles {
                    handle.join().map_err(|_| {
                        pg_core::PgError::InvalidOp(
                            "sharded Parallel Muon local worker panicked".into(),
                        )
                    })??;
                }
                Ok(())
            })?;
        } else {
            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let sharded_replica = &mut parallel_muon.replicas[rank];
                sharded_parallel_muon_local_rank_step_graph_or_eager(
                    &replica.gpu_model.kernels,
                    &replica.grad_norm_scratch,
                    sharded_replica,
                    rank,
                    bank_count,
                    train_config,
                    step,
                    lr_scale,
                    apply_global_clip,
                    fused_global_clip,
                    bf16_shadow_all_gather,
                )?;
            }
        }
        record_sharded_muon_phase_ms(
            &runtime.replicas,
            &mut bank_phase_events,
            &mut bank_phase_host_start,
            timing,
            |timing| &mut timing.cuda_bank_update_stage_muon_ms,
        )?;

        // Param all-gather. Same side-stream collective pattern: side stream waits on the
        // main-stream Muon-step event before reading shard_param; main stream
        // waits on the all-gather completion event before the copy_fwd kernel
        // that writes the gathered param back into the model bank.
        let use_side_comms_gather = runtime.comm_side_comms.is_some();
        if use_side_comms_gather {
            nccl_side_stream_main_to_side(
                &runtime.replicas,
                &runtime.comm_side_streams,
                &runtime.comm_side_events,
                "bank param all-gather",
            )?;
        }
        let active_comms_gather = if use_side_comms_gather {
            runtime
                .comm_side_comms
                .as_ref()
                .expect("checked use_side_comms_gather")
        } else {
            &runtime.comms
        };
        cudarc::nccl::group_start()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
        for bank_idx in 0..bank_count {
            for rank in 0..world_size {
                let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                if bf16_shadow_all_gather {
                    active_comms_gather[rank].all_gather_tensor_bf16(
                        &buffers.shard_param_bf16,
                        &mut buffers.padded_param_bf16,
                    )?;
                } else {
                    active_comms_gather[rank]
                        .all_gather_tensor_f32(&buffers.shard_param, &mut buffers.padded_param)?;
                }
            }
        }
        cudarc::nccl::group_end()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
        if use_side_comms_gather {
            nccl_side_stream_side_to_main(
                &runtime.replicas,
                &runtime.comm_side_streams,
                &runtime.comm_side_events,
                "bank param all-gather",
            )?;
        }
        record_sharded_muon_phase_ms(
            &runtime.replicas,
            &mut bank_phase_events,
            &mut bank_phase_host_start,
            timing,
            |timing| &mut timing.cuda_bank_update_stage_param_gather_ms,
        )?;

        for bank_idx in 0..bank_count {
            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let kernels = &replica.gpu_model.kernels;
                let buffers = &parallel_muon.replicas[rank].banks[bank_idx];
                if bf16_shadow_all_gather {
                    let param_bf16 = bank_param_bf16(replica, bank_idx)?;
                    let gathered_real = buffers
                        .padded_param_bf16
                        .slice_range(0, buffers.real_batch)?;
                    kernels.copy_u16_fwd(
                        pg_kernels::gpu_kernels::CudaPtr(gathered_real.cu_ptr(kernels.stream())?),
                        pg_kernels::gpu_kernels::CudaPtr(param_bf16.cu_ptr(kernels.stream())?),
                        param_bf16.numel() as u32,
                    )?;
                } else {
                    let param = bank_param(replica, bank_idx)?;
                    let gathered_real = buffers.padded_param.slice_range(0, buffers.real_batch)?;
                    kernels.copy_fwd(
                        pg_kernels::gpu_kernels::CudaPtr(gathered_real.cu_ptr(kernels.stream())?),
                        pg_kernels::gpu_kernels::CudaPtr(param.cu_ptr(kernels.stream())?),
                        param.numel() as u32,
                    )?;
                }
            }
        }
        record_sharded_muon_phase_ms(
            &runtime.replicas,
            &mut bank_phase_events,
            &mut bank_phase_host_start,
            timing,
            |timing| &mut timing.cuda_bank_update_stage_copy_back_ms,
        )?;
    }
    let bank_host_ms = bank_t0.elapsed().as_secs_f64() * 1000.0;
    if let Some(starts) = bank_start_events {
        let ends = record_replica_events(runtime)?;
        timing.cuda_bank_update_ms += max_elapsed_replica_events_ms(starts, ends)?;
    } else {
        timing.cuda_bank_update_ms += bank_host_ms;
    }

    let non_bank_start_events = if event_timing {
        Some(record_replica_events(runtime)?)
    } else {
        None
    };
    let non_bank_t0 = Instant::now();
    let adamw_bf16_shadow = gpu_adamw_bf16_shadow_update_enabled();
    std::thread::scope(|scope| -> PgResult<()> {
        let mut handles = Vec::with_capacity(runtime.replicas.len());
        for replica in runtime.replicas.iter_mut() {
            handles.push(scope.spawn(move || {
                cuda_fast_apply_non_bank_updates_unclipped_no_shadow_refresh(
                    replica,
                    train_config,
                    step,
                    lr_scale,
                )?;
                if bf16_shadow_all_gather {
                    if adamw_bf16_shadow {
                        replica
                            .gpu_model
                            .refresh_bf16_qkv_shadow_after_sharded_bank_update()
                    } else {
                        replica
                            .gpu_model
                            .refresh_bf16_non_bank_shadows_after_sharded_bank_update()
                    }
                } else {
                    replica.gpu_model.refresh_bf16_shadows()
                }
            }));
        }
        for handle in handles {
            handle.join().map_err(|_| {
                pg_core::PgError::InvalidOp("non-bank update worker panicked".into())
            })??;
        }
        Ok(())
    })?;
    let non_bank_host_ms = non_bank_t0.elapsed().as_secs_f64() * 1000.0;
    if let Some(starts) = non_bank_start_events {
        let ends = record_replica_events(runtime)?;
        // This intentionally omits CPU scalar downloads; use the default host
        // timing mode when auditing host synchronization overhead.
        timing.cuda_non_bank_update_ms += max_elapsed_replica_events_ms(starts, ends)?;
    } else {
        timing.cuda_non_bank_update_ms += non_bank_host_ms;
    }
    runtime.distributed_sync = true;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_sync_f32_banks_from_sharded_muon(
    runtime: &mut CudaDistributedRuntime,
) -> PgResult<()> {
    let world_size = runtime.replicas.len();
    let bank_count = runtime
        .parallel_muon
        .as_ref()
        .ok_or_else(|| {
            pg_core::PgError::InvalidOp("sharded Parallel Muon runtime was not initialized".into())
        })?
        .replicas
        .first()
        .map(|replica| replica.banks.len())
        .unwrap_or(0);

    {
        let parallel_muon = runtime
            .parallel_muon
            .as_mut()
            .expect("validated sharded Parallel Muon runtime");
        cudarc::nccl::group_start()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
        for bank_idx in 0..bank_count {
            for rank in 0..world_size {
                let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                runtime.comms[rank]
                    .all_gather_tensor_f32(&buffers.shard_param, &mut buffers.padded_param)?;
            }
        }
        cudarc::nccl::group_end()
            .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;

        for bank_idx in 0..bank_count {
            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let kernels = &replica.gpu_model.kernels;
                let buffers = &parallel_muon.replicas[rank].banks[bank_idx];
                let param = bank_param(replica, bank_idx)?;
                let gathered_real = buffers.padded_param.slice_range(0, buffers.real_batch)?;
                kernels.copy_fwd(
                    pg_kernels::gpu_kernels::CudaPtr(gathered_real.cu_ptr(kernels.stream())?),
                    pg_kernels::gpu_kernels::CudaPtr(param.cu_ptr(kernels.stream())?),
                    param.numel() as u32,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_launch_replica_backward(
    replica: &mut CudaSingleFastRuntime,
    rank_idx: usize,
    rank_batches: Option<&Vec<(Vec<u32>, Vec<u32>)>>,
    compute_step_loss: bool,
    runtime_seq_len: usize,
    event_timing: bool,
    observer: Option<&mut dyn pg_model::gpu::GpuBackwardLayerObserver>,
) -> PgResult<ReplicaBackwardLaunchResult> {
    let stream = replica.gpu_model.gemm.stream().clone();
    let backward_host_t0 = Instant::now();
    let backward_start_event = if event_timing {
        Some(
            stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| {
                    pg_core::PgError::InvalidOp(format!("cuda backward event record failed: {e:?}"))
                })?,
        )
    } else {
        None
    };

    let graph_handles_zero_grads =
        !compute_step_loss && observer.is_none() && cuda_backward_graph_enabled();
    let mut result = ReplicaBackwardLaunchResult::default();
    if !graph_handles_zero_grads {
        let zero_host_t0 = Instant::now();
        let zero_start_event = if event_timing {
            Some(
                stream
                    .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                    .map_err(|e| {
                        pg_core::PgError::InvalidOp(format!(
                            "cuda zero-grads event record failed: {e:?}"
                        ))
                    })?,
            )
        } else {
            None
        };
        zero_gpu_grads(
            &replica.gpu_model.kernels,
            &mut replica.gpu_grads,
            &replica.gpu_model.config,
        )?;

        if let Some(start) = zero_start_event {
            let end = stream
                .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|e| {
                    pg_core::PgError::InvalidOp(format!(
                        "cuda zero-grads event record failed: {e:?}"
                    ))
                })?;
            result.cuda_zero_grads_ms = start.elapsed_ms(&end).map_err(|e| {
                pg_core::PgError::InvalidOp(format!("cuda zero-grads event elapsed failed: {e:?}"))
            })? as f64;
        } else {
            result.cuda_zero_grads_ms = zero_host_t0.elapsed().as_secs_f64() * 1000.0;
        }
    }

    let compute_loss = compute_step_loss && rank_idx == 0;
    let mut host_batch_flatten_calls = 0usize;
    let mut host_batch_flatten_bytes = 0usize;
    if let Some(rank_batches) = rank_batches {
        if record_strict_device_batch_enabled() {
            return Err(pg_core::PgError::InvalidOp(
                "record runtime requires device-resident batches; refusing per-rank host microbatch flatten in cuda_distributed_step".into(),
            ));
        }
        host_batch_flatten_calls = 1;
        host_batch_flatten_bytes = rank_batches
            .iter()
            .map(|(input_ids, targets)| {
                (input_ids.len() + targets.len()) * std::mem::size_of::<u32>()
            })
            .sum();
        flatten_microbatches_into(
            rank_batches,
            &mut replica.host_input_ids,
            &mut replica.host_targets,
        );
    }
    let mut local_timing = RunTiming::default();
    result.loss = cuda_fast_accumulate_runtime_grads(
        replica,
        compute_loss,
        runtime_seq_len.max(1),
        Some(&mut local_timing),
        observer,
    )?;
    result.cuda_h2d_ms = local_timing.cuda_h2d_ms;
    result.host_batch_flatten_calls = host_batch_flatten_calls;
    result.host_to_device_batch_bytes =
        host_batch_flatten_bytes + local_timing.host_to_device_batch_bytes;
    result.device_batch_ready_steps = local_timing.device_batch_ready_steps;
    result.device_batch_missing_steps = local_timing.device_batch_missing_steps;
    result.loss_count = usize::from(compute_loss);

    if let Some(start) = backward_start_event {
        let end = stream
            .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|e| {
                pg_core::PgError::InvalidOp(format!("cuda backward event record failed: {e:?}"))
            })?;
        result.cuda_backward_ms = start.elapsed_ms(&end).map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda backward event elapsed failed: {e:?}"))
        })? as f64;
    } else {
        result.cuda_backward_ms = backward_host_t0.elapsed().as_secs_f64() * 1000.0;
    }
    Ok(result)
}

#[cfg(feature = "cuda")]
fn replica_backward_graph_ready(replica: &CudaSingleFastRuntime, runtime_seq_len: usize) -> bool {
    !cuda_backward_graph_enabled()
        || (replica.backward_graph.is_some()
            && replica.backward_graph_seq_len == runtime_seq_len
            && replica.backward_graph_warmup_steps_done >= cuda_backward_graph_warmup_steps())
}

#[cfg(feature = "cuda")]
fn synchronize_distributed_replica_streams(
    runtime: &CudaDistributedRuntime,
    label: &str,
) -> PgResult<()> {
    for (rank, replica) in runtime.replicas.iter().enumerate() {
        replica.gpu_model.gemm.stream().synchronize().map_err(|e| {
            pg_core::PgError::InvalidOp(format!(
                "{label}: rank {rank} stream synchronize failed: {e:?}"
            ))
        })?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_distributed_step(
    runtime: &mut CudaDistributedRuntime,
    batches: Option<&Vec<Vec<(Vec<u32>, Vec<u32>)>>>,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
    distributed_optimizer_backend: DistributedOptimizerBackend,
    compute_step_loss: bool,
    runtime_seq_len: usize,
    timing: &mut RunTiming,
) -> PgResult<f32> {
    if let Some(batches) = batches {
        if runtime.replicas.len() != batches.len() {
            return Err(pg_core::PgError::InvalidOp(format!(
                "cuda_distributed_step got {} batches for {} replicas",
                batches.len(),
                runtime.replicas.len()
            )));
        }
    }

    let mut total_loss = 0.0f32;
    let mut loss_count = 0usize;
    let event_timing = cuda_event_timing_enabled();
    let use_backward_bucket_overlap = backward_nccl_bucket_overlap_enabled()
        && distributed_optimizer_backend == DistributedOptimizerBackend::ShardedParallelMuon;
    if use_backward_bucket_overlap {
        if compute_step_loss {
            return Err(pg_core::PgError::InvalidOp(
                "PG_NCCL_BACKWARD_BUCKET_OVERLAP=1 requires no-loss backward; disable step-loss timing for this record path".into(),
            ));
        }
        if runtime.comm_side_comms.is_none() {
            return Err(pg_core::PgError::InvalidOp(
                "PG_NCCL_BACKWARD_BUCKET_OVERLAP=1 requires side-stream NCCL communicators".into(),
            ));
        }
        if runtime
            .parallel_muon
            .as_ref()
            .map(|parallel_muon| parallel_muon.replicas.len())
            != Some(runtime.replicas.len())
        {
            return Err(pg_core::PgError::InvalidOp(
                "PG_NCCL_BACKWARD_BUCKET_OVERLAP=1 requires initialized sharded Parallel Muon replicas".into(),
            ));
        }
    }

    let graph_setup_pending = cuda_backward_graph_enabled()
        && !use_backward_bucket_overlap
        && !compute_step_loss
        && runtime
            .replicas
            .iter()
            .any(|replica| !replica_backward_graph_ready(replica, runtime_seq_len));

    let replica_results = if graph_setup_pending {
        let mut results = Vec::with_capacity(runtime.replicas.len());
        for rank_idx in 0..runtime.replicas.len() {
            if !replica_backward_graph_ready(&runtime.replicas[rank_idx], runtime_seq_len) {
                synchronize_distributed_replica_streams(
                    runtime,
                    "cuda backward graph setup isolation sync",
                )?;
            }
            let rank_batches = batches.map(|all_batches| &all_batches[rank_idx]);
            let result = cuda_distributed_launch_replica_backward(
                &mut runtime.replicas[rank_idx],
                rank_idx,
                rank_batches,
                compute_step_loss,
                runtime_seq_len,
                event_timing,
                None,
            )?;
            results.push(result);
        }
        PgResult::Ok(results)?
    } else if use_backward_bucket_overlap {
        let world_size = runtime.replicas.len();
        let bucket_layers = backward_nccl_bucket_layers();
        let num_layers = runtime
            .replicas
            .first()
            .map(|replica| replica.gpu_model.config.num_layers)
            .unwrap_or(0);
        let side_comms = runtime
            .comm_side_comms
            .as_ref()
            .expect("checked side-stream comms for backward buckets");
        let side_streams = &runtime.comm_side_streams;
        let side_events = &runtime.comm_side_events;
        let sharded_replicas = &mut runtime
            .parallel_muon
            .as_mut()
            .expect("checked sharded Parallel Muon for backward buckets")
            .replicas;

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(runtime.replicas.len());
            for (rank_idx, (replica, sharded_replica)) in runtime
                .replicas
                .iter_mut()
                .zip(sharded_replicas.iter_mut())
                .enumerate()
            {
                let rank_batches = batches.map(|all_batches| &all_batches[rank_idx]);
                let comm = &side_comms[rank_idx];
                let side_stream = &side_streams[rank_idx];
                let side_events = &side_events[rank_idx];
                handles.push(
                    scope.spawn(move || -> PgResult<ReplicaBackwardLaunchResult> {
                        let mut observer = BackwardBucketOverlapObserver {
                            rank: rank_idx,
                            world_size,
                            num_layers,
                            bucket_layers,
                            pending_layers: Vec::with_capacity(bucket_layers.max(1)),
                            bucket_flushes: 0,
                            bucket_layers_reduced: 0,
                            bucket_matrix_reductions: 0,
                            comm,
                            side_stream,
                            side_events,
                            sharded: sharded_replica,
                        };
                        let mut result = cuda_distributed_launch_replica_backward(
                            replica,
                            rank_idx,
                            rank_batches,
                            compute_step_loss,
                            runtime_seq_len,
                            event_timing,
                            Some(&mut observer),
                        )?;
                        result.backward_bucket_flushes = observer.bucket_flushes;
                        result.backward_bucket_layers = observer.bucket_layers_reduced;
                        result.backward_bucket_matrix_reductions =
                            observer.bucket_matrix_reductions;
                        Ok(result)
                    }),
                );
            }

            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                let result = handle.join().map_err(|_| {
                    pg_core::PgError::InvalidOp("cuda distributed worker thread panicked".into())
                })??;
                results.push(result);
            }
            PgResult::Ok(results)
        })?
    } else {
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(runtime.replicas.len());
            for (rank_idx, replica) in runtime.replicas.iter_mut().enumerate() {
                let rank_batches = batches.map(|all_batches| &all_batches[rank_idx]);
                handles.push(
                    scope.spawn(move || -> PgResult<ReplicaBackwardLaunchResult> {
                        cuda_distributed_launch_replica_backward(
                            replica,
                            rank_idx,
                            rank_batches,
                            compute_step_loss,
                            runtime_seq_len,
                            event_timing,
                            None,
                        )
                    }),
                );
            }

            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                let result = handle.join().map_err(|_| {
                    pg_core::PgError::InvalidOp("cuda distributed worker thread panicked".into())
                })??;
                results.push(result);
            }
            PgResult::Ok(results)
        })?
    };
    let mut max_backward_ms = 0.0f64;
    let mut max_backward_bucket_flushes = 0usize;
    let mut max_backward_bucket_layers = 0usize;
    let mut max_backward_bucket_matrix_reductions = 0usize;
    for result in replica_results {
        total_loss += result.loss;
        loss_count += result.loss_count;
        timing.cuda_zero_grads_ms += result.cuda_zero_grads_ms;
        timing.cuda_h2d_ms += result.cuda_h2d_ms;
        timing.host_batch_flatten_calls += result.host_batch_flatten_calls;
        timing.host_to_device_batch_bytes += result.host_to_device_batch_bytes;
        timing.device_batch_ready_steps += result.device_batch_ready_steps;
        timing.device_batch_missing_steps += result.device_batch_missing_steps;
        max_backward_ms = max_backward_ms.max(result.cuda_backward_ms);
        max_backward_bucket_flushes =
            max_backward_bucket_flushes.max(result.backward_bucket_flushes);
        max_backward_bucket_layers = max_backward_bucket_layers.max(result.backward_bucket_layers);
        max_backward_bucket_matrix_reductions =
            max_backward_bucket_matrix_reductions.max(result.backward_bucket_matrix_reductions);
    }
    if use_backward_bucket_overlap {
        println!(
            "backward_nccl_bucket_overlap_runtime_json={{\"event\":\"backward_nccl_bucket_overlap_runtime\",\"step\":{},\"enabled\":true,\"bucket_layers_config\":{},\"bucket_flushes_per_rank_max\":{},\"layers_reduced_per_rank_max\":{},\"matrix_reductions_per_rank_max\":{},\"num_layers\":{},\"world_size\":{}}}",
            step,
            backward_nccl_bucket_layers(),
            max_backward_bucket_flushes,
            max_backward_bucket_layers,
            max_backward_bucket_matrix_reductions,
            runtime
                .replicas
                .first()
                .map(|replica| replica.gpu_model.config.num_layers)
                .unwrap_or(0),
            runtime.replicas.len()
        );
    }
    timing.cuda_backward_ms += max_backward_ms;
    accumulate_gpu_backward_stage_timing(&runtime.replicas, timing);
    match distributed_optimizer_backend {
        DistributedOptimizerBackend::AllReduceReplicatedMuon => {
            let sync_start_events = if event_timing {
                Some(record_replica_events(runtime)?)
            } else {
                None
            };
            let sync_t0 = Instant::now();
            cuda_distributed_all_reduce_average(runtime, 1)?;
            let sync_host_ms = sync_t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(starts) = sync_start_events {
                let ends = record_replica_events(runtime)?;
                timing.cuda_non_bank_sync_ms += max_elapsed_replica_events_ms(starts, ends)?;
            } else {
                timing.cuda_non_bank_sync_ms += sync_host_ms;
            }
            let update_start_events = if event_timing {
                Some(record_replica_events(runtime)?)
            } else {
                None
            };
            let update_t0 = Instant::now();
            for replica in runtime.replicas.iter_mut() {
                cuda_fast_apply_updates(replica, train_config, step, lr_scale)?;
            }
            let update_host_ms = update_t0.elapsed().as_secs_f64() * 1000.0;
            if let Some(starts) = update_start_events {
                let ends = record_replica_events(runtime)?;
                timing.cuda_bank_update_ms += max_elapsed_replica_events_ms(starts, ends)?;
            } else {
                timing.cuda_bank_update_ms += update_host_ms;
            }
        }
        DistributedOptimizerBackend::ShardedParallelMuon => {
            cuda_distributed_sharded_parallel_muon_step(
                runtime,
                train_config,
                step,
                lr_scale,
                1,
                timing,
            )?;
        }
    }
    Ok(total_loss / loss_count.max(1) as f32)
}

#[cfg(feature = "cuda")]
fn download_gpu_f32(tensor: &pg_core::GpuTensor) -> PgResult<Vec<f32>> {
    let bytes = tensor.to_host_bytes()?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(pg_core::PgError::InvalidOp(format!(
            "GPU f32 download returned {} bytes, not divisible by {}",
            bytes.len(),
            std::mem::size_of::<f32>()
        )));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn validate_executable_variant(run_spec: &RunSpec, mode: RunMode) -> PgResult<()> {
    let _ = (run_spec, mode);
    Ok(())
}

fn is_record_shaped_mode(mode: RunMode) -> bool {
    matches!(mode, RunMode::RecordShapedProxy | RunMode::Record)
}

fn record_timing_skip_steps() -> usize {
    std::env::var("PG_RECORD_TIMING_SKIP_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0)
}

fn run_mode_label(mode: RunMode) -> &'static str {
    match mode {
        RunMode::Smoke => "smoke",
        RunMode::Proxy => "proxy",
        RunMode::RecordShapedProxy => "record-shaped-proxy",
        RunMode::Record => "record",
    }
}

fn attention_dtype_label(backend: AttentionBackend) -> &'static str {
    match backend {
        AttentionBackend::CudnnSdpaBf16 => "bf16",
        AttentionBackend::NaiveF32 | AttentionBackend::FlashF32 => "f32",
    }
}

fn attention_backend_impl_label(backend: AttentionBackend) -> &'static str {
    match backend {
        AttentionBackend::NaiveF32 => "cuda_nvrtc_naive_f32",
        AttentionBackend::FlashF32 => "cuda_nvrtc_online_f32",
        AttentionBackend::CudnnSdpaBf16 => "cudnn_frontend_sdpa_bf16_f32_bridge",
    }
}

fn model_compute_dtype_label(run_spec: &RunSpec) -> &'static str {
    // F32 remains the authoritative activation/parameter-gradient storage.
    // The BF16 target currently uses BF16 shadows for projection/output GEMMs
    // and cuDNN SDPA scratch, not a full BF16 activation graph.
    match run_spec.model.compute_precision {
        ModelComputePrecision::F32Tf32 => "f32_tf32",
        ModelComputePrecision::Bf16TensorCore => "f32_activation_bf16_projection_gemm",
    }
}

fn model_precision_target_label(run_spec: &RunSpec) -> &'static str {
    match run_spec.model.compute_precision {
        ModelComputePrecision::F32Tf32 => "f32_tf32",
        ModelComputePrecision::Bf16TensorCore => "bf16_tensor_core",
    }
}

fn model_precision_target_met(run_spec: &RunSpec) -> bool {
    match run_spec.model.compute_precision {
        ModelComputePrecision::F32Tf32 => true,
        ModelComputePrecision::Bf16TensorCore => {
            bf16_hot_graph_primary_paths_complete_for_audit(run_spec)
                && f32_hot_path_bridge_count_for_audit(run_spec) == 0
        }
    }
}

fn attention_precision_bridge_enabled(backend: AttentionBackend) -> bool {
    matches!(backend, AttentionBackend::CudnnSdpaBf16)
}

fn cudnn_sdpa_deterministic_enabled() -> bool {
    matches!(
        std::env::var("PG_CUDNN_SDPA_DETERMINISTIC")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn backward_backend_label(backend: TrainBackend) -> &'static str {
    match backend {
        TrainBackend::Cpu => "cpu_reference",
        TrainBackend::CudaSingleParity => "cuda_forward_cpu_grad_mirror",
        TrainBackend::CudaSingle | TrainBackend::CudaDistributed => "gpu_explicit",
    }
}

fn escape_json_str(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn json_str_field(name: &str, value: &str) -> String {
    format!("\"{}\":\"{}\"", name, escape_json_str(value))
}

fn json_f64_field(name: &str, value: f64) -> String {
    if value.is_finite() {
        format!("\"{}\":{:.6}", name, value)
    } else {
        format!("\"{}\":null", name)
    }
}

fn json_opt_f64_field(name: &str, value: Option<f64>) -> String {
    match value {
        Some(v) if v.is_finite() => format!("\"{}\":{:.6}", name, v),
        _ => format!("\"{}\":null", name),
    }
}

fn json_opt_usize_field(name: &str, value: Option<usize>) -> String {
    match value {
        Some(v) => format!("\"{}\":{}", name, v),
        None => format!("\"{}\":null", name),
    }
}

fn json_opt_bool_field(name: &str, value: Option<bool>) -> String {
    match value {
        Some(v) => format!("\"{}\":{}", name, v),
        None => format!("\"{}\":null", name),
    }
}

fn json_opt_str_field(name: &str, value: Option<&str>) -> String {
    match value {
        Some(v) => format!("\"{}\":\"{}\"", name, escape_json_str(v)),
        None => format!("\"{}\":null", name),
    }
}

#[allow(clippy::too_many_arguments)]
fn record_artifact_audit_json(
    run_spec: &RunSpec,
    artifact_model_bytes: Option<usize>,
    artifact_code_bytes: Option<usize>,
    artifact_total_bytes: Option<usize>,
    artifact_budget_ok: Option<bool>,
    artifact_model_sha256: Option<&str>,
    artifact_code_sha256: Option<&str>,
    caseops_byte_sidecar_sha256: Option<&str>,
) -> String {
    let mut fields = Vec::with_capacity(16);
    fields.push(json_str_field("event", "record_artifact_audit"));
    fields.push(json_opt_usize_field(
        "artifact_model_bytes",
        artifact_model_bytes,
    ));
    fields.push(json_opt_usize_field(
        "artifact_code_bytes",
        artifact_code_bytes,
    ));
    fields.push(json_opt_usize_field(
        "artifact_total_bytes",
        artifact_total_bytes,
    ));
    fields.push(format!(
        "\"artifact_total_limit\":{}",
        run_spec.quant.target_artifact_bytes
    ));
    fields.push(format!(
        "\"artifact_budget_known\":{}",
        artifact_model_bytes.is_some() && artifact_code_bytes.is_some()
    ));
    fields.push(json_opt_bool_field(
        "artifact_budget_ok",
        artifact_budget_ok,
    ));
    fields.push(json_opt_str_field(
        "artifact_model_sha256",
        artifact_model_sha256,
    ));
    fields.push(json_opt_str_field(
        "artifact_code_sha256",
        artifact_code_sha256,
    ));
    fields.push(json_opt_str_field(
        "caseops_byte_sidecar_sha256",
        caseops_byte_sidecar_sha256,
    ));
    fields.push("\"strict_decimal_bytes\":true".to_string());
    format!("{{{}}}", fields.join(","))
}

/// Emit the full [`VariantResult`] as a single-line JSON object.
///
/// Used by `deploy/run_record_ab.py` to do step-time A/B deltas without
/// having to regex-parse the legacy `key=value` lines that `main.rs` prints.
/// Look for the line prefixed with `run_timing_json=` in stdout.
pub fn run_timing_json(result: &VariantResult) -> String {
    let mut fields: Vec<String> = Vec::with_capacity(96);
    fields.push(json_str_field("event", "run_timing"));
    fields.push(json_str_field("run_name", &result.run_name));
    fields.push(json_str_field("mode", &format!("{:?}", result.mode)));
    fields.push(json_str_field(
        "train_backend",
        &format!("{:?}", result.train_backend),
    ));
    fields.push(json_str_field(
        "variant_fingerprint",
        &result.variant_fingerprint,
    ));
    fields.push(json_str_field(
        "attention_backend",
        &result.attention_backend,
    ));
    fields.push(json_str_field(
        "distributed_optimizer_backend",
        &result.distributed_optimizer_backend,
    ));
    fields.push(json_str_field(
        "eval_adaptation_backend",
        &result.eval_adaptation_backend,
    ));
    fields.push(json_str_field("timing_backend", &result.timing_backend));
    fields.push(format!("\"steps_completed\":{}", result.steps_completed));
    fields.push(format!("\"timing_steps\":{}", result.timing_steps));
    fields.push(format!("\"rank\":{}", result.rank));
    fields.push(format!("\"world_size\":{}", result.world_size));
    fields.push(format!("\"seq_len\":{}", result.seq_len));
    fields.push(format!(
        "\"global_batch_tokens\":{}",
        result.global_batch_tokens
    ));
    fields.push(format!(
        "\"local_microbatches_per_step\":{}",
        result.local_microbatches_per_step
    ));
    fields.push(format!(
        "\"tokens_seen_global\":{}",
        result.tokens_seen_global
    ));
    fields.push(format!("\"distributed_sync\":{}", result.distributed_sync));
    fields.push(format!(
        "\"frontier_record_ready\":{}",
        result.frontier_record_ready
    ));
    fields.push(format!(
        "\"leaderboard_algorithm_ready\":{}",
        result.leaderboard_algorithm_ready
    ));
    fields.push(format!("\"record_shape\":{}", result.record_shape));
    fields.push(format!(
        "\"record_attention_grade\":{}",
        result.record_attention_grade
    ));
    fields.push(format!(
        "\"microbatch_serial_loop\":{}",
        result.microbatch_serial_loop
    ));
    fields.push(json_str_field(
        "bank_update_backend",
        &result.bank_update_backend,
    ));
    fields.push(json_str_field(
        "train_data_source",
        &result.train_data_source,
    ));
    fields.push(json_str_field("bpb_byte_source", &result.bpb_byte_source));
    fields.push(json_opt_str_field(
        "proxy_metric_source",
        result.proxy_metric_source.as_deref(),
    ));
    fields.push(format!("\"train_loss\":{:.6}", result.train_loss));
    fields.push(json_str_field(
        "train_loss_source",
        &result.train_loss_source,
    ));
    fields.push(json_opt_f64_field("proxy_bpb", result.proxy_bpb));
    fields.push(json_opt_f64_field("eval_loss", result.eval_loss));
    fields.push(json_opt_f64_field("final_bpb", result.final_bpb));
    fields.push(json_opt_usize_field("eval_tokens", result.eval_tokens));
    fields.push(json_opt_usize_field(
        "artifact_bytes",
        result.artifact_bytes,
    ));
    fields.push(json_opt_usize_field(
        "submission_code_bytes",
        result.submission_code_bytes,
    ));
    fields.push(json_opt_usize_field(
        "submission_total_bytes",
        result.submission_total_bytes,
    ));
    fields.push(json_opt_bool_field(
        "artifact_budget_ok",
        result.artifact_budget_ok,
    ));
    fields.push(json_opt_str_field(
        "artifact_model_sha256",
        result.artifact_model_sha256.as_deref(),
    ));
    fields.push(json_opt_str_field(
        "artifact_code_sha256",
        result.artifact_code_sha256.as_deref(),
    ));
    fields.push(json_opt_str_field(
        "caseops_byte_sidecar_sha256",
        result.caseops_byte_sidecar_sha256.as_deref(),
    ));
    fields.push(format!(
        "\"host_batch_flatten_calls\":{}",
        result.host_batch_flatten_calls
    ));
    fields.push(format!(
        "\"host_to_device_batch_bytes\":{}",
        result.host_to_device_batch_bytes
    ));
    fields.push(format!(
        "\"device_batch_ready_steps\":{}",
        result.device_batch_ready_steps
    ));
    fields.push(format!(
        "\"device_batch_missing_steps\":{}",
        result.device_batch_missing_steps
    ));
    fields.push(format!(
        "\"device_to_host_scalar_reads\":{}",
        result.device_to_host_scalar_reads
    ));
    fields.push(json_f64_field("ms_per_step", result.ms_per_step));
    fields.push(json_f64_field(
        "wallclock_seconds",
        result.wallclock_seconds,
    ));
    fields.push(json_f64_field(
        "timing_measured_ms_per_step",
        result.timing_measured_ms_per_step,
    ));
    fields.push(format!(
        "\"timing_recurrent_active_steps\":{}",
        result.timing_recurrent_active_steps
    ));
    fields.push(format!(
        "\"timing_recurrent_inactive_steps\":{}",
        result.timing_recurrent_inactive_steps
    ));
    fields.push(json_f64_field(
        "timing_recurrent_active_ms",
        result.timing_recurrent_active_ms,
    ));
    fields.push(json_f64_field(
        "timing_recurrent_inactive_ms",
        result.timing_recurrent_inactive_ms,
    ));
    fields.push(json_f64_field(
        "timing_recurrent_active_ms_per_step",
        if result.timing_recurrent_active_steps == 0 {
            0.0
        } else {
            result.timing_recurrent_active_ms / result.timing_recurrent_active_steps as f64
        },
    ));
    fields.push(json_f64_field(
        "timing_recurrent_inactive_ms_per_step",
        if result.timing_recurrent_inactive_steps == 0 {
            0.0
        } else {
            result.timing_recurrent_inactive_ms / result.timing_recurrent_inactive_steps as f64
        },
    ));
    let timing_steps_f = result.timing_steps.max(1) as f64;
    let per_step = |total: f64| -> f64 {
        if result.timing_steps == 0 {
            0.0
        } else {
            total / timing_steps_f
        }
    };
    let pairs: &[(&str, f64)] = &[
        ("timing_data_sampling_ms", result.timing_data_sampling_ms),
        ("timing_train_step_ms", result.timing_train_step_ms),
        (
            "timing_cuda_zero_grads_ms",
            result.timing_cuda_zero_grads_ms,
        ),
        ("timing_cuda_h2d_ms", result.timing_cuda_h2d_ms),
        ("timing_cuda_backward_ms", result.timing_cuda_backward_ms),
        (
            "timing_cuda_backward_forward_ms",
            result.timing_cuda_backward_forward_ms,
        ),
        (
            "timing_cuda_backward_forward_embed_ms",
            result.timing_cuda_backward_forward_embed_ms,
        ),
        (
            "timing_cuda_backward_forward_encoder_ms",
            result.timing_cuda_backward_forward_encoder_ms,
        ),
        (
            "timing_cuda_backward_forward_encoder_layer_max_ms",
            result.timing_cuda_backward_forward_encoder_layer_max_ms,
        ),
        (
            "timing_cuda_backward_forward_decoder_ms",
            result.timing_cuda_backward_forward_decoder_ms,
        ),
        (
            "timing_cuda_backward_forward_decoder_layer_max_ms",
            result.timing_cuda_backward_forward_decoder_layer_max_ms,
        ),
        (
            "timing_cuda_backward_forward_logits_ms",
            result.timing_cuda_backward_forward_logits_ms,
        ),
        (
            "timing_cuda_backward_forward_block_pre_attn_ms",
            result.timing_cuda_backward_forward_block_pre_attn_ms,
        ),
        (
            "timing_cuda_backward_forward_block_attention_ms",
            result.timing_cuda_backward_forward_block_attention_ms,
        ),
        (
            "timing_cuda_backward_forward_block_post_attn_ms",
            result.timing_cuda_backward_forward_block_post_attn_ms,
        ),
        (
            "timing_cuda_backward_forward_block_mlp_ms",
            result.timing_cuda_backward_forward_block_mlp_ms,
        ),
        (
            "timing_cuda_backward_block_recompute_ms",
            result.timing_cuda_backward_block_recompute_ms,
        ),
        (
            "timing_cuda_backward_block_mlp_ms",
            result.timing_cuda_backward_block_mlp_ms,
        ),
        (
            "timing_cuda_backward_block_mlp_residual_ms",
            result.timing_cuda_backward_block_mlp_residual_ms,
        ),
        (
            "timing_cuda_backward_block_mlp_down_ms",
            result.timing_cuda_backward_block_mlp_down_ms,
        ),
        (
            "timing_cuda_backward_block_mlp_act_ms",
            result.timing_cuda_backward_block_mlp_act_ms,
        ),
        (
            "timing_cuda_backward_block_mlp_up_ms",
            result.timing_cuda_backward_block_mlp_up_ms,
        ),
        (
            "timing_cuda_backward_block_mlp_norm_ms",
            result.timing_cuda_backward_block_mlp_norm_ms,
        ),
        (
            "timing_cuda_backward_block_attn_out_ms",
            result.timing_cuda_backward_block_attn_out_ms,
        ),
        (
            "timing_cuda_backward_block_attn_out_residual_ms",
            result.timing_cuda_backward_block_attn_out_residual_ms,
        ),
        (
            "timing_cuda_backward_block_attn_out_proj_ms",
            result.timing_cuda_backward_block_attn_out_proj_ms,
        ),
        (
            "timing_cuda_backward_block_attn_out_gate_xsa_ms",
            result.timing_cuda_backward_block_attn_out_gate_xsa_ms,
        ),
        (
            "timing_cuda_backward_block_attention_ms",
            result.timing_cuda_backward_block_attention_ms,
        ),
        (
            "timing_cuda_backward_block_attention_sdpa_ms",
            result.timing_cuda_backward_block_attention_sdpa_ms,
        ),
        (
            "timing_cuda_backward_block_attention_xsa_accum_ms",
            result.timing_cuda_backward_block_attention_xsa_accum_ms,
        ),
        (
            "timing_cuda_backward_block_qkv_ms",
            result.timing_cuda_backward_block_qkv_ms,
        ),
        (
            "timing_cuda_backward_block_qkv_rope_ms",
            result.timing_cuda_backward_block_qkv_rope_ms,
        ),
        (
            "timing_cuda_backward_block_qkv_proj_ms",
            result.timing_cuda_backward_block_qkv_proj_ms,
        ),
        (
            "timing_cuda_backward_block_qkv_ve_ms",
            result.timing_cuda_backward_block_qkv_ve_ms,
        ),
        (
            "timing_cuda_backward_block_qkv_norm_resid_ms",
            result.timing_cuda_backward_block_qkv_norm_resid_ms,
        ),
        (
            "timing_cuda_backward_output_ms",
            result.timing_cuda_backward_output_ms,
        ),
        (
            "timing_cuda_backward_decoder_ms",
            result.timing_cuda_backward_decoder_ms,
        ),
        (
            "timing_cuda_backward_encoder_ms",
            result.timing_cuda_backward_encoder_ms,
        ),
        (
            "timing_cuda_backward_tail_ms",
            result.timing_cuda_backward_tail_ms,
        ),
        (
            "timing_cuda_non_bank_sync_ms",
            result.timing_cuda_non_bank_sync_ms,
        ),
        (
            "timing_cuda_bank_update_ms",
            result.timing_cuda_bank_update_ms,
        ),
        (
            "timing_cuda_bank_update_stage_pack_ms",
            result.timing_cuda_bank_update_stage_pack_ms,
        ),
        (
            "timing_cuda_bank_update_stage_grad_collectives_ms",
            result.timing_cuda_bank_update_stage_grad_collectives_ms,
        ),
        (
            "timing_cuda_bank_update_stage_scale_ms",
            result.timing_cuda_bank_update_stage_scale_ms,
        ),
        (
            "timing_cuda_bank_update_stage_norm_ms",
            result.timing_cuda_bank_update_stage_norm_ms,
        ),
        (
            "timing_cuda_bank_update_stage_clip_ms",
            result.timing_cuda_bank_update_stage_clip_ms,
        ),
        (
            "timing_cuda_bank_update_stage_muon_ms",
            result.timing_cuda_bank_update_stage_muon_ms,
        ),
        (
            "timing_cuda_bank_update_stage_param_gather_ms",
            result.timing_cuda_bank_update_stage_param_gather_ms,
        ),
        (
            "timing_cuda_bank_update_stage_copy_back_ms",
            result.timing_cuda_bank_update_stage_copy_back_ms,
        ),
        (
            "timing_cuda_non_bank_update_ms",
            result.timing_cuda_non_bank_update_ms,
        ),
        (
            "timing_post_train_sync_ms",
            result.timing_post_train_sync_ms,
        ),
        (
            "timing_artifact_export_ms",
            result.timing_artifact_export_ms,
        ),
        ("timing_eval_ms", result.timing_eval_ms),
    ];
    for (name, total) in pairs {
        fields.push(json_f64_field(name, *total));
        fields.push(json_f64_field(
            &format!("{}_per_step", name),
            per_step(*total),
        ));
    }
    format!("{{{}}}", fields.join(","))
}

#[allow(clippy::too_many_arguments)]
fn record_path_audit_json(
    run_spec: &RunSpec,
    mode: RunMode,
    batch_plan: &StepBatchPlan,
    world_size: usize,
    frontier_record_ready: bool,
    leaderboard_algorithm_ready: bool,
    leaderboard_algorithm_gaps: &[&'static str],
    record_attention_grade: bool,
    microbatch_serial_loop: bool,
    bank_update_backend: &str,
) -> String {
    let local_batch = batch_plan.local_microbatches_per_step;
    let local_tokens_per_rank = batch_plan.microbatch_tokens * local_batch;
    let mut fields = Vec::with_capacity(30);
    fields.push(json_str_field("event", "record_path_audit"));
    fields.push(json_str_field("mode", run_mode_label(mode)));
    fields.push(format!("\"record_shape\":{}", is_record_shaped_mode(mode)));
    fields.push(format!("\"seq_len\":{}", batch_plan.microbatch_tokens));
    fields.push(format!(
        "\"global_batch_tokens\":{}",
        batch_plan.global_batch_tokens
    ));
    fields.push(format!("\"world_size\":{}", world_size));
    fields.push(format!("\"local_batch\":{}", local_batch));
    fields.push(format!("\"local_batch_sequences\":{}", local_batch));
    fields.push(format!(
        "\"local_tokens_per_rank\":{}",
        local_tokens_per_rank
    ));
    fields.push(format!(
        "\"effective_global_batch_tokens\":{}",
        batch_plan.microbatch_tokens * local_batch * world_size.max(1)
    ));
    let record_max_ms = record_max_ms_per_step_for_submission(run_spec);
    fields.push(format!("\"record_max_ms_per_step\":{record_max_ms:.3}"));
    let target_steps_per_600s = if record_max_ms > 0.0 {
        600_000.0 / record_max_ms
    } else {
        0.0
    };
    fields.push(format!(
        "\"record_target_steps_per_600s\":{target_steps_per_600s:.3}"
    ));
    fields.push(json_str_field(
        "attention_backend",
        &format!("{:?}", run_spec.model.attention_backend),
    ));
    fields.push(json_str_field(
        "attention_backend_impl",
        attention_backend_impl_label(run_spec.model.attention_backend),
    ));
    fields.push(json_str_field(
        "attention_dtype",
        attention_dtype_label(run_spec.model.attention_backend),
    ));
    fields.push(json_str_field(
        "model_compute_dtype",
        model_compute_dtype_label(run_spec),
    ));
    fields.push(json_str_field(
        "model_family",
        &format!("{:?}", run_spec.model.family),
    ));
    fields.push(format!("\"num_layers\":{}", run_spec.model.num_layers));
    fields.push(format!("\"model_dim\":{}", run_spec.model.model_dim));
    fields.push(format!("\"num_heads\":{}", run_spec.model.num_heads));
    fields.push(format!("\"num_kv_heads\":{}", run_spec.model.num_kv_heads));
    fields.push(format!("\"mlp_mult\":{}", run_spec.model.mlp_mult));
    fields.push(format!("\"qk_gain_init\":{}", run_spec.model.qk_gain_init));
    fields.push(format!(
        "\"logit_softcap\":{}",
        run_spec.model.logit_softcap
    ));
    fields.push(json_str_field(
        "model_precision_target",
        model_precision_target_label(run_spec),
    ));
    fields.push(format!(
        "\"model_precision_target_met\":{}",
        model_precision_target_met(run_spec)
    ));
    fields.push(format!(
        "\"bf16_final_norm_to_ce\":{}",
        bf16_final_norm_to_ce_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_output_to_projection\":{}",
        bf16_attention_output_to_projection_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_mlp_backward_inputs\":{}",
        bf16_mlp_backward_inputs_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_qkv_backward\":{}",
        bf16_qkv_backward_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_hot_graph_primary_paths_complete\":{}",
        bf16_hot_graph_primary_paths_complete_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"f32_hot_path_bridge_count\":{}",
        f32_hot_path_bridge_count_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"attention_precision_bridge\":{}",
        attention_precision_bridge_enabled(run_spec.model.attention_backend)
    ));
    fields.push(format!(
        "\"cudnn_sdpa_deterministic_backward\":{}",
        cudnn_sdpa_deterministic_enabled()
    ));
    fields.push(format!(
        "\"record_attention_grade\":{}",
        record_attention_grade
    ));
    fields.push(json_str_field(
        "backward_backend",
        backward_backend_label(run_spec.train.backend),
    ));
    fields.push(json_str_field("optimizer_backend", bank_update_backend));
    fields.push(json_str_field(
        "distributed_optimizer_backend",
        &format!("{:?}", run_spec.train.distributed_optimizer_backend),
    ));
    let sharded_parallel_muon = run_spec.train.backend == TrainBackend::CudaDistributed
        && run_spec.train.distributed_optimizer_backend
            == DistributedOptimizerBackend::ShardedParallelMuon;
    fields.push(format!(
        "\"sharded_parallel_muon_reduce_scatter\":{}",
        sharded_parallel_muon
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_local_shard_update\":{}",
        sharded_parallel_muon
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_all_gather\":{}",
        sharded_parallel_muon
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_grouped_grad_collectives\":{}",
        sharded_parallel_muon && sharded_grouped_grad_collectives_enabled_for_audit()
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_grouped_all_gather\":{}",
        sharded_parallel_muon
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_fused_global_clip\":{}",
        sharded_parallel_muon && sharded_parallel_muon_fused_global_clip_enabled_for_audit()
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_bf16_shadow_all_gather\":{}",
        sharded_parallel_muon && sharded_parallel_muon_bf16_shadow_all_gather_enabled_for_audit()
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_local_graph\":{}",
        sharded_parallel_muon && sharded_parallel_muon_local_graph_enabled()
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_pre_norm_graph\":{}",
        sharded_parallel_muon && sharded_parallel_muon_pre_norm_graph_enabled()
    ));
    fields.push(format!(
        "\"sharded_parallel_muon_host_scalar_sync\":{}",
        false
    ));
    fields.push(format!(
        "\"replicated_allreduce_packed_all_grads\":{}",
        run_spec.train.backend == TrainBackend::CudaDistributed
            && run_spec.train.distributed_optimizer_backend
                == DistributedOptimizerBackend::AllReduceReplicatedMuon
    ));
    fields.push(json_str_field(
        "sharded_parallel_muon_bank_grad_wire_dtype",
        if sharded_bank_grad_bf16_wire_enabled_for_audit() {
            "bf16"
        } else {
            "f32"
        },
    ));
    fields.push(json_str_field(
        "sharded_parallel_muon_bank_grad_collective",
        if backward_nccl_bucket_overlap_enabled_for_audit() {
            "bucketed_reduce_to_owner"
        } else {
            "reduce_scatter"
        },
    ));
    fields.push(format!(
        "\"backward_nccl_bucket_layers\":{}",
        if backward_nccl_bucket_overlap_enabled_for_audit() {
            backward_nccl_bucket_layers()
        } else {
            0
        }
    ));
    fields.push(format!(
        "\"nccl_side_stream_collectives\":{}",
        nccl_side_stream_collectives_enabled_for_audit()
    ));
    fields.push(format!(
        "\"backward_nccl_bucket_overlap\":{}",
        backward_nccl_bucket_overlap_enabled_for_audit()
    ));
    fields.push(format!(
        "\"backward_nccl_bucket_overlap_validated\":{}",
        backward_nccl_bucket_overlap_validated_for_audit()
    ));
    fields.push(format!(
        "\"distributed_configured\":{}",
        run_spec.train.backend == TrainBackend::CudaDistributed
    ));
    fields.push(format!(
        "\"distributed_sync_expected\":{}",
        run_spec.train.backend == TrainBackend::CudaDistributed
    ));
    fields.push(format!(
        "\"microbatch_serial_loop\":{}",
        microbatch_serial_loop
    ));
    fields.push(format!(
        "\"host_scalar_updates\":{}",
        gpu_host_scalar_updates_enabled_for_audit()
    ));
    fields.push("\"device_scalar_params\":true".to_string());
    fields.push(format!("\"microbatch_count\":{}", local_batch));
    let host_batch_segments_per_rank =
        if matches!(run_spec.train.backend, TrainBackend::CudaDistributed) {
            1
        } else {
            local_batch
        };
    fields.push(format!(
        "\"host_batch_segments_per_rank\":{}",
        host_batch_segments_per_rank
    ));
    fields.push(format!("\"gemm_m_dimension\":{}", local_tokens_per_rank));
    fields.push(format!("\"attention_batch_dimension\":{}", local_batch));
    fields.push(format!(
        "\"value_embedding_layers\":{}",
        run_spec.model.to_model_config().ve_layers.len()
    ));
    fields.push(format!(
        "\"caseops_byte_sidecar\":{}",
        run_spec.model.caseops.enabled && run_spec.model.caseops.byte_sidecar
    ));
    fields.push(format!(
        "\"sparse_attn_gate_enabled\":{}",
        run_spec.model.sparse_attn_gate.enabled
    ));
    fields.push(format!(
        "\"sparse_attn_gate_width\":{}",
        run_spec.model.sparse_attn_gate.width
    ));
    fields.push(format!(
        "\"sparse_attn_gate_scale\":{}",
        run_spec.model.sparse_attn_gate.scale
    ));
    fields.push(format!("\"matrix_lr\":{}", run_spec.train.matrix_lr));
    fields.push(format!("\"min_lr_scale\":{}", run_spec.train.min_lr_scale));
    fields.push(format!(
        "\"grad_clip_norm\":{}",
        run_spec.train.grad_clip_norm
    ));
    fields.push(format!("\"warmup_steps\":{}", run_spec.train.warmup_steps));
    fields.push(format!(
        "\"warmdown_iters\":{}",
        run_spec.train.to_train_config().warmdown_iters
    ));
    fields.push(format!(
        "\"warmdown_frac\":{}",
        run_spec.train.warmdown_frac
    ));
    fields.push(format!("\"adam_beta2\":{}", run_spec.train.adam_beta2));
    fields.push(format!(
        "\"quant_matrix_bits\":{}",
        run_spec.quant.matrix_bits
    ));
    fields.push(format!(
        "\"quant_embed_bits\":{}",
        run_spec.quant.embed_bits
    ));
    fields.push(format!(
        "\"quant_attn_gate_bits\":{}",
        run_spec.quant.attn_gate_bits
    ));
    fields.push(format!(
        "\"quant_mlp_clip_sigmas\":{}",
        run_spec.quant.mlp_clip_sigmas
    ));
    fields.push(format!(
        "\"quant_attn_clip_sigmas\":{}",
        run_spec.quant.attn_clip_sigmas
    ));
    fields.push(format!(
        "\"quant_embed_clip_sigmas\":{}",
        run_spec.quant.embed_clip_sigmas
    ));
    fields.push(format!(
        "\"quant_gptq_calibration_batches\":{}",
        run_spec.quant.gptq_calibration_batches
    ));
    fields.push(json_str_field(
        "quant_compression",
        &format!("{:?}", run_spec.quant.compression),
    ));
    fields.push(format!("\"lqer_enabled\":{}", run_spec.quant.lqer.enabled));
    fields.push(format!("\"lqer_rank\":{}", run_spec.quant.lqer.rank));
    fields.push(format!("\"lqer_top_k\":{}", run_spec.quant.lqer.top_k));
    fields.push(format!("\"lqer_a_bits\":{}", run_spec.quant.lqer.a_bits));
    fields.push(format!("\"lqer_b_bits\":{}", run_spec.quant.lqer.b_bits));
    fields.push(format!(
        "\"lqer_group_size\":{}",
        run_spec.quant.lqer.group_size
    ));
    fields.push(format!(
        "\"bf16_forward_projection_gemm\":{}",
        bf16_forward_projection_gemm_enabled(run_spec)
    ));
    fields.push(format!(
        "\"primary_block_forward_bf16_gemm\":{}",
        bf16_primary_forward_projection_gemm_enabled(run_spec)
    ));
    fields.push(format!(
        "\"bf16_backward_projection_gemm\":{}",
        bf16_backward_projection_gemm_enabled(run_spec)
    ));
    fields.push(format!(
        "\"experimental_fused_qkv_projection\":{}",
        experimental_fused_qkv_projection_enabled()
    ));
    fields.push(format!(
        "\"qkv_dx_beta_accum\":{}",
        qkv_dx_beta_accum_enabled_for_audit()
    ));
    fields.push(format!(
        "\"chunked_q_gain_backward\":{}",
        chunked_q_gain_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"q_gain_backward_chunk_tokens\":{}",
        if chunked_q_gain_backward_enabled_for_audit() {
            q_gain_backward_chunk_tokens_for_audit()
        } else {
            0
        }
    ));
    fields.push(format!(
        "\"chunked_residual_mix_backward\":{}",
        chunked_residual_mix_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"chunked_qkv_norm_resid_backward\":{}",
        chunked_qkv_norm_resid_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"bf16_backward_chain_qkv_norm_resid_reducer\":\"{}\"",
        bf16_backward_chain_qkv_norm_resid_reducer_for_audit()
    ));
    fields.push(format!(
        "\"direct_compact_qkv_norm_resid_backward\":{}",
        direct_compact_qkv_norm_resid_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"split_qkv_norm_resid_backward\":{}",
        split_qkv_norm_resid_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"split_compact_qkv_norm_resid_backward\":{}",
        split_compact_qkv_norm_resid_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"qkv_norm_resid_backward_rows_per_chunk\":{}",
        if chunked_qkv_norm_resid_backward_enabled_for_audit()
            || split_qkv_norm_resid_backward_enabled_for_audit()
        {
            qkv_norm_resid_backward_rows_per_chunk_for_audit()
        } else {
            0
        }
    ));
    fields.push(format!(
        "\"split_residual_mix_grad\":{}",
        split_residual_mix_grad_enabled_for_audit()
    ));
    fields.push(format!(
        "\"residual_scale_reduce\":{}",
        residual_scale_reduce_enabled_for_audit()
    ));
    fields.push(format!(
        "\"chunked_residual_scale_backward\":{}",
        chunked_residual_scale_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"tiled_residual_scale_backward\":{}",
        tiled_residual_scale_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"residual_scale_backward_rows_per_chunk\":{}",
        residual_scale_backward_rows_per_chunk_for_audit()
    ));
    fields.push(format!(
        "\"bf16_qkv_dx_output\":{}",
        bf16_qkv_dx_output_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"fused_qk_rope_gain_backward\":{}",
        fused_qk_rope_gain_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"fused_qk_rope_gain_forward\":{}",
        fused_qk_rope_gain_forward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"fused_residual_mix_norm\":{}",
        fused_residual_mix_norm_enabled_for_audit()
    ));
    fields.push(format!(
        "\"fused_mlp_activation_bf16\":{}",
        fused_mlp_activation_bf16_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_mlp_up_output\":{}",
        bf16_mlp_up_output_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_mlp_down_dx\":{}",
        bf16_mlp_down_dx_enabled_for_audit()
    ));
    fields.push(format!(
        "\"vec2_mlp_activation_backward\":{}",
        vec2_mlp_activation_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"vec4_mlp_activation_backward\":{}",
        vec4_mlp_activation_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"fast_mlp_activation_backward\":{}",
        fast_mlp_activation_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"bf16_norm_side_outputs\":{}",
        bf16_norm_side_outputs_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_norm_grad_path\":{}",
        bf16_norm_grad_path_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_residual_projection_output\":{}",
        bf16_residual_projection_output_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_projection_output\":{}",
        bf16_attention_projection_output_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_backward_tail\":{}",
        bf16_attention_backward_tail_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_tail_qkv_pack\":{}",
        bf16_attention_tail_qkv_pack_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_tail_direct_qkv_pack\":{}",
        bf16_attention_tail_direct_qkv_pack_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_backward_bhsd_do\":{}",
        bf16_attention_backward_bhsd_do_enabled_for_audit(run_spec)
    ));
    fields.push(json_str_field(
        "record_profile",
        &format!("{:?}", run_spec.runtime.record_profile),
    ));
    fields.push(json_str_field(
        "backward_chain_profile",
        &format!("{:?}", run_spec.runtime.backward_chain_profile),
    ));
    fields.push(json_str_field(
        "recurrent_backward_profile",
        &format!("{:?}", run_spec.runtime.recurrent_backward_profile),
    ));
    fields.push(format!(
        "\"runtime_recurrent_straight_through_layers\":{}",
        run_spec.runtime.recurrent_straight_through_layers
    ));
    fields.push(format!(
        "\"runtime_skip_recurrent_bank_grads\":{}",
        run_spec.runtime.skip_recurrent_bank_grads
    ));
    fields.push(format!(
        "\"runtime_skip_recurrent_pass1_bank_grads\":{}",
        run_spec.runtime.skip_recurrent_pass1_bank_grads
    ));
    fields.push(format!(
        "\"recurrent_pass1_forward_bf16_cache\":{}",
        run_spec.model.recurrence.enabled
            && run_spec.runtime.backward_chain_profile != BackwardChainProfile::Off
    ));
    fields.push(json_str_field(
        "recurrent_pass1_forward_cache_semantics",
        "bf16_prepack_no_f32_backward_acts_when_st",
    ));
    fields.push(format!(
        "\"recurrence_active_required\":{}",
        run_spec.runtime.recurrence_active_required
    ));
    fields.push(format!(
        "\"recurrent_active_steps_min\":{}",
        run_spec.runtime.recurrent_active_steps_min
    ));
    fields.push(json_str_field(
        "cuda_graph_profile",
        &format!("{:?}", run_spec.runtime.cuda_graph_profile),
    ));
    fields.push(json_str_field(
        "nccl_overlap_mode",
        &format!("{:?}", run_spec.runtime.nccl_overlap_mode),
    ));
    fields.push(format!(
        "\"device_batch_required\":{}",
        run_spec.runtime.require_device_batch
    ));
    fields.push(format!(
        "\"runtime_token_ring_full_schedule\":{}",
        run_spec.runtime.token_ring_full_schedule
    ));
    fields.push(format!(
        "\"runtime_bigram_embedding_merge\":{}",
        run_spec.runtime.bigram_embedding_merge
    ));
    fields.push(format!(
        "\"runtime_combined_qkv_rope_tail_backward\":{}",
        run_spec.runtime.combined_qkv_rope_tail_backward
    ));
    fields.push(format!(
        "\"runtime_graph_side_gemm_capture\":{}",
        run_spec.runtime.graph_side_gemm_capture
    ));
    fields.push(format!(
        "\"runtime_sharded_muon_local_graph\":{}",
        run_spec.runtime.sharded_muon_local_graph
    ));
    fields.push(format!(
        "\"runtime_sharded_muon_pre_norm_graph\":{}",
        run_spec.runtime.sharded_muon_pre_norm_graph
    ));
    fields.push(format!(
        "\"runtime_sharded_muon_fused_global_clip\":{}",
        run_spec.runtime.sharded_muon_fused_global_clip
    ));
    fields.push(format!(
        "\"runtime_sharded_muon_bf16_shadow_all_gather\":{}",
        run_spec.runtime.sharded_muon_bf16_shadow_all_gather
    ));
    fields.push(json_str_field(
        "gpu_lora_targets",
        &run_spec.eval.ttt_lora_targets.label(),
    ));
    fields.push(format!(
        "\"gpu_lora_targets_lm_head\":{}",
        run_spec.eval.ttt_lora_targets.lm_head
    ));
    fields.push(format!(
        "\"gpu_lora_targets_q\":{}",
        run_spec.eval.ttt_lora_targets.q
    ));
    fields.push(format!(
        "\"gpu_lora_targets_k\":{}",
        run_spec.eval.ttt_lora_targets.k
    ));
    fields.push(format!(
        "\"gpu_lora_targets_v\":{}",
        run_spec.eval.ttt_lora_targets.v
    ));
    fields.push(format!(
        "\"gpu_lora_targets_o\":{}",
        run_spec.eval.ttt_lora_targets.o
    ));
    fields.push(format!(
        "\"gpu_lora_targets_mlp\":{}",
        run_spec.eval.ttt_lora_targets.mlp
    ));
    fields.push(format!(
        "\"gpu_lora_targets_runtime_supported\":{}",
        run_spec.eval.ttt_lora_targets.rust_gpu_runtime_supported()
    ));
    fields.push(format!(
        "\"bf16_backward_chain_requested\":{}",
        bf16_backward_chain_requested_for_audit()
    ));
    fields.push(format!(
        "\"bf16_backward_chain_strict\":{}",
        bf16_backward_chain_strict_for_audit()
    ));
    fields.push(format!(
        "\"bf16_backward_chain_complete\":{}",
        bf16_backward_chain_complete_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"final_norm_bf16_side_output\":{}",
        final_norm_bf16_side_output_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_attention_output_bridge_to_f32\":{}",
        bf16_attention_output_bridge_to_f32_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_sparse_xsa_forward\":{}",
        bf16_sparse_xsa_forward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"sparse_xsa_warphead_backward\":{}",
        sparse_xsa_warphead_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"sparse_xsa_grouped_kv_backward\":{}",
        sparse_xsa_grouped_kv_backward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"compact_attn_gate_grad_input\":{}",
        compact_attn_gate_grad_input_enabled_for_audit()
    ));
    fields.push(format!(
        "\"recompute_residual_mix_norm_inputs\":{}",
        recompute_residual_mix_norm_inputs_enabled_for_audit()
    ));
    fields.push(format!(
        "\"fused_attention_residual_from_base\":{}",
        fused_attention_residual_from_base_enabled_for_audit()
    ));
    fields.push(format!(
        "\"fused_parallel_attn_residual_rms_norm\":{}",
        fused_parallel_attn_residual_rms_norm_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"batched_muon_newton_schulz\":{}",
        batched_muon_newton_schulz_enabled_for_audit()
    ));
    fields.push(json_str_field(
        "muon_newton_schulz_profile",
        &muon_ns_profile_for_audit(),
    ));
    fields.push(format!(
        "\"muon_newton_schulz_steps\":{}",
        muon_ns_steps_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"polar_express_newton_schulz\":{}",
        polar_express_muon_enabled_for_audit()
    ));
    fields.push(format!(
        "\"cudnn_saved_bf16_attention\":{}",
        cudnn_saved_bf16_attention_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"cudnn_prepacked_bf16_attention\":{}",
        cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"cudnn_prepacked_bf16_qk_fresh_producer\":{}",
        cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec)
            && fused_qk_rope_gain_forward_enabled_for_audit()
    ));
    fields.push(format!(
        "\"prepacked_bf16_qkv_freshness_checked\":{}",
        cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"cudnn_prepacked_bf16_poison_check\":{}",
        cudnn_prepacked_bf16_poison_check_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"skip_f32_attention_saved_activations\":{}",
        skip_f32_attention_saved_activations_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"lean_bf16_saved_layer_cache\":{}",
        lean_bf16_saved_layer_cache_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"direct_saved_layer_activations\":{}",
        direct_saved_layer_activations_enabled_for_audit()
    ));
    fields.push(format!(
        "\"saved_bf16_layer_activation_shadows\":{}",
        saved_bf16_layer_activation_shadows_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"bf16_output_projection_gemm\":{}",
        bf16_output_projection_gemm_enabled(run_spec)
    ));
    fields.push(format!(
        "\"bf16_output_backward_gemm\":{}",
        bf16_output_backward_gemm_enabled(run_spec)
    ));
    fields.push(format!(
        "\"overlap_linear_backward_gemms\":{}",
        overlap_linear_backward_gemms_enabled_for_audit()
    ));
    fields.push(format!(
        "\"overlap_mlp_down_backward_gemms\":{}",
        overlap_linear_backward_gemm_role_enabled_for_audit("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS")
    ));
    fields.push(format!(
        "\"overlap_mlp_up_backward_gemms\":{}",
        overlap_linear_backward_gemm_role_enabled_for_audit("PG_GPU_OVERLAP_MLP_UP_BWD_GEMMS")
    ));
    fields.push(format!(
        "\"overlap_qkv_backward_gemms\":{}",
        overlap_linear_backward_gemm_role_enabled_for_audit("PG_GPU_OVERLAP_QKV_BWD_GEMMS")
    ));
    fields.push(format!(
        "\"overlap_attn_out_backward_gemms\":{}",
        overlap_linear_backward_gemm_role_enabled_for_audit("PG_GPU_OVERLAP_ATTN_OUT_BWD_GEMMS")
    ));
    fields.push(format!(
        "\"bf16_output_logits\":{}",
        bf16_output_logits_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"fused_output_cross_entropy\":{}",
        fused_output_cross_entropy_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"production_fused_output_projection_ce\":{}",
        production_fused_output_projection_ce_enabled_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"production_fused_output_projection_ce_validated\":{}",
        production_fused_output_projection_ce_validated_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"output_ce_backend\":\"{}\"",
        output_ce_backend_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"output_ce_implementation\":\"{}\"",
        output_ce_implementation_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"output_projection_recompute_passes\":{}",
        output_projection_recompute_passes_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"chunked_bf16_output_ce_cache\":{}",
        chunked_bf16_output_ce_cache_enabled_for_audit(run_spec)
    ));
    let tiled_output_ce = tiled_output_cross_entropy_enabled_for_audit(run_spec);
    fields.push(format!(
        "\"tiled_output_cross_entropy\":{}",
        tiled_output_ce
    ));
    fields.push(format!(
        "\"output_loss_backend\":\"{}\"",
        output_loss_backend_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"output_ce_tile_vocab\":{}",
        output_ce_tile_vocab_for_audit()
    ));
    fields.push(format!(
        "\"output_ce_chunk_tokens\":{}",
        output_ce_chunk_tokens_for_audit()
    ));
    fields.push(format!(
        "\"materializes_full_logits\":{}",
        output_path_materializes_full_logits_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"materializes_full_bf16_logits\":{}",
        chunked_bf16_output_ce_cache_enabled_for_audit(run_spec)
            && output_path_materializes_full_logits_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"smear_gate_boundary_token_id\":{}",
        run_spec
            .model
            .smear_gate_boundary_token_id
            .map(|id| id.to_string())
            .unwrap_or_else(|| "null".to_string())
    ));
    fields.push(format!(
        "\"smeargate_bos_doc_mask\":{}",
        run_spec.model.smear_gate_boundary_token_id.is_some()
    ));
    fields.push(format!(
        "\"gpu_lora_prefix_doc_ttt_schedule\":{}",
        run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
            && run_spec.eval.phased_ttt_prefix_docs > 0
            && run_spec.model.smear_gate_boundary_token_id.is_some()
    ));
    fields.push(format!(
        "\"gpu_lora_prefix_docs\":{}",
        run_spec.eval.phased_ttt_prefix_docs
    ));
    fields.push(format!("\"gpu_lora_rank\":{}", run_spec.eval.lora_rank));
    fields.push(format!("\"gpu_lora_alpha\":{}", run_spec.eval.lora_alpha));
    fields.push(format!(
        "\"gpu_lora_phased_ttt_phases\":{}",
        run_spec.eval.phased_ttt_phases
    ));
    fields.push(format!(
        "\"gpu_lora_phased_ttt_weight_decay\":{}",
        run_spec.eval.phased_ttt_weight_decay
    ));
    fields.push(format!(
        "\"gpu_lora_ttt_beta2\":{}",
        run_spec.eval.ttt_beta2
    ));
    fields.push(format!(
        "\"frontier_record_ready\":{}",
        frontier_record_ready
    ));
    fields.push(format!(
        "\"leaderboard_algorithm_ready\":{}",
        leaderboard_algorithm_ready
    ));
    fields.push(json_str_field(
        "leaderboard_algorithm_gaps",
        &leaderboard_algorithm_gaps.join("; "),
    ));
    fields.push(json_str_field(
        "timing_backend",
        cuda_timing_backend_label(),
    ));
    fields.push(format!(
        "\"cuda_backward_graph\":{}",
        cuda_backward_graph_enabled()
    ));
    fields.push(format!(
        "\"save_layer_activations\":{}",
        gpu_saved_layer_activations_enabled()
    ));
    fields.push(json_str_field(
        "save_layer_activation_mode",
        &gpu_saved_layer_activations_mode_for_audit(),
    ));
    fields.push(json_str_field(
        "gemm_compute_mode",
        gemm_compute_mode_label(),
    ));
    fields.push(json_str_field(
        "bf16_gemm_compute_mode",
        bf16_gemm_compute_mode_label(),
    ));
    fields.push(format!(
        "\"train_data_configured\":{}",
        run_spec.train.train_data_pattern.is_some()
    ));
    fields.push(format!(
        "\"validation_data_configured\":{}",
        run_spec.train.validation_data_pattern.is_some()
    ));
    let gpu_resident_synthetic_sampler =
        gpu_resident_synthetic_record_sampler_active_for_audit(run_spec, mode);
    let gpu_token_ring_sampler =
        is_record_shaped_mode(mode) && gpu_token_ring_sampler_configured_for_audit(run_spec);
    let gpu_token_full_schedule =
        gpu_token_ring_sampler && gpu_token_ring_full_schedule_enabled_for_audit();
    let gpu_resident_sampler = gpu_resident_synthetic_sampler || gpu_token_ring_sampler;
    fields.push(format!(
        "\"record_host_batch_u32_preload\":{}",
        is_record_shaped_mode(mode)
            && run_spec.train.backend == TrainBackend::CudaDistributed
            && !gpu_shifted_u16_batch_upload_enabled_for_audit()
            && !gpu_resident_sampler
    ));
    fields.push(format!(
        "\"record_host_shifted_u16_span_upload\":{}",
        is_record_shaped_mode(mode)
            && run_spec.train.backend == TrainBackend::CudaDistributed
            && gpu_shifted_u16_batch_upload_enabled_for_audit()
    ));
    fields.push(format!(
        "\"gpu_shifted_target_construction\":{}",
        gpu_shifted_u16_batch_upload_enabled_for_audit()
    ));
    fields.push(format!(
        "\"gpu_resident_data_sampler\":{}",
        gpu_resident_sampler
    ));
    fields.push(format!(
        "\"gpu_resident_sampler_kind\":\"{}\"",
        if gpu_resident_synthetic_sampler {
            "synthetic_record_sequence"
        } else if gpu_token_full_schedule {
            "token_schedule_shard"
        } else if gpu_token_ring_sampler {
            "token_ring_shard"
        } else {
            "none"
        }
    ));
    fields.push(format!(
        "\"gpu_token_ring_sampler_steps\":{}",
        if gpu_token_ring_sampler {
            gpu_token_ring_sampler_steps_for_runtime()
        } else {
            0
        }
    ));
    let gpu_token_ring_hbm_bytes = if gpu_token_ring_sampler {
        (local_tokens_per_rank + 1)
            .saturating_mul(gpu_token_ring_sampler_steps_for_runtime())
            .saturating_mul(std::mem::size_of::<u16>())
    } else {
        0
    };
    fields.push(format!(
        "\"gpu_token_ring_sampler_hbm_bytes_per_rank\":{}",
        gpu_token_ring_hbm_bytes
    ));
    fields.push(format!(
        "\"host_input_copy_per_step\":{}",
        !gpu_resident_sampler && !gpu_shifted_u16_batch_upload_enabled_for_audit()
    ));
    fields.push(format!(
        "\"host_input_ring_refill\":{}",
        gpu_token_ring_sampler && !gpu_token_full_schedule
    ));
    fields.push(format!(
        "\"host_input_schedule_upload_once\":{}",
        gpu_token_full_schedule
    ));
    fields.push(format!(
        "\"gpu_token_ring_full_schedule\":{}",
        gpu_token_full_schedule
    ));
    fields.push(format!(
        "\"gpu_resident_data_sampler_final\":{}",
        gpu_record_data_sampler_final_for_audit(run_spec, mode)
    ));
    fields.push(format!(
        "\"record_requires_gpu_data_sampler\":{}",
        record_require_gpu_data_sampler_for_audit()
    ));
    fields.push(format!(
        "\"tokenizer_vocab_configured\":{}",
        run_spec.eval.tokenizer_vocab_path.is_some()
    ));
    fields.push(format!(
        "\"ttt_seq_len\":{}",
        run_spec
            .eval
            .ttt_seq_len
            .map(|value| value.to_string())
            .unwrap_or_else(|| "null".to_string())
    ));
    fields.push(json_str_field(
        "ttt_mask",
        &format!("{:?}", run_spec.eval.ttt_mask),
    ));
    fields.push(format!(
        "\"asym_logit_enabled\":{}",
        run_spec.model.asym_logit.enabled
    ));
    fields.push(format!(
        "\"ngram_tilt_enabled\":{}",
        run_spec.eval.ngram_tilt.enabled
    ));
    fields.push(format!(
        "\"ngram_token_only\":{}",
        ngram_tilt_token_only_for_audit(run_spec)
    ));
    fields.push(format!(
        "\"ngram_precompute_inside_eval_timer\":{}",
        run_spec.eval.ngram_tilt.precompute_inside_eval_timer
    ));
    fields.push(format!(
        "\"score_first_ttt_verified\":{}",
        run_spec.eval.legal_score_first
            && (!run_spec.eval.qttt
                || run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased)
    ));
    fields.push(format!(
        "\"train_shards\":{}",
        run_spec
            .train
            .train_data_pattern
            .as_deref()
            .and_then(|pattern| simple_glob_paths(pattern).ok())
            .map(|paths| paths.len())
            .unwrap_or(0)
    ));
    fields.push(format!("\"val_docs\":{}", 50_000));
    fields.push("\"val_tokens\":null".to_string());
    fields.push(format!(
        "\"canonical_caseops_dataset\":{}",
        run_spec.eval.caseops_byte_sidecar_pattern.is_some()
    ));
    let caseops_sidecar_sha256 = run_spec
        .eval
        .caseops_byte_sidecar_pattern
        .as_deref()
        .and_then(|pattern| sha256_file_set(pattern).ok().flatten());
    fields.push(json_opt_str_field(
        "caseops_byte_sidecar_sha256",
        caseops_sidecar_sha256.as_deref(),
    ));
    let audit_code_bytes = current_executable_bytes();
    fields.push(format!(
        "\"artifact_total_limit\":{}",
        run_spec.quant.target_artifact_bytes
    ));
    fields.push(format!(
        "\"artifact_code_bytes\":{}",
        audit_code_bytes
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "null".to_string())
    ));
    fields.push(format!(
        "\"artifact_code_bytes_known\":{}",
        audit_code_bytes.is_some()
    ));
    fields.push("\"artifact_model_bytes_known\":false".to_string());
    fields.push("\"artifact_total_bytes_known\":false".to_string());
    fields.push("\"artifact_budget_known\":false".to_string());
    fields.push("\"strict_decimal_bytes\":true".to_string());
    format!("{{{}}}", fields.join(","))
}

fn attention_backend_record_grade(run_spec: &RunSpec) -> bool {
    matches!(
        run_spec.model.attention_backend,
        AttentionBackend::CudnnSdpaBf16
    ) && cudnn_frontend_sdpa_available()
}

fn bf16_output_projection_gemm_enabled(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_OUTPUT_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_forward_projection_gemm_enabled(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_FORWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_primary_forward_projection_gemm_enabled(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_PRIMARY_FORWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_backward_projection_gemm_enabled(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_BACKWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn bf16_backward_chain_requested_for_audit() -> bool {
    env_flag_enabled("PG_GPU_BF16_BACKWARD_CHAIN", false)
}

fn bf16_backward_chain_strict_for_audit() -> bool {
    bf16_backward_chain_requested_for_audit()
        && env_flag_enabled("PG_GPU_BF16_BACKWARD_CHAIN_STRICT", true)
}

fn bf16_backward_chain_qkv_norm_resid_reducer_for_audit() -> &'static str {
    if let Ok(raw) = std::env::var("PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER") {
        match raw.to_ascii_lowercase().as_str() {
            "direct" | "direct_compact" | "one_pass" | "one_pass_compact" => {
                return "direct_compact";
            }
            "split" | "split_compact" | "split_reduce" => return "split_compact",
            "chunked" | "chunked_compact" | "chunked_reduce" => return "chunked_compact",
            _ => return "direct_compact",
        }
    }
    if !bf16_backward_chain_requested_for_audit() {
        return "legacy_or_disabled";
    }
    if env_flag_enabled("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", false) {
        "split_compact"
    } else if env_flag_enabled("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD", false) {
        "chunked_compact"
    } else {
        "direct_compact"
    }
}

fn experimental_fused_qkv_projection_enabled() -> bool {
    bf16_backward_chain_requested_for_audit() || env_flag_enabled("PG_GPU_FUSED_QKV_PROJ", false)
}

fn fused_qkv_projection_record_ok() -> bool {
    matches!(
        std::env::var("PG_GPU_FUSED_QKV_PROJ_RECORD_OK")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn qkv_dx_beta_accum_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_QKV_DX_BETA_ACCUM")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn chunked_q_gain_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_CHUNKED_Q_GAIN_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn chunked_residual_mix_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_CHUNKED_RESIDUAL_MIX_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn chunked_qkv_norm_resid_backward_enabled_for_audit() -> bool {
    if bf16_backward_chain_requested_for_audit() {
        return bf16_backward_chain_qkv_norm_resid_reducer_for_audit() == "chunked_compact";
    }
    env_flag_enabled("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD", false)
}

fn split_qkv_norm_resid_backward_enabled_for_audit() -> bool {
    if bf16_backward_chain_requested_for_audit() {
        return bf16_backward_chain_qkv_norm_resid_reducer_for_audit() == "split_compact";
    }
    env_flag_enabled("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", false)
}

fn split_compact_qkv_norm_resid_backward_enabled_for_audit() -> bool {
    split_qkv_norm_resid_backward_enabled_for_audit()
        && compact_attn_gate_grad_input_enabled_for_audit()
}

fn direct_compact_qkv_norm_resid_backward_enabled_for_audit() -> bool {
    bf16_backward_chain_qkv_norm_resid_reducer_for_audit() == "direct_compact"
        && compact_attn_gate_grad_input_enabled_for_audit()
}

fn qkv_norm_resid_backward_rows_per_chunk_for_audit() -> usize {
    std::env::var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|rows| *rows >= 256)
        .unwrap_or(1024)
}

fn split_residual_mix_grad_enabled_for_audit() -> bool {
    // Experimental only: v43 H100 record-shaped A/B regressed this path.
    // The audit field remains useful to prove the slow split reducer is off.
    matches!(
        std::env::var("PG_GPU_SPLIT_RESIDUAL_MIX_GRAD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn residual_scale_reduce_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_RESIDUAL_SCALE_REDUCE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn chunked_residual_scale_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn tiled_residual_scale_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_TILED_RESIDUAL_SCALE_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn residual_scale_backward_rows_per_chunk_for_audit() -> usize {
    std::env::var("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|rows| *rows >= 64)
        .unwrap_or(256)
}

fn bf16_mlp_down_dx_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_BF16_MLP_DOWN_DX")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn vec2_mlp_activation_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_VEC2_MLP_ACT_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn vec4_mlp_activation_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_VEC4_MLP_ACT_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn fast_mlp_activation_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_FAST_MLP_ACT_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sharded_parallel_muon_fused_global_clip_enabled_for_audit() -> bool {
    // v72/v73 H100 record-shaped runs showed this saves isolated bank-update time
    // but regresses total step time. Keep it available for A/Bs, but do not let it
    // replace the measured-better standalone shard clipping path by default.
    matches!(
        std::env::var("PG_GPU_SHARDED_MUON_FUSED_GLOBAL_CLIP")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sharded_parallel_muon_bf16_shadow_all_gather_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_SHARDED_MUON_BF16_SHADOW_ALL_GATHER")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sharded_parallel_muon_phase_timing_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SHARDED_MUON_PHASE_TIMING")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sharded_parallel_muon_parallel_local_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SHARDED_MUON_PARALLEL_LOCAL")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn bf16_qkv_dx_output_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_backward_projection_gemm_enabled(run_spec)
        && (bf16_backward_chain_requested_for_audit()
            || env_flag_enabled("PG_GPU_BF16_QKV_DX_OUTPUT", false))
}

fn fused_qk_rope_gain_backward_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_FUSED_QK_ROPE_GAIN_BWD")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn fused_qk_rope_gain_forward_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_FUSED_QK_ROPE_GAIN_FWD")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn fused_residual_mix_norm_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_FUSED_RESIDUAL_MIX_NORM")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn fused_mlp_activation_bf16_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_forward_projection_gemm_enabled(run_spec)
        && !matches!(
            std::env::var("PG_GPU_FUSED_MLP_ACT_BF16")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_mlp_up_output_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_primary_forward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_MLP_UP_OUTPUT")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn bf16_norm_side_outputs_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_primary_forward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_NORM_SIDE_OUTPUTS")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn bf16_norm_grad_path_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_backward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_NORM_GRAD_PATH")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn bf16_residual_projection_output_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_primary_forward_projection_gemm_enabled(run_spec)
        && bf16_backward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn bf16_attention_projection_output_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_primary_forward_projection_gemm_enabled(run_spec)
        && bf16_backward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_ATTN_PROJ_OUTPUT")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn bf16_attention_backward_tail_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && run_spec.model.attention_backend == AttentionBackend::CudnnSdpaBf16
        && bf16_backward_projection_gemm_enabled(run_spec)
        && fused_qk_rope_gain_backward_enabled_for_audit()
        && cudnn_saved_bf16_attention_enabled_for_audit(run_spec)
        && (bf16_backward_chain_requested_for_audit()
            || env_flag_enabled("PG_GPU_BF16_ATTN_BACKWARD_TAIL", false))
}

fn bf16_attention_tail_qkv_pack_enabled_for_audit(run_spec: &RunSpec) -> bool {
    bf16_attention_backward_tail_enabled_for_audit(run_spec)
        && bf16_qkv_dx_output_enabled_for_audit(run_spec)
        && experimental_fused_qkv_projection_enabled()
        && (bf16_backward_chain_requested_for_audit()
            || env_flag_enabled("PG_GPU_BF16_ATTN_TAIL_QKV_PACK", false))
}

fn bf16_attention_tail_direct_qkv_pack_enabled_for_audit(run_spec: &RunSpec) -> bool {
    bf16_attention_tail_qkv_pack_enabled_for_audit(run_spec)
        && !matches!(
            std::env::var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_attention_backward_bhsd_do_enabled_for_audit(run_spec: &RunSpec) -> bool {
    bf16_attention_tail_direct_qkv_pack_enabled_for_audit(run_spec)
        && run_spec.model.sparse_attn_gate.enabled
        && run_spec.model.xsa_last_n > 0
        && !matches!(
            std::env::var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_backward_chain_complete_for_audit(run_spec: &RunSpec) -> bool {
    let sparse_xsa_requires_bhsd_do = run_spec.model.sparse_attn_gate.enabled
        && run_spec.model.xsa_last_n >= run_spec.model.num_layers;
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && run_spec.model.attention_backend == AttentionBackend::CudnnSdpaBf16
        && bf16_backward_projection_gemm_enabled(run_spec)
        && cudnn_saved_bf16_attention_enabled_for_audit(run_spec)
        && cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec)
        && direct_saved_layer_activations_enabled_for_audit()
        && skip_f32_attention_saved_activations_enabled_for_audit(run_spec)
        && fused_qk_rope_gain_forward_enabled_for_audit()
        && fused_qk_rope_gain_backward_enabled_for_audit()
        && bf16_qkv_dx_output_enabled_for_audit(run_spec)
        && experimental_fused_qkv_projection_enabled()
        && bf16_attention_backward_tail_enabled_for_audit(run_spec)
        && bf16_attention_tail_qkv_pack_enabled_for_audit(run_spec)
        && bf16_attention_tail_direct_qkv_pack_enabled_for_audit(run_spec)
        && (!sparse_xsa_requires_bhsd_do
            || bf16_attention_backward_bhsd_do_enabled_for_audit(run_spec))
        && (direct_compact_qkv_norm_resid_backward_enabled_for_audit()
            || split_compact_qkv_norm_resid_backward_enabled_for_audit()
            || (chunked_qkv_norm_resid_backward_enabled_for_audit()
                && compact_attn_gate_grad_input_enabled_for_audit()))
        && (!run_spec.model.sparse_attn_gate.enabled
            || compact_attn_gate_grad_input_enabled_for_audit())
        && !recompute_residual_mix_norm_inputs_enabled_for_audit()
        && !split_residual_mix_grad_enabled_for_audit()
}

fn final_norm_bf16_side_output_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_forward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_FINAL_NORM_BF16_OUTPUT")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn bf16_final_norm_to_ce_for_audit(run_spec: &RunSpec) -> bool {
    final_norm_bf16_side_output_enabled_for_audit(run_spec)
        && (chunked_bf16_output_ce_cache_enabled_for_audit(run_spec)
            || tiled_output_cross_entropy_enabled_for_audit(run_spec)
            || fused_exact_output_ce_enabled_for_audit(run_spec)
            || bf16_output_logits_enabled_for_audit(run_spec))
        && !output_path_materializes_full_logits_for_audit(run_spec)
}

fn bf16_attention_output_to_projection_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && run_spec.model.attention_backend == AttentionBackend::CudnnSdpaBf16
        && !bf16_attention_output_bridge_to_f32_for_audit(run_spec)
}

fn bf16_mlp_backward_inputs_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_backward_projection_gemm_enabled(run_spec)
        && bf16_norm_side_outputs_enabled_for_audit(run_spec)
        && bf16_norm_grad_path_enabled_for_audit(run_spec)
        && bf16_mlp_up_output_enabled_for_audit(run_spec)
        // This is deliberately strict. The current fastest profile leaves this
        // off because the BF16 dX subpath regressed on H100; until a better MLP
        // backward cut lands, the audit should show one remaining F32 bridge.
        && bf16_mlp_down_dx_enabled_for_audit()
}

fn bf16_qkv_backward_for_audit(run_spec: &RunSpec) -> bool {
    bf16_backward_chain_complete_for_audit(run_spec)
}

fn bf16_hot_graph_primary_paths_complete_for_audit(run_spec: &RunSpec) -> bool {
    bf16_final_norm_to_ce_for_audit(run_spec)
        && bf16_attention_output_to_projection_for_audit(run_spec)
        && bf16_mlp_backward_inputs_for_audit(run_spec)
        && bf16_qkv_backward_for_audit(run_spec)
}

fn f32_hot_path_bridge_count_for_audit(run_spec: &RunSpec) -> usize {
    if run_spec.model.compute_precision != ModelComputePrecision::Bf16TensorCore {
        return 0;
    }
    [
        bf16_final_norm_to_ce_for_audit(run_spec),
        bf16_attention_output_to_projection_for_audit(run_spec),
        bf16_mlp_backward_inputs_for_audit(run_spec),
        bf16_qkv_backward_for_audit(run_spec),
        bf16_residual_projection_output_enabled_for_audit(run_spec),
        bf16_norm_grad_path_enabled_for_audit(run_spec),
    ]
    .into_iter()
    .filter(|ready| !ready)
    .count()
}

fn bf16_attention_output_bridge_to_f32_for_audit(run_spec: &RunSpec) -> bool {
    let direct_plain_attention = run_spec.model.xsa_last_n == 0
        && !run_spec.model.attn_out_gate.enabled
        && !run_spec.model.sparse_attn_gate.enabled;
    let direct_sparse_xsa = run_spec.model.xsa_last_n >= run_spec.model.num_layers
        && run_spec.model.sparse_attn_gate.enabled
        && !run_spec.model.attn_out_gate.enabled
        && cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec)
        && bf16_sparse_xsa_forward_enabled_for_audit();
    !(bf16_attention_projection_output_enabled_for_audit(run_spec)
        && cudnn_saved_bf16_attention_enabled_for_audit(run_spec)
        && (direct_plain_attention || direct_sparse_xsa))
}

fn bf16_sparse_xsa_forward_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_BF16_SPARSE_XSA_FWD")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn sparse_xsa_warphead_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_SPARSE_XSA_WARPHEAD_BWD")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sparse_xsa_grouped_kv_backward_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_SPARSE_XSA_GROUPED_KV_BWD")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn compact_attn_gate_grad_input_enabled_for_audit() -> bool {
    bf16_backward_chain_requested_for_audit()
        || env_flag_enabled("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT", false)
}

fn recompute_residual_mix_norm_inputs_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn fused_attention_residual_from_base_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_FUSED_ATTN_RESIDUAL_FROM_BASE")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn fused_parallel_attn_residual_rms_norm_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.parallel_residual.enabled
        && fused_attention_residual_from_base_enabled_for_audit()
        && !matches!(
            std::env::var("PG_GPU_FUSED_PARALLEL_ATTN_RESID_RMS")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn batched_muon_newton_schulz_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_MUON_BATCHED_NS")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn muon_ns_profile_for_audit() -> String {
    std::env::var("PG_GPU_MUON_NS_PROFILE")
        .or_else(|_| std::env::var("PG_MUON_NS_PROFILE"))
        .unwrap_or_else(|_| "polar_express".to_string())
        .to_ascii_lowercase()
}

fn muon_ns_steps_for_audit(run_spec: &RunSpec) -> usize {
    std::env::var("PG_GPU_MUON_NS_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|steps| *steps > 0)
        .unwrap_or(run_spec.train.to_train_config().newton_schulz_steps)
}

fn polar_express_muon_enabled_for_audit() -> bool {
    matches!(
        muon_ns_profile_for_audit().as_str(),
        "polar" | "polar_express" | "polarns" | "polar_ns"
    )
}

fn cudnn_saved_bf16_attention_enabled_for_audit(run_spec: &RunSpec) -> bool {
    matches!(
        run_spec.model.attention_backend,
        AttentionBackend::CudnnSdpaBf16
    ) && !matches!(
        std::env::var("PG_GPU_CUDNN_SAVED_BF16_ATTN")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn cudnn_prepacked_bf16_attention_requested_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_CUDNN_PREPACKED_BF16_ATTN")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec: &RunSpec) -> bool {
    cudnn_saved_bf16_attention_enabled_for_audit(run_spec)
        && fused_qk_rope_gain_forward_enabled_for_audit()
        && cudnn_prepacked_bf16_attention_requested_for_audit()
}

fn cudnn_prepacked_bf16_poison_check_enabled_for_audit(run_spec: &RunSpec) -> bool {
    cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec)
        && matches!(
            std::env::var("PG_GPU_CUDNN_PREPACKED_BF16_POISON")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn skip_f32_attention_saved_activations_enabled_for_audit(run_spec: &RunSpec) -> bool {
    cudnn_saved_bf16_attention_enabled_for_audit(run_spec)
        && direct_saved_layer_activations_enabled_for_audit()
        && run_spec.model.xsa_last_n >= run_spec.model.num_layers
        && !matches!(
            std::env::var("PG_GPU_SKIP_F32_ATTN_SAVED_ACTS")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn lean_bf16_saved_layer_cache_enabled_for_audit(run_spec: &RunSpec) -> bool {
    skip_f32_attention_saved_activations_enabled_for_audit(run_spec)
        && bf16_primary_forward_projection_gemm_enabled(run_spec)
        && bf16_backward_projection_gemm_enabled(run_spec)
        && !matches!(
            std::env::var("PG_GPU_LEAN_BF16_SAVED_ACTS")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn direct_saved_layer_activations_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_DIRECT_SAVED_ACTS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn gpu_saved_layer_activations_mode_for_audit() -> String {
    std::env::var("PG_GPU_SAVE_LAYER_ACTS").unwrap_or_else(|_| "off".to_string())
}

fn saved_bf16_layer_activation_shadows_enabled_for_audit(run_spec: &RunSpec) -> bool {
    direct_saved_layer_activations_enabled_for_audit()
        && bf16_primary_forward_projection_gemm_enabled(run_spec)
        && bf16_backward_projection_gemm_enabled(run_spec)
        && !matches!(
            gpu_saved_layer_activations_mode_for_audit()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_output_backward_gemm_enabled(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn overlap_linear_backward_gemms_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn overlap_linear_backward_gemm_role_enabled_for_audit(env_name: &str) -> bool {
    overlap_linear_backward_gemms_enabled_for_audit()
        || matches!(
            std::env::var(env_name)
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn fused_output_cross_entropy_enabled_for_audit(run_spec: &RunSpec) -> bool {
    bf16_output_backward_gemm_enabled(run_spec)
        && !matches!(
            std::env::var("PG_GPU_FUSED_CE_LOSS_BWD")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

fn bf16_output_logits_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_output_projection_gemm_enabled(run_spec)
        && bf16_output_backward_gemm_enabled(run_spec)
        && !fused_exact_output_ce_enabled_for_audit(run_spec)
        && !tiled_output_cross_entropy_enabled_for_audit(run_spec)
        && !chunked_bf16_output_ce_cache_enabled_for_audit(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_LOGITS")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn output_ce_chunk_tokens_for_audit() -> usize {
    std::env::var("PG_GPU_OUTPUT_CE_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(8192)
}

fn output_ce_tile_vocab_for_audit() -> usize {
    std::env::var("PG_GPU_OUTPUT_CE_TILE_VOCAB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(512)
}

fn q_gain_backward_chunk_tokens_for_audit() -> usize {
    std::env::var("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= 256)
        .unwrap_or(256)
}

fn output_ce_backend_env_override_for_audit() -> Option<OutputCeBackend> {
    let raw = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok()?;
    match raw.to_ascii_lowercase().as_str() {
        "chunked_bf16_cache" | "chunked" | "cache" => Some(OutputCeBackend::ChunkedBf16Cache),
        "tiled_repeated_gemm" | "tiled" | "tiled_ce" => Some(OutputCeBackend::TiledRepeatedGemm),
        "fused_exact_wmma" | "fused_exact" | "fused" => Some(OutputCeBackend::FusedExactWmma),
        "auto" | "" => None,
        _ => None,
    }
}

fn effective_output_ce_backend_for_audit(run_spec: &RunSpec) -> Option<OutputCeBackend> {
    if run_spec.model.compute_precision != ModelComputePrecision::Bf16TensorCore
        || !bf16_output_projection_gemm_enabled(run_spec)
        || !bf16_output_backward_gemm_enabled(run_spec)
    {
        return None;
    }
    output_ce_backend_env_override_for_audit()
        .or_else(legacy_output_ce_backend_env_override_for_audit)
        .or(Some(run_spec.model.output_ce_backend))
}

fn legacy_output_ce_backend_env_override_for_audit() -> Option<OutputCeBackend> {
    if matches!(
        std::env::var("PG_GPU_FUSED_EXACT_OUTPUT_CE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) {
        return Some(OutputCeBackend::FusedExactWmma);
    }
    if matches!(
        std::env::var("PG_GPU_TILED_OUTPUT_CE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) {
        return Some(OutputCeBackend::TiledRepeatedGemm);
    }
    if matches!(
        std::env::var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) {
        return Some(OutputCeBackend::ChunkedBf16Cache);
    }
    None
}

fn chunked_bf16_output_ce_cache_enabled_for_audit(run_spec: &RunSpec) -> bool {
    effective_output_ce_backend_for_audit(run_spec) == Some(OutputCeBackend::ChunkedBf16Cache)
}

fn tiled_output_cross_entropy_enabled_for_audit(run_spec: &RunSpec) -> bool {
    effective_output_ce_backend_for_audit(run_spec) == Some(OutputCeBackend::TiledRepeatedGemm)
        && run_spec.model.vocab_size % output_ce_tile_vocab_for_audit() == 0
}

fn fused_exact_output_ce_enabled_for_audit(run_spec: &RunSpec) -> bool {
    effective_output_ce_backend_for_audit(run_spec) == Some(OutputCeBackend::FusedExactWmma)
        && run_spec.model.vocab_size % output_ce_tile_vocab_for_audit() == 0
}

fn output_path_materializes_full_logits_for_audit(run_spec: &RunSpec) -> bool {
    if fused_exact_output_ce_enabled_for_audit(run_spec) {
        return false;
    }
    if chunked_bf16_output_ce_cache_enabled_for_audit(run_spec) {
        let chunk_tokens = output_ce_chunk_tokens_for_audit();
        return chunk_tokens >= run_spec.train.batch_tokens / run_spec.train.world_size.max(1);
    }
    if !tiled_output_cross_entropy_enabled_for_audit(run_spec) {
        return true;
    }
    // A full-vocab "tile" avoids the persistent activation logits field, but
    // still allocates and writes a [tokens, vocab] scratch tile. For record
    // readiness, only sub-vocab tiling counts as removing full logits.
    output_ce_tile_vocab_for_audit() >= run_spec.model.vocab_size
}

fn output_loss_backend_for_audit(run_spec: &RunSpec) -> &'static str {
    if fused_exact_output_ce_enabled_for_audit(run_spec) {
        "fused_exact_wmma"
    } else if chunked_bf16_output_ce_cache_enabled_for_audit(run_spec) {
        if output_path_materializes_full_logits_for_audit(run_spec) {
            "full_batch_bf16_logits_cache"
        } else {
            "chunked_bf16_logits_cache"
        }
    } else if !tiled_output_cross_entropy_enabled_for_audit(run_spec) {
        if bf16_output_logits_enabled_for_audit(run_spec) {
            "full_logits_bf16_single_gemm"
        } else {
            "full_logits_single_gemm"
        }
    } else if output_path_materializes_full_logits_for_audit(run_spec) {
        "full_vocab_tile_repeated_gemm"
    } else {
        "tiled_repeated_gemm"
    }
}

fn output_ce_backend_for_audit(run_spec: &RunSpec) -> &'static str {
    match effective_output_ce_backend_for_audit(run_spec) {
        Some(OutputCeBackend::ChunkedBf16Cache) => "chunked_bf16_cache",
        Some(OutputCeBackend::TiledRepeatedGemm) => "tiled_repeated_gemm",
        Some(OutputCeBackend::FusedExactWmma) => "fused_exact_wmma",
        None => "full_logits",
    }
}

fn output_ce_implementation_for_audit(run_spec: &RunSpec) -> &'static str {
    if fused_exact_output_ce_enabled_for_audit(run_spec) {
        "fused_exact_cublas_tile_warp_rows_v1"
    } else {
        output_loss_backend_for_audit(run_spec)
    }
}

fn output_projection_recompute_passes_for_audit(run_spec: &RunSpec) -> usize {
    if fused_exact_output_ce_enabled_for_audit(run_spec)
        || tiled_output_cross_entropy_enabled_for_audit(run_spec)
    {
        1
    } else {
        0
    }
}

fn production_fused_output_projection_ce_enabled_for_audit(run_spec: &RunSpec) -> bool {
    fused_exact_output_ce_enabled_for_audit(run_spec)
}

fn production_fused_output_projection_ce_validated_for_audit(run_spec: &RunSpec) -> bool {
    production_fused_output_projection_ce_enabled_for_audit(run_spec)
        && matches!(
            std::env::var("PG_GPU_FUSED_EXACT_OUTPUT_CE_VALIDATED")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn gpu_host_scalar_updates_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_GPU_HOST_SCALAR_UPDATES")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

fn gpu_shifted_u16_batch_upload_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_SHIFTED_U16_BATCH_UPLOAD")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn gpu_resident_synthetic_sampler_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn gpu_token_ring_sampler_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_TOKEN_RING_SAMPLER")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn gpu_token_ring_sampler_steps_for_runtime() -> usize {
    std::env::var("PG_GPU_TOKEN_RING_STEPS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|steps| *steps > 0)
        .unwrap_or(16)
}

fn gpu_token_ring_full_schedule_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_GPU_TOKEN_RING_FULL_SCHEDULE")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_token_ring_sampler_alloc_steps_for_runtime() -> usize {
    if gpu_token_ring_sampler_enabled_for_audit() {
        gpu_token_ring_sampler_steps_for_runtime()
    } else {
        1
    }
}

fn gpu_token_ring_sampler_configured_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.train.backend == TrainBackend::CudaDistributed
        && run_spec.train.train_data_pattern.is_some()
        && gpu_token_ring_sampler_enabled_for_audit()
}

fn gpu_record_data_sampler_final_for_audit(run_spec: &RunSpec, mode: RunMode) -> bool {
    is_record_shaped_mode(mode)
        && gpu_token_ring_sampler_configured_for_audit(run_spec)
        && gpu_token_ring_full_schedule_enabled_for_audit()
}

fn record_require_gpu_data_sampler_for_audit() -> bool {
    matches!(
        std::env::var("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Phase 0 runtime gate: when enabled, any time the H2D host-copy fallback
/// fires inside [`cuda_fast_accumulate_runtime_grads`] we return an error
/// instead of silently uploading from `host_input_ids` / `host_targets`. This
/// catches code-path regressions that reintroduce host sampling on the record
/// hot path (closes the spirit of the P1-7 finding without committing the
/// behavior change required by Phase 4a).
#[cfg(feature = "cuda")]
fn record_strict_device_batch_enabled() -> bool {
    matches!(
        std::env::var("PG_RECORD_REQUIRE_DEVICE_BATCH")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn gpu_resident_synthetic_record_sampler_configured_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.train.backend == TrainBackend::CudaDistributed
        && run_spec.train.train_data_pattern.is_none()
        && gpu_resident_synthetic_sampler_enabled_for_audit()
}

fn gpu_resident_synthetic_record_sampler_active_for_audit(
    run_spec: &RunSpec,
    mode: RunMode,
) -> bool {
    is_record_shaped_mode(mode)
        && gpu_resident_synthetic_record_sampler_configured_for_audit(run_spec)
}

fn record_shaped_artifact_export_enabled() -> bool {
    matches!(
        std::env::var("PG_RECORD_SHAPED_EXPORT_ARTIFACT")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sharded_bank_grad_bf16_wire_enabled_for_audit() -> bool {
    matches!(
        std::env::var("PG_NCCL_BF16_BANK_GRAD_WIRE")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn sharded_grouped_grad_collectives_enabled_for_audit() -> bool {
    !matches!(
        std::env::var("PG_NCCL_GROUP_SHARDED_GRAD_COLLECTIVES")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

/// Returns true when the runtime should issue bank-gradient collectives from
/// the backward observer as layer buckets become ready. This path is still
/// experimental: the bucket size and wire dtype must be H100-tuned before it
/// can be treated as the record default.
///
/// `PG_NCCL_BUCKET_OVERLAP` remains accepted as a compatibility alias for the
/// existing Modal/deploy flag. New callers should prefer
/// `PG_NCCL_SIDE_STREAM_COLLECTIVES`.
fn nccl_side_stream_collectives_enabled() -> bool {
    let is_enabled = |name: &str| {
        matches!(
            std::env::var(name)
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    };
    is_enabled("PG_NCCL_SIDE_STREAM_COLLECTIVES")
        || is_enabled("PG_NCCL_BUCKET_OVERLAP")
        || is_enabled("PG_NCCL_BACKWARD_BUCKET_OVERLAP")
}

fn nccl_side_stream_collectives_enabled_for_audit() -> bool {
    nccl_side_stream_collectives_enabled()
}

fn backward_nccl_bucket_overlap_enabled() -> bool {
    matches!(
        std::env::var("PG_NCCL_BACKWARD_BUCKET_OVERLAP")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn backward_nccl_bucket_layers() -> usize {
    std::env::var("PG_NCCL_BACKWARD_BUCKET_LAYERS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|layers| *layers > 0)
        .unwrap_or(1)
}

fn backward_nccl_bucket_overlap_enabled_for_audit() -> bool {
    backward_nccl_bucket_overlap_enabled()
}

fn backward_nccl_bucket_overlap_validated_for_audit() -> bool {
    backward_nccl_bucket_overlap_enabled_for_audit()
        && matches!(
            std::env::var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn cudnn_frontend_sdpa_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        pg_kernels::flash_attn::CudnnFrontendAttention::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

fn frontier_record_gaps(run_spec: &RunSpec) -> Vec<&'static str> {
    let mut gaps = Vec::new();
    match run_spec.model.attention_backend {
        AttentionBackend::NaiveF32 => {
            gaps.push("attention_backend=naive_f32 is debug/parity-only and cannot be used for record-shaped performance");
        }
        AttentionBackend::FlashF32 => {
            gaps.push("attention_backend=flash_f32 is scalar f32 online attention, not production BF16/FP16 FlashAttention/cuDNN SDPA");
        }
        AttentionBackend::CudnnSdpaBf16 => {
            if !cudnn_frontend_sdpa_available() {
                gaps.push("attention_backend=cudnn_sdpa_bf16 requested but cudnn_frontend.h was unavailable at build time, so no fused cuDNN frontend SDPA runtime was compiled");
            }
        }
    }
    if run_spec.train.distributed_optimizer_backend
        != DistributedOptimizerBackend::ShardedParallelMuon
    {
        gaps.push("distributed_optimizer_backend must be sharded_parallel_muon; current all-reduce replicated path is not true Parallel Muon");
    }
    if !polar_express_muon_enabled_for_audit() {
        gaps.push("frontier record target requires Polar Express Newton-Schulz coefficients for Muon; legacy/simple NS is only a parity fallback");
    }
    if matches!(
        run_spec.model.attention_backend,
        AttentionBackend::CudnnSdpaBf16
    ) {
        if !cudnn_saved_bf16_attention_enabled_for_audit(run_spec) {
            gaps.push("record-grade cuDNN SDPA requires saved BF16 attention tensors for backward; PG_GPU_CUDNN_SAVED_BF16_ATTN is disabled");
        }
        if !cudnn_prepacked_bf16_attention_enabled_for_audit(run_spec) {
            gaps.push("record-grade cuDNN SDPA requires prepacked BF16 Q/K/V from the fused QK/RoPE/Gain producer; enable PG_GPU_CUDNN_PREPACKED_BF16_ATTN with the fused producer");
        }
        if bf16_attention_output_bridge_to_f32_for_audit(run_spec) {
            gaps.push("record-grade cuDNN SDPA must keep the attention output on the BF16 projection path; current config still bridges BF16 attention output back to F32");
        }
        if !bf16_attention_backward_tail_enabled_for_audit(run_spec) {
            gaps.push("record-grade cuDNN SDPA backward must keep dQ/dK in the BF16 tail through QK/RoPE/gain backward; current config promotes attention gradients back to F32 immediately");
        }
        if bf16_attention_backward_tail_enabled_for_audit(run_spec)
            && !bf16_attention_tail_qkv_pack_enabled_for_audit(run_spec)
        {
            gaps.push("record-grade cuDNN SDPA backward tail must feed fused QKV backward from BF16-packed dQ/dK/dV; current config still repacks through F32 before the QKV GEMM");
        }
        if bf16_attention_tail_direct_qkv_pack_enabled_for_audit(run_spec)
            && run_spec.model.sparse_attn_gate.enabled
            && run_spec.model.xsa_last_n >= run_spec.model.num_layers
            && !bf16_attention_backward_bhsd_do_enabled_for_audit(run_spec)
        {
            gaps.push("record-grade sparse-XSA cuDNN SDPA backward should feed BF16 [B,H,S,D] dO directly into cuDNN; current config still materializes F32 dO and converts it before SDPA backward");
        }
        if gpu_saved_layer_activations_mode_for_audit().to_ascii_lowercase() != "all"
            || !direct_saved_layer_activations_enabled_for_audit()
            || !skip_f32_attention_saved_activations_enabled_for_audit(run_spec)
        {
            gaps.push("record-grade cuDNN SDPA backward requires direct all-layer BF16 saved activations; otherwise backward can fall back to F32 recompute or saved F32 attention state");
        }
        if !bf16_backward_chain_complete_for_audit(run_spec) {
            gaps.push("record-grade BF16 backward chain is incomplete; require BF16 SDPA dQ/dK/dV, direct QK/RoPE/gain packing, fused BF16 QKV backward, and compact/split QKV norm-resid without F32 fallback");
        }
    }
    match run_spec.model.compute_precision {
        ModelComputePrecision::F32Tf32 => {
            gaps.push("frontier record target requires compute_precision=bf16_tensor_core; current target is f32_tf32");
        }
        ModelComputePrecision::Bf16TensorCore => {
            let bridge_count = f32_hot_path_bridge_count_for_audit(run_spec);
            if !bf16_hot_graph_primary_paths_complete_for_audit(run_spec) || bridge_count > 0 {
                gaps.push("compute_precision=bf16_tensor_core is requested, but the path-level BF16 audit still reports incomplete primary BF16 hot paths or remaining F32 bridges");
            }
        }
    }
    if production_fused_output_projection_ce_enabled_for_audit(run_spec) {
        if !production_fused_output_projection_ce_validated_for_audit(run_spec) {
            gaps.push("fused exact output projection + softcapped CE is implemented and opt-in, but it has not passed the record-readiness timing gate; H100 A/B must beat chunked BF16 CE before this clears the output-CE blocker");
        }
    } else if chunked_bf16_output_ce_cache_enabled_for_audit(run_spec) {
        if output_path_materializes_full_logits_for_audit(run_spec) {
            gaps.push("PG_GPU_CHUNKED_OUTPUT_CE_CACHE is enabled, but output_ce_chunk_tokens covers the full local batch, so the output path still materializes a full [batch_tokens, vocab] BF16 scratch tensor");
        }
    } else if !tiled_output_cross_entropy_enabled_for_audit(run_spec) {
        gaps.push("current GPU output path still materializes full [batch_tokens, vocab] logits; the remaining record cut is a real fused output projection + softcapped CE/backward kernel or the chunked BF16 CE cache bridge");
    } else if output_path_materializes_full_logits_for_audit(run_spec) {
        gaps.push("PG_GPU_TILED_OUTPUT_CE is enabled, but output_ce_tile_vocab is >= vocab_size, so the output path still materializes a full [batch_tokens, vocab] scratch tile");
    } else {
        gaps.push("PG_GPU_TILED_OUTPUT_CE avoids persistent logits but repeats output GEMMs and is not the production fused output projection + softcapped CE/backward kernel required by the final record engine");
    }
    if gpu_host_scalar_updates_enabled_for_audit() {
        gaps.push("PG_GPU_HOST_SCALAR_UPDATES=1 enables legacy per-step host scalar mirror synchronization; record mode should keep trainable scalar params device-resident and sync them only for export");
    }
    if backward_nccl_bucket_overlap_enabled_for_audit()
        && !backward_nccl_bucket_overlap_validated_for_audit()
    {
        gaps.push("backward NCCL bucket overlap is implemented and opt-in, but it has not passed the record-readiness timing gate; keep it experimental until H100 A/B shows a step-time win");
    }
    if run_spec.model.smear_gate && run_spec.model.smear_gate_boundary_token_id.is_none() {
        gaps.push("SmearGate must be BOS/document-boundary masked before record-shaped frontier runs; unmasked previous-token mixing can leak across packed documents");
    }
    if gpu_resident_synthetic_record_sampler_configured_for_audit(run_spec) {
        gaps.push("record-shaped proxy uses a GPU-resident synthetic sampler, but final FineWeb training still needs a real GPU-resident or pinned/ring data sampler before CUDA-graphable record mode is complete");
    } else if gpu_token_ring_sampler_configured_for_audit(run_spec)
        && gpu_token_ring_full_schedule_enabled_for_audit()
    {
        // Full-schedule token ring is the accepted <=120ms record-readiness data path.
    } else if gpu_token_ring_sampler_configured_for_audit(run_spec) {
        gaps.push("record data path uses a GPU token-ring sampler to remove per-step input/target H2D copies, but full-schedule preload is not enabled, so host ring refills can still block CUDA graph capture");
    } else {
        gaps.push("record data path is still host-sampled with per-step H2D input/target copies; final CUDA-graphable record engine needs a GPU-resident sampler");
    }
    if !run_spec.model.recurrence.enabled {
        gaps.push("frontier record target requires recurrence enabled");
    }
    if !(run_spec.model.parallel_residual.enabled
        && run_spec.model.parallel_residual.split_attention_mlp)
    {
        gaps.push("frontier record target requires split parallel residuals");
    }
    if !run_spec.model.sparse_attn_gate.enabled {
        gaps.push("frontier record target requires PR1787/PR1797 SparseAttnGate enabled");
    }
    if run_spec.model.sparse_attn_gate.width != 12 {
        gaps.push("frontier record target expects PR1787/PR1797 SparseAttnGate width 12");
    }
    if run_spec.eval.adaptation_backend != EvalAdaptationBackend::GpuLoraPhased {
        gaps.push("frontier record target requires GPU LoRA/phased legal score-first TTT");
    }
    gaps
}

fn leaderboard_algorithm_gaps(run_spec: &RunSpec) -> Vec<&'static str> {
    let mut gaps = Vec::new();
    match run_spec.runtime.record_profile {
        RecordProfile::Baseline => {}
        RecordProfile::Frontier2014Clean => {
            if run_spec.model.train_seq_len < 3072 || run_spec.model.eval_seq_len < 3072 {
                gaps.push("frontier_2014_clean requires 3072-token final train/eval context");
            }
            if run_spec.train.seq_schedule.len() < 3 {
                gaps.push("frontier_2014_clean requires the 1024/2048/3072 progressive train context schedule");
            }
            if run_spec.eval.ttt_seq_len != Some(3072) {
                gaps.push("frontier_2014_clean requires eval.ttt_seq_len=3072");
            }
            if run_spec.eval.ttt_mask != TttMask::NoQv {
                gaps.push("frontier_2014_clean requires no_qv TTT masking");
            }
            if run_spec.eval.ttt_lora_targets
                != pg_model::spec::TttLoraTargetsSpec::upstream_pr2014_no_qv()
            {
                gaps.push("frontier_2014_clean requires upstream no_qv LoRA targets: lm_head,k,o,mlp enabled and q/v disabled");
            }
            if run_spec.eval.phased_ttt_phases != 1 {
                gaps.push("frontier_2014_clean requires one score-first TTT phase");
            }
            if !run_spec.model.bigram.enabled {
                gaps.push("frontier_2014_clean requires BigramHash enabled");
            }
            if run_spec.model.bigram.enabled && !run_spec.runtime.bigram_embedding_merge {
                gaps.push("frontier_2014_clean requires runtime.bigram_embedding_merge=true so BigramHash does not add an unfused forward pass");
            }
        }
        RecordProfile::Frontier2135Audit => {
            if (run_spec.model.qk_gain_init - 5.0).abs() > f32::EPSILON {
                gaps.push("frontier_2135_audit requires qk_gain_init=5.0");
            }
            if (run_spec.model.recurrence.enable_at_frac - 0.35).abs() > 1e-6 {
                gaps.push("frontier_2135_audit requires recurrence enable_at_frac=0.35; late recurrence is a speed probe, not a quality record target");
            }
            if run_spec.model.recurrence.enable_at_step != Some(1748) {
                gaps.push("frontier_2135_audit requires recurrence enable_at_step=1748 so active-recurrence timing and BPB evidence are reproducible");
            }
            if run_spec.model.recurrence.start_layer != 3
                || run_spec.model.recurrence.repeat_layers != 3
            {
                gaps.push("frontier_2135_audit requires recurrence over layers 3-5");
            }
            if run_spec.model.parallel_residual.start_layer != 8 {
                gaps.push(
                    "frontier_2135_audit requires decoder-only parallel residual from layer 8",
                );
            }
            if run_spec.model.bigram.enabled {
                gaps.push(
                    "frontier_2135_audit inherits the PR2130 stack and must not enable BigramHash",
                );
            }
            if run_spec.model.eval_seq_len < 2560 || run_spec.eval.ttt_seq_len != Some(2560) {
                gaps.push("frontier_2135_audit requires 2560-token eval/TTT context");
            }
            if run_spec.eval.ttt_mask != TttMask::KOff {
                gaps.push("frontier_2135_audit requires K-off TTT masking");
            }
            if run_spec.eval.ttt_lora_targets
                != pg_model::spec::TttLoraTargetsSpec::upstream_pr2135()
            {
                gaps.push("frontier_2135_audit requires upstream PR2135 LoRA targets: lm_head,q,v,o,mlp enabled and k disabled");
            }
            if run_spec
                .eval
                .lora_lr
                .map(|value| (value - 0.00008).abs() > 1e-9)
                .unwrap_or(true)
            {
                gaps.push("frontier_2135_audit requires eval.lora_lr=8e-5; using the training matrix LR for LoRA-TTT is not PR2135-compatible");
            }
            if run_spec.quant.gptq_calibration_batches != 32 {
                gaps.push("frontier_2135_audit requires GPTQ calibration batches=32");
            }
            if !run_spec.model.asym_logit.enabled {
                gaps.push("frontier_2135_audit requires AsymLogit enabled");
            }
            if !run_spec.eval.ngram_tilt.enabled {
                gaps.push("frontier_2135_audit requires token-only n-gram tilt enabled");
            }
            if !ngram_tilt_token_only_for_audit(run_spec) {
                gaps.push("frontier_2135_audit requires token-only n-gram tilt with within/word/agreement channels disabled");
            }
        }
        RecordProfile::Frontier2135SpeedProbe => {
            gaps.push("frontier_2135_speed_probe is diagnostic only and is not a leaderboard-quality record target");
        }
    }
    if matches!(
        run_spec.runtime.record_profile,
        RecordProfile::Frontier2014Clean | RecordProfile::Frontier2135Audit
    ) {
        if run_spec.runtime.backward_chain_profile != BackwardChainProfile::Bf16DirectCompact {
            gaps.push("frontier record targets require runtime.backward_chain_profile=bf16_direct_compact");
        }
        if !run_spec.runtime.require_device_batch || !run_spec.runtime.token_ring_full_schedule {
            gaps.push("frontier record targets require device-batch and full-schedule token-ring runtime ownership");
        }
        if !run_spec.runtime.recurrence_active_required {
            gaps.push("frontier record targets require runtime.recurrence_active_required=true so speed-only inactive-recurrence runs cannot pass as quality records");
        }
        if run_spec.runtime.recurrent_active_steps_min == 0 {
            gaps.push("frontier record targets require runtime.recurrent_active_steps_min>0");
        }
        if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
            && !run_spec.eval.ttt_lora_targets.rust_gpu_runtime_supported()
        {
            gaps.push("Rust GPU LoRA-TTT currently implements q-only adapters; requested frontier LoRA targets are not yet implemented in the Rust runtime");
        }
        if pg_quant::layout::compiled_layout_for_quant_spec(&run_spec.quant).is_none() {
            gaps.push(
                "frontier record targets require QuantSpec to match a compiled quantization layout",
            );
        }
        if matches!(
            run_spec.runtime.recurrent_backward_profile,
            RecurrentBackwardProfile::Pass1StraightThrough
                | RecurrentBackwardProfile::AllStraightThrough
        ) {
            gaps.push("frontier record targets cannot use straight-through recurrent backward without separate BPB validation; keep it as a speed probe");
        }
        if run_spec.runtime.skip_recurrent_pass1_bank_grads {
            gaps.push("frontier record targets cannot skip recurrent pass-1 bank gradients without separate BPB validation; keep it as a speed probe");
        }
        if run_spec.runtime.skip_recurrent_bank_grads {
            gaps.push("frontier record targets cannot skip recurrent bank gradients without separate BPB validation; keep it as a speed probe");
        }
    }
    if !run_spec.model.caseops.enabled || !run_spec.model.caseops.byte_sidecar {
        gaps.push("PR1787/PR1797 algorithm target requires CaseOps with byte sidecar");
    }
    if run_spec.model.caseops.enabled
        && run_spec.model.caseops.byte_sidecar
        && run_spec.eval.caseops_byte_sidecar_pattern.is_none()
    {
        gaps.push("CaseOps byte sidecar is enabled, but eval.caseops_byte_sidecar_pattern is not configured");
    }
    if !run_spec.model.sparse_attn_gate.enabled {
        gaps.push("PR1787/PR1797 algorithm target requires SparseAttnGate");
    }
    if run_spec.model.sparse_attn_gate.width != 12 {
        gaps.push("PR1787/PR1797 SparseAttnGate target expects width 12");
    }
    if !run_spec.model.smear_gate {
        gaps.push("PR1797 algorithm target requires SmearGate");
    }
    if run_spec.model.smear_gate && run_spec.model.smear_gate_boundary_token_id.is_none() {
        gaps.push("SmearGate must be BOS/document-boundary masked; unmasked previous-token mixing can leak across packed documents");
    }
    if !run_spec.quant.lqer.enabled {
        gaps.push("PR1797 algorithm target requires LQER asymmetric post-GPTQ correction");
    }
    gaps
}

fn ngram_tilt_token_only_for_audit(run_spec: &RunSpec) -> bool {
    !run_spec.eval.ngram_tilt.enabled
        || (run_spec.eval.ngram_tilt.token_boost != 0.0
            && run_spec.eval.ngram_tilt.within_boost == 0.0
            && run_spec.eval.ngram_tilt.word_boost == 0.0
            && run_spec.eval.ngram_tilt.agree_add_boost == 0.0)
}

fn allow_frontier_record_gaps_for_development() -> bool {
    std::env::var("PG_ALLOW_FRONTIER_RECORD_GAPS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn record_max_ms_per_step_for_submission(run_spec: &RunSpec) -> f64 {
    if let Some(max_ms) = run_spec.runtime.max_ms_per_step {
        return max_ms;
    }
    std::env::var("PG_RECORD_MAX_MS_PER_STEP")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(130.0)
}

#[cfg(feature = "cuda")]
fn eval_gpu_lora_phased_from_train(
    model: &GptModel,
    plan: &ExecutionPlan,
    tokens: &[u32],
    token_bytes: &[f32],
) -> PgResult<(f64, f64)> {
    let cfg = pg_eval::gpu_lora_ttt::GpuLoraPhasedTttConfig::from_plan(plan, tokens.len());
    pg_eval::gpu_lora_ttt::eval_gpu_lora_phased_ttt_distributed(
        model,
        plan,
        tokens,
        token_bytes,
        &cfg,
        plan.run_spec.train.world_size.max(1),
    )
}

#[cfg(not(feature = "cuda"))]
fn eval_gpu_lora_phased_from_train(
    _model: &GptModel,
    _plan: &ExecutionPlan,
    _tokens: &[u32],
    _token_bytes: &[f32],
) -> PgResult<(f64, f64)> {
    Err(pg_core::PgError::InvalidOp(
        "eval_adaptation_backend=gpu_lora_phased requires pg-train --features cuda".into(),
    ))
}

fn validate_backend_request(run_spec: &RunSpec, mode: RunMode) -> PgResult<()> {
    if is_record_shaped_mode(mode) && run_spec.train.backend != TrainBackend::CudaDistributed {
        return Err(pg_core::PgError::InvalidOp(
            "record-shaped modes require --backend cuda-distributed; cpu and cuda-single backends are not valid for the real H100 train surface".into(),
        ));
    }
    if is_record_shaped_mode(mode) && run_spec.train.fast_bank_updates {
        return Err(pg_core::PgError::InvalidOp(
            "record-shaped modes forbid --fast-bank-updates because it bypasses Muon".into(),
        ));
    }
    let allow_synthetic_record_shape_sampler = mode == RunMode::RecordShapedProxy
        && gpu_resident_synthetic_record_sampler_configured_for_audit(run_spec);
    if is_record_shaped_mode(mode)
        && run_spec.train.train_data_pattern.is_none()
        && !allow_synthetic_record_shape_sampler
    {
        return Err(pg_core::PgError::InvalidOp(
            "record-shaped modes require --train-data; synthetic data is not representative of the submission train surface".into(),
        ));
    }
    if is_record_shaped_mode(mode)
        && cudnn_prepacked_bf16_attention_requested_for_audit()
        && !fused_qk_rope_gain_forward_enabled_for_audit()
    {
        return Err(pg_core::PgError::InvalidOp(
            "record-shaped modes refuse PG_GPU_CUDNN_PREPACKED_BF16_ATTN when PG_GPU_FUSED_QK_ROPE_GAIN_FWD is disabled; prepacked BF16 Q/K requires the fused producer".into(),
        ));
    }
    if mode == RunMode::Record {
        if experimental_fused_qkv_projection_enabled() && !fused_qkv_projection_record_ok() {
            return Err(pg_core::PgError::InvalidOp(
                "record mode refuses PG_GPU_FUSED_QKV_PROJ unless PG_GPU_FUSED_QKV_PROJ_RECORD_OK=1; promote the packed QKV path only after the BF16 parity and record-shaped timing gates pass".into(),
            ));
        }
        if run_spec.train.validation_data_pattern.is_none() {
            return Err(pg_core::PgError::InvalidOp(
                "record mode requires --val-data so train -> artifact -> full legal eval is exercised in the same submission path".into(),
            ));
        }
        if run_spec.model.caseops.enabled && run_spec.model.caseops.byte_sidecar {
            if run_spec.eval.caseops_byte_sidecar_pattern.is_none() {
                return Err(pg_core::PgError::InvalidOp(
                    "record mode with CaseOps requires --caseops-byte-sidecar; score bytes must come from the lossless CaseOps byte sidecar".into(),
                ));
            }
        } else if run_spec.eval.tokenizer_vocab_path.is_none() {
            return Err(pg_core::PgError::InvalidOp(
                "record mode requires --tokenizer-vocab; placeholder BPB bytes are not leaderboard-valid".into(),
            ));
        }
        if run_spec.eval.max_tokens.is_some() {
            return Err(pg_core::PgError::InvalidOp(
                "record mode cannot set --eval-max-tokens; leaderboard eval must score the full validation stream".into(),
            ));
        }
    }
    if is_record_shaped_mode(mode) {
        let gaps = frontier_record_gaps(run_spec);
        if !gaps.is_empty()
            && mode != RunMode::Record
            && (run_spec.allow_unsupported_variants || allow_frontier_record_gaps_for_development())
        {
            log::warn!(
                "record-shaped proxy is running with frontier/submission gaps: {}",
                gaps.join("; ")
            );
        } else if !gaps.is_empty() {
            return Err(pg_core::PgError::InvalidOp(format!(
                "record-shaped mode is not frontier/submission-ready: {}. Set PG_ALLOW_FRONTIER_RECORD_GAPS=1 or pass --allow-unsupported-variants only for non-submission development proxy runs; real record mode always fails closed.",
                gaps.join("; ")
            )));
        }
    }
    if mode == RunMode::Record {
        let algorithm_gaps = leaderboard_algorithm_gaps(run_spec);
        if !algorithm_gaps.is_empty() {
            return Err(pg_core::PgError::InvalidOp(format!(
                "record mode is not leaderboard-algorithm-ready: {}. Use record-shaped-proxy for systems benchmarking until these P1 algorithm gaps are closed.",
                algorithm_gaps.join("; ")
            )));
        }
    }
    if run_spec.train.world_size > 1
        && matches!(
            run_spec.train.backend,
            TrainBackend::CudaSingle | TrainBackend::CudaSingleParity
        )
    {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-single backends require world_size=1; use cuda-distributed for multi-rank execution".into(),
        ));
    }
    if run_spec.train.backend == TrainBackend::CudaSingleParity && !matches!(mode, RunMode::Smoke) {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-single-parity is a smoke/debug backend only and cannot produce proxy or record metrics".into(),
        ));
    }
    if run_spec.train.backend == TrainBackend::CudaDistributed && run_spec.train.world_size < 2 {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-distributed requires --world-size >= 2".into(),
        ));
    }
    if run_spec.train.backend == TrainBackend::CudaDistributed && run_spec.train.rank != 0 {
        return Err(pg_core::PgError::InvalidOp(
            "cuda-distributed currently runs as one local multi-GPU process, so rank must be 0"
                .into(),
        ));
    }
    if run_spec.train.backend != TrainBackend::CudaDistributed
        && run_spec.train.distributed_optimizer_backend
            != DistributedOptimizerBackend::AllReduceReplicatedMuon
    {
        return Err(pg_core::PgError::InvalidOp(
            "non-distributed backends must use distributed_optimizer_backend=all_reduce_replicated_muon".into(),
        ));
    }
    #[cfg(not(feature = "cuda"))]
    if matches!(
        run_spec.train.backend,
        TrainBackend::CudaSingle | TrainBackend::CudaSingleParity | TrainBackend::CudaDistributed
    ) {
        return Err(pg_core::PgError::InvalidOp(
            "selected CUDA backend requires building with --features cuda".into(),
        ));
    }
    match run_spec.train.backend {
        TrainBackend::Cpu => Ok(()),
        TrainBackend::CudaSingle => Ok(()),
        TrainBackend::CudaSingleParity => Ok(()),
        TrainBackend::CudaDistributed => Ok(()),
    }
}

fn load_bpb_luts(
    run_spec: &RunSpec,
    vocab_size: usize,
    mode: RunMode,
) -> PgResult<(BpbLuts, &'static str)> {
    if run_spec.model.caseops.enabled && run_spec.model.caseops.byte_sidecar {
        if run_spec.eval.caseops_byte_sidecar_pattern.is_none()
            && matches!(mode, RunMode::Record)
            && run_spec.train.validation_data_pattern.is_some()
        {
            return Err(pg_core::PgError::InvalidOp(
                "record mode with CaseOps requires --caseops-byte-sidecar; tokenizer vocab byte estimates are not CaseOps-safe".into(),
            ));
        }
        return Ok((BpbLuts::placeholder(vocab_size), "caseops_byte_sidecar"));
    }
    if let Some(path) = run_spec.eval.tokenizer_vocab_path.as_deref() {
        let luts = BpbLuts::from_vocab_file(std::path::Path::new(path))?;
        if matches!(mode, RunMode::Record) && luts.base_bytes.len() != vocab_size {
            return Err(pg_core::PgError::InvalidOp(format!(
                "record mode tokenizer vocab has {} pieces but model vocab_size is {}; use the submission tokenizer for final BPB",
                luts.base_bytes.len(),
                vocab_size
            )));
        }
        Ok((luts, "tokenizer_vocab"))
    } else if matches!(mode, RunMode::Record) && run_spec.train.validation_data_pattern.is_some() {
        Err(pg_core::PgError::InvalidOp(
            "record mode evaluation requires --tokenizer-vocab; placeholder byte counts are not submission-valid".into(),
        ))
    } else {
        Ok((BpbLuts::placeholder(vocab_size), "placeholder"))
    }
}

fn eval_target_byte_counts(
    run_spec: &RunSpec,
    tokens: &[u32],
    bpb_luts: &BpbLuts,
    max_tokens: Option<usize>,
) -> PgResult<Vec<f32>> {
    if let Some(pattern) = run_spec.eval.caseops_byte_sidecar_pattern.as_deref() {
        let sidecar =
            pg_data::token_stream::load_validation_byte_sidecar_limited(pattern, max_tokens)?;
        if sidecar.len() != tokens.len() {
            return Err(pg_core::PgError::DataFormat(format!(
                "CaseOps byte sidecar length {} does not match validation token length {}",
                sidecar.len(),
                tokens.len()
            )));
        }
        if sidecar.len() < 2 {
            return Ok(Vec::new());
        }
        return Ok(sidecar[1..].to_vec());
    }
    Ok(bpb_luts.pair_byte_counts_u32(tokens))
}

fn count_params(model: &GptModel) -> usize {
    let mut total = 0;
    total += model.tok_emb.len();
    total += model.bigram_embed.len();
    total += model.bigram_proj.len();
    total += 1;
    total += model.smear_gate.len();
    total += model.skip_weights.len();
    total += model.qo_bank.len();
    total += model.kv_bank.len();
    total += model.mlp_up_bank.len();
    total += model.mlp_down_bank.len();
    for bp in &model.blocks {
        total += bp.attn_scale.len();
        total += bp.mlp_scale.len();
        total += bp.resid_mix.len();
        total += bp.q_gain.len();
        total += bp.attn_gate_weight.len();
        total += bp.attn_gate_bias.len();
        total += bp.sparse_attn_gate_weight.len();
    }
    total += model.ve_embed.len();
    total += model.ve_proj.len();
    total += 1;
    total += model.ve_layer_scales.len();
    total
}

fn smoke_bank_step(param: &mut [f32], grad: &[f32], lr: f32, weight_decay: f32) {
    let decay = 1.0 - lr * weight_decay;
    for (p, &g) in param.iter_mut().zip(grad.iter()) {
        *p = decay * *p - lr * g;
    }
}

fn step_batch_plan(
    run_spec: &RunSpec,
    mode: RunMode,
    model_config: &pg_model::ModelConfig,
    world_size: usize,
) -> PgResult<StepBatchPlan> {
    step_batch_plan_with_seq_len(
        run_spec,
        mode,
        model_config,
        world_size,
        run_spec.train.seq_len,
    )
}

fn step_batch_plan_for_train_step(
    run_spec: &RunSpec,
    mode: RunMode,
    model_config: &pg_model::ModelConfig,
    world_size: usize,
    step: usize,
    total_steps: usize,
) -> PgResult<StepBatchPlan> {
    let seq_len = active_train_seq_len_for_step(run_spec, step, total_steps);
    step_batch_plan_with_seq_len(run_spec, mode, model_config, world_size, seq_len)
}

fn active_train_seq_len_for_step(run_spec: &RunSpec, step: usize, total_steps: usize) -> usize {
    if run_spec.train.seq_schedule.is_empty() || total_steps == 0 {
        return run_spec.train.seq_len;
    }
    let progress = (step + 1) as f32 / total_steps.max(1) as f32;
    let mut entries = run_spec.train.seq_schedule.clone();
    entries.sort_by(|a, b| a.frac.total_cmp(&b.frac));
    entries
        .iter()
        .find(|entry| progress <= entry.frac)
        .map(|entry| entry.seq_len)
        .or_else(|| entries.last().map(|entry| entry.seq_len))
        .unwrap_or(run_spec.train.seq_len)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn active_recurrence_for_train_step(
    run_spec: &RunSpec,
    step: usize,
    total_steps: usize,
    elapsed_seconds: f32,
    max_wallclock_seconds: f32,
) -> bool {
    if !run_spec.model.recurrence.enabled {
        return false;
    }
    if let Some(enable_at_step) = run_spec.model.recurrence.enable_at_step {
        return step + 1 >= enable_at_step;
    }
    let threshold = run_spec.model.recurrence.enable_at_frac;
    if threshold <= 0.0 {
        return true;
    }
    let progress = if max_wallclock_seconds > 0.0 {
        elapsed_seconds / max_wallclock_seconds
    } else if total_steps > 0 {
        (step + 1) as f32 / total_steps as f32
    } else {
        0.0
    };
    progress >= threshold
}

fn step_batch_plan_with_seq_len(
    run_spec: &RunSpec,
    mode: RunMode,
    model_config: &pg_model::ModelConfig,
    world_size: usize,
    requested_seq_len: usize,
) -> PgResult<StepBatchPlan> {
    let sequence_tokens = run_spec
        .train
        .seq_len
        .min(requested_seq_len)
        .min(model_config.train_seq_len)
        .max(1);
    let microbatch_tokens = match mode {
        RunMode::Smoke => sequence_tokens.min(16),
        RunMode::Proxy => sequence_tokens.min(64),
        RunMode::RecordShapedProxy | RunMode::Record => sequence_tokens,
    }
    .max(1);
    let minimum_global_tokens = microbatch_tokens * world_size.max(1);
    let global_batch_tokens = match mode {
        RunMode::Smoke | RunMode::Proxy => minimum_global_tokens,
        RunMode::RecordShapedProxy | RunMode::Record => run_spec.train.batch_tokens,
    };
    if global_batch_tokens < minimum_global_tokens {
        return Err(pg_core::PgError::InvalidOp(format!(
            "batch_tokens {} is smaller than one global microbatch {}",
            global_batch_tokens, minimum_global_tokens
        )));
    }
    if global_batch_tokens % minimum_global_tokens != 0 {
        return Err(pg_core::PgError::InvalidOp(format!(
            "batch_tokens {} must be divisible by world_size * seq_len ({} * {} = {})",
            global_batch_tokens, world_size, microbatch_tokens, minimum_global_tokens
        )));
    }
    Ok(StepBatchPlan {
        microbatch_tokens,
        global_batch_tokens,
        local_microbatches_per_step: global_batch_tokens / minimum_global_tokens,
    })
}

fn scale_cpu_grads(grads: &mut GradBuffers, scale: f32) {
    fn scale_slice(values: &mut [f32], scale: f32) {
        for value in values {
            *value *= scale;
        }
    }

    scale_slice(&mut grads.tok_emb, scale);
    scale_slice(&mut grads.bigram_embed, scale);
    scale_slice(&mut grads.bigram_proj, scale);
    grads.bigram_scale *= scale;
    scale_slice(&mut grads.smear_gate, scale);
    scale_slice(&mut grads.skip_weights, scale);
    scale_slice(&mut grads.qo_bank, scale);
    scale_slice(&mut grads.kv_bank, scale);
    scale_slice(&mut grads.mlp_up_bank, scale);
    scale_slice(&mut grads.mlp_down_bank, scale);
    for values in &mut grads.block_attn_scale {
        scale_slice(values, scale);
    }
    for values in &mut grads.block_mlp_scale {
        scale_slice(values, scale);
    }
    for values in &mut grads.block_resid_mix {
        scale_slice(values, scale);
    }
    for values in &mut grads.block_q_gain {
        scale_slice(values, scale);
    }
    for values in &mut grads.block_attn_gate_weight {
        scale_slice(values, scale);
    }
    for values in &mut grads.block_attn_gate_bias {
        scale_slice(values, scale);
    }
    for values in &mut grads.block_sparse_attn_gate_weight {
        scale_slice(values, scale);
    }
    scale_slice(&mut grads.ve_embed, scale);
    scale_slice(&mut grads.ve_proj, scale);
    grads.ve_scale *= scale;
    scale_slice(&mut grads.ve_layer_scales, scale);
}

fn current_executable_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("PG_SUBMISSION_CODE_BYTES") {
        if let Ok(bytes) = value.parse::<usize>() {
            return Some(bytes);
        }
        return None;
    }
    if let Ok(path) = std::env::var("PG_SUBMISSION_CODE_DIR") {
        if let Ok(bytes) = directory_regular_file_bytes(std::path::Path::new(&path)) {
            return Some(bytes);
        }
        return None;
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len() as usize)
}

fn current_executable_sha256() -> Option<String> {
    if let Ok(value) = std::env::var("PG_SUBMISSION_CODE_SHA256") {
        return Some(value);
    }
    if let Ok(path) = std::env::var("PG_SUBMISSION_CODE_DIR") {
        return directory_regular_file_sha256(std::path::Path::new(&path)).ok();
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| sha256_file(&path).ok())
}

fn directory_regular_file_sha256(path: &std::path::Path) -> std::io::Result<String> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.is_file() {
        return sha256_file(path).map_err(|err| std::io::Error::other(err.to_string()));
    }
    let mut paths = Vec::new();
    collect_regular_files(path, &mut paths)?;
    paths.sort();
    let mut state = Sha256State::new();
    for file in paths {
        state.update(file.to_string_lossy().as_bytes());
        state.update(b"\0");
        let digest = sha256_file(&file).map_err(|err| std::io::Error::other(err.to_string()))?;
        state.update(digest.as_bytes());
        state.update(b"\0");
    }
    Ok(hex_digest(state.finalize()))
}

fn collect_regular_files(path: &std::path::Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.is_file() {
        out.push(path.to_path_buf());
    } else if metadata.is_dir() {
        for entry in std::fs::read_dir(path)? {
            collect_regular_files(&entry?.path(), out)?;
        }
    }
    Ok(())
}

fn directory_regular_file_bytes(path: &std::path::Path) -> std::io::Result<usize> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len() as usize);
    }
    if !metadata.is_dir() {
        return Ok(0);
    }

    let mut total = 0usize;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        total += directory_regular_file_bytes(&entry.path())?;
    }
    Ok(total)
}

fn flatten_params_into(model: &GptModel, flat: &mut [f32]) {
    let mut pos = 0;
    fn copy(src: &[f32], dst: &mut [f32], pos: &mut usize) {
        dst[*pos..*pos + src.len()].copy_from_slice(src);
        *pos += src.len();
    }

    copy(&model.tok_emb, flat, &mut pos);
    copy(&model.bigram_embed, flat, &mut pos);
    copy(&model.bigram_proj, flat, &mut pos);
    flat[pos] = model.bigram_scale;
    pos += 1;
    copy(&model.smear_gate, flat, &mut pos);
    copy(&model.skip_weights, flat, &mut pos);
    copy(&model.qo_bank, flat, &mut pos);
    copy(&model.kv_bank, flat, &mut pos);
    copy(&model.mlp_up_bank, flat, &mut pos);
    copy(&model.mlp_down_bank, flat, &mut pos);
    for bp in &model.blocks {
        copy(&bp.attn_scale, flat, &mut pos);
        copy(&bp.mlp_scale, flat, &mut pos);
        copy(&bp.resid_mix, flat, &mut pos);
        copy(&bp.q_gain, flat, &mut pos);
        copy(&bp.attn_gate_weight, flat, &mut pos);
        copy(&bp.attn_gate_bias, flat, &mut pos);
        copy(&bp.sparse_attn_gate_weight, flat, &mut pos);
    }
    copy(&model.ve_embed, flat, &mut pos);
    copy(&model.ve_proj, flat, &mut pos);
    flat[pos] = model.ve_scale;
    pos += 1;
    copy(&model.ve_layer_scales, flat, &mut pos);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_tiled_output_ce_env<T>(
        enabled: Option<&str>,
        tile_vocab: Option<&str>,
        f: impl FnOnce() -> T,
    ) -> T {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_backend = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok();
        let old_enabled = std::env::var("PG_GPU_TILED_OUTPUT_CE").ok();
        let old_tile = std::env::var("PG_GPU_OUTPUT_CE_TILE_VOCAB").ok();
        if matches!(enabled, Some("1" | "true" | "TRUE" | "yes" | "on")) {
            unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", "tiled_repeated_gemm") };
        } else {
            unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", "chunked_bf16_cache") };
        }
        match enabled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match tile_vocab {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_TILE_VOCAB", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_TILE_VOCAB") },
        }
        let out = f();
        match old_backend {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND") },
        }
        match old_enabled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match old_tile {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_TILE_VOCAB", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_TILE_VOCAB") },
        }
        out
    }

    #[test]
    fn record_requires_distributed_backend() {
        for backend in [
            TrainBackend::Cpu,
            TrainBackend::CudaSingle,
            TrainBackend::CudaSingleParity,
        ] {
            let mut spec = RunSpec::default();
            spec.train.backend = backend;
            let err = validate_backend_request(&spec, RunMode::Record).unwrap_err();
            assert!(
                err.to_string().contains("cuda-distributed"),
                "unexpected error for {:?}: {}",
                backend,
                err
            );
        }
    }

    #[test]
    fn record_shaped_proxy_requires_distributed_backend() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingle;
        let err = validate_backend_request(&spec, RunMode::RecordShapedProxy).unwrap_err();
        assert!(err.to_string().contains("cuda-distributed"));
    }

    #[test]
    fn cuda_single_requires_world_size_one() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingle;
        spec.train.world_size = 2;
        let err = validate_backend_request(&spec, RunMode::Smoke).unwrap_err();
        assert!(err.to_string().contains("world_size=1"));
    }

    #[test]
    fn cuda_single_parity_stays_smoke_only() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingleParity;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("smoke/debug"));
    }

    #[test]
    fn cuda_single_smoke_support_matches_build() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingle;
        let result = validate_backend_request(&spec, RunMode::Smoke);
        if cfg!(feature = "cuda") {
            assert!(result.is_ok(), "expected cuda-single smoke to be allowed");
        } else {
            let err = result.unwrap_err();
            assert!(
                err.to_string()
                    .contains("requires building with --features cuda")
            );
        }
    }

    #[test]
    fn cuda_distributed_requires_world_size_two_or_more() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 1;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("world-size >= 2"));
    }

    #[test]
    fn cuda_distributed_requires_rank_zero() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 2;
        spec.train.rank = 1;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("rank must be 0"));
    }

    #[test]
    fn sharded_parallel_muon_requires_distributed_backend() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaSingle;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        let err = validate_backend_request(&spec, RunMode::Proxy).unwrap_err();
        assert!(err.to_string().contains("non-distributed backends"));
    }

    #[test]
    fn record_requires_real_train_data() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.train_data_pattern = None;
        let err = validate_backend_request(&spec, RunMode::Record).unwrap_err();
        assert!(err.to_string().contains("--train-data"));
    }

    #[test]
    fn nccl_side_stream_audit_does_not_claim_bucket_overlap() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        // Side-stream collectives are useful plumbing, but they are not true
        // backward bucket overlap until reduce-scatter fires while layer
        // backward is still running. The audit must keep those states separate.
        let prev = std::env::var("PG_NCCL_BUCKET_OVERLAP").ok();
        let prev_side = std::env::var("PG_NCCL_SIDE_STREAM_COLLECTIVES").ok();
        let prev_backward = std::env::var("PG_NCCL_BACKWARD_BUCKET_OVERLAP").ok();
        let prev_validated = std::env::var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED").ok();

        unsafe {
            std::env::set_var("PG_NCCL_BUCKET_OVERLAP", "0");
            std::env::remove_var("PG_NCCL_SIDE_STREAM_COLLECTIVES");
            std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP");
            std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED");
        }
        assert!(!nccl_side_stream_collectives_enabled());
        assert!(!nccl_side_stream_collectives_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_validated_for_audit());

        unsafe {
            std::env::set_var("PG_NCCL_BUCKET_OVERLAP", "1");
        }
        assert!(nccl_side_stream_collectives_enabled());
        assert!(nccl_side_stream_collectives_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_validated_for_audit());

        unsafe {
            std::env::set_var("PG_NCCL_BUCKET_OVERLAP", "true");
        }
        assert!(nccl_side_stream_collectives_enabled());
        assert!(nccl_side_stream_collectives_enabled_for_audit());

        unsafe {
            std::env::remove_var("PG_NCCL_BUCKET_OVERLAP");
            std::env::set_var("PG_NCCL_SIDE_STREAM_COLLECTIVES", "true");
        }
        assert!(nccl_side_stream_collectives_enabled());
        assert!(nccl_side_stream_collectives_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_validated_for_audit());

        unsafe {
            std::env::remove_var("PG_NCCL_BUCKET_OVERLAP");
            std::env::remove_var("PG_NCCL_SIDE_STREAM_COLLECTIVES");
            std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP", "true");
        }
        assert!(nccl_side_stream_collectives_enabled());
        assert!(nccl_side_stream_collectives_enabled_for_audit());
        assert!(backward_nccl_bucket_overlap_enabled_for_audit());
        assert!(!backward_nccl_bucket_overlap_validated_for_audit());

        unsafe {
            std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED", "true");
        }
        assert!(backward_nccl_bucket_overlap_validated_for_audit());

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_BUCKET_OVERLAP", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_BUCKET_OVERLAP") },
        }
        match prev_side {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_SIDE_STREAM_COLLECTIVES", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_SIDE_STREAM_COLLECTIVES") },
        }
        match prev_backward {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP") },
        }
        match prev_validated {
            Some(value) => unsafe {
                std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED", value)
            },
            None => unsafe { std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED") },
        }
    }

    #[test]
    fn record_audit_marks_bucket_overlap_as_bf16_per_layer_reduce_when_requested() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_bucket = std::env::var("PG_NCCL_BUCKET_OVERLAP").ok();
        let prev_side = std::env::var("PG_NCCL_SIDE_STREAM_COLLECTIVES").ok();
        let prev_backward = std::env::var("PG_NCCL_BACKWARD_BUCKET_OVERLAP").ok();
        let prev_validated = std::env::var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED").ok();
        let prev_bf16_wire = std::env::var("PG_NCCL_BF16_BANK_GRAD_WIRE").ok();

        unsafe {
            std::env::remove_var("PG_NCCL_BUCKET_OVERLAP");
            std::env::remove_var("PG_NCCL_SIDE_STREAM_COLLECTIVES");
            std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP", "1");
            std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED");
            std::env::set_var("PG_NCCL_BF16_BANK_GRAD_WIRE", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &[],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"nccl_side_stream_collectives\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"backward_nccl_bucket_overlap\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"backward_nccl_bucket_overlap_validated\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"sharded_parallel_muon_bank_grad_wire_dtype\":\"bf16\""),
            "{json}"
        );
        assert!(
            json.contains(
                "\"sharded_parallel_muon_bank_grad_collective\":\"bucketed_reduce_to_owner\""
            ),
            "{json}"
        );
        assert!(json.contains("\"backward_nccl_bucket_layers\":1"), "{json}");

        match prev_bucket {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_BUCKET_OVERLAP", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_BUCKET_OVERLAP") },
        }
        match prev_side {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_SIDE_STREAM_COLLECTIVES", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_SIDE_STREAM_COLLECTIVES") },
        }
        match prev_backward {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP") },
        }
        match prev_validated {
            Some(value) => unsafe {
                std::env::set_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED", value)
            },
            None => unsafe { std::env::remove_var("PG_NCCL_BACKWARD_BUCKET_OVERLAP_VALIDATED") },
        }
        match prev_bf16_wire {
            Some(value) => unsafe { std::env::set_var("PG_NCCL_BF16_BANK_GRAD_WIRE", value) },
            None => unsafe { std::env::remove_var("PG_NCCL_BF16_BANK_GRAD_WIRE") },
        }
    }

    #[test]
    fn record_rejects_non_frontier_capabilities() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        spec.train.validation_data_pattern = Some("/tmp/val/*.bin".into());
        spec.eval.tokenizer_vocab_path = Some("/tmp/tokenizer.vocab".into());
        let err = validate_backend_request(&spec, RunMode::Record).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("not frontier/submission-ready"), "{msg}");
        assert!(msg.contains("sharded_parallel_muon"), "{msg}");
        assert!(msg.contains("GPU LoRA/phased"), "{msg}");
    }

    #[test]
    fn record_rejects_frontier_gap_override() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        spec.train.validation_data_pattern = Some("/tmp/val/*.bin".into());
        spec.eval.tokenizer_vocab_path = Some("/tmp/tokenizer.vocab".into());
        spec.allow_unsupported_variants = true;
        let err = validate_backend_request(&spec, RunMode::Record).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("real record mode always fails closed"),
            "{msg}"
        );
    }

    #[test]
    fn record_shaped_proxy_allows_explicit_frontier_gap_override() {
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        spec.allow_unsupported_variants = true;
        let result = validate_backend_request(&spec, RunMode::RecordShapedProxy);
        if cfg!(feature = "cuda") {
            assert!(result.is_ok(), "{result:?}");
        } else {
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("requires building with --features cuda"),
                "{msg}"
            );
        }
    }

    #[test]
    fn record_shaped_proxy_allows_explicit_synthetic_gpu_sampler_benchmark() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER").ok();
        unsafe {
            std::env::set_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.train_data_pattern = None;
        spec.allow_unsupported_variants = true;

        let result = validate_backend_request(&spec, RunMode::RecordShapedProxy);
        if cfg!(feature = "cuda") {
            assert!(result.is_ok(), "{result:?}");
        } else {
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("requires building with --features cuda"),
                "{msg}"
            );
        }

        let err = validate_backend_request(&spec, RunMode::Record)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("record-shaped modes require --train-data")
                || err.contains("record mode requires --val-data"),
            "real record mode must not silently accept synthetic train data: {err}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER", value) },
            None => unsafe { std::env::remove_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER") },
        }
    }

    #[test]
    fn record_requires_full_tokenizer_backed_eval() {
        let mut spec = RunSpec::for_family(pg_model::VariantFamily::HybridCompetitiveSp8192);
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;

        let msg = validate_backend_request(&spec, RunMode::Record)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("--val-data"), "{msg}");

        spec.train.validation_data_pattern = Some("/tmp/val/*.bin".into());
        let msg = validate_backend_request(&spec, RunMode::Record)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("--caseops-byte-sidecar"), "{msg}");

        spec.eval.caseops_byte_sidecar_pattern = Some("/tmp/val_bytes/*.bin".into());
        spec.eval.max_tokens = Some(4096);
        let msg = validate_backend_request(&spec, RunMode::Record)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("--eval-max-tokens"), "{msg}");
    }

    #[test]
    fn frontier_gap_detector_rejects_scalar_attention_even_with_other_capabilities() {
        let mut spec = RunSpec::for_family(pg_model::VariantFamily::HybridCompetitiveSp8192);
        spec.model.attention_backend = AttentionBackend::FlashF32;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        let gaps = frontier_record_gaps(&spec);
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("scalar f32 online attention")),
            "{gaps:?}"
        );
        assert!(
            gaps.iter().any(|gap| gap.contains("compute_precision")),
            "{gaps:?}"
        );
    }

    #[test]
    fn frontier_gap_detector_reflects_cudnn_runtime_availability() {
        let mut spec = RunSpec::for_family(pg_model::VariantFamily::HybridCompetitiveSp8192);
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        let gaps = frontier_record_gaps(&spec);
        if cudnn_frontend_sdpa_available() {
            assert!(
                gaps.iter().any(|gap| gap.contains("prepacked BF16 Q/K/V")),
                "{gaps:?}"
            );
        } else {
            assert!(
                gaps.iter()
                    .any(|gap| gap.contains("unavailable at build time")),
                "{gaps:?}"
            );
        }
        assert!(
            gaps.iter().any(|gap| gap.contains("compute_precision")),
            "{gaps:?}"
        );
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("full [batch_tokens, vocab] logits")),
            "{gaps:?}"
        );
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("bridges BF16 attention output back to F32")),
            "{gaps:?}"
        );
    }

    #[test]
    fn frontier_gap_detector_distinguishes_synthetic_sampler_from_final_sampler() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER").ok();
        unsafe {
            std::env::set_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER", "1");
        }

        let mut spec = RunSpec::for_family(pg_model::VariantFamily::HybridCompetitiveSp8192);
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.train_data_pattern = None;
        let gaps = frontier_record_gaps(&spec);
        assert!(
            gaps.iter().any(|gap| gap.contains("synthetic sampler")),
            "synthetic GPU sampler is a record-shaped benchmark tool, not the final FineWeb sampler: {gaps:?}"
        );
        assert!(
            !gaps
                .iter()
                .any(|gap| gap.contains("per-step H2D input/target copies")),
            "synthetic GPU sampler should not be described as per-step host input copies: {gaps:?}"
        );

        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        let gaps = frontier_record_gaps(&spec);
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("per-step H2D input/target copies")),
            "real train-data mode still needs a final GPU-resident sampler: {gaps:?}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER", value) },
            None => unsafe { std::env::remove_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER") },
        }
    }

    #[test]
    fn frontier_gap_detector_requires_polar_express_muon() {
        let old_gpu_profile = std::env::var("PG_GPU_MUON_NS_PROFILE").ok();
        let old_profile = std::env::var("PG_MUON_NS_PROFILE").ok();

        unsafe {
            std::env::set_var("PG_GPU_MUON_NS_PROFILE", "simple");
            std::env::remove_var("PG_MUON_NS_PROFILE");
        }
        let mut spec = RunSpec::for_family(pg_model::VariantFamily::HybridCompetitiveSp8192);
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        let gaps = frontier_record_gaps(&spec);
        assert!(
            gaps.iter().any(|gap| gap.contains("Polar Express")),
            "legacy/simple NS should remain a frontier record gap: {gaps:?}"
        );

        unsafe {
            std::env::set_var("PG_GPU_MUON_NS_PROFILE", "polar_express");
        }
        let gaps = frontier_record_gaps(&spec);
        assert!(
            !gaps.iter().any(|gap| gap.contains("Polar Express")),
            "Polar Express profile should satisfy the Muon coefficient gate: {gaps:?}"
        );

        match old_gpu_profile {
            Some(value) => unsafe { std::env::set_var("PG_GPU_MUON_NS_PROFILE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_MUON_NS_PROFILE") },
        }
        match old_profile {
            Some(value) => unsafe { std::env::set_var("PG_MUON_NS_PROFILE", value) },
            None => unsafe { std::env::remove_var("PG_MUON_NS_PROFILE") },
        }
    }

    #[test]
    fn record_batch_plan_uses_batch_tokens() {
        let mut spec = RunSpec::default();
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::Record, &config, 8).unwrap();
        assert_eq!(plan.microbatch_tokens, 2048);
        assert_eq!(plan.global_batch_tokens, 786_432);
        assert_eq!(plan.local_microbatches_per_step, 48);
    }

    #[test]
    fn record_shaped_proxy_batch_plan_matches_record_shape() {
        let mut spec = RunSpec::default();
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        assert_eq!(plan.microbatch_tokens, 2048);
        assert_eq!(plan.global_batch_tokens, 786_432);
        assert_eq!(plan.local_microbatches_per_step, 48);
    }

    #[test]
    fn train_seq_schedule_changes_microbatch_shape_without_changing_tokens() {
        let mut spec = RunSpec::default();
        spec.train.seq_len = 3072;
        spec.model.train_seq_len = 3072;
        spec.train.batch_tokens = 786_432;
        spec.train.seq_schedule = vec![
            pg_model::TrainSeqScheduleEntry {
                frac: 0.10,
                seq_len: 1024,
            },
            pg_model::TrainSeqScheduleEntry {
                frac: 0.70,
                seq_len: 2048,
            },
            pg_model::TrainSeqScheduleEntry {
                frac: 1.00,
                seq_len: 3072,
            },
        ];
        let config = spec.model.to_model_config();

        let early =
            step_batch_plan_for_train_step(&spec, RunMode::Record, &config, 8, 0, 100).unwrap();
        assert_eq!(early.microbatch_tokens, 1024);
        assert_eq!(early.global_batch_tokens, 786_432);
        assert_eq!(early.local_microbatches_per_step, 96);

        let middle =
            step_batch_plan_for_train_step(&spec, RunMode::Record, &config, 8, 10, 100).unwrap();
        assert_eq!(middle.microbatch_tokens, 2048);
        assert_eq!(middle.local_microbatches_per_step, 48);

        let late =
            step_batch_plan_for_train_step(&spec, RunMode::Record, &config, 8, 70, 100).unwrap();
        assert_eq!(late.microbatch_tokens, 3072);
        assert_eq!(late.local_microbatches_per_step, 32);
    }

    #[test]
    fn runtime_bf16_direct_compact_profile_owns_mlp_down_dx() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let keys = [
            "PG_GPU_BF16_BACKWARD_CHAIN",
            "PG_GPU_BF16_BACKWARD_CHAIN_STRICT",
            "PG_GPU_BF16_MLP_DOWN_DX",
        ];
        let old = keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in keys {
            unsafe { std::env::remove_var(key) };
        }

        let mut spec = RunSpec::default();
        spec.runtime.backward_chain_profile = BackwardChainProfile::Bf16DirectCompact;
        apply_runtime_profile_env(&spec, RunMode::RecordShapedProxy).unwrap();

        assert_eq!(
            std::env::var("PG_GPU_BF16_BACKWARD_CHAIN").as_deref(),
            Ok("1")
        );
        assert_eq!(
            std::env::var("PG_GPU_BF16_BACKWARD_CHAIN_STRICT").as_deref(),
            Ok("1")
        );
        assert_eq!(std::env::var("PG_GPU_BF16_MLP_DOWN_DX").as_deref(), Ok("1"));

        for (key, value) in old {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }

    #[test]
    fn record_audit_json_declares_record_shape_and_backends() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let sampler_keys = [
            "PG_GPU_TOKEN_RING_SAMPLER",
            "PG_GPU_TOKEN_RING_STEPS",
            "PG_GPU_RESIDENT_SYNTHETIC_SAMPLER",
            "PG_GPU_SHIFTED_U16_BATCH_UPLOAD",
        ];
        let old_sampler_env = sampler_keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        unsafe {
            for key in sampler_keys {
                std::env::remove_var(key);
            }
        }
        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(json.contains("\"event\":\"record_path_audit\""), "{json}");
        assert!(json.contains("\"mode\":\"record-shaped-proxy\""), "{json}");
        assert!(json.contains("\"seq_len\":2048"), "{json}");
        assert!(json.contains("\"global_batch_tokens\":786432"), "{json}");
        assert!(json.contains("\"local_batch\":48"), "{json}");
        assert!(json.contains("\"local_tokens_per_rank\":98304"), "{json}");
        assert!(
            json.contains("\"effective_global_batch_tokens\":786432"),
            "{json}"
        );
        assert!(
            json.contains("\"attention_backend_impl\":\"cudnn_frontend_sdpa_bf16_f32_bridge\""),
            "{json}"
        );
        assert!(json.contains("\"attention_dtype\":\"bf16\""), "{json}");
        assert!(
            json.contains("\"model_compute_dtype\":\"f32_tf32\""),
            "{json}"
        );
        assert!(
            json.contains("\"model_precision_target\":\"f32_tf32\""),
            "{json}"
        );
        assert!(
            json.contains("\"model_precision_target_met\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"attention_precision_bridge\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"cudnn_sdpa_deterministic_backward\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"backward_backend\":\"gpu_explicit\""),
            "{json}"
        );
        assert!(json.contains("\"microbatch_serial_loop\":false"), "{json}");
        assert!(
            json.contains("\"host_batch_segments_per_rank\":1"),
            "{json}"
        );
        assert!(
            json.contains("\"record_host_batch_u32_preload\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"muon_newton_schulz_profile\":\"polar_express\""),
            "{json}"
        );
        assert!(
            json.contains("\"polar_express_newton_schulz\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_resident_data_sampler\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_lora_prefix_doc_ttt_schedule\":false"),
            "{json}"
        );
        assert!(json.contains("\"gpu_lora_prefix_docs\":2000"), "{json}");
        assert!(
            json.contains("\"timing_backend\":\"host_wallclock_cuda_boundary\""),
            "{json}"
        );
        assert!(
            json.contains("\"leaderboard_algorithm_ready\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"artifact_model_bytes_known\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"artifact_total_bytes_known\":false"),
            "{json}"
        );
        for (key, value) in old_sampler_env {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }

    #[test]
    fn record_audit_marks_gpu_resident_synthetic_sampler_only_for_synthetic_record_shape() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER").ok();
        unsafe {
            std::env::set_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.train_data_pattern = None;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            false,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"gpu_resident_data_sampler\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_resident_sampler_kind\":\"synthetic_record_sequence\""),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_resident_data_sampler_final\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"host_input_copy_per_step\":false"),
            "{json}"
        );

        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            false,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"gpu_resident_data_sampler\":false"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER", value) },
            None => unsafe { std::env::remove_var("PG_GPU_RESIDENT_SYNTHETIC_SAMPLER") },
        }
    }

    #[test]
    fn record_audit_enables_primary_bf16_forward_by_default() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_qkv_dx = std::env::var("PG_GPU_BF16_QKV_DX_OUTPUT").ok();
        let old_backend = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok();
        let old_fused_ce = std::env::var("PG_GPU_FUSED_EXACT_OUTPUT_CE").ok();
        let old_tiled_ce = std::env::var("PG_GPU_TILED_OUTPUT_CE").ok();
        let old_chunked_ce = std::env::var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE").ok();
        unsafe {
            std::env::remove_var("PG_GPU_BF16_QKV_DX_OUTPUT");
            std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND");
            std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE");
            std::env::remove_var("PG_GPU_TILED_OUTPUT_CE");
            std::env::remove_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"bf16_forward_projection_gemm\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"primary_block_forward_bf16_gemm\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"experimental_fused_qkv_projection\":false"),
            "{json}"
        );
        assert!(json.contains("\"qkv_dx_beta_accum\":false"), "{json}");
        assert!(json.contains("\"bf16_qkv_dx_output\":false"), "{json}");
        assert!(
            json.contains("\"direct_saved_layer_activations\":false"),
            "{json}"
        );

        match old_qkv_dx {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_QKV_DX_OUTPUT", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_QKV_DX_OUTPUT") },
        }
        match old_backend {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND") },
        }
        match old_fused_ce {
            Some(value) => unsafe { std::env::set_var("PG_GPU_FUSED_EXACT_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE") },
        }
        match old_tiled_ce {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match old_chunked_ce {
            Some(value) => unsafe { std::env::set_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE") },
        }
    }

    #[test]
    fn record_audit_marks_gpu_token_ring_sampler_for_real_train_data() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_enabled = std::env::var("PG_GPU_TOKEN_RING_SAMPLER").ok();
        let prev_steps = std::env::var("PG_GPU_TOKEN_RING_STEPS").ok();
        let prev_full_schedule = std::env::var("PG_GPU_TOKEN_RING_FULL_SCHEDULE").ok();
        let prev_require_sampler = std::env::var("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER").ok();
        unsafe {
            std::env::set_var("PG_GPU_TOKEN_RING_SAMPLER", "1");
            std::env::set_var("PG_GPU_TOKEN_RING_STEPS", "8");
            std::env::remove_var("PG_GPU_TOKEN_RING_FULL_SCHEDULE");
            std::env::remove_var("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.train_data_pattern = Some("/tmp/train/*.bin".into());
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"gpu_resident_data_sampler\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_resident_sampler_kind\":\"token_ring_shard\""),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_token_ring_sampler_steps\":8"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_token_ring_sampler_hbm_bytes_per_rank\":1572880"),
            "{json}"
        );
        assert!(
            json.contains("\"host_input_copy_per_step\":false"),
            "{json}"
        );
        assert!(json.contains("\"host_input_ring_refill\":true"), "{json}");
        assert!(
            json.contains("\"host_input_schedule_upload_once\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_token_ring_full_schedule\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"gpu_resident_data_sampler_final\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"record_requires_gpu_data_sampler\":false"),
            "{json}"
        );

        unsafe {
            std::env::set_var("PG_GPU_TOKEN_RING_FULL_SCHEDULE", "1");
            std::env::set_var("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER", "1");
        }
        let full_schedule_json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            full_schedule_json.contains("\"gpu_resident_sampler_kind\":\"token_schedule_shard\""),
            "{full_schedule_json}"
        );
        assert!(
            full_schedule_json.contains("\"host_input_ring_refill\":false"),
            "{full_schedule_json}"
        );
        assert!(
            full_schedule_json.contains("\"host_input_schedule_upload_once\":true"),
            "{full_schedule_json}"
        );
        assert!(
            full_schedule_json.contains("\"gpu_token_ring_full_schedule\":true"),
            "{full_schedule_json}"
        );
        assert!(
            full_schedule_json.contains("\"gpu_resident_data_sampler_final\":true"),
            "{full_schedule_json}"
        );
        assert!(
            full_schedule_json.contains("\"record_requires_gpu_data_sampler\":true"),
            "{full_schedule_json}"
        );

        match prev_enabled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TOKEN_RING_SAMPLER", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TOKEN_RING_SAMPLER") },
        }
        match prev_steps {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TOKEN_RING_STEPS", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TOKEN_RING_STEPS") },
        }
        match prev_full_schedule {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TOKEN_RING_FULL_SCHEDULE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TOKEN_RING_FULL_SCHEDULE") },
        }
        match prev_require_sampler {
            Some(value) => unsafe {
                std::env::set_var("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER", value)
            },
            None => unsafe { std::env::remove_var("PG_RECORD_REQUIRE_GPU_DATA_SAMPLER") },
        }
    }

    #[test]
    fn record_audit_tracks_bf16_attention_bhsd_do_tail() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_tail = std::env::var("PG_GPU_BF16_ATTN_BACKWARD_TAIL").ok();
        let prev_qkv_pack = std::env::var("PG_GPU_BF16_ATTN_TAIL_QKV_PACK").ok();
        let prev_direct = std::env::var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK").ok();
        let prev_bhsd_do = std::env::var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO").ok();
        let prev_qkv_dx = std::env::var("PG_GPU_BF16_QKV_DX_OUTPUT").ok();
        let prev_fused_qkv = std::env::var("PG_GPU_FUSED_QKV_PROJ").ok();

        unsafe {
            std::env::set_var("PG_GPU_BF16_ATTN_BACKWARD_TAIL", "1");
            std::env::set_var("PG_GPU_BF16_ATTN_TAIL_QKV_PACK", "1");
            std::env::set_var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK", "1");
            std::env::set_var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO", "1");
            std::env::set_var("PG_GPU_BF16_QKV_DX_OUTPUT", "1");
            std::env::set_var("PG_GPU_FUSED_QKV_PROJ", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.xsa_last_n = spec.model.num_layers;
        spec.model.sparse_attn_gate.enabled = true;

        assert!(bf16_attention_backward_bhsd_do_enabled_for_audit(&spec));
        unsafe {
            std::env::set_var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO", "0");
        }
        assert!(!bf16_attention_backward_bhsd_do_enabled_for_audit(&spec));

        match prev_tail {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_ATTN_BACKWARD_TAIL", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_ATTN_BACKWARD_TAIL") },
        }
        match prev_qkv_pack {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_ATTN_TAIL_QKV_PACK", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_ATTN_TAIL_QKV_PACK") },
        }
        match prev_direct {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK") },
        }
        match prev_bhsd_do {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO") },
        }
        match prev_qkv_dx {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_QKV_DX_OUTPUT", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_QKV_DX_OUTPUT") },
        }
        match prev_fused_qkv {
            Some(value) => unsafe { std::env::set_var("PG_GPU_FUSED_QKV_PROJ", value) },
            None => unsafe { std::env::remove_var("PG_GPU_FUSED_QKV_PROJ") },
        }
    }

    #[test]
    fn record_audit_tracks_coherent_bf16_backward_chain() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let keys = [
            "PG_GPU_BF16_BACKWARD_CHAIN",
            "PG_GPU_BF16_BACKWARD_CHAIN_STRICT",
            "PG_GPU_BF16_ATTN_BACKWARD_TAIL",
            "PG_GPU_BF16_ATTN_TAIL_QKV_PACK",
            "PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK",
            "PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO",
            "PG_GPU_BF16_QKV_DX_OUTPUT",
            "PG_GPU_FUSED_QKV_PROJ",
            "PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER",
            "PG_GPU_SPLIT_QKV_NORM_RESID_BWD",
            "PG_GPU_CHUNKED_QKV_NORM_RESID_BWD",
            "PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT",
            "PG_GPU_CUDNN_PREPACKED_BF16_ATTN",
            "PG_GPU_DIRECT_SAVED_ACTS",
            "PG_GPU_SAVE_LAYER_ACTS",
            "PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS",
            "PG_GPU_SPLIT_RESIDUAL_MIX_GRAD",
            "PG_GPU_FINAL_NORM_BF16_OUTPUT",
            "PG_GPU_OUTPUT_CE_BACKEND",
            "PG_GPU_CHUNKED_OUTPUT_CE_CACHE",
            "PG_GPU_OUTPUT_CE_CHUNK_TOKENS",
            "PG_GPU_BF16_ATTN_PROJ_OUTPUT",
            "PG_GPU_BF16_MLP_UP_OUTPUT",
            "PG_GPU_BF16_NORM_SIDE_OUTPUTS",
            "PG_GPU_BF16_NORM_GRAD_PATH",
            "PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT",
            "PG_GPU_BF16_MLP_DOWN_DX",
        ];
        let old = keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        unsafe {
            for key in keys {
                std::env::remove_var(key);
            }
            std::env::set_var("PG_GPU_BF16_BACKWARD_CHAIN", "1");
            std::env::set_var("PG_GPU_CUDNN_PREPACKED_BF16_ATTN", "1");
            std::env::set_var("PG_GPU_DIRECT_SAVED_ACTS", "1");
            std::env::set_var("PG_GPU_SAVE_LAYER_ACTS", "all");
            std::env::set_var("PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS", "0");
            std::env::set_var("PG_GPU_SPLIT_RESIDUAL_MIX_GRAD", "0");
            std::env::set_var("PG_GPU_FINAL_NORM_BF16_OUTPUT", "1");
            std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", "chunked_bf16_cache");
            std::env::set_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE", "1");
            std::env::set_var("PG_GPU_OUTPUT_CE_CHUNK_TOKENS", "8192");
            std::env::set_var("PG_GPU_BF16_ATTN_PROJ_OUTPUT", "1");
            std::env::set_var("PG_GPU_BF16_MLP_UP_OUTPUT", "1");
            std::env::set_var("PG_GPU_BF16_NORM_SIDE_OUTPUTS", "1");
            std::env::set_var("PG_GPU_BF16_NORM_GRAD_PATH", "1");
            std::env::set_var("PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT", "1");
            std::env::set_var("PG_GPU_BF16_MLP_DOWN_DX", "0");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.xsa_last_n = spec.model.num_layers;
        spec.model.sparse_attn_gate.enabled = true;

        assert!(bf16_backward_chain_requested_for_audit());
        assert!(bf16_backward_chain_strict_for_audit());
        assert!(experimental_fused_qkv_projection_enabled());
        assert!(bf16_qkv_dx_output_enabled_for_audit(&spec));
        assert!(bf16_attention_backward_tail_enabled_for_audit(&spec));
        assert!(bf16_attention_tail_qkv_pack_enabled_for_audit(&spec));
        assert!(bf16_attention_tail_direct_qkv_pack_enabled_for_audit(&spec));
        assert!(bf16_attention_backward_bhsd_do_enabled_for_audit(&spec));
        assert_eq!(
            bf16_backward_chain_qkv_norm_resid_reducer_for_audit(),
            "direct_compact"
        );
        assert!(direct_compact_qkv_norm_resid_backward_enabled_for_audit());
        assert!(!split_qkv_norm_resid_backward_enabled_for_audit());
        assert!(!split_compact_qkv_norm_resid_backward_enabled_for_audit());
        assert!(bf16_backward_chain_complete_for_audit(&spec));
        assert!(bf16_final_norm_to_ce_for_audit(&spec));
        assert!(bf16_attention_output_to_projection_for_audit(&spec));
        assert!(bf16_qkv_backward_for_audit(&spec));
        assert!(!bf16_mlp_backward_inputs_for_audit(&spec));
        assert!(!bf16_hot_graph_primary_paths_complete_for_audit(&spec));
        assert_eq!(f32_hot_path_bridge_count_for_audit(&spec), 1);

        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &[],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"bf16_backward_chain_requested\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"bf16_backward_chain_strict\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"bf16_backward_chain_complete\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"experimental_fused_qkv_projection\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"bf16_attention_tail_direct_qkv_pack\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"bf16_backward_chain_qkv_norm_resid_reducer\":\"direct_compact\""),
            "{json}"
        );
        assert!(
            json.contains("\"direct_compact_qkv_norm_resid_backward\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"split_compact_qkv_norm_resid_backward\":false"),
            "{json}"
        );
        assert!(json.contains("\"bf16_final_norm_to_ce\":true"), "{json}");
        assert!(
            json.contains("\"bf16_attention_output_to_projection\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"bf16_mlp_backward_inputs\":false"),
            "{json}"
        );
        assert!(json.contains("\"bf16_qkv_backward\":true"), "{json}");
        assert!(
            json.contains("\"bf16_hot_graph_primary_paths_complete\":false"),
            "{json}"
        );
        assert!(json.contains("\"f32_hot_path_bridge_count\":1"), "{json}");
        assert!(
            json.contains("\"model_precision_target_met\":false"),
            "{json}"
        );

        unsafe {
            std::env::set_var("PG_GPU_BF16_MLP_DOWN_DX", "1");
        }
        assert!(bf16_mlp_backward_inputs_for_audit(&spec));
        assert!(bf16_hot_graph_primary_paths_complete_for_audit(&spec));
        assert_eq!(f32_hot_path_bridge_count_for_audit(&spec), 0);
        assert!(model_precision_target_met(&spec));

        unsafe {
            std::env::set_var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK", "0");
        }
        assert!(!bf16_backward_chain_complete_for_audit(&spec));

        for (key, value) in old {
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }

    #[test]
    fn record_audit_tracks_q_gain_backward_chunk_tokens() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_chunked = std::env::var("PG_GPU_CHUNKED_Q_GAIN_BWD").ok();
        let prev_tokens = std::env::var("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS").ok();
        unsafe {
            std::env::set_var("PG_GPU_CHUNKED_Q_GAIN_BWD", "1");
            std::env::set_var("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS", "512");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(json.contains("\"chunked_q_gain_backward\":true"), "{json}");
        assert!(
            json.contains("\"q_gain_backward_chunk_tokens\":512"),
            "{json}"
        );

        match prev_chunked {
            Some(value) => unsafe { std::env::set_var("PG_GPU_CHUNKED_Q_GAIN_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_CHUNKED_Q_GAIN_BWD") },
        }
        match prev_tokens {
            Some(value) => unsafe { std::env::set_var("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS", value) },
            None => unsafe { std::env::remove_var("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS") },
        }
    }

    #[test]
    fn record_audit_tracks_tiled_residual_scale_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_chunked = std::env::var("PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD").ok();
        let prev_tiled = std::env::var("PG_GPU_TILED_RESIDUAL_SCALE_BWD").ok();
        let prev_rows = std::env::var("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK").ok();
        unsafe {
            std::env::set_var("PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD", "1");
            std::env::set_var("PG_GPU_TILED_RESIDUAL_SCALE_BWD", "1");
            std::env::set_var("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK", "512");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"chunked_residual_scale_backward\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"tiled_residual_scale_backward\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"residual_scale_backward_rows_per_chunk\":512"),
            "{json}"
        );

        match prev_chunked {
            Some(value) => unsafe { std::env::set_var("PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD") },
        }
        match prev_tiled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_RESIDUAL_SCALE_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_RESIDUAL_SCALE_BWD") },
        }
        match prev_rows {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK") },
        }
    }

    #[test]
    fn record_audit_tracks_chunked_qkv_norm_resid_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_enabled = std::env::var("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD").ok();
        let prev_rows = std::env::var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK").ok();
        unsafe {
            std::env::set_var("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD", "1");
            std::env::set_var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK", "1024");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"chunked_qkv_norm_resid_backward\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"qkv_norm_resid_backward_rows_per_chunk\":1024"),
            "{json}"
        );

        match prev_enabled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD") },
        }
        match prev_rows {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK") },
        }
    }

    #[test]
    fn record_audit_tracks_split_qkv_norm_resid_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_enabled = std::env::var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD").ok();
        let prev_rows = std::env::var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK").ok();
        unsafe {
            std::env::set_var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", "1");
            std::env::set_var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK", "2048");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"split_qkv_norm_resid_backward\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"qkv_norm_resid_backward_rows_per_chunk\":2048"),
            "{json}"
        );

        match prev_enabled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD") },
        }
        match prev_rows {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK") },
        }
    }

    #[test]
    fn record_audit_tracks_vec2_mlp_activation_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_VEC2_MLP_ACT_BWD").ok();
        unsafe {
            std::env::set_var("PG_GPU_VEC2_MLP_ACT_BWD", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"vec2_mlp_activation_backward\":true"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_VEC2_MLP_ACT_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_VEC2_MLP_ACT_BWD") },
        }
    }

    #[test]
    fn record_audit_tracks_vec4_mlp_activation_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_VEC4_MLP_ACT_BWD").ok();
        unsafe {
            std::env::set_var("PG_GPU_VEC4_MLP_ACT_BWD", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"vec4_mlp_activation_backward\":true"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_VEC4_MLP_ACT_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_VEC4_MLP_ACT_BWD") },
        }
    }

    #[test]
    fn record_audit_tracks_fast_mlp_activation_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_FAST_MLP_ACT_BWD").ok();
        unsafe {
            std::env::set_var("PG_GPU_FAST_MLP_ACT_BWD", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"fast_mlp_activation_backward\":true"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_FAST_MLP_ACT_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_FAST_MLP_ACT_BWD") },
        }
    }

    #[test]
    fn record_audit_tracks_sparse_xsa_grouped_kv_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_SPARSE_XSA_GROUPED_KV_BWD").ok();
        unsafe {
            std::env::set_var("PG_GPU_SPARSE_XSA_GROUPED_KV_BWD", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"sparse_xsa_grouped_kv_backward\":true"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_SPARSE_XSA_GROUPED_KV_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_SPARSE_XSA_GROUPED_KV_BWD") },
        }
    }

    #[test]
    fn record_audit_tracks_compact_attn_gate_grad_input() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT").ok();
        unsafe {
            std::env::set_var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.sparse_attn_gate.enabled = true;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"compact_attn_gate_grad_input\":true"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT") },
        }
    }

    #[test]
    fn record_audit_tracks_split_compact_qkv_norm_resid_backward() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev_compact = std::env::var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT").ok();
        let prev_split = std::env::var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD").ok();
        unsafe {
            std::env::set_var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT", "1");
            std::env::set_var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.sparse_attn_gate.enabled = true;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"split_compact_qkv_norm_resid_backward\":true"),
            "{json}"
        );

        match prev_compact {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT") },
        }
        match prev_split {
            Some(value) => unsafe { std::env::set_var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", value) },
            None => unsafe { std::env::remove_var("PG_GPU_SPLIT_QKV_NORM_RESID_BWD") },
        }
    }

    #[test]
    fn record_audit_tracks_linear_backward_gemm_overlap() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let prev = std::env::var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS").ok();
        let prev_mlp_down = std::env::var("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS").ok();
        unsafe {
            std::env::set_var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS", "1");
            std::env::remove_var("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.attention_backend = AttentionBackend::CudnnSdpaBf16;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        let config = spec.model.to_model_config();
        let plan = step_batch_plan(&spec, RunMode::RecordShapedProxy, &config, 8).unwrap();
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"overlap_linear_backward_gemms\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"overlap_mlp_down_backward_gemms\":true"),
            "{json}"
        );

        unsafe {
            std::env::set_var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS", "0");
            std::env::set_var("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS", "1");
        }
        let json = record_path_audit_json(
            &spec,
            RunMode::RecordShapedProxy,
            &plan,
            8,
            false,
            false,
            &["test_gap"],
            true,
            false,
            "nccl_reduce_scatter_all_gather_parallel_muon_ns5",
        );
        assert!(
            json.contains("\"overlap_linear_backward_gemms\":false"),
            "{json}"
        );
        assert!(
            json.contains("\"overlap_mlp_down_backward_gemms\":true"),
            "{json}"
        );
        assert!(
            json.contains("\"overlap_mlp_up_backward_gemms\":false"),
            "{json}"
        );

        match prev {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS") },
        }
        match prev_mlp_down {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS") },
        }
    }

    #[test]
    fn record_audit_treats_full_vocab_output_tile_as_full_logits() {
        with_tiled_output_ce_env(Some("1"), Some("8192"), || {
            let mut spec = RunSpec::default();
            spec.train.backend = TrainBackend::CudaDistributed;
            spec.train.world_size = 8;
            spec.train.seq_len = 2048;
            spec.train.batch_tokens = 786_432;
            spec.train.distributed_optimizer_backend =
                DistributedOptimizerBackend::ShardedParallelMuon;
            spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
            spec.model.vocab_size = 8192;
            assert!(tiled_output_cross_entropy_enabled_for_audit(&spec));
            assert!(output_path_materializes_full_logits_for_audit(&spec));
            assert_eq!(
                output_loss_backend_for_audit(&spec),
                "full_vocab_tile_repeated_gemm"
            );
            assert!(!production_fused_output_projection_ce_enabled_for_audit(
                &spec
            ));
        });
    }

    #[test]
    fn record_audit_treats_sub_vocab_output_tile_as_no_full_logits() {
        with_tiled_output_ce_env(Some("1"), Some("1024"), || {
            let mut spec = RunSpec::default();
            spec.train.backend = TrainBackend::CudaDistributed;
            spec.train.world_size = 8;
            spec.train.seq_len = 2048;
            spec.train.batch_tokens = 786_432;
            spec.train.distributed_optimizer_backend =
                DistributedOptimizerBackend::ShardedParallelMuon;
            spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
            spec.model.vocab_size = 8192;
            assert!(tiled_output_cross_entropy_enabled_for_audit(&spec));
            assert!(!output_path_materializes_full_logits_for_audit(&spec));
            assert_eq!(output_loss_backend_for_audit(&spec), "tiled_repeated_gemm");
            assert!(!production_fused_output_projection_ce_enabled_for_audit(
                &spec
            ));
        });
    }

    #[test]
    fn record_audit_labels_bf16_full_logits_as_intermediate_path() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_backend = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok();
        let old_tiled = std::env::var("PG_GPU_TILED_OUTPUT_CE").ok();
        let old_bf16_logits = std::env::var("PG_GPU_BF16_LOGITS").ok();
        let old_bf16_bwd = std::env::var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM").ok();
        unsafe {
            std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", "chunked_bf16_cache");
            std::env::set_var("PG_GPU_TILED_OUTPUT_CE", "0");
            std::env::set_var("PG_GPU_BF16_LOGITS", "1");
            std::env::set_var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM", "0");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.vocab_size = 8192;

        assert!(!bf16_output_logits_enabled_for_audit(&spec));
        assert!(output_path_materializes_full_logits_for_audit(&spec));
        assert_eq!(
            output_loss_backend_for_audit(&spec),
            "full_logits_single_gemm"
        );
        assert!(!production_fused_output_projection_ce_enabled_for_audit(
            &spec
        ));

        match old_backend {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND") },
        }
        match old_tiled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match old_bf16_logits {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_LOGITS", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_LOGITS") },
        }
        match old_bf16_bwd {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM") },
        }
    }

    #[test]
    fn record_audit_labels_fused_exact_output_ce_as_production_opt_in() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_backend = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok();
        let old_legacy = std::env::var("PG_GPU_FUSED_EXACT_OUTPUT_CE").ok();
        let old_validated = std::env::var("PG_GPU_FUSED_EXACT_OUTPUT_CE_VALIDATED").ok();
        unsafe {
            std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", "fused_exact_wmma");
            std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE");
            std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE_VALIDATED");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.vocab_size = 8192;
        spec.model.output_ce_backend = OutputCeBackend::FusedExactWmma;

        assert!(fused_exact_output_ce_enabled_for_audit(&spec));
        assert!(!output_path_materializes_full_logits_for_audit(&spec));
        assert_eq!(output_loss_backend_for_audit(&spec), "fused_exact_wmma");
        assert_eq!(
            output_ce_implementation_for_audit(&spec),
            "fused_exact_cublas_tile_warp_rows_v1"
        );
        assert_eq!(output_projection_recompute_passes_for_audit(&spec), 1);
        assert!(production_fused_output_projection_ce_enabled_for_audit(
            &spec
        ));
        assert!(!production_fused_output_projection_ce_validated_for_audit(
            &spec
        ));
        assert!(
            frontier_record_gaps(&spec)
                .iter()
                .any(|gap| gap.contains("fused exact output projection"))
        );

        match old_backend {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND") },
        }
        match old_legacy {
            Some(value) => unsafe { std::env::set_var("PG_GPU_FUSED_EXACT_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE") },
        }
        match old_validated {
            Some(value) => unsafe {
                std::env::set_var("PG_GPU_FUSED_EXACT_OUTPUT_CE_VALIDATED", value)
            },
            None => unsafe { std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE_VALIDATED") },
        }
    }

    #[test]
    fn output_ce_legacy_env_overrides_spec_default_for_ab_runs() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_backend = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok();
        let old_fused = std::env::var("PG_GPU_FUSED_EXACT_OUTPUT_CE").ok();
        let old_tiled = std::env::var("PG_GPU_TILED_OUTPUT_CE").ok();
        let old_chunked = std::env::var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE").ok();
        unsafe {
            std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND");
            std::env::set_var("PG_GPU_FUSED_EXACT_OUTPUT_CE", "1");
            std::env::remove_var("PG_GPU_TILED_OUTPUT_CE");
            std::env::remove_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE");
        }

        let mut spec = RunSpec::default();
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.vocab_size = 8192;
        spec.model.output_ce_backend = OutputCeBackend::ChunkedBf16Cache;

        assert_eq!(output_ce_backend_for_audit(&spec), "fused_exact_wmma");
        assert!(fused_exact_output_ce_enabled_for_audit(&spec));
        assert!(!chunked_bf16_output_ce_cache_enabled_for_audit(&spec));

        unsafe {
            std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", "tiled_repeated_gemm");
        }
        assert_eq!(output_ce_backend_for_audit(&spec), "tiled_repeated_gemm");
        assert!(tiled_output_cross_entropy_enabled_for_audit(&spec));

        match old_backend {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_BACKEND", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_BACKEND") },
        }
        match old_fused {
            Some(value) => unsafe { std::env::set_var("PG_GPU_FUSED_EXACT_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_FUSED_EXACT_OUTPUT_CE") },
        }
        match old_tiled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match old_chunked {
            Some(value) => unsafe { std::env::set_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE") },
        }
    }

    #[test]
    fn frontier_audit_requires_smear_gate_document_boundary() {
        let mut spec = RunSpec::default();
        spec.model.smear_gate = true;
        spec.model.smear_gate_boundary_token_id = Some(1);

        let gaps = frontier_record_gaps(&spec);
        assert!(
            !gaps.iter().any(|gap| gap.contains("SmearGate must be BOS")),
            "boundary-aware SmearGate should not be reported as a leakage gap: {gaps:?}"
        );

        spec.model.smear_gate_boundary_token_id = None;
        let gaps = frontier_record_gaps(&spec);
        assert!(
            gaps.iter().any(|gap| gap.contains("SmearGate must be BOS")),
            "unmasked SmearGate must be a frontier record gap: {gaps:?}"
        );
    }

    #[test]
    fn recurrence_activation_uses_wallclock_fraction_when_budgeted() {
        let mut spec = RunSpec::default();
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.enable_at_frac = 0.35;

        assert!(!active_recurrence_for_train_step(
            &spec, 100, 24, 60.0, 600.0
        ));
        assert!(active_recurrence_for_train_step(
            &spec, 100, 24, 211.0, 600.0
        ));
    }

    #[test]
    fn recurrence_activation_falls_back_to_iteration_fraction_without_wallclock() {
        let mut spec = RunSpec::default();
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.enable_at_frac = 0.35;

        assert!(!active_recurrence_for_train_step(&spec, 33, 100, 0.0, 0.0));
        assert!(active_recurrence_for_train_step(&spec, 34, 100, 0.0, 0.0));
    }

    #[test]
    fn recurrence_activation_prefers_explicit_step_gate() {
        let mut spec = RunSpec::default();
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.enable_at_frac = 0.35;
        spec.model.recurrence.enable_at_step = Some(1748);

        assert!(!active_recurrence_for_train_step(
            &spec, 1746, 4994, 300.0, 600.0
        ));
        assert!(active_recurrence_for_train_step(
            &spec, 1747, 4994, 0.0, 600.0
        ));
    }

    #[test]
    fn frontier2135_quality_profile_rejects_speed_only_recurrence() {
        let mut spec = RunSpec::default();
        spec.runtime.record_profile = RecordProfile::Frontier2135Audit;
        spec.runtime.backward_chain_profile = BackwardChainProfile::Bf16DirectCompact;
        spec.runtime.require_device_batch = true;
        spec.runtime.token_ring_full_schedule = true;
        spec.runtime.recurrence_active_required = true;
        spec.runtime.recurrent_active_steps_min = 2000;
        spec.model.qk_gain_init = 5.0;
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 3;
        spec.model.recurrence.repeat_layers = 3;
        spec.model.recurrence.enable_at_frac = 0.35;
        spec.model.recurrence.enable_at_step = Some(1748);
        spec.model.parallel_residual.enabled = true;
        spec.model.parallel_residual.split_attention_mlp = true;
        spec.model.parallel_residual.start_layer = 8;
        spec.model.bigram.enabled = false;
        spec.model.eval_seq_len = 2560;
        spec.eval.ttt_seq_len = Some(2560);
        spec.eval.ttt_mask = TttMask::KOff;
        spec.eval.ttt_lora_targets = pg_model::spec::TttLoraTargetsSpec::upstream_pr2135();
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        spec.eval.lora_lr = Some(0.00008);
        spec.quant.gptq_calibration_batches = 32;
        spec.quant.matrix_bits = 6;
        spec.quant.mlp_bits = 6;
        spec.quant.embed_bits = 7;
        spec.model.asym_logit.enabled = true;
        spec.eval.ngram_tilt.enabled = true;
        spec.eval.ngram_tilt.token_boost = 1.0;
        spec.eval.ngram_tilt.within_boost = 0.0;
        spec.eval.ngram_tilt.word_boost = 0.0;
        spec.eval.ngram_tilt.agree_add_boost = 0.0;
        spec.model.caseops.enabled = true;
        spec.model.caseops.byte_sidecar = true;
        spec.eval.caseops_byte_sidecar_pattern = Some("/tmp/bytes_*.bin".into());
        spec.model.sparse_attn_gate.enabled = true;
        spec.model.sparse_attn_gate.width = 12;
        spec.model.smear_gate = true;
        spec.model.smear_gate_boundary_token_id = Some(1);
        spec.quant.lqer.enabled = true;

        let gaps = leaderboard_algorithm_gaps(&spec);
        assert!(
            gaps.is_empty(),
            "fully specified frontier_2135 audit profile should have no algorithm gaps: {gaps:?}"
        );

        spec.runtime.recurrent_backward_profile = RecurrentBackwardProfile::Pass1StraightThrough;
        let gaps = leaderboard_algorithm_gaps(&spec);
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("straight-through recurrent backward")),
            "pass1 straight-through recurrence must be marked as a speed probe until BPB validates it: {gaps:?}"
        );

        spec.runtime.recurrent_backward_profile = RecurrentBackwardProfile::Full;
        spec.runtime.skip_recurrent_bank_grads = true;
        let gaps = leaderboard_algorithm_gaps(&spec);
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("skip recurrent bank gradients")),
            "skipping recurrent bank gradients must stay marked as a speed probe until BPB validates it: {gaps:?}"
        );

        spec.runtime.skip_recurrent_bank_grads = false;
        spec.runtime.skip_recurrent_pass1_bank_grads = true;
        let gaps = leaderboard_algorithm_gaps(&spec);
        assert!(
            gaps.iter()
                .any(|gap| gap.contains("skip recurrent pass-1 bank gradients")),
            "skipping pass-1 recurrent bank gradients must stay marked as a speed probe until BPB validates it: {gaps:?}"
        );

        spec.runtime.skip_recurrent_pass1_bank_grads = false;
        spec.model.recurrence.enable_at_frac = 0.999;
        let gaps = leaderboard_algorithm_gaps(&spec);
        assert!(
            gaps.iter().any(|gap| gap.contains("late recurrence")),
            "late-recurrence speed probes must not pass as frontier_2135 quality targets: {gaps:?}"
        );
    }

    #[test]
    fn record_batch_plan_rejects_non_divisible_batch() {
        let mut spec = RunSpec::default();
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_431;
        let config = spec.model.to_model_config();
        let err = step_batch_plan(&spec, RunMode::Record, &config, 8).unwrap_err();
        assert!(err.to_string().contains("must be divisible"));
    }

    #[test]
    fn directory_regular_file_bytes_counts_nested_record_package() {
        let root = std::env::temp_dir().join(format!(
            "pg_train_byte_count_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let nested = root.join("src");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("train.rs"), [0u8; 7]).unwrap();
        std::fs::write(nested.join("kernel.cu"), [0u8; 11]).unwrap();
        let bytes = directory_regular_file_bytes(&root).unwrap();
        let _ = std::fs::remove_dir_all(&root);
        assert_eq!(bytes, 18);
    }

    #[test]
    fn executable_variants_include_recurrence_and_parallel_residual() {
        let mut spec = RunSpec::default();
        spec.model.recurrence.enabled = true;
        spec.model.recurrence.start_layer = 2;
        spec.model.recurrence.repeat_layers = 2;
        spec.model.parallel_residual.enabled = true;
        spec.model.parallel_residual.split_attention_mlp = true;
        assert!(validate_executable_variant(&spec, RunMode::Proxy).is_ok());
    }
}
