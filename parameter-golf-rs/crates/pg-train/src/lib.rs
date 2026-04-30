use std::time::Instant;

use pg_model::backward::GradBuffers;
use pg_model::{
    AttentionBackend, DistributedOptimizerBackend, EvalAdaptationBackend, ExecutionPlan,
    ForwardBuffer, GptModel, ModelComputePrecision, RunMode, RunSpec, TrainBackend,
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
    cuda_non_bank_update_ms: f64,
    post_train_sync_ms: f64,
    artifact_export_ms: f64,
    eval_ms: f64,
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
    input_ids: pg_core::GpuTensor,
    targets: pg_core::GpuTensor,
    host_input_ids: Vec<u32>,
    host_targets: Vec<u32>,
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
            input_ids: pg_core::GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens],
                pg_core::DType::U32,
            )?,
            targets: pg_core::GpuTensor::zeros_gpu(stream.clone(), &[tokens], pg_core::DType::U32)?,
            host_input_ids: Vec::with_capacity(tokens),
            host_targets: Vec::with_capacity(tokens),
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
struct CudaDistributedRuntime {
    replicas: Vec<CudaSingleFastRuntime>,
    comms: Vec<pg_core::nccl::NcclComm>,
    parallel_muon: Option<ShardedParallelMuonRuntime>,
    all_grad_sync: Vec<AllGradSyncBuffers>,
    non_bank_sync: Vec<NonBankSyncBuffers>,
    distributed_sync: bool,
}

#[cfg(feature = "cuda")]
struct ShardedParallelMuonRuntime {
    replicas: Vec<ShardedParallelMuonReplica>,
}

#[cfg(feature = "cuda")]
struct ShardedParallelMuonReplica {
    banks: Vec<ShardedBankBuffers>,
    muon: GpuMuon,
}

#[cfg(feature = "cuda")]
struct ShardedBankBuffers {
    padded_grad: pg_core::GpuTensor,
    padded_grad_bf16: pg_core::GpuTensor,
    shard_grad: pg_core::GpuTensor,
    shard_grad_bf16: pg_core::GpuTensor,
    shard_param: pg_core::GpuTensor,
    padded_param: pg_core::GpuTensor,
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
                        padded_param: pg_core::GpuTensor::zeros_gpu(
                            stream.clone(),
                            &padded_shape,
                            pg_core::DType::F32,
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
    runtime
        .replicas
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

impl VariantRunner {
    pub fn new(run_spec: RunSpec) -> PgResult<Self> {
        let plan = ExecutionPlan::from_run_spec(&run_spec)?;
        Ok(Self { run_spec, plan })
    }

    pub fn run(&self, mode: RunMode) -> PgResult<VariantResult> {
        let model_config = self.run_spec.model.to_model_config();
        let train_config = self.run_spec.train.to_train_config();
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
            RunMode::RecordShapedProxy => train_config.total_iterations.min(8),
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
            let synthetic_distributed_batches = || {
                (0..world_size)
                    .map(|rank_idx| {
                        let local_tokens =
                            batch_plan.microbatch_tokens * batch_plan.local_microbatches_per_step;
                        let mut x = Vec::with_capacity(local_tokens);
                        let mut y = Vec::with_capacity(local_tokens);
                        for micro_idx in 0..batch_plan.local_microbatches_per_step {
                            let offset = step * batch_plan.global_batch_tokens
                                + micro_idx * batch_plan.microbatch_tokens * world_size
                                + rank_idx * batch_plan.microbatch_tokens;
                            x.extend(
                                (0..batch_plan.microbatch_tokens)
                                    .map(|i| ((offset + i) % model_config.vocab_size) as u32),
                            );
                            y.extend(
                                (1..=batch_plan.microbatch_tokens)
                                    .map(|i| ((offset + i) % model_config.vocab_size) as u32),
                            );
                        }
                        vec![(x, y)]
                    })
                    .collect::<Vec<_>>()
            };
            let distributed_batches: Option<Vec<Vec<(Vec<u32>, Vec<u32>)>>> =
                if self.run_spec.train.backend == TrainBackend::CudaDistributed {
                    if is_record_shaped_mode(mode) {
                        #[cfg(feature = "cuda")]
                        if let (Some(loaders), Some(runtime)) = (
                            distributed_data_loaders.as_mut(),
                            cuda_distributed_runtime.as_mut(),
                        ) {
                            for (loader, replica) in
                                loaders.iter_mut().zip(runtime.replicas.iter_mut())
                            {
                                loader.next_batch_u32_into(
                                    batch_plan.global_batch_tokens,
                                    &mut replica.host_input_ids,
                                    &mut replica.host_targets,
                                )?;
                            }
                            distributed_batches_preloaded = true;
                            None
                        } else {
                            Some(synthetic_distributed_batches())
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            Some(synthetic_distributed_batches())
                        }
                    } else if let Some(loaders) = distributed_data_loaders.as_mut() {
                        Some(
                            loaders
                                .iter_mut()
                                .map(|loader| {
                                    let (x, y) = loader.next_batch(
                                        batch_plan.global_batch_tokens,
                                        batch_plan.microbatch_tokens,
                                    )?;
                                    Ok::<_, pg_core::PgError>(vec![(
                                        x.into_iter().map(|v| v as u32).collect(),
                                        y.into_iter().map(|v| v as u32).collect(),
                                    )])
                                })
                                .collect::<PgResult<Vec<_>>>()?,
                        )
                    } else {
                        Some(synthetic_distributed_batches())
                    }
                } else {
                    None
                };
            let local_batches: Vec<(Vec<u32>, Vec<u32>)> = if distributed_batches_preloaded {
                Vec::new()
            } else if let Some(batches) = distributed_batches.as_ref() {
                batches[0].clone()
            } else if let Some(loader) = data_loader.as_mut() {
                (0..batch_plan.local_microbatches_per_step)
                    .map(|_| {
                        let global_tokens = batch_plan.microbatch_tokens * world_size;
                        let (x, y) =
                            loader.next_batch(global_tokens, batch_plan.microbatch_tokens)?;
                        Ok::<_, pg_core::PgError>((
                            x.into_iter().map(|v| v as u32).collect(),
                            y.into_iter().map(|v| v as u32).collect(),
                        ))
                    })
                    .collect::<PgResult<Vec<_>>>()?
            } else {
                (0..batch_plan.local_microbatches_per_step)
                    .map(|micro_idx| {
                        let offset = step * batch_plan.global_batch_tokens
                            + micro_idx * batch_plan.microbatch_tokens * world_size
                            + rank * batch_plan.microbatch_tokens;
                        (
                            (0..batch_plan.microbatch_tokens)
                                .map(|i| ((offset + i) % model_config.vocab_size) as u32)
                                .collect(),
                            (1..=batch_plan.microbatch_tokens)
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
                        batch_plan.microbatch_tokens,
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
                grads.clip_grad_norm(train_config.grad_clip_norm);

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
                timing_measured_wallclock_ms += step_wall_t0.elapsed().as_secs_f64() * 1000.0;
                timing_steps += 1;
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
            let max_ms = record_max_ms_per_step_for_submission();
            if max_ms > 0.0 && timing_measured_ms_per_step > max_ms {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "record run is too slow for a leaderboard submission: measured_ms_per_step={timing_measured_ms_per_step:.3} max_ms_per_step={max_ms:.3}. Set PG_RECORD_MAX_MS_PER_STEP=0 only for non-submission debugging."
                )));
            }
        }

        let needs_cpu_model_after_training =
            !matches!(mode, RunMode::Smoke | RunMode::RecordShapedProxy);
        let sync_t0 = Instant::now();
        if needs_cpu_model_after_training {
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
            RunMode::Smoke | RunMode::RecordShapedProxy => None,
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
        let submission_code_bytes = artifact_bytes.map(|_| current_executable_bytes());
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
) -> PgResult<()> {
    let zero = |tensor: &mut pg_core::GpuTensor| tensor.zero_bytes();

    zero(&mut grads.tok_emb)?;
    zero(&mut grads.bigram_embed)?;
    zero(&mut grads.bigram_proj)?;
    zero(&mut grads.bigram_scale)?;
    zero(&mut grads.smear_gate)?;
    zero(&mut grads.skip_weights)?;
    zero(&mut grads.qo_bank)?;
    zero(&mut grads.kv_bank)?;
    zero(&mut grads.mlp_up_bank)?;
    zero(&mut grads.mlp_down_bank)?;
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
    for tensor in &mut grads.block_attn_gate_weight {
        zero(tensor)?;
    }
    for tensor in &mut grads.block_attn_gate_bias {
        zero(tensor)?;
    }
    for tensor in &mut grads.block_sparse_attn_gate_weight {
        zero(tensor)?;
    }
    zero(&mut grads.ve_embed)?;
    zero(&mut grads.ve_proj)?;
    zero(&mut grads.ve_scale)?;
    zero(&mut grads.ve_layer_scales)?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_single_fast_step(
    runtime: &mut CudaSingleFastRuntime,
    _model: &mut GptModel,
    _plan: &ExecutionPlan,
    microbatches: &[(Vec<u32>, Vec<u32>)],
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<f32> {
    zero_gpu_grads(&runtime.gpu_model.kernels, &mut runtime.gpu_grads)?;
    let seq_len = microbatches
        .first()
        .map(|(input_ids, _)| input_ids.len())
        .unwrap_or(1)
        .max(1);
    flatten_microbatches_into(
        microbatches,
        &mut runtime.host_input_ids,
        &mut runtime.host_targets,
    );
    let loss_sum = cuda_fast_accumulate_runtime_grads(runtime, true, seq_len, None)?;
    cuda_fast_apply_updates(runtime, train_config, step, lr_scale)?;
    Ok(loss_sum)
}

#[cfg(feature = "cuda")]
fn cuda_fast_accumulate_runtime_grads(
    runtime: &mut CudaSingleFastRuntime,
    compute_loss: bool,
    runtime_seq_len: usize,
    mut timing: Option<&mut RunTiming>,
) -> PgResult<f32> {
    let h2d_t0 = Instant::now();
    runtime
        .input_ids
        .copy_from_host_bytes(bytemuck::cast_slice(&runtime.host_input_ids))?;
    runtime
        .targets
        .copy_from_host_bytes(bytemuck::cast_slice(&runtime.host_targets))?;
    if let Some(timing) = timing.as_deref_mut() {
        timing.cuda_h2d_ms += h2d_t0.elapsed().as_secs_f64() * 1000.0;
    }

    if compute_loss {
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
    }

    stream
        .begin_capture(
            cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        )
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda backward graph capture begin failed: {e:?}"))
        })?;
    let capture_result = runtime.gpu_model.backward_with_state_seq_len_no_loss(
        &runtime.input_ids,
        &runtime.targets,
        &mut runtime.gpu_buf,
        &mut runtime.backward_state,
        &mut runtime.gpu_grads,
        runtime_seq_len,
    );
    let graph = stream
        .end_capture(
            cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        )
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda backward graph capture end failed: {e:?}"))
        })?;
    capture_result?;
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
fn collect_gpu_non_bank_grad_refs(
    grads: &pg_model::gpu::GpuGradBuffers,
) -> Vec<&pg_core::GpuTensor> {
    let mut grad_refs: Vec<&pg_core::GpuTensor> = vec![
        &grads.tok_emb,
        &grads.bigram_embed,
        &grads.bigram_proj,
        &grads.bigram_scale,
        &grads.smear_gate,
        &grads.skip_weights,
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
    cuda_fast_apply_updates_inner(runtime, train_config, step, lr_scale, true, true)
}

#[cfg(feature = "cuda")]
fn cuda_fast_apply_non_bank_updates_unclipped(
    runtime: &mut CudaSingleFastRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
) -> PgResult<()> {
    cuda_fast_apply_updates_inner(runtime, train_config, step, lr_scale, false, false)
}

#[cfg(feature = "cuda")]
fn cuda_fast_apply_updates_inner(
    runtime: &mut CudaSingleFastRuntime,
    train_config: &pg_model::TrainConfig,
    step: usize,
    lr_scale: f32,
    update_banks: bool,
    clip_grads: bool,
) -> PgResult<()> {
    let (gpu_optimizer, grad_norm_scratch, gpu_model, gpu_grads) = (
        &mut runtime.gpu_optimizer,
        &mut runtime.grad_norm_scratch,
        &runtime.gpu_model,
        &runtime.gpu_grads,
    );
    if clip_grads {
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

    runtime.gpu_optimizer.adamw_step(
        &runtime.gpu_model.kernels,
        &runtime.gpu_model.weights.tok_emb,
        &runtime.gpu_grads.tok_emb,
        &mut runtime.state_tok_emb,
        embed_hyper,
    )?;
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
    runtime.gpu_model.refresh_bf16_shadows()?;

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
            let end = offset + tensor.numel();
            let dst = packed.slice_range(offset, end)?;
            copy_gpu_tensor(kernels, tensor, &dst)?;
            offset = end;
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
            let end = offset + tensor.numel();
            let src = packed.slice_range(offset, end)?;
            copy_gpu_tensor(kernels, &src, tensor)?;
            offset = end;
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
            let end = offset + tensor.numel();
            let dst = packed.slice_range(offset, end)?;
            copy_gpu_tensor(kernels, tensor, &dst)?;
            offset = end;
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
            let end = offset + tensor.numel();
            let src = packed.slice_range(offset, end)?;
            copy_gpu_tensor(kernels, &src, tensor)?;
            offset = end;
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

    cudarc::nccl::group_start()
        .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
    for (buffers, comm) in runtime.all_grad_sync.iter_mut().zip(runtime.comms.iter()) {
        comm.all_reduce_sum_tensor_f32_in_place(&mut buffers.packed_grad)?;
    }
    cudarc::nccl::group_end()
        .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;

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
fn scale_and_unpack_non_bank_sync_buffers(
    runtime: &mut CudaDistributedRuntime,
    scale: f32,
) -> PgResult<()> {
    for rank in 0..runtime.replicas.len() {
        let replica = &mut runtime.replicas[rank];
        let packed = &runtime.non_bank_sync[rank].packed_grad;
        scale_gpu_tensor(&replica.gpu_model.kernels, packed, scale)?;
        unpack_non_bank_gpu_grads(&replica.gpu_model.kernels, packed, &mut replica.gpu_grads)?;
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
                let param = bank_param(replica, bank_idx)?;
                let grad = bank_grad(replica, bank_idx)?;
                let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
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

                let shard_start = rank * buffers.chunk_batch;
                let shard_end = (shard_start + buffers.chunk_batch).min(buffers.real_batch);
                if shard_start < shard_end {
                    let real_count = shard_end - shard_start;
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

        if sharded_grouped_grad_collectives_enabled_for_audit() {
            cudarc::nccl::group_start()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
            for (buffers, comm) in runtime.non_bank_sync.iter_mut().zip(runtime.comms.iter()) {
                comm.all_reduce_sum_tensor_f32_in_place(&mut buffers.packed_grad)?;
            }
            for bank_idx in 0..bank_count {
                for rank in 0..world_size {
                    let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                    if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                        runtime.comms[rank].reduce_scatter_sum_tensor_bf16(
                            &buffers.padded_grad_bf16,
                            &mut buffers.shard_grad_bf16,
                        )?;
                    } else {
                        runtime.comms[rank].reduce_scatter_sum_tensor_f32(
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
            for (buffers, comm) in runtime.non_bank_sync.iter_mut().zip(runtime.comms.iter()) {
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
                        runtime.comms[rank].reduce_scatter_sum_tensor_bf16(
                            &buffers.padded_grad_bf16,
                            &mut buffers.shard_grad_bf16,
                        )?;
                    } else {
                        runtime.comms[rank].reduce_scatter_sum_tensor_f32(
                            &buffers.padded_grad,
                            &mut buffers.shard_grad,
                        )?;
                    }
                }
                cudarc::nccl::group_end()
                    .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;
            }
        }
    }

    scale_and_unpack_non_bank_sync_buffers(runtime, inv_total)?;
    {
        let parallel_muon = runtime
            .parallel_muon
            .as_mut()
            .expect("validated sharded Parallel Muon runtime");
        for bank_idx in 0..bank_count {
            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                if sharded_bank_grad_bf16_wire_enabled_for_audit() {
                    replica.gpu_model.kernels.bf16_to_f32(
                        pg_kernels::gpu_kernels::CudaPtr(
                            buffers
                                .shard_grad_bf16
                                .cu_ptr(replica.gpu_model.kernels.stream())?,
                        ),
                        pg_kernels::gpu_kernels::CudaPtr(
                            buffers
                                .shard_grad
                                .cu_ptr(replica.gpu_model.kernels.stream())?,
                        ),
                        buffers.shard_grad.numel() as u32,
                    )?;
                }
                scale_gpu_tensor(&replica.gpu_model.kernels, &buffers.shard_grad, inv_total)?;
            }
        }
    }
    // Exact global-norm clipping for sharded Parallel Muon:
    // - non-bank grads are replicated after all-reduce, so each rank contributes
    //   1/world_size of their squared norm before the scalar all-reduce.
    // - bank grads are reduce-scattered, so each rank contributes only its shard.
    for rank in 0..world_size {
        let replica = &mut runtime.replicas[rank];
        let kernels = &replica.gpu_model.kernels;
        scale_gpu_tensor(kernels, &replica.grad_norm_scratch, 0.0)?;
        let scratch =
            pg_kernels::gpu_kernels::CudaPtr(replica.grad_norm_scratch.cu_ptr(kernels.stream())?);
        let non_bank_alpha = 1.0f32 / world_size as f32;
        for grad in collect_gpu_non_bank_grad_refs(&replica.gpu_grads) {
            kernels.dot_accumulate(
                pg_kernels::gpu_kernels::CudaPtr(grad.cu_ptr(kernels.stream())?),
                pg_kernels::gpu_kernels::CudaPtr(grad.cu_ptr(kernels.stream())?),
                scratch,
                non_bank_alpha,
                grad.numel() as u32,
            )?;
        }
        let parallel_muon = runtime
            .parallel_muon
            .as_ref()
            .expect("validated sharded Parallel Muon runtime");
        for buffers in &parallel_muon.replicas[rank].banks {
            let active_batch = sharded_bank_real_batch(buffers, rank);
            if active_batch == 0 {
                continue;
            }
            let shard_grad = buffers.shard_grad.slice_range(0, active_batch)?;
            kernels.dot_accumulate(
                pg_kernels::gpu_kernels::CudaPtr(shard_grad.cu_ptr(kernels.stream())?),
                pg_kernels::gpu_kernels::CudaPtr(shard_grad.cu_ptr(kernels.stream())?),
                scratch,
                1.0,
                shard_grad.numel() as u32,
            )?;
        }
    }
    cudarc::nccl::group_start()
        .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
    for rank in 0..world_size {
        runtime.comms[rank]
            .all_reduce_sum_tensor_f32_in_place(&mut runtime.replicas[rank].grad_norm_scratch)?;
    }
    cudarc::nccl::group_end()
        .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;

    for rank in 0..world_size {
        let replica = &runtime.replicas[rank];
        let kernels = &replica.gpu_model.kernels;
        let scratch =
            pg_kernels::gpu_kernels::CudaPtr(replica.grad_norm_scratch.cu_ptr(kernels.stream())?);
        for grad in collect_gpu_non_bank_grad_refs(&replica.gpu_grads) {
            kernels.clip_by_global_norm(
                pg_kernels::gpu_kernels::CudaPtr(grad.cu_ptr(kernels.stream())?),
                scratch,
                train_config.grad_clip_norm,
                grad.numel() as u32,
            )?;
        }
        let parallel_muon = runtime
            .parallel_muon
            .as_ref()
            .expect("validated sharded Parallel Muon runtime");
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

    {
        let parallel_muon = runtime
            .parallel_muon
            .as_mut()
            .expect("validated sharded Parallel Muon runtime");
        for bank_idx in 0..bank_count {
            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let sharded_replica = &mut parallel_muon.replicas[rank];
                let buffers = &mut sharded_replica.banks[bank_idx];
                let active_batch = sharded_bank_real_batch(buffers, rank);
                if active_batch == 0 {
                    continue;
                }
                let shard_param = buffers.shard_param.slice_range(0, active_batch)?;
                let shard_grad = buffers.shard_grad.slice_range(0, active_batch)?;
                sharded_replica.muon.lr = train_config.matrix_lr * lr_scale;
                sharded_replica.muon.momentum = train_config.muon_momentum_at(step);
                sharded_replica.muon.weight_decay = train_config.muon_wd;
                sharded_replica.muon.step_bank(
                    &replica.gpu_model.kernels,
                    bank_idx,
                    &shard_param,
                    &shard_grad,
                )?;
            }
            cudarc::nccl::group_start()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_start failed: {e:?}")))?;
            for rank in 0..world_size {
                let buffers = &mut parallel_muon.replicas[rank].banks[bank_idx];
                runtime.comms[rank]
                    .all_gather_tensor_f32(&buffers.shard_param, &mut buffers.padded_param)?;
            }
            cudarc::nccl::group_end()
                .map_err(|e| pg_core::PgError::Nccl(format!("group_end failed: {e:?}")))?;

            for rank in 0..world_size {
                let replica = &runtime.replicas[rank];
                let kernels = &replica.gpu_model.kernels;
                let param = bank_param(replica, bank_idx)?;
                let buffers = &parallel_muon.replicas[rank].banks[bank_idx];
                let gathered_real = buffers.padded_param.slice_range(0, buffers.real_batch)?;
                kernels.copy_fwd(
                    pg_kernels::gpu_kernels::CudaPtr(gathered_real.cu_ptr(kernels.stream())?),
                    pg_kernels::gpu_kernels::CudaPtr(param.cu_ptr(kernels.stream())?),
                    param.numel() as u32,
                )?;
            }
        }
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
    for replica in runtime.replicas.iter_mut() {
        cuda_fast_apply_non_bank_updates_unclipped(replica, train_config, step, lr_scale)?;
        replica.gpu_model.refresh_bf16_shadows()?;
    }
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
    let replica_results = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(runtime.replicas.len());
        for (rank_idx, replica) in runtime.replicas.iter_mut().enumerate() {
            let rank_batches = batches.map(|all_batches| &all_batches[rank_idx]);
            handles.push(
                scope.spawn(move || -> PgResult<ReplicaBackwardLaunchResult> {
                    let stream = replica.gpu_model.gemm.stream().clone();
                    let backward_host_t0 = Instant::now();
                    let backward_start_event = if event_timing {
                        Some(
                            stream
                                .record_event(Some(
                                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                                ))
                                .map_err(|e| {
                                    pg_core::PgError::InvalidOp(format!(
                                        "cuda backward event record failed: {e:?}"
                                    ))
                                })?,
                        )
                    } else {
                        None
                    };

                    let zero_host_t0 = Instant::now();
                    let zero_start_event = if event_timing {
                        Some(
                            stream
                                .record_event(Some(
                                    cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                                ))
                                .map_err(|e| {
                                    pg_core::PgError::InvalidOp(format!(
                                        "cuda zero-grads event record failed: {e:?}"
                                    ))
                                })?,
                        )
                    } else {
                        None
                    };
                    zero_gpu_grads(&replica.gpu_model.kernels, &mut replica.gpu_grads)?;

                    let mut result = ReplicaBackwardLaunchResult::default();
                    if let Some(start) = zero_start_event {
                        let end = stream
                            .record_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                            .map_err(|e| {
                                pg_core::PgError::InvalidOp(format!(
                                    "cuda zero-grads event record failed: {e:?}"
                                ))
                            })?;
                        result.cuda_zero_grads_ms = start.elapsed_ms(&end).map_err(|e| {
                            pg_core::PgError::InvalidOp(format!(
                                "cuda zero-grads event elapsed failed: {e:?}"
                            ))
                        })? as f64;
                    } else {
                        result.cuda_zero_grads_ms = zero_host_t0.elapsed().as_secs_f64() * 1000.0;
                    }

                    let compute_loss = compute_step_loss && rank_idx == 0;
                    if let Some(rank_batches) = rank_batches {
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
                    )?;
                    result.cuda_h2d_ms = local_timing.cuda_h2d_ms;
                    result.loss_count = usize::from(compute_loss);

                    if let Some(start) = backward_start_event {
                        let end = stream
                            .record_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                            .map_err(|e| {
                                pg_core::PgError::InvalidOp(format!(
                                    "cuda backward event record failed: {e:?}"
                                ))
                            })?;
                        result.cuda_backward_ms = start.elapsed_ms(&end).map_err(|e| {
                            pg_core::PgError::InvalidOp(format!(
                                "cuda backward event elapsed failed: {e:?}"
                            ))
                        })? as f64;
                    } else {
                        result.cuda_backward_ms = backward_host_t0.elapsed().as_secs_f64() * 1000.0;
                    }
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
    })?;
    let mut max_backward_ms = 0.0f64;
    for result in replica_results {
        total_loss += result.loss;
        loss_count += result.loss_count;
        timing.cuda_zero_grads_ms += result.cuda_zero_grads_ms;
        timing.cuda_h2d_ms += result.cuda_h2d_ms;
        max_backward_ms = max_backward_ms.max(result.cuda_backward_ms);
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
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
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
    matches!(
        run_spec.model.compute_precision,
        ModelComputePrecision::F32Tf32
    )
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
    let record_max_ms = record_max_ms_per_step_for_submission();
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
        "model_precision_target",
        model_precision_target_label(run_spec),
    ));
    fields.push(format!(
        "\"model_precision_target_met\":{}",
        model_precision_target_met(run_spec)
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
    fields.push(format!(
        "\"backward_nccl_bucket_overlap\":{}",
        backward_nccl_bucket_overlap_enabled_for_audit()
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
        "\"chunked_residual_mix_backward\":{}",
        chunked_residual_mix_backward_enabled_for_audit()
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
        "\"materializes_full_logits\":{}",
        output_path_materializes_full_logits_for_audit(run_spec)
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
        "\"gpu_lora_prefix_doc_ttt_schedule\":{}",
        run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
            && run_spec.eval.phased_ttt_prefix_docs > 0
            && run_spec.model.smear_gate_boundary_token_id.is_some()
    ));
    fields.push(format!(
        "\"gpu_lora_prefix_docs\":{}",
        run_spec.eval.phased_ttt_prefix_docs
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
    fields.push(format!(
        "\"record_host_batch_u32_preload\":{}",
        is_record_shaped_mode(mode) && run_spec.train.backend == TrainBackend::CudaDistributed
    ));
    fields.push("\"gpu_resident_data_sampler\":false".to_string());
    fields.push("\"host_input_copy_per_step\":true".to_string());
    fields.push(format!(
        "\"tokenizer_vocab_configured\":{}",
        run_spec.eval.tokenizer_vocab_path.is_some()
    ));
    fields.push("\"artifact_model_bytes\":null".to_string());
    fields.push("\"artifact_code_bytes\":null".to_string());
    fields.push("\"artifact_total_bytes\":null".to_string());
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

fn experimental_fused_qkv_projection_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_FUSED_QKV_PROJ")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
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

fn bf16_qkv_dx_output_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_backward_projection_gemm_enabled(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_QKV_DX_OUTPUT")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
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
        && !tiled_output_cross_entropy_enabled_for_audit(run_spec)
        && matches!(
            std::env::var("PG_GPU_BF16_LOGITS")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn output_ce_tile_vocab_for_audit() -> usize {
    std::env::var("PG_GPU_OUTPUT_CE_TILE_VOCAB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(512)
}

fn tiled_output_cross_entropy_enabled_for_audit(run_spec: &RunSpec) -> bool {
    run_spec.model.compute_precision == ModelComputePrecision::Bf16TensorCore
        && bf16_output_projection_gemm_enabled(run_spec)
        && bf16_output_backward_gemm_enabled(run_spec)
        && run_spec.model.vocab_size % output_ce_tile_vocab_for_audit() == 0
        && matches!(
            std::env::var("PG_GPU_TILED_OUTPUT_CE")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

fn output_path_materializes_full_logits_for_audit(run_spec: &RunSpec) -> bool {
    if !tiled_output_cross_entropy_enabled_for_audit(run_spec) {
        return true;
    }
    // A full-vocab "tile" avoids the persistent activation logits field, but
    // still allocates and writes a [tokens, vocab] scratch tile. For record
    // readiness, only sub-vocab tiling counts as removing full logits.
    output_ce_tile_vocab_for_audit() >= run_spec.model.vocab_size
}

fn output_loss_backend_for_audit(run_spec: &RunSpec) -> &'static str {
    if !tiled_output_cross_entropy_enabled_for_audit(run_spec) {
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

fn production_fused_output_projection_ce_enabled_for_audit(_run_spec: &RunSpec) -> bool {
    // The current "fused CE" path fuses softcapped CE loss/backward only after
    // logits already exist. It does not fuse the tied output projection with CE,
    // so record mode must not count it as the production no-logits output path.
    false
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

fn backward_nccl_bucket_overlap_enabled_for_audit() -> bool {
    false
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
        if gpu_saved_layer_activations_mode_for_audit().to_ascii_lowercase() != "all"
            || !direct_saved_layer_activations_enabled_for_audit()
            || !skip_f32_attention_saved_activations_enabled_for_audit(run_spec)
        {
            gaps.push("record-grade cuDNN SDPA backward requires direct all-layer BF16 saved activations; otherwise backward can fall back to F32 recompute or saved F32 attention state");
        }
    }
    match run_spec.model.compute_precision {
        ModelComputePrecision::F32Tf32 => {
            gaps.push("frontier record target requires compute_precision=bf16_tensor_core; current target is f32_tf32");
        }
        ModelComputePrecision::Bf16TensorCore => {
            gaps.push("compute_precision=bf16_tensor_core is requested, but Rust still stores activations/gradients in f32; projection/output GEMMs use BF16 shadows, but a full BF16/FP16 activation graph is not implemented yet");
        }
    }
    if !production_fused_output_projection_ce_enabled_for_audit(run_spec) {
        if !tiled_output_cross_entropy_enabled_for_audit(run_spec) {
            gaps.push("current GPU output path still materializes full [batch_tokens, vocab] logits; the measured tiled-CE workaround is not sufficient, so the remaining record cut is a real fused output projection + softcapped CE/backward kernel");
        } else if output_path_materializes_full_logits_for_audit(run_spec) {
            gaps.push("PG_GPU_TILED_OUTPUT_CE is enabled, but output_ce_tile_vocab is >= vocab_size, so the output path still materializes a full [batch_tokens, vocab] scratch tile");
        } else {
            gaps.push("PG_GPU_TILED_OUTPUT_CE avoids persistent logits but repeats output GEMMs and is not the production fused output projection + softcapped CE/backward kernel required by the final record engine");
        }
    }
    if gpu_host_scalar_updates_enabled_for_audit() {
        gaps.push("PG_GPU_HOST_SCALAR_UPDATES=1 enables legacy per-step host scalar mirror synchronization; record mode should keep trainable scalar params device-resident and sync them only for export");
    }
    if !backward_nccl_bucket_overlap_enabled_for_audit() {
        gaps.push("distributed bank communication is still launched after full backward; final record runtime still needs bucketed reduce-scatter overlap with backward");
    }
    if run_spec.model.smear_gate && run_spec.model.smear_gate_boundary_token_id.is_none() {
        gaps.push("SmearGate must be BOS/document-boundary masked before record-shaped frontier runs; unmasked previous-token mixing can leak across packed documents");
    }
    gaps.push("record data path is still host-sampled with per-step H2D input/target copies; final CUDA-graphable record engine needs a GPU-resident sampler");
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

fn allow_frontier_record_gaps_for_development() -> bool {
    std::env::var("PG_ALLOW_FRONTIER_RECORD_GAPS")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn record_max_ms_per_step_for_submission() -> f64 {
    std::env::var("PG_RECORD_MAX_MS_PER_STEP")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(150.0)
}

#[cfg(feature = "cuda")]
fn eval_gpu_lora_phased_from_train(
    model: &GptModel,
    plan: &ExecutionPlan,
    tokens: &[u32],
    token_bytes: &[f32],
) -> PgResult<(f64, f64)> {
    let cfg = pg_eval::gpu_lora_ttt::GpuLoraPhasedTttConfig::from_plan(plan, tokens.len());
    pg_eval::gpu_lora_ttt::eval_gpu_lora_phased_ttt(model, plan, tokens, token_bytes, &cfg)
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
    if is_record_shaped_mode(mode) && run_spec.train.train_data_pattern.is_none() {
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
    let sequence_tokens = run_spec
        .train
        .seq_len
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

fn current_executable_bytes() -> usize {
    if let Ok(value) = std::env::var("PG_SUBMISSION_CODE_BYTES") {
        if let Ok(bytes) = value.parse::<usize>() {
            return bytes;
        }
    }
    if let Ok(path) = std::env::var("PG_SUBMISSION_CODE_DIR") {
        if let Ok(bytes) = directory_regular_file_bytes(std::path::Path::new(&path)) {
            return bytes;
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len() as usize)
        .unwrap_or(0)
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
        let old_enabled = std::env::var("PG_GPU_TILED_OUTPUT_CE").ok();
        let old_tile = std::env::var("PG_GPU_OUTPUT_CE_TILE_VOCAB").ok();
        match enabled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match tile_vocab {
            Some(value) => unsafe { std::env::set_var("PG_GPU_OUTPUT_CE_TILE_VOCAB", value) },
            None => unsafe { std::env::remove_var("PG_GPU_OUTPUT_CE_TILE_VOCAB") },
        }
        let out = f();
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
    fn record_audit_json_declares_record_shape_and_backends() {
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
        assert!(json.contains("\"artifact_model_bytes\":null"), "{json}");
    }

    #[test]
    fn record_audit_enables_primary_bf16_forward_by_default() {
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
        let old_tiled = std::env::var("PG_GPU_TILED_OUTPUT_CE").ok();
        let old_bf16_logits = std::env::var("PG_GPU_BF16_LOGITS").ok();
        unsafe {
            std::env::set_var("PG_GPU_TILED_OUTPUT_CE", "0");
            std::env::set_var("PG_GPU_BF16_LOGITS", "1");
        }

        let mut spec = RunSpec::default();
        spec.train.backend = TrainBackend::CudaDistributed;
        spec.train.world_size = 8;
        spec.train.seq_len = 2048;
        spec.train.batch_tokens = 786_432;
        spec.train.distributed_optimizer_backend = DistributedOptimizerBackend::ShardedParallelMuon;
        spec.model.compute_precision = ModelComputePrecision::Bf16TensorCore;
        spec.model.vocab_size = 8192;

        assert!(bf16_output_logits_enabled_for_audit(&spec));
        assert!(output_path_materializes_full_logits_for_audit(&spec));
        assert_eq!(
            output_loss_backend_for_audit(&spec),
            "full_logits_bf16_single_gemm"
        );
        assert!(!production_fused_output_projection_ce_enabled_for_audit(
            &spec
        ));

        match old_tiled {
            Some(value) => unsafe { std::env::set_var("PG_GPU_TILED_OUTPUT_CE", value) },
            None => unsafe { std::env::remove_var("PG_GPU_TILED_OUTPUT_CE") },
        }
        match old_bf16_logits {
            Some(value) => unsafe { std::env::set_var("PG_GPU_BF16_LOGITS", value) },
            None => unsafe { std::env::remove_var("PG_GPU_BF16_LOGITS") },
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
