#[cfg(feature = "cuda")]
use pg_core::{DType, GpuTensor, PgError, PgResult};
#[cfg(feature = "cuda")]
use pg_model::gpu::{GpuActivations, GpuBackwardState, GpuGradBuffers, GpuModel};
#[cfg(feature = "cuda")]
use pg_model::{ExecutionPlan, GptModel};

#[cfg(feature = "cuda")]
use crate::sliding::build_ttt_chunks;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GpuLoraPhasedTttConfig {
    pub stride: usize,
    pub seq_len: usize,
    pub chunk_tokens: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub prefix_docs: usize,
    pub phases: usize,
    pub weight_decay: f32,
    pub lr: f32,
}

#[cfg(feature = "cuda")]
impl GpuLoraPhasedTttConfig {
    pub fn from_plan(plan: &ExecutionPlan, token_count: usize) -> Self {
        let seq_len = plan
            .run_spec
            .model
            .eval_seq_len
            .min(token_count.saturating_sub(1))
            .max(1);
        Self {
            stride: plan.eval_plan.stride,
            seq_len,
            chunk_tokens: plan.eval_plan.chunk_tokens,
            lora_rank: plan.eval_plan.lora_rank,
            lora_alpha: plan.eval_plan.lora_alpha,
            prefix_docs: plan.eval_plan.phased_ttt_prefix_docs,
            phases: plan.eval_plan.phased_ttt_phases.max(1),
            weight_decay: plan.eval_plan.phased_ttt_weight_decay,
            // The frontier LoRA-TTT recipes tune this with the matrix-update
            // learning rate. Keeping it spec-driven prevents eval-time
            // adaptation from silently diverging from the record config.
            lr: plan.run_spec.train.matrix_lr,
        }
    }
}

#[cfg(feature = "cuda")]
pub fn eval_gpu_lora_phased_ttt(
    cpu_model: &GptModel,
    plan: &ExecutionPlan,
    val_tokens: &[u32],
    base_bytes: &[f32],
    cfg: &GpuLoraPhasedTttConfig,
) -> PgResult<(f64, f64)> {
    if !plan.eval_plan.legal_score_first {
        return Err(PgError::InvalidOp(
            "gpu_lora_phased_ttt requires legal_score_first=true".into(),
        ));
    }
    if cfg.stride == 0 {
        return Err(PgError::InvalidOp(
            "gpu_lora_phased_ttt requires stride > 0".into(),
        ));
    }
    if val_tokens.len() < 2 {
        return Ok((0.0, 0.0));
    }
    let audit = ttt_audit_enabled();
    let ctx = cudarc::driver::CudaContext::new(0)
        .map_err(|e| PgError::InvalidOp(format!("CUDA context init failed: {e:?}")))?;
    let stream = ctx.default_stream();
    let mut model = GpuModel::from_cpu_reference(cpu_model, plan, ctx, stream.clone())?;
    model.enable_q_lora(cfg.lora_rank, cfg.lora_alpha)?;

    let seq_len = cfg.seq_len.min(val_tokens.len() - 1).max(1);
    let mut input_gpu = GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::U32)?;
    let mut target_gpu = GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::U32)?;
    let losses_gpu = GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::F32)?;
    let mut activations = GpuActivations::new(&cpu_model.config, seq_len, stream.clone())?;
    let mut backward_state = GpuBackwardState::new(&cpu_model.config, seq_len, stream.clone())?;
    let mut grads = GpuGradBuffers::new(&cpu_model.config, stream.clone())?;
    let mut host_workspace = GpuLoraHostWorkspace::new(seq_len);

    let total_tokens = val_tokens.len() - 1;
    let chunks = build_ttt_chunks(total_tokens, cfg.chunk_tokens, cfg.stride, seq_len);
    let num_chunks = chunks.len().max(1);
    if audit {
        println!(
            "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_start\",\"score_first\":true,\"future_token_access\":false,\"tokens\":{},\"seq_len\":{},\"stride\":{},\"chunk_tokens\":{},\"chunks\":{},\"lora_rank\":{},\"lora_alpha\":{},\"phases\":{},\"prefix_docs\":{},\"weight_decay\":{:.6}}}",
            total_tokens,
            seq_len,
            cfg.stride,
            cfg.chunk_tokens,
            num_chunks,
            cfg.lora_rank,
            cfg.lora_alpha,
            cfg.phases,
            cfg.prefix_docs,
            cfg.weight_decay,
        );
    }
    let mut total_loss = 0.0f64;
    let mut total_scored = 0u64;
    let mut total_bytes = 0.0f64;
    let mut current_phase = 0usize;

    for (ci, chunk) in chunks.iter().enumerate() {
        let phase = (ci * cfg.phases / num_chunks).min(cfg.phases - 1);
        if phase != current_phase {
            // Warm-start A across phases and reset B so each phase begins from
            // zero-delta score-first semantics with the accumulated subspace.
            model.reset_q_lora_b()?;
            current_phase = phase;
        }

        let (loss, scored, bytes) = score_chunk_gpu(
            &model,
            val_tokens,
            base_bytes,
            chunk,
            cfg.stride,
            seq_len,
            &mut input_gpu,
            &mut target_gpu,
            &losses_gpu,
            &mut activations,
            &mut host_workspace,
        )?;
        total_loss += loss;
        total_scored += scored;
        total_bytes += bytes;

        let update_after_score = ci + 1 < chunks.len() && scored > 0;
        if audit {
            println!(
                "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_chunk\",\"chunk_id\":{},\"chunk_start\":{},\"chunk_end\":{},\"phase\":{},\"loss_sum\":{:.9},\"tokens_scored_before_update\":{},\"cumulative_tokens_scored\":{},\"ttt_update_after_score\":{},\"future_token_access\":false}}",
                ci,
                chunk.chunk_start,
                chunk.chunk_end,
                phase,
                loss,
                scored,
                total_scored,
                update_after_score,
            );
        }

        if update_after_score {
            let phase_lr_scale = 0.5f32.powi(phase as i32);
            train_chunk_gpu_lora(
                &model,
                val_tokens,
                chunk.chunk_start,
                chunk.chunk_end.min(val_tokens.len() - 1),
                seq_len,
                cfg.lr * phase_lr_scale,
                cfg.weight_decay,
                &mut input_gpu,
                &mut target_gpu,
                &mut activations,
                &mut backward_state,
                &mut grads,
                &mut host_workspace,
            )?;
        }
    }

    let val_loss = if total_scored > 0 {
        total_loss / total_scored as f64
    } else {
        0.0
    };
    let bits_per_token = val_loss / 2.0f64.ln();
    let tokens_per_byte = if total_bytes > 0.0 {
        total_scored as f64 / total_bytes
    } else {
        1.0
    };
    if audit {
        println!(
            "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_done\",\"score_first\":true,\"future_token_access\":false,\"tokens_scored\":{},\"bytes_scored\":{:.3},\"val_loss\":{:.9},\"bpb\":{:.9}}}",
            total_scored,
            total_bytes,
            val_loss,
            bits_per_token * tokens_per_byte,
        );
    }
    Ok((val_loss, bits_per_token * tokens_per_byte))
}

#[cfg(feature = "cuda")]
fn ttt_audit_enabled() -> bool {
    matches!(
        std::env::var("PG_TTT_AUDIT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
struct GpuLoraHostWorkspace {
    input: Vec<u32>,
    target: Vec<u32>,
}

#[cfg(feature = "cuda")]
impl GpuLoraHostWorkspace {
    fn new(seq_len: usize) -> Self {
        Self {
            input: vec![0u32; seq_len],
            target: vec![0u32; seq_len],
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn score_chunk_gpu(
    model: &GpuModel,
    val_tokens: &[u32],
    base_bytes: &[f32],
    chunk: &crate::sliding::TttChunk,
    stride: usize,
    seq_len: usize,
    input_gpu: &mut GpuTensor,
    target_gpu: &mut GpuTensor,
    losses_gpu: &GpuTensor,
    activations: &mut GpuActivations,
    host: &mut GpuLoraHostWorkspace,
) -> PgResult<(f64, u64, f64)> {
    let total_tokens = val_tokens.len() - 1;
    let mut loss_sum = 0.0f64;
    let mut token_count = 0u64;
    let mut byte_count = 0.0f64;

    for &ws in &chunk.windows {
        let end = (ws + seq_len).min(total_tokens);
        let wlen = end - ws;
        if wlen == 0 {
            continue;
        }
        host.input.fill(0);
        host.target.fill(0);
        host.input[..wlen].copy_from_slice(&val_tokens[ws..end]);
        host.target[..wlen].copy_from_slice(&val_tokens[ws + 1..end + 1]);
        input_gpu.copy_from_host_bytes(bytemuck::cast_slice(&host.input))?;
        target_gpu.copy_from_host_bytes(bytemuck::cast_slice(&host.target))?;

        model.forward_with_seq_len(input_gpu, activations, seq_len)?;
        model.cross_entropy_losses(&activations.logits, target_gpu, losses_gpu, seq_len)?;
        let losses = losses_gpu.to_host_bytes()?;
        let losses = bytemuck::cast_slice::<u8, f32>(&losses);
        let score_start = if ws == 0 {
            0
        } else {
            wlen.saturating_sub(stride)
        };
        for (t, nll) in losses.iter().take(wlen).enumerate().skip(score_start) {
            loss_sum += *nll as f64;
            token_count += 1;
            let tok_idx = ws + t;
            byte_count += base_bytes.get(tok_idx).copied().unwrap_or(1.0) as f64;
        }
    }
    Ok((loss_sum, token_count, byte_count))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn train_chunk_gpu_lora(
    model: &GpuModel,
    val_tokens: &[u32],
    chunk_start: usize,
    chunk_end: usize,
    seq_len: usize,
    lr: f32,
    weight_decay: f32,
    input_gpu: &mut GpuTensor,
    target_gpu: &mut GpuTensor,
    activations: &mut GpuActivations,
    backward_state: &mut GpuBackwardState,
    grads: &mut GpuGradBuffers,
    host: &mut GpuLoraHostWorkspace,
) -> PgResult<()> {
    let mut ws = chunk_start;
    while ws + seq_len < chunk_end + 1 && ws + seq_len < val_tokens.len() {
        host.input.copy_from_slice(&val_tokens[ws..ws + seq_len]);
        host.target
            .copy_from_slice(&val_tokens[ws + 1..ws + seq_len + 1]);
        input_gpu.copy_from_host_bytes(bytemuck::cast_slice(&host.input))?;
        target_gpu.copy_from_host_bytes(bytemuck::cast_slice(&host.target))?;

        grads.zero(&model.kernels)?;
        model.zero_q_lora_grads()?;
        model.backward_with_state_seq_len_no_loss(
            input_gpu,
            target_gpu,
            activations,
            backward_state,
            grads,
            seq_len,
        )?;
        model.step_q_lora_sgd(lr, weight_decay)?;
        ws += seq_len;
    }
    Ok(())
}
