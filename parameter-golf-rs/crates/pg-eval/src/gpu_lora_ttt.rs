#[cfg(feature = "cuda")]
use pg_core::{DType, GpuTensor, PgError, PgResult};
#[cfg(feature = "cuda")]
use pg_model::gpu::{
    GpuActivations, GpuBackwardState, GpuGradBuffers, GpuModel, GpuQProjectionLoraHostState,
};
#[cfg(feature = "cuda")]
use pg_model::{ExecutionPlan, GptModel};

#[cfg(feature = "cuda")]
use crate::sliding::build_ttt_chunks;

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GpuLoraPhasedTttConfig {
    pub stride: usize,
    pub seq_len: usize,
    pub chunk_tokens: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub prefix_docs: usize,
    pub boundary_token_id: Option<u32>,
    pub phases: usize,
    pub weight_decay: f32,
    pub beta2: f32,
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
            boundary_token_id: plan.run_spec.model.smear_gate_boundary_token_id,
            phases: plan.eval_plan.phased_ttt_phases.max(1),
            weight_decay: plan.eval_plan.phased_ttt_weight_decay,
            beta2: plan.eval_plan.ttt_beta2,
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
    let mut loss_sum_gpu = GpuTensor::zeros_gpu(stream.clone(), &[1], DType::F32)?;
    let mut activations = GpuActivations::new_for_plan(plan, seq_len, stream.clone())?;
    let mut backward_state = GpuBackwardState::new_for_plan(plan, seq_len, stream.clone())?;
    let mut grads = GpuGradBuffers::new(&cpu_model.config, stream.clone())?;
    let mut host_workspace = GpuLoraHostWorkspace::new(seq_len);

    let total_tokens = val_tokens.len() - 1;
    let chunks = build_scheduled_ttt_chunks(total_tokens, val_tokens, cfg)?;
    let num_chunks = chunks.len().max(1);
    let prefix_token_end = chunks
        .iter()
        .filter(|chunk| chunk.prefix_warmup)
        .map(|chunk| chunk.chunk.chunk_end)
        .max()
        .unwrap_or(0)
        .min(total_tokens);
    let prefix_docs_seen = if cfg.prefix_docs == 0 {
        0
    } else {
        count_document_starts_until(val_tokens, cfg.boundary_token_id, prefix_token_end)?
    };
    let mutation_guard = ttt_score_mutation_guard_enabled(audit);
    let deadline = TttEvalDeadline::from_env();
    if audit {
        println!(
            "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_start\",\"score_first\":true,\"future_token_access\":false,\"score_phase_lora_mutation_guard\":{},\"tokens\":{},\"seq_len\":{},\"stride\":{},\"chunk_tokens\":{},\"chunks\":{},\"lora_rank\":{},\"lora_alpha\":{},\"phases\":{},\"prefix_docs\":{},\"prefix_docs_seen\":{},\"prefix_token_end\":{},\"boundary_token_id\":{},\"weight_decay\":{:.6},\"ttt_beta2\":{:.6},\"tiled_output_cross_entropy\":{},\"chunked_bf16_output_ce_cache\":{},\"materializes_full_logits\":{},\"forward_hidden_without_logits\":{},\"loss_window_reduction_gpu\":true,\"loss_scalar_downloads\":\"one_per_ttt_chunk\"}}",
            mutation_guard,
            total_tokens,
            seq_len,
            cfg.stride,
            cfg.chunk_tokens,
            num_chunks,
            cfg.lora_rank,
            cfg.lora_alpha,
            cfg.phases,
            cfg.prefix_docs,
            prefix_docs_seen,
            prefix_token_end,
            cfg.boundary_token_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "null".to_string()),
            cfg.weight_decay,
            cfg.beta2,
            model.uses_tiled_output_ce(),
            model.uses_chunked_bf16_output_ce_cache(),
            !model.uses_tiled_output_ce() && !model.uses_chunked_bf16_output_ce_cache(),
            model.uses_tiled_output_ce() || model.uses_chunked_bf16_output_ce_cache(),
        );
    }
    let mut total_loss = 0.0f64;
    let mut total_scored = 0u64;
    let mut total_bytes = 0.0f64;
    let mut current_phase = 0usize;

    for (ci, scheduled) in chunks.iter().enumerate() {
        deadline.check(ci)?;
        let chunk = &scheduled.chunk;
        let phase = scheduled.phase.min(cfg.phases - 1);
        if phase != current_phase {
            // Warm-start A across phases and reset B so each phase begins from
            // zero-delta score-first semantics with the accumulated subspace.
            model.reset_q_lora_b()?;
            current_phase = phase;
        }

        let lora_state_before_score = if mutation_guard {
            Some(model.q_lora_state_to_host()?)
        } else {
            None
        };
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
            &mut loss_sum_gpu,
            &mut activations,
            &mut backward_state,
            &mut host_workspace,
        )?;
        if let Some(before) = lora_state_before_score.as_ref() {
            assert_q_lora_state_unchanged(&model.q_lora_state_to_host()?, before, ci, 0)?;
        }
        total_loss += loss;
        total_scored += scored;
        total_bytes += bytes;

        let update_after_score = ci + 1 < chunks.len() && scored > 0;
        let update_start = if update_after_score {
            chunk.chunk_start
        } else {
            chunk.chunk_end
        };
        let update_end = if update_after_score {
            chunk.chunk_end.min(val_tokens.len() - 1)
        } else {
            chunk.chunk_end
        };
        if audit {
            println!(
                "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_chunk\",\"chunk_id\":{},\"chunk_start\":{},\"chunk_end\":{},\"phase\":{},\"prefix_warmup\":{},\"loss_sum\":{:.9},\"tokens_scored_before_update\":{},\"cumulative_tokens_scored\":{},\"ttt_update_after_score\":{},\"update_start\":{},\"update_end\":{},\"update_tokens\":{},\"future_token_access\":false}}",
                ci,
                chunk.chunk_start,
                chunk.chunk_end,
                phase,
                scheduled.prefix_warmup,
                loss,
                scored,
                total_scored,
                update_after_score,
                update_start,
                update_end,
                update_end.saturating_sub(update_start),
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
        deadline.check(ci)?;
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
pub fn eval_gpu_lora_phased_ttt_distributed(
    cpu_model: &GptModel,
    plan: &ExecutionPlan,
    val_tokens: &[u32],
    base_bytes: &[f32],
    cfg: &GpuLoraPhasedTttConfig,
    world_size: usize,
) -> PgResult<(f64, f64)> {
    if world_size <= 1 {
        return eval_gpu_lora_phased_ttt(cpu_model, plan, val_tokens, base_bytes, cfg);
    }
    if !plan.eval_plan.legal_score_first {
        return Err(PgError::InvalidOp(
            "distributed gpu_lora_phased_ttt requires legal_score_first=true".into(),
        ));
    }
    if cfg.stride == 0 {
        return Err(PgError::InvalidOp(
            "distributed gpu_lora_phased_ttt requires stride > 0".into(),
        ));
    }
    if val_tokens.len() < 2 {
        return Ok((0.0, 0.0));
    }

    let device_count = cudarc::driver::CudaContext::device_count()
        .map_err(|e| PgError::InvalidOp(format!("CUDA device_count failed: {e:?}")))?
        as usize;
    if world_size > device_count {
        return Err(PgError::InvalidOp(format!(
            "distributed gpu_lora_phased_ttt requested world_size {world_size} but only {device_count} CUDA devices are visible",
        )));
    }

    let seq_len = cfg.seq_len.min(val_tokens.len() - 1).max(1);
    let mut replicas = (0..world_size)
        .map(|rank| GpuLoraEvalReplica::new(rank, cpu_model, plan, cfg, seq_len))
        .collect::<PgResult<Vec<_>>>()?;
    let streams = replicas
        .iter()
        .map(|replica| replica.model.gemm.stream().clone())
        .collect::<Vec<_>>();
    let comms = pg_core::nccl::NcclComm::from_local_devices(streams)?;

    let audit = ttt_audit_enabled();
    let total_tokens = val_tokens.len() - 1;
    let chunks = build_scheduled_ttt_chunks(total_tokens, val_tokens, cfg)?;
    let num_chunks = chunks.len().max(1);
    let prefix_token_end = chunks
        .iter()
        .filter(|chunk| chunk.prefix_warmup)
        .map(|chunk| chunk.chunk.chunk_end)
        .max()
        .unwrap_or(0)
        .min(total_tokens);
    let prefix_docs_seen = if cfg.prefix_docs == 0 {
        0
    } else {
        count_document_starts_until(val_tokens, cfg.boundary_token_id, prefix_token_end)?
    };
    let mutation_guard = ttt_score_mutation_guard_enabled(audit);
    let deadline = TttEvalDeadline::from_env();
    if audit {
        println!(
            "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_start\",\"distributed_eval\":true,\"world_size\":{},\"score_parallelism\":\"chunk_windows\",\"ttt_update_parallelism\":\"packed_data_parallel_lora_gradient_allreduce\",\"lora_grad_packed_all_reduce\":true,\"lora_grad_grouped_all_reduce\":true,\"fully_distributed_ttt_update\":true,\"fully_sharded_ttt_update\":false,\"score_first\":true,\"future_token_access\":false,\"score_phase_lora_mutation_guard\":{},\"tokens\":{},\"seq_len\":{},\"stride\":{},\"chunk_tokens\":{},\"chunks\":{},\"lora_rank\":{},\"lora_alpha\":{},\"phases\":{},\"prefix_docs\":{},\"prefix_docs_seen\":{},\"prefix_token_end\":{},\"boundary_token_id\":{},\"weight_decay\":{:.6},\"ttt_beta2\":{:.6},\"tiled_output_cross_entropy\":{},\"chunked_bf16_output_ce_cache\":{},\"materializes_full_logits\":{},\"forward_hidden_without_logits\":{},\"loss_window_reduction_gpu\":true,\"loss_scalar_downloads\":\"one_per_rank_per_ttt_chunk\"}}",
            world_size,
            mutation_guard,
            total_tokens,
            seq_len,
            cfg.stride,
            cfg.chunk_tokens,
            num_chunks,
            cfg.lora_rank,
            cfg.lora_alpha,
            cfg.phases,
            cfg.prefix_docs,
            prefix_docs_seen,
            prefix_token_end,
            cfg.boundary_token_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "null".to_string()),
            cfg.weight_decay,
            cfg.beta2,
            replicas[0].model.uses_tiled_output_ce(),
            replicas[0].model.uses_chunked_bf16_output_ce_cache(),
            !replicas[0].model.uses_tiled_output_ce()
                && !replicas[0].model.uses_chunked_bf16_output_ce_cache(),
            replicas[0].model.uses_tiled_output_ce()
                || replicas[0].model.uses_chunked_bf16_output_ce_cache(),
        );
    }

    let mut total_loss = 0.0f64;
    let mut total_scored = 0u64;
    let mut total_bytes = 0.0f64;
    let mut current_phase = 0usize;

    for (ci, scheduled) in chunks.iter().enumerate() {
        deadline.check(ci)?;
        let chunk = &scheduled.chunk;
        let phase = scheduled.phase.min(cfg.phases - 1);
        if phase != current_phase {
            for replica in &replicas {
                replica.model.reset_q_lora_b()?;
            }
            current_phase = phase;
        }

        let lora_states_before_score = if mutation_guard {
            Some(
                replicas
                    .iter()
                    .map(|replica| replica.model.q_lora_state_to_host())
                    .collect::<PgResult<Vec<_>>>()?,
            )
        } else {
            None
        };
        let mut windows_by_rank = vec![Vec::new(); world_size];
        for (wi, &window_start) in chunk.windows.iter().enumerate() {
            windows_by_rank[wi % world_size].push(window_start);
        }
        let shard_results: PgResult<Vec<(f64, u64, f64)>> = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(world_size);
            for (rank, replica) in replicas.iter_mut().enumerate() {
                let windows = windows_by_rank[rank].clone();
                handles.push(scope.spawn(move || {
                    replica.score_windows(val_tokens, base_bytes, &windows, cfg.stride)
                }));
            }
            let mut results = Vec::with_capacity(world_size);
            for handle in handles {
                let result = handle.join().map_err(|_| {
                    PgError::InvalidOp(
                        "distributed gpu_lora_phased_ttt scoring worker panicked".into(),
                    )
                })??;
                results.push(result);
            }
            Ok(results)
        });
        let shard_results = shard_results?;
        if let Some(before) = lora_states_before_score.as_ref() {
            for (rank, replica) in replicas.iter().enumerate() {
                assert_q_lora_state_unchanged(
                    &replica.model.q_lora_state_to_host()?,
                    &before[rank],
                    ci,
                    rank,
                )?;
            }
        }
        let (loss, scored, bytes) = shard_results.into_iter().fold(
            (0.0f64, 0u64, 0.0f64),
            |(loss_acc, scored_acc, bytes_acc), (loss, scored, bytes)| {
                (loss_acc + loss, scored_acc + scored, bytes_acc + bytes)
            },
        );
        total_loss += loss;
        total_scored += scored;
        total_bytes += bytes;

        let update_after_score = ci + 1 < chunks.len() && scored > 0;
        let update_start = if update_after_score {
            chunk.chunk_start
        } else {
            chunk.chunk_end
        };
        let update_end = if update_after_score {
            chunk.chunk_end.min(val_tokens.len() - 1)
        } else {
            chunk.chunk_end
        };
        if audit {
            println!(
                "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_chunk\",\"distributed_eval\":true,\"world_size\":{},\"ttt_update_parallelism\":\"packed_data_parallel_lora_gradient_allreduce\",\"lora_grad_packed_all_reduce\":true,\"lora_grad_grouped_all_reduce\":true,\"fully_distributed_ttt_update\":true,\"fully_sharded_ttt_update\":false,\"chunk_id\":{},\"chunk_start\":{},\"chunk_end\":{},\"phase\":{},\"prefix_warmup\":{},\"loss_sum\":{:.9},\"tokens_scored_before_update\":{},\"cumulative_tokens_scored\":{},\"ttt_update_after_score\":{},\"update_start\":{},\"update_end\":{},\"update_tokens\":{},\"future_token_access\":false}}",
                world_size,
                ci,
                chunk.chunk_start,
                chunk.chunk_end,
                phase,
                scheduled.prefix_warmup,
                loss,
                scored,
                total_scored,
                update_after_score,
                update_start,
                update_end,
                update_end.saturating_sub(update_start),
            );
        }

        if update_after_score {
            let phase_lr_scale = 0.5f32.powi(phase as i32);
            train_chunk_distributed_lora(
                &mut replicas,
                &comms,
                val_tokens,
                chunk.chunk_start,
                chunk.chunk_end.min(val_tokens.len() - 1),
                cfg.lr * phase_lr_scale,
                cfg.weight_decay,
            )?;
            debug_verify_q_lora_replicas_synced(&mut replicas)?;
        }
        deadline.check(ci)?;
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
            "ttt_audit_json={{\"event\":\"gpu_lora_phased_ttt_done\",\"distributed_eval\":true,\"world_size\":{},\"score_first\":true,\"future_token_access\":false,\"tokens_scored\":{},\"bytes_scored\":{:.3},\"val_loss\":{:.9},\"bpb\":{:.9}}}",
            world_size,
            total_scored,
            total_bytes,
            val_loss,
            bits_per_token * tokens_per_byte,
        );
    }
    Ok((val_loss, bits_per_token * tokens_per_byte))
}

#[cfg(feature = "cuda")]
fn train_chunk_distributed_lora(
    replicas: &mut [GpuLoraEvalReplica],
    comms: &[pg_core::nccl::NcclComm],
    val_tokens: &[u32],
    chunk_start: usize,
    chunk_end: usize,
    lr: f32,
    weight_decay: f32,
) -> PgResult<()> {
    let world_size = replicas.len();
    if world_size == 0 || comms.len() != world_size {
        return Err(PgError::InvalidOp(format!(
            "distributed LoRA TTT update requires matching replicas/comms, got replicas={} comms={}",
            replicas.len(),
            comms.len()
        )));
    }
    let seq_len = replicas[0].seq_len;
    let mut windows = Vec::new();
    let mut ws = chunk_start;
    while ws + seq_len < chunk_end + 1 && ws + seq_len < val_tokens.len() {
        windows.push(ws);
        ws += seq_len;
    }
    for group in windows.chunks(world_size) {
        for replica in replicas.iter() {
            replica.model.zero_q_lora_grads()?;
        }
        std::thread::scope(|scope| -> PgResult<()> {
            let mut handles = Vec::with_capacity(world_size);
            for (rank, replica) in replicas.iter_mut().enumerate() {
                let window = group.get(rank).copied();
                handles.push(scope.spawn(move || {
                    if let Some(ws) = window {
                        replica.accumulate_lora_grad_window(val_tokens, ws)
                    } else {
                        Ok(())
                    }
                }));
            }
            for handle in handles {
                handle.join().map_err(|_| {
                    PgError::InvalidOp(
                        "distributed gpu_lora_phased_ttt update worker panicked".into(),
                    )
                })??;
            }
            Ok(())
        })?;
        all_reduce_q_lora_grads(replicas, comms)?;
        let inv_participants = 1.0f32 / group.len().max(1) as f32;
        for replica in replicas.iter() {
            replica.model.scale_q_lora_grads(inv_participants)?;
            replica.model.step_q_lora_sgd(lr, weight_decay)?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn all_reduce_q_lora_grads(
    replicas: &mut [GpuLoraEvalReplica],
    comms: &[pg_core::nccl::NcclComm],
) -> PgResult<()> {
    for replica in replicas.iter_mut() {
        replica.model.pack_q_lora_grads(&replica.lora_grad_pack)?;
    }
    cudarc::nccl::group_start().map_err(|e| PgError::Nccl(format!("group_start failed: {e:?}")))?;
    for (rank, replica) in replicas.iter_mut().enumerate() {
        comms[rank].all_reduce_sum_tensor_f32_in_place(&mut replica.lora_grad_pack)?;
    }
    cudarc::nccl::group_end().map_err(|e| PgError::Nccl(format!("group_end failed: {e:?}")))?;
    for replica in replicas.iter_mut() {
        replica.model.unpack_q_lora_grads(&replica.lora_grad_pack)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn debug_verify_q_lora_replicas_synced(replicas: &mut [GpuLoraEvalReplica]) -> PgResult<()> {
    if !debug_distributed_ttt_sync_enabled() || replicas.len() <= 1 {
        return Ok(());
    }
    let reference = replicas[0].model.q_lora_state_to_host()?;
    for (rank, replica) in replicas.iter().enumerate().skip(1) {
        let state = replica.model.q_lora_state_to_host()?;
        if state.a.len() != reference.a.len() || state.b.len() != reference.b.len() {
            return Err(PgError::InvalidOp(format!(
                "LoRA replica sync debug failed at rank {rank}: layer count mismatch"
            )));
        }
        for layer in 0..reference.a.len() {
            if state.a[layer] != reference.a[layer] || state.b[layer] != reference.b[layer] {
                return Err(PgError::InvalidOp(format!(
                    "LoRA replica sync debug failed at rank {rank} layer {layer}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn assert_q_lora_state_unchanged(
    after: &GpuQProjectionLoraHostState,
    before: &GpuQProjectionLoraHostState,
    chunk_id: usize,
    rank: usize,
) -> PgResult<()> {
    if after.rank != before.rank
        || (after.alpha - before.alpha).abs() > f32::EPSILON
        || after.a.len() != before.a.len()
        || after.b.len() != before.b.len()
    {
        return Err(PgError::InvalidOp(format!(
            "LoRA score-phase mutation guard failed before comparison at chunk {chunk_id} rank {rank}: state metadata changed"
        )));
    }
    for layer in 0..before.a.len() {
        if after.a[layer] != before.a[layer] || after.b[layer] != before.b[layer] {
            return Err(PgError::InvalidOp(format!(
                "LoRA score-phase mutation guard failed at chunk {chunk_id} rank {rank} layer {layer}: adapter state changed before score-before-update phase completed"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn ttt_score_mutation_guard_enabled(audit: bool) -> bool {
    std::env::var("PG_TTT_ASSERT_SCORE_NO_MUTATION")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(audit)
}

#[cfg(feature = "cuda")]
fn debug_distributed_ttt_sync_enabled() -> bool {
    matches!(
        std::env::var("PG_DEBUG_DISTRIBUTED_TTT_SYNC")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
struct TttEvalDeadline {
    start: Instant,
    max_seconds: f64,
}

#[cfg(feature = "cuda")]
impl TttEvalDeadline {
    fn from_env() -> Self {
        let max_seconds = std::env::var("PG_EVAL_MAX_WALLCLOCK_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(600.0);
        let guard_seconds = std::env::var("PG_EVAL_DEADLINE_GUARD_SECONDS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(2.0);
        Self {
            start: Instant::now(),
            max_seconds: (max_seconds - guard_seconds).max(0.0),
        }
    }

    fn check(&self, chunk_id: usize) -> PgResult<()> {
        if self.max_seconds <= 0.0 {
            return Ok(());
        }
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > self.max_seconds {
            return Err(PgError::InvalidOp(format!(
                "gpu_lora_phased_ttt exceeded eval wallclock deadline before/after chunk {chunk_id}: elapsed_seconds={elapsed:.3} deadline_seconds={:.3}",
                self.max_seconds
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
struct ScheduledTttChunk {
    chunk: crate::sliding::TttChunk,
    phase: usize,
    prefix_warmup: bool,
}

#[cfg(feature = "cuda")]
fn build_scheduled_ttt_chunks(
    total_tokens: usize,
    val_tokens: &[u32],
    cfg: &GpuLoraPhasedTttConfig,
) -> PgResult<Vec<ScheduledTttChunk>> {
    if cfg.prefix_docs == 0 {
        let chunks = build_ttt_chunks(total_tokens, cfg.chunk_tokens, cfg.stride, cfg.seq_len);
        let num_chunks = chunks.len().max(1);
        return Ok(chunks
            .into_iter()
            .enumerate()
            .map(|(ci, chunk)| ScheduledTttChunk {
                chunk,
                phase: (ci * cfg.phases / num_chunks).min(cfg.phases - 1),
                prefix_warmup: false,
            })
            .collect());
    }

    let prefix_end = prefix_doc_token_end(
        val_tokens,
        total_tokens,
        cfg.boundary_token_id,
        cfg.prefix_docs,
    )?;
    let mut scheduled = Vec::new();
    for chunk in build_ttt_chunks_for_range(
        total_tokens,
        cfg.chunk_tokens,
        cfg.stride,
        cfg.seq_len,
        0,
        prefix_end,
    ) {
        scheduled.push(ScheduledTttChunk {
            chunk,
            phase: 0,
            prefix_warmup: true,
        });
    }

    let suffix_chunks = build_ttt_chunks_for_range(
        total_tokens,
        cfg.chunk_tokens,
        cfg.stride,
        cfg.seq_len,
        prefix_end,
        total_tokens,
    );
    let suffix_count = suffix_chunks.len().max(1);
    for (si, chunk) in suffix_chunks.into_iter().enumerate() {
        let phase = if cfg.phases <= 1 {
            0
        } else {
            1 + (si * (cfg.phases - 1) / suffix_count).min(cfg.phases - 2)
        };
        scheduled.push(ScheduledTttChunk {
            chunk,
            phase,
            prefix_warmup: false,
        });
    }
    Ok(scheduled)
}

#[cfg(feature = "cuda")]
fn build_ttt_chunks_for_range(
    total_tokens: usize,
    chunk_tokens: usize,
    stride: usize,
    seq_len: usize,
    range_start: usize,
    range_end: usize,
) -> Vec<crate::sliding::TttChunk> {
    if range_end <= range_start {
        return Vec::new();
    }
    let chunk_tokens = chunk_tokens.max(1);
    let num_chunks = (range_end - range_start + chunk_tokens - 1) / chunk_tokens;
    let mut chunks: Vec<crate::sliding::TttChunk> = (0..num_chunks)
        .map(|ci| {
            let chunk_start = range_start + ci * chunk_tokens;
            crate::sliding::TttChunk {
                chunk_start,
                chunk_end: (chunk_start + chunk_tokens).min(range_end),
                windows: Vec::new(),
            }
        })
        .collect();

    for ws in (0..total_tokens).step_by(stride.max(1)) {
        let end = (ws + seq_len).min(total_tokens);
        let wlen = end.saturating_sub(ws);
        if wlen == 0 || (wlen < stride && ws != 0) {
            continue;
        }
        let score_start = if ws == 0 {
            0
        } else {
            wlen.saturating_sub(stride)
        };
        let scored_start = ws + score_start;
        if scored_start < range_start || scored_start >= range_end {
            continue;
        }
        let ci = ((scored_start - range_start) / chunk_tokens).min(num_chunks - 1);
        chunks[ci].windows.push(ws);
    }

    chunks
}

#[cfg(feature = "cuda")]
fn prefix_doc_token_end(
    val_tokens: &[u32],
    total_tokens: usize,
    boundary_token_id: Option<u32>,
    prefix_docs: usize,
) -> PgResult<usize> {
    if prefix_docs == 0 || total_tokens == 0 {
        return Ok(0);
    }
    let boundary = boundary_token_id.ok_or_else(|| {
        PgError::InvalidOp(
            "gpu_lora_phased_ttt prefix-doc warmup requires model.smear_gate_boundary_token_id/BOS token".into(),
        )
    })?;
    let mut docs_seen = 0usize;
    if val_tokens.first().copied() != Some(boundary) {
        docs_seen = 1;
    }
    for (idx, &token) in val_tokens.iter().take(total_tokens).enumerate() {
        if token == boundary {
            docs_seen += 1;
            if docs_seen > prefix_docs {
                return Ok(idx);
            }
        }
    }
    Ok(total_tokens)
}

#[cfg(feature = "cuda")]
fn count_document_starts_until(
    val_tokens: &[u32],
    boundary_token_id: Option<u32>,
    token_end: usize,
) -> PgResult<usize> {
    let boundary = boundary_token_id.ok_or_else(|| {
        PgError::InvalidOp(
            "gpu_lora_phased_ttt prefix-doc audit requires model.smear_gate_boundary_token_id/BOS token".into(),
        )
    })?;
    let mut docs = if val_tokens.first().copied() == Some(boundary) {
        0
    } else {
        1
    };
    docs += val_tokens
        .iter()
        .take(token_end)
        .filter(|&&token| token == boundary)
        .count();
    Ok(docs)
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
struct GpuLoraEvalReplica {
    model: GpuModel,
    input_gpu: GpuTensor,
    target_gpu: GpuTensor,
    losses_gpu: GpuTensor,
    loss_sum_gpu: GpuTensor,
    activations: GpuActivations,
    backward_state: GpuBackwardState,
    grads: GpuGradBuffers,
    lora_grad_pack: GpuTensor,
    host_workspace: GpuLoraHostWorkspace,
    seq_len: usize,
}

#[cfg(feature = "cuda")]
impl GpuLoraEvalReplica {
    fn new(
        device_ordinal: usize,
        cpu_model: &GptModel,
        plan: &ExecutionPlan,
        cfg: &GpuLoraPhasedTttConfig,
        seq_len: usize,
    ) -> PgResult<Self> {
        let ctx = cudarc::driver::CudaContext::new(device_ordinal).map_err(|e| {
            PgError::InvalidOp(format!(
                "CUDA context init failed for eval device {device_ordinal}: {e:?}"
            ))
        })?;
        let stream = ctx.default_stream();
        let mut model = GpuModel::from_cpu_reference(cpu_model, plan, ctx, stream.clone())?;
        model.enable_q_lora(cfg.lora_rank, cfg.lora_alpha)?;
        let lora_grad_numel = model.q_lora_grad_numel()?;
        Ok(Self {
            model,
            input_gpu: GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::U32)?,
            target_gpu: GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::U32)?,
            losses_gpu: GpuTensor::zeros_gpu(stream.clone(), &[seq_len], DType::F32)?,
            loss_sum_gpu: GpuTensor::zeros_gpu(stream.clone(), &[1], DType::F32)?,
            activations: GpuActivations::new_for_plan(plan, seq_len, stream.clone())?,
            backward_state: GpuBackwardState::new_for_plan(plan, seq_len, stream.clone())?,
            grads: GpuGradBuffers::new(&cpu_model.config, stream.clone())?,
            lora_grad_pack: GpuTensor::zeros_gpu(stream.clone(), &[lora_grad_numel], DType::F32)?,
            host_workspace: GpuLoraHostWorkspace::new(seq_len),
            seq_len,
        })
    }

    fn score_windows(
        &mut self,
        val_tokens: &[u32],
        base_bytes: &[f32],
        windows: &[usize],
        stride: usize,
    ) -> PgResult<(f64, u64, f64)> {
        let chunk = crate::sliding::TttChunk {
            chunk_start: 0,
            chunk_end: val_tokens.len().saturating_sub(1),
            windows: windows.to_vec(),
        };
        score_chunk_gpu(
            &self.model,
            val_tokens,
            base_bytes,
            &chunk,
            stride,
            self.seq_len,
            &mut self.input_gpu,
            &mut self.target_gpu,
            &self.losses_gpu,
            &mut self.loss_sum_gpu,
            &mut self.activations,
            &mut self.backward_state,
            &mut self.host_workspace,
        )
    }

    fn accumulate_lora_grad_window(&mut self, val_tokens: &[u32], ws: usize) -> PgResult<()> {
        self.host_workspace
            .input
            .copy_from_slice(&val_tokens[ws..ws + self.seq_len]);
        self.host_workspace
            .target
            .copy_from_slice(&val_tokens[ws + 1..ws + self.seq_len + 1]);
        self.input_gpu
            .copy_from_host_bytes(bytemuck::cast_slice(&self.host_workspace.input))?;
        self.target_gpu
            .copy_from_host_bytes(bytemuck::cast_slice(&self.host_workspace.target))?;

        self.grads.zero(&self.model.kernels)?;
        self.model.zero_q_lora_grads()?;
        self.model.backward_with_state_seq_len_no_loss(
            &self.input_gpu,
            &self.target_gpu,
            &mut self.activations,
            &mut self.backward_state,
            &mut self.grads,
            self.seq_len,
        )
    }
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
    loss_sum_gpu: &mut GpuTensor,
    activations: &mut GpuActivations,
    backward_state: &mut GpuBackwardState,
    host: &mut GpuLoraHostWorkspace,
) -> PgResult<(f64, u64, f64)> {
    let total_tokens = val_tokens.len() - 1;
    let mut token_count = 0u64;
    let mut byte_count = 0.0f64;
    loss_sum_gpu.zero_bytes()?;

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

        if model.uses_tiled_output_ce() {
            model.forward_hidden_with_seq_len(input_gpu, activations, seq_len)?;
        } else {
            model.forward_with_seq_len(input_gpu, activations, seq_len)?;
        }
        model.cross_entropy_losses_with_state(
            activations,
            backward_state,
            target_gpu,
            losses_gpu,
            seq_len,
        )?;
        let score_start = if ws == 0 {
            0
        } else {
            wlen.saturating_sub(stride)
        };
        model.kernels.loss_window_accumulate(
            pg_kernels::gpu_kernels::CudaPtr(losses_gpu.cu_ptr(model.gemm.stream())?),
            pg_kernels::gpu_kernels::CudaPtr(loss_sum_gpu.cu_ptr(model.gemm.stream())?),
            score_start as u32,
            wlen as u32,
        )?;
        for t in score_start..wlen {
            token_count += 1;
            let tok_idx = ws + t;
            byte_count += base_bytes.get(tok_idx).copied().unwrap_or(1.0) as f64;
        }
    }
    let loss_sum_bytes = loss_sum_gpu.to_host_bytes()?;
    let loss_sum = bytemuck::cast_slice::<u8, f32>(&loss_sum_bytes)
        .first()
        .copied()
        .unwrap_or(0.0) as f64;
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn test_cfg(prefix_docs: usize) -> GpuLoraPhasedTttConfig {
        GpuLoraPhasedTttConfig {
            stride: 2,
            seq_len: 4,
            chunk_tokens: 4,
            lora_rank: 2,
            lora_alpha: 4.0,
            prefix_docs,
            boundary_token_id: Some(1),
            phases: 3,
            weight_decay: 1.0,
            beta2: 0.99,
            lr: 0.01,
        }
    }

    #[test]
    fn prefix_doc_end_stops_at_next_bos() {
        let tokens = vec![1, 10, 11, 1, 20, 21, 1, 30, 31, 1, 40];
        assert_eq!(prefix_doc_token_end(&tokens, 10, Some(1), 1).unwrap(), 3);
        assert_eq!(prefix_doc_token_end(&tokens, 10, Some(1), 2).unwrap(), 6);
        assert_eq!(prefix_doc_token_end(&tokens, 10, Some(1), 3).unwrap(), 9);
        assert_eq!(prefix_doc_token_end(&tokens, 10, Some(1), 99).unwrap(), 10);
    }

    #[test]
    fn prefix_doc_schedule_anchors_first_phase_to_prefix_docs() {
        let tokens = vec![1, 10, 11, 1, 20, 21, 1, 30, 31, 1, 40, 41];
        let scheduled = build_scheduled_ttt_chunks(11, &tokens, &test_cfg(2)).unwrap();
        assert!(
            scheduled.iter().any(|chunk| chunk.prefix_warmup),
            "prefix-doc warmup chunks should be present"
        );
        assert!(
            scheduled
                .iter()
                .filter(|chunk| chunk.prefix_warmup)
                .all(|chunk| chunk.phase == 0 && chunk.chunk.chunk_end <= 6),
            "prefix chunks must stay in phase 0 and end at the third BOS boundary"
        );
        assert!(
            scheduled
                .iter()
                .filter(|chunk| !chunk.prefix_warmup)
                .all(|chunk| chunk.phase >= 1 && chunk.chunk.chunk_start >= 6),
            "suffix chunks must start after the prefix boundary and use later phases"
        );
    }

    #[test]
    fn prefix_doc_schedule_requires_boundary_token() {
        let mut cfg = test_cfg(2);
        cfg.boundary_token_id = None;
        let err = match build_scheduled_ttt_chunks(8, &[1, 2, 3, 4, 5, 6, 7, 8, 9], &cfg) {
            Ok(_) => panic!("prefix-doc scheduling unexpectedly accepted a missing boundary token"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("prefix-doc warmup requires"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn score_phase_mutation_guard_accepts_identical_lora_state() {
        let state = GpuQProjectionLoraHostState {
            rank: 2,
            alpha: 4.0,
            a: vec![vec![1, 2, 3, 4]],
            b: vec![vec![5, 6, 7, 8]],
        };
        assert!(assert_q_lora_state_unchanged(&state, &state, 7, 0).is_ok());
    }

    #[test]
    fn score_phase_mutation_guard_rejects_adapter_changes() {
        let before = GpuQProjectionLoraHostState {
            rank: 2,
            alpha: 4.0,
            a: vec![vec![1, 2, 3, 4]],
            b: vec![vec![5, 6, 7, 8]],
        };
        let mut after = before.clone();
        after.b[0][2] ^= 0x80;
        let err = assert_q_lora_state_unchanged(&after, &before, 7, 3)
            .expect_err("mutated LoRA state should fail the score-phase guard")
            .to_string();
        assert!(
            err.contains("chunk 7 rank 3 layer 0"),
            "unexpected error: {err}"
        );
    }
}
