/// GPU model runner — orchestrates forward/backward on CUDA.
///
/// Maps the CPU-verified model logic to GPU kernels:
///   - cuBLASLt for all GEMM (QKV projections, MLP, output, Newton-Schulz)
///   - F32 CUDA SDPA parity kernels today; production BF16 fused SDPA is gated
///   - CUDA element-wise kernels (RMSNorm, RoPE, activations, residuals)
///   - NCCL all-reduce multi-GPU today; sharded Parallel Muon is gated
///
/// Memory layout: all parameter banks are contiguous F32 on device today.
/// Activations are allocated from BufferPool (zero runtime malloc).
///
/// This module requires the `cuda` feature.

#[cfg(feature = "cuda")]
use std::cell::Cell;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

#[cfg(feature = "cuda")]
use pg_core::{DType, GpuTensor, PgError, PgResult};

use crate::config::ModelConfig;
#[cfg(feature = "cuda")]
use crate::{AttentionBackend, ExecutionPlan, GptModel, ModelComputePrecision, OutputCeBackend};

#[cfg(feature = "cuda")]
fn decode_f32_host_bytes(bytes: &[u8]) -> PgResult<Vec<f32>> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(PgError::InvalidOp(format!(
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

/// GPU weight storage — all parameters resident on a single device.
#[cfg(feature = "cuda")]
pub struct GpuWeights {
    // Parameter banks — F32 is still the authoritative storage. BF16 shadows
    // are refreshed after optimizer updates and used only by gated tensor-core
    // forward GEMMs.
    pub qo_bank: GpuTensor,            // [2*n, d, d]
    pub qo_bank_bf16: GpuTensor,       // [2*n, d, d]
    pub kv_bank: GpuTensor,            // [2*n, kv, d]
    pub kv_bank_bf16: GpuTensor,       // [2*n, kv, d]
    pub qkv_bank: GpuTensor,           // [n, d + 2*kv, d], derived hot-path shadow
    pub qkv_bank_bf16: GpuTensor,      // [n, d + 2*kv, d], derived hot-path shadow
    pub mlp_up_bank: GpuTensor,        // [n, mlp, d]
    pub mlp_up_bank_bf16: GpuTensor,   // [n, mlp, d]
    pub mlp_down_bank: GpuTensor,      // [n, d, mlp]
    pub mlp_down_bank_bf16: GpuTensor, // [n, d, mlp]

    // Embeddings — F32 today.
    pub tok_emb: GpuTensor,      // [vocab, d]
    pub tok_emb_bf16: GpuTensor, // [vocab, d] shadow for tensor-core output projection

    // Scalar/vector params — F32 device tensors (small, precision-sensitive)
    pub attn_scales: Vec<GpuTensor>,              // per-layer [d]
    pub mlp_scales: Vec<GpuTensor>,               // per-layer [d]
    pub resid_mix: Vec<GpuTensor>,                // per-layer [2, d]
    pub q_gains: Vec<GpuTensor>,                  // per-layer [h]
    pub attn_gate_weights: Vec<GpuTensor>,        // per-layer [h, gate_width]
    pub attn_gate_biases: Vec<GpuTensor>,         // per-layer [h]
    pub sparse_attn_gate_weights: Vec<GpuTensor>, // per-layer [h, sparse_gate_width]

    // Misc
    pub bigram_embed: GpuTensor,
    pub bigram_proj: GpuTensor,
    // Host mirrors are updated only for export/debug. The forward/backward hot
    // path consumes the device scalar tensors below to avoid per-step D2H sync.
    pub bigram_scale: f32,
    pub bigram_scale_param: GpuTensor,
    pub smear_gate: GpuTensor,
    pub skip_weights: GpuTensor,

    // Value Embedding (shared)
    pub ve_embed: GpuTensor,
    pub ve_proj: GpuTensor,
    pub ve_scale: f32,
    pub ve_scale_param: GpuTensor,
    pub ve_layer_scales: GpuTensor,
    pub ve_layer_scales_host: Vec<f32>,

    // RoPE tables — precomputed F32
    pub rope_cos: GpuTensor,
    pub rope_sin: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuWeights {
    pub fn from_cpu(cpu: &crate::model::GptModel, stream: Arc<CudaStream>) -> PgResult<Self> {
        let c = &cpu.config;
        let l = c.num_layers;
        let d = c.model_dim;
        let md = c.mlp_dim;
        let kv = c.kv_dim();

        fn to_gpu(s: &Arc<CudaStream>, data: &[f32], shape: &[usize]) -> PgResult<GpuTensor> {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            GpuTensor::from_host_data_gpu(s.clone(), bytes, shape, DType::F32)
        }

        fn to_gpu_bf16(s: &Arc<CudaStream>, data: &[f32], shape: &[usize]) -> PgResult<GpuTensor> {
            GpuTensor::from_host_data_gpu(s.clone(), &f32_to_bf16_bytes(data), shape, DType::BF16)
        }

        let mut attn_scales = Vec::new();
        let mut mlp_scales = Vec::new();
        let mut resid_mix = Vec::new();
        let mut q_gains = Vec::new();
        let mut attn_gate_weights = Vec::new();
        let mut attn_gate_biases = Vec::new();
        let mut sparse_attn_gate_weights = Vec::new();

        for i in 0..l {
            let b = &cpu.blocks[i];
            attn_scales.push(to_gpu(&stream, &b.attn_scale, &[d])?);
            mlp_scales.push(to_gpu(&stream, &b.mlp_scale, &[d])?);
            resid_mix.push(to_gpu(&stream, &b.resid_mix, &[2, d])?);
            q_gains.push(to_gpu(&stream, &b.q_gain, &[c.num_heads])?);
            let gate_width = c.attn_out_gate_width.max(1);
            attn_gate_weights.push(to_gpu(
                &stream,
                &b.attn_gate_weight,
                &[c.num_heads, gate_width],
            )?);
            attn_gate_biases.push(to_gpu(&stream, &b.attn_gate_bias, &[c.num_heads])?);
            let sparse_gate_width = c.sparse_attn_gate_width.max(1);
            sparse_attn_gate_weights.push(to_gpu(
                &stream,
                &b.sparse_attn_gate_weight,
                &[c.num_heads, sparse_gate_width],
            )?);
        }

        let qkv_host = pack_qkv_bank_host(&cpu.qo_bank, &cpu.kv_bank, l, d, kv);

        Ok(Self {
            qo_bank: to_gpu(&stream, &cpu.qo_bank, &[2 * l, d, d])?,
            qo_bank_bf16: to_gpu_bf16(&stream, &cpu.qo_bank, &[2 * l, d, d])?,
            kv_bank: to_gpu(&stream, &cpu.kv_bank, &[2 * l, kv, d])?,
            kv_bank_bf16: to_gpu_bf16(&stream, &cpu.kv_bank, &[2 * l, kv, d])?,
            qkv_bank: to_gpu(&stream, &qkv_host, &[l, d + 2 * kv, d])?,
            qkv_bank_bf16: to_gpu_bf16(&stream, &qkv_host, &[l, d + 2 * kv, d])?,
            mlp_up_bank: to_gpu(&stream, &cpu.mlp_up_bank, &[l, md, d])?,
            mlp_up_bank_bf16: to_gpu_bf16(&stream, &cpu.mlp_up_bank, &[l, md, d])?,
            mlp_down_bank: to_gpu(&stream, &cpu.mlp_down_bank, &[l, d, md])?,
            mlp_down_bank_bf16: to_gpu_bf16(&stream, &cpu.mlp_down_bank, &[l, d, md])?,
            tok_emb: to_gpu(&stream, &cpu.tok_emb, &[c.vocab_size, d])?,
            tok_emb_bf16: to_gpu_bf16(&stream, &cpu.tok_emb, &[c.vocab_size, d])?,
            attn_scales,
            mlp_scales,
            resid_mix,
            q_gains,
            attn_gate_weights,
            attn_gate_biases,
            sparse_attn_gate_weights,
            bigram_embed: to_gpu(
                &stream,
                &cpu.bigram_embed,
                &[c.bigram_vocab_size, c.bigram_dim],
            )?,
            bigram_proj: to_gpu(&stream, &cpu.bigram_proj, &[d, c.bigram_dim])?,
            bigram_scale: cpu.bigram_scale,
            bigram_scale_param: to_gpu(&stream, &[cpu.bigram_scale], &[1])?,
            smear_gate: to_gpu(&stream, &cpu.smear_gate, &[d])?,
            skip_weights: to_gpu(&stream, &cpu.skip_weights, &[c.num_skip_weights(), d])?,
            ve_embed: if c.ve_enabled {
                to_gpu(&stream, &cpu.ve_embed, &[c.vocab_size, c.ve_dim])?
            } else {
                to_gpu(&stream, &[0.0], &[1])?
            },
            ve_proj: if c.ve_enabled {
                to_gpu(&stream, &cpu.ve_proj, &[kv, c.ve_dim])?
            } else {
                to_gpu(&stream, &[0.0], &[1])?
            },
            ve_scale: cpu.ve_scale,
            ve_scale_param: to_gpu(&stream, &[cpu.ve_scale], &[1])?,
            ve_layer_scales: if c.ve_enabled {
                to_gpu(&stream, &cpu.ve_layer_scales, &[cpu.ve_layer_scales.len()])?
            } else {
                to_gpu(&stream, &[0.0], &[1])?
            },
            ve_layer_scales_host: if c.ve_enabled {
                cpu.ve_layer_scales.clone()
            } else {
                Vec::new()
            },
            rope_cos: to_gpu(&stream, &cpu.rope_cos, &[c.train_seq_len, c.rope_dims / 2])?,
            rope_sin: to_gpu(&stream, &cpu.rope_sin, &[c.train_seq_len, c.rope_dims / 2])?,
        })
    }

    pub fn sync_from_cpu(&mut self, cpu: &crate::model::GptModel) -> PgResult<()> {
        fn sync_f32(tensor: &mut GpuTensor, data: &[f32]) -> PgResult<()> {
            tensor.copy_from_host_bytes(bytemuck::cast_slice(data))
        }

        let c = &cpu.config;
        let l = c.num_layers;
        if self.attn_scales.len() != l
            || self.mlp_scales.len() != l
            || self.resid_mix.len() != l
            || self.q_gains.len() != l
            || self.attn_gate_weights.len() != l
            || self.attn_gate_biases.len() != l
            || self.sparse_attn_gate_weights.len() != l
        {
            return Err(pg_core::PgError::InvalidOp(
                "GPU weight layout no longer matches CPU model layer count".into(),
            ));
        }

        sync_f32(&mut self.qo_bank, &cpu.qo_bank)?;
        self.qo_bank_bf16
            .copy_from_host_bytes(&f32_to_bf16_bytes(&cpu.qo_bank))?;
        sync_f32(&mut self.kv_bank, &cpu.kv_bank)?;
        self.kv_bank_bf16
            .copy_from_host_bytes(&f32_to_bf16_bytes(&cpu.kv_bank))?;
        let qkv_host = pack_qkv_bank_host(&cpu.qo_bank, &cpu.kv_bank, l, c.model_dim, c.kv_dim());
        sync_f32(&mut self.qkv_bank, &qkv_host)?;
        self.qkv_bank_bf16
            .copy_from_host_bytes(&f32_to_bf16_bytes(&qkv_host))?;
        sync_f32(&mut self.mlp_up_bank, &cpu.mlp_up_bank)?;
        self.mlp_up_bank_bf16
            .copy_from_host_bytes(&f32_to_bf16_bytes(&cpu.mlp_up_bank))?;
        sync_f32(&mut self.mlp_down_bank, &cpu.mlp_down_bank)?;
        self.mlp_down_bank_bf16
            .copy_from_host_bytes(&f32_to_bf16_bytes(&cpu.mlp_down_bank))?;
        sync_f32(&mut self.tok_emb, &cpu.tok_emb)?;
        self.tok_emb_bf16
            .copy_from_host_bytes(&f32_to_bf16_bytes(&cpu.tok_emb))?;
        for i in 0..l {
            let b = &cpu.blocks[i];
            sync_f32(&mut self.attn_scales[i], &b.attn_scale)?;
            sync_f32(&mut self.mlp_scales[i], &b.mlp_scale)?;
            sync_f32(&mut self.resid_mix[i], &b.resid_mix)?;
            sync_f32(&mut self.q_gains[i], &b.q_gain)?;
            sync_f32(&mut self.attn_gate_weights[i], &b.attn_gate_weight)?;
            sync_f32(&mut self.attn_gate_biases[i], &b.attn_gate_bias)?;
            sync_f32(
                &mut self.sparse_attn_gate_weights[i],
                &b.sparse_attn_gate_weight,
            )?;
        }
        sync_f32(&mut self.bigram_embed, &cpu.bigram_embed)?;
        sync_f32(&mut self.bigram_proj, &cpu.bigram_proj)?;
        self.bigram_scale = cpu.bigram_scale;
        sync_f32(&mut self.bigram_scale_param, &[cpu.bigram_scale])?;
        sync_f32(&mut self.smear_gate, &cpu.smear_gate)?;
        sync_f32(&mut self.skip_weights, &cpu.skip_weights)?;
        self.ve_scale = cpu.ve_scale;
        sync_f32(&mut self.ve_scale_param, &[cpu.ve_scale])?;
        if cpu.config.ve_enabled {
            sync_f32(&mut self.ve_embed, &cpu.ve_embed)?;
            sync_f32(&mut self.ve_proj, &cpu.ve_proj)?;
            sync_f32(&mut self.ve_layer_scales, &cpu.ve_layer_scales)?;
            self.ve_layer_scales_host.clone_from(&cpu.ve_layer_scales);
        } else {
            sync_f32(&mut self.ve_embed, &[0.0])?;
            sync_f32(&mut self.ve_proj, &[0.0])?;
            sync_f32(&mut self.ve_layer_scales, &[0.0])?;
            self.ve_layer_scales_host.clear();
        }
        Ok(())
    }

    pub fn sync_to_cpu(&self, cpu: &mut crate::model::GptModel) -> PgResult<()> {
        fn download_f32(tensor: &GpuTensor) -> PgResult<Vec<f32>> {
            decode_f32_host_bytes(&tensor.to_host_bytes()?)
        }

        let c = &cpu.config;
        let l = c.num_layers;
        if self.attn_scales.len() != l
            || self.mlp_scales.len() != l
            || self.resid_mix.len() != l
            || self.q_gains.len() != l
            || self.attn_gate_weights.len() != l
            || self.attn_gate_biases.len() != l
            || self.sparse_attn_gate_weights.len() != l
        {
            return Err(pg_core::PgError::InvalidOp(
                "GPU weight layout no longer matches CPU model layer count".into(),
            ));
        }

        cpu.qo_bank = download_f32(&self.qo_bank)?;
        cpu.kv_bank = download_f32(&self.kv_bank)?;
        cpu.mlp_up_bank = download_f32(&self.mlp_up_bank)?;
        cpu.mlp_down_bank = download_f32(&self.mlp_down_bank)?;
        cpu.tok_emb = download_f32(&self.tok_emb)?;
        for i in 0..l {
            cpu.blocks[i].attn_scale = download_f32(&self.attn_scales[i])?;
            cpu.blocks[i].mlp_scale = download_f32(&self.mlp_scales[i])?;
            cpu.blocks[i].resid_mix = download_f32(&self.resid_mix[i])?;
            cpu.blocks[i].q_gain = download_f32(&self.q_gains[i])?;
            cpu.blocks[i].attn_gate_weight = download_f32(&self.attn_gate_weights[i])?;
            cpu.blocks[i].attn_gate_bias = download_f32(&self.attn_gate_biases[i])?;
            cpu.blocks[i].sparse_attn_gate_weight =
                download_f32(&self.sparse_attn_gate_weights[i])?;
        }
        cpu.bigram_embed = download_f32(&self.bigram_embed)?;
        cpu.bigram_proj = download_f32(&self.bigram_proj)?;
        cpu.bigram_scale = download_f32(&self.bigram_scale_param)?[0];
        cpu.smear_gate = download_f32(&self.smear_gate)?;
        cpu.skip_weights = download_f32(&self.skip_weights)?;
        cpu.ve_scale = download_f32(&self.ve_scale_param)?[0];
        if cpu.config.ve_enabled {
            cpu.ve_embed = download_f32(&self.ve_embed)?;
            cpu.ve_proj = download_f32(&self.ve_proj)?;
            cpu.ve_layer_scales = download_f32(&self.ve_layer_scales)?;
        } else {
            cpu.ve_embed.fill(0.0);
            cpu.ve_proj.fill(0.0);
            cpu.ve_layer_scales.clear();
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn f32_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for &value in data {
        let bits = value.to_bits();
        // Round-to-nearest-even before truncating to BF16.
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        let bf16 = (rounded >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

#[cfg(feature = "cuda")]
fn pack_qkv_bank_host(
    qo_bank: &[f32],
    kv_bank: &[f32],
    layers: usize,
    d: usize,
    kv: usize,
) -> Vec<f32> {
    let qkv = d + 2 * kv;
    let mut out = vec![0.0f32; layers * qkv * d];
    for layer in 0..layers {
        let dst_layer = layer * qkv * d;
        let q_src = layer * d * d;
        out[dst_layer..dst_layer + d * d].copy_from_slice(&qo_bank[q_src..q_src + d * d]);

        let k_src = layer * kv * d;
        let k_dst = dst_layer + d * d;
        out[k_dst..k_dst + kv * d].copy_from_slice(&kv_bank[k_src..k_src + kv * d]);

        let v_src = (layers + layer) * kv * d;
        let v_dst = dst_layer + (d + kv) * d;
        out[v_dst..v_dst + kv * d].copy_from_slice(&kv_bank[v_src..v_src + kv * d]);
    }
    out
}

/// GPU activation buffers — pre-allocated from pool.
#[cfg(feature = "cuda")]
pub struct GpuActivations {
    pub x: GpuTensor,
    pub x_in: GpuTensor,
    pub x0: GpuTensor,
    pub attn_norm: GpuTensor,
    pub mlp_norm: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub qkv_out: GpuTensor,
    pub qkv_aux_bf16: GpuTensor,
    pub ve_out: GpuTensor,
    pub ve_embed_out: GpuTensor,
    pub attn_out: GpuTensor,
    pub xsa_out: GpuTensor,
    pub attn_gated: GpuTensor,
    pub attn_gate_values: GpuTensor,
    pub attn_gate_grad_input: GpuTensor,
    pub attn_gate_grad_input_compact: GpuTensor,
    pub proj_out: GpuTensor,
    pub mlp_up: GpuTensor,
    pub mlp_act: GpuTensor,
    pub mlp_out: GpuTensor,
    pub bigram_out: GpuTensor,
    pub bigram_proj_out: GpuTensor,
    pub lora_tmp: GpuTensor,
    pub lora_delta: GpuTensor,
    pub lora_grad_tmp: GpuTensor,
    pub x_in_bf16: GpuTensor,
    pub x_aux_bf16: GpuTensor,
    pub wide_bf16: GpuTensor,
    pub logits: GpuTensor,

    pub encoder_skips: Vec<GpuTensor>,
}

#[cfg(feature = "cuda")]
impl GpuActivations {
    pub fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        Self::new_with_materialized_logits(config, tokens, stream, true, DType::F32)
    }

    fn new_with_materialized_logits(
        config: &ModelConfig,
        tokens: usize,
        stream: Arc<CudaStream>,
        materialize_logits: bool,
        logits_dtype: DType,
    ) -> PgResult<Self> {
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        let vocab = config.vocab_size;
        let bigram_dim = config.bigram_dim.max(1);
        let ve_dim = config.ve_dim.max(1);
        let compact_gate_width = config
            .attn_out_gate_width
            .max(config.sparse_attn_gate_width)
            .max(1);

        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        let mut encoder_skips = Vec::with_capacity(config.num_encoder_layers());
        for _ in 0..config.num_encoder_layers() {
            encoder_skips.push(zeros(&[tokens, d])?);
        }

        Ok(Self {
            x: zeros(&[tokens, d])?,
            x_in: zeros(&[tokens, d])?,
            x0: zeros(&[tokens, d])?,
            attn_norm: zeros(&[tokens, d])?,
            mlp_norm: zeros(&[tokens, d])?,
            q: zeros(&[tokens, config.num_heads, config.head_dim])?,
            k: zeros(&[tokens, config.num_kv_heads, config.head_dim])?,
            v: zeros(&[tokens, config.num_kv_heads, config.head_dim])?,
            qkv_out: zeros(&[tokens, d + 2 * kv])?,
            qkv_aux_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d + 2 * kv], DType::BF16)?,
            ve_out: zeros(&[tokens, kv])?,
            ve_embed_out: zeros(&[tokens, ve_dim])?,
            attn_out: zeros(&[tokens, config.num_heads, config.head_dim])?,
            xsa_out: zeros(&[tokens, config.num_heads, config.head_dim])?,
            attn_gated: zeros(&[tokens, config.num_heads, config.head_dim])?,
            attn_gate_values: zeros(&[tokens, config.num_heads])?,
            attn_gate_grad_input: zeros(&[tokens, d])?,
            attn_gate_grad_input_compact: zeros(&[tokens, compact_gate_width])?,
            proj_out: zeros(&[tokens, d])?,
            mlp_up: zeros(&[tokens, mlp])?,
            mlp_act: zeros(&[tokens, mlp])?,
            mlp_out: zeros(&[tokens, d])?,
            bigram_out: zeros(&[tokens, bigram_dim])?,
            bigram_proj_out: zeros(&[tokens, d])?,
            lora_tmp: zeros(&[tokens, d])?,
            lora_delta: zeros(&[tokens, d])?,
            lora_grad_tmp: zeros(&[tokens, d])?,
            x_in_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?,
            x_aux_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?,
            wide_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, mlp], DType::BF16)?,
            logits: if materialize_logits {
                GpuTensor::zeros_gpu(stream.clone(), &[tokens, vocab], logits_dtype)?
            } else {
                zeros(&[1])?
            },
            encoder_skips,
        })
    }

    pub fn new_for_plan(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        Self::new_for_plan_with_lm_head_lora(plan, tokens, stream, false)
    }

    pub fn new_for_plan_for_ttt(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        Self::new_for_plan_with_lm_head_lora(
            plan,
            tokens,
            stream,
            plan.eval_plan.ttt_lora_targets.lm_head,
        )
    }

    fn new_for_plan_with_lm_head_lora(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
        lm_head_lora: bool,
    ) -> PgResult<Self> {
        let config = plan.run_spec.model.to_model_config();
        let materialize_logits = lm_head_lora
            || matches!(
                selected_output_ce_backend_for_config(
                    &config,
                    plan.run_spec.model.compute_precision,
                    Some(plan.run_spec.model.output_ce_backend),
                ),
                GpuOutputCeBackend::FullLogits
            );
        let logits_dtype = if materialize_logits
            && gpu_bf16_logits_eligible_for_config(&config, plan.run_spec.model.compute_precision)
        {
            DType::BF16
        } else {
            DType::F32
        };
        Self::new_with_materialized_logits(
            &config,
            tokens,
            stream,
            materialize_logits,
            logits_dtype,
        )
    }
}

/// GPU gradient buffers matching the CPU `GradBuffers` parameter layout.
///
/// These buffers are preallocated once per runtime and filled by the CUDA
/// backward path. Record-shaped modes must reuse them rather than allocating
/// gradient storage inside the steady-state step.
#[cfg(feature = "cuda")]
pub struct GpuGradBuffers {
    pub tok_emb: GpuTensor,
    pub bigram_embed: GpuTensor,
    pub bigram_proj: GpuTensor,
    pub bigram_scale: GpuTensor,
    pub smear_gate: GpuTensor,
    pub skip_weights: GpuTensor,
    pub qo_bank: GpuTensor,
    pub kv_bank: GpuTensor,
    pub mlp_up_bank: GpuTensor,
    pub mlp_down_bank: GpuTensor,
    pub block_attn_scale: Vec<GpuTensor>,
    pub block_mlp_scale: Vec<GpuTensor>,
    pub block_resid_mix: Vec<GpuTensor>,
    pub block_q_gain: Vec<GpuTensor>,
    pub block_attn_gate_weight: Vec<GpuTensor>,
    pub block_attn_gate_bias: Vec<GpuTensor>,
    pub block_sparse_attn_gate_weight: Vec<GpuTensor>,
    pub ve_embed: GpuTensor,
    pub ve_proj: GpuTensor,
    pub ve_scale: GpuTensor,
    pub ve_layer_scales: GpuTensor,
}

#[cfg(feature = "cuda")]
pub trait GpuBackwardLayerObserver {
    fn after_layer(
        &mut self,
        model: &GpuModel,
        layer: usize,
        grads: &GpuGradBuffers,
    ) -> PgResult<()>;
}

#[cfg(feature = "cuda")]
impl GpuGradBuffers {
    pub fn new(config: &ModelConfig, stream: Arc<CudaStream>) -> PgResult<Self> {
        let n = config.num_layers;
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        Ok(Self {
            tok_emb: zeros(&[config.vocab_size, d])?,
            bigram_embed: zeros(&[config.bigram_vocab_size, config.bigram_dim.max(1)])?,
            bigram_proj: zeros(&[d, config.bigram_dim.max(1)])?,
            bigram_scale: zeros(&[1])?,
            smear_gate: zeros(&[d])?,
            skip_weights: zeros(&[config.num_skip_weights(), d])?,
            qo_bank: zeros(&[2 * n, d, d])?,
            kv_bank: zeros(&[2 * n, kv, d])?,
            mlp_up_bank: zeros(&[n, mlp, d])?,
            mlp_down_bank: zeros(&[n, d, mlp])?,
            block_attn_scale: (0..n).map(|_| zeros(&[d])).collect::<PgResult<_>>()?,
            block_mlp_scale: (0..n).map(|_| zeros(&[d])).collect::<PgResult<_>>()?,
            block_resid_mix: (0..n).map(|_| zeros(&[2, d])).collect::<PgResult<_>>()?,
            block_q_gain: (0..n)
                .map(|_| zeros(&[config.num_heads]))
                .collect::<PgResult<_>>()?,
            block_attn_gate_weight: (0..n)
                .map(|_| zeros(&[config.num_heads, config.attn_out_gate_width.max(1)]))
                .collect::<PgResult<_>>()?,
            block_attn_gate_bias: (0..n)
                .map(|_| zeros(&[config.num_heads]))
                .collect::<PgResult<_>>()?,
            block_sparse_attn_gate_weight: (0..n)
                .map(|_| zeros(&[config.num_heads, config.sparse_attn_gate_width.max(1)]))
                .collect::<PgResult<_>>()?,
            ve_embed: if config.ve_enabled {
                zeros(&[config.vocab_size, config.ve_dim.max(1)])?
            } else {
                zeros(&[1])?
            },
            ve_proj: if config.ve_enabled {
                zeros(&[kv, config.ve_dim.max(1)])?
            } else {
                zeros(&[1])?
            },
            ve_scale: zeros(&[1])?,
            ve_layer_scales: if config.ve_enabled {
                zeros(&[config.ve_layers.len().max(1)])?
            } else {
                zeros(&[1])?
            },
        })
    }

    pub fn zero(&self, kernels: &pg_kernels::gpu_kernels::GpuKernels) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let zero = |tensor: &GpuTensor| {
            kernels.scale_inplace(
                CudaPtr(tensor.cu_ptr(kernels.stream())?),
                0.0,
                tensor.numel() as u32,
            )
        };

        zero(&self.tok_emb)?;
        zero(&self.bigram_embed)?;
        zero(&self.bigram_proj)?;
        zero(&self.bigram_scale)?;
        zero(&self.smear_gate)?;
        zero(&self.skip_weights)?;
        zero(&self.qo_bank)?;
        zero(&self.kv_bank)?;
        zero(&self.mlp_up_bank)?;
        zero(&self.mlp_down_bank)?;
        for tensor in &self.block_attn_scale {
            zero(tensor)?;
        }
        for tensor in &self.block_mlp_scale {
            zero(tensor)?;
        }
        for tensor in &self.block_resid_mix {
            zero(tensor)?;
        }
        for tensor in &self.block_q_gain {
            zero(tensor)?;
        }
        for tensor in &self.block_attn_gate_weight {
            zero(tensor)?;
        }
        for tensor in &self.block_attn_gate_bias {
            zero(tensor)?;
        }
        for tensor in &self.block_sparse_attn_gate_weight {
            zero(tensor)?;
        }
        zero(&self.ve_embed)?;
        zero(&self.ve_proj)?;
        zero(&self.ve_scale)?;
        zero(&self.ve_layer_scales)?;
        Ok(())
    }
}

/// Saved GPU forward boundary states required for correctness-first backward.
///
/// This mirrors the CPU `ForwardCache` at the layer-boundary level while still
/// allowing block-internal activations to be recomputed later.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bf16QkvProducer {
    None,
    FusedNormQkvRopeGain,
    ExplicitPackAfterF32Qkv,
}

#[cfg(feature = "cuda")]
struct Bf16QkvFreshness {
    valid: Cell<bool>,
    producer: Cell<Bf16QkvProducer>,
    step: Cell<u64>,
    layer: Cell<usize>,
    tokens: Cell<usize>,
    seq_len: Cell<usize>,
    num_heads: Cell<usize>,
    num_kv_heads: Cell<usize>,
    head_dim: Cell<usize>,
}

#[cfg(feature = "cuda")]
impl Bf16QkvFreshness {
    fn new() -> Self {
        Self {
            valid: Cell::new(false),
            producer: Cell::new(Bf16QkvProducer::None),
            step: Cell::new(0),
            layer: Cell::new(usize::MAX),
            tokens: Cell::new(0),
            seq_len: Cell::new(0),
            num_heads: Cell::new(0),
            num_kv_heads: Cell::new(0),
            head_dim: Cell::new(0),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mark(
        &self,
        producer: Bf16QkvProducer,
        step: u64,
        layer: usize,
        tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) {
        self.producer.set(producer);
        self.step.set(step);
        self.layer.set(layer);
        self.tokens.set(tokens);
        self.seq_len.set(seq_len);
        self.num_heads.set(num_heads);
        self.num_kv_heads.set(num_kv_heads);
        self.head_dim.set(head_dim);
        self.valid.set(true);
    }

    #[allow(clippy::too_many_arguments)]
    fn require(
        &self,
        producer: Bf16QkvProducer,
        step: u64,
        layer: usize,
        tokens: usize,
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> PgResult<()> {
        if self.valid.get()
            && self.producer.get() == producer
            && self.step.get() == step
            && self.layer.get() == layer
            && self.tokens.get() == tokens
            && self.seq_len.get() == seq_len
            && self.num_heads.get() == num_heads
            && self.num_kv_heads.get() == num_kv_heads
            && self.head_dim.get() == head_dim
        {
            return Ok(());
        }
        Err(PgError::InvalidOp(format!(
            "stale prepacked BF16 QKV for layer {layer}: valid={}, producer={:?}, step={}, layer={}, tokens={}, seq_len={}, heads={}, kv_heads={}, head_dim={}; expected producer={:?}, step={step}, tokens={tokens}, seq_len={seq_len}, heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}",
            self.valid.get(),
            self.producer.get(),
            self.step.get(),
            self.layer.get(),
            self.tokens.get(),
            self.seq_len.get(),
            self.num_heads.get(),
            self.num_kv_heads.get(),
            self.head_dim.get(),
            producer,
        )))
    }
}

#[cfg(feature = "cuda")]
pub struct GpuForwardCache {
    pub layer_x: Vec<GpuTensor>,
    pub x0: GpuTensor,
    pub x_final: GpuTensor,
    pub skips: Vec<GpuTensor>,
    pub x_post_embed: GpuTensor,
    pub x_post_norm: GpuTensor,
    saved_layers: Vec<Option<GpuLayerForwardCache>>,
    recurrent_pass1_layers: Vec<Option<GpuLayerForwardCache>>,
    recurrent_mid_x: Vec<Option<GpuTensor>>,
    forward_generation: Cell<u64>,
}

#[cfg(feature = "cuda")]
struct GpuLayerForwardCache {
    lean_bf16_direct: bool,
    x_in: GpuTensor,
    attn_norm: GpuTensor,
    attn_norm_bf16: GpuTensor,
    q_pre_norm: GpuTensor,
    k_pre_norm: GpuTensor,
    q_post_rope: GpuTensor,
    q: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    q_bhsd_bf16: GpuTensor,
    k_bhsd_bf16: GpuTensor,
    v_bhsd_bf16: GpuTensor,
    bf16_qkv_freshness: Bf16QkvFreshness,
    ve_embed_out: GpuTensor,
    ve_out: GpuTensor,
    attn_out: GpuTensor,
    attn_out_bhsd_bf16: GpuTensor,
    attn_stats: GpuTensor,
    xsa_out: GpuTensor,
    attn_gated: GpuTensor,
    attn_gate_values: GpuTensor,
    attn_weight_input_bf16: GpuTensor,
    proj_out: GpuTensor,
    proj_out_bf16: GpuTensor,
    x_after_attn: GpuTensor,
    mlp_norm: GpuTensor,
    mlp_norm_bf16: GpuTensor,
    mlp_up: GpuTensor,
    mlp_up_bf16: GpuTensor,
    mlp_act: GpuTensor,
    mlp_act_bf16: GpuTensor,
    mlp_out: GpuTensor,
    mlp_out_bf16: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuLayerForwardCache {
    fn new_with_options(
        config: &ModelConfig,
        layer: usize,
        tokens: usize,
        stream: Arc<CudaStream>,
        lean_bf16_direct: bool,
    ) -> PgResult<Self> {
        let d = config.model_dim;
        let h = config.num_heads;
        let hkv = config.num_kv_heads;
        let hd = config.head_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        let ve_dim = config.ve_dim.max(1);
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);
        let maybe_f32 = |save: bool, shape: &[usize]| {
            if save {
                GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32)
            } else {
                GpuTensor::zeros_gpu(stream.clone(), &[1], DType::F32)
            }
        };
        let needs_ve_cache = !config.ve_layers.is_empty();
        let needs_gate_cache = config.attn_out_gate_enabled || config.sparse_attn_gate_enabled;
        let save_f32_attention = !lean_bf16_direct;
        let save_f32_xsa = !lean_bf16_direct || needs_gate_cache;
        let save_f32_gates = needs_gate_cache;
        let save_f32_bf16_gemm_inputs = !lean_bf16_direct;
        let save_attn_proj_bf16 =
            lean_bf16_direct && gpu_bf16_attention_projection_output_env_enabled();
        let save_x_in = !lean_bf16_direct
            || !gpu_recompute_residual_mix_norm_inputs_enabled()
            || gpu_bf16_norm_grad_path_env_enabled();
        let save_x_after_attn =
            !lean_bf16_direct || !config.parallel_residual_enabled_for_layer(layer);
        let save_ve_cache = !lean_bf16_direct || needs_ve_cache;
        Ok(Self {
            lean_bf16_direct: lean_bf16_direct,
            x_in: maybe_f32(save_x_in, &[tokens, d])?,
            attn_norm: maybe_f32(save_f32_bf16_gemm_inputs || needs_gate_cache, &[tokens, d])?,
            attn_norm_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?,
            q_pre_norm: zeros(&[tokens, h, hd])?,
            k_pre_norm: zeros(&[tokens, hkv, hd])?,
            q_post_rope: zeros(&[tokens, h, hd])?,
            q: maybe_f32(save_f32_attention, &[tokens, h, hd])?,
            k: maybe_f32(save_f32_attention, &[tokens, hkv, hd])?,
            v: maybe_f32(save_f32_attention, &[tokens, hkv, hd])?,
            q_bhsd_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, h, hd], DType::BF16)?,
            k_bhsd_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, hkv, hd], DType::BF16)?,
            v_bhsd_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, hkv, hd], DType::BF16)?,
            bf16_qkv_freshness: Bf16QkvFreshness::new(),
            ve_embed_out: maybe_f32(save_ve_cache, &[tokens, ve_dim])?,
            ve_out: maybe_f32(save_ve_cache, &[tokens, kv])?,
            attn_out: maybe_f32(save_f32_attention || needs_gate_cache, &[tokens, h, hd])?,
            attn_out_bhsd_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens, h, hd],
                DType::BF16,
            )?,
            attn_stats: zeros(&[tokens, h])?,
            xsa_out: maybe_f32(save_f32_xsa, &[tokens, h, hd])?,
            attn_gated: maybe_f32(save_f32_gates, &[tokens, h, hd])?,
            attn_gate_values: maybe_f32(save_f32_gates, &[tokens, h])?,
            attn_weight_input_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens, d],
                DType::BF16,
            )?,
            proj_out: zeros(&[tokens, d])?,
            proj_out_bf16: if save_attn_proj_bf16 {
                GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?
            } else {
                GpuTensor::zeros_gpu(stream.clone(), &[1], DType::BF16)?
            },
            x_after_attn: maybe_f32(save_x_after_attn, &[tokens, d])?,
            mlp_norm: maybe_f32(save_f32_bf16_gemm_inputs, &[tokens, d])?,
            mlp_norm_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?,
            mlp_up: maybe_f32(!lean_bf16_direct, &[tokens, mlp])?,
            mlp_up_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, mlp], DType::BF16)?,
            mlp_act: maybe_f32(!lean_bf16_direct, &[tokens, mlp])?,
            mlp_act_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, mlp], DType::BF16)?,
            mlp_out: zeros(&[tokens, d])?,
            mlp_out_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?,
        })
    }
}

#[cfg(feature = "cuda")]
impl GpuForwardCache {
    pub fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        Self::new_with_saved_layer_mask(config, tokens, stream, None)
    }

    fn new_with_saved_layer_mask(
        config: &ModelConfig,
        tokens: usize,
        stream: Arc<CudaStream>,
        save_layer_mask: Option<Vec<bool>>,
    ) -> PgResult<Self> {
        Self::new_with_saved_layer_mask_and_options(config, tokens, stream, save_layer_mask, false)
    }

    fn new_with_saved_layer_mask_and_options(
        config: &ModelConfig,
        tokens: usize,
        stream: Arc<CudaStream>,
        save_layer_mask: Option<Vec<bool>>,
        lean_bf16_direct_layers: bool,
    ) -> PgResult<Self> {
        let d = config.model_dim;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);
        let save_mask = if let Some(mask) = save_layer_mask {
            if mask.len() != config.num_layers {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "saved layer mask has {} entries for {} layers",
                    mask.len(),
                    config.num_layers
                )));
            }
            mask
        } else {
            vec![false; config.num_layers]
        };

        let saved_layers = save_mask
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, save)| {
                if save {
                    GpuLayerForwardCache::new_with_options(
                        config,
                        layer,
                        tokens,
                        stream.clone(),
                        lean_bf16_direct_layers,
                    )
                    .map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<PgResult<Vec<_>>>()?;

        let recurrent_pass1_layers = save_mask
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, save)| {
                if save && config.is_recurrent_layer(layer) {
                    GpuLayerForwardCache::new_with_options(
                        config,
                        layer,
                        tokens,
                        stream.clone(),
                        lean_bf16_direct_layers,
                    )
                    .map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<PgResult<Vec<_>>>()?;

        let recurrent_mid_x = save_mask
            .iter()
            .copied()
            .enumerate()
            .map(|(layer, save)| {
                if save && config.is_recurrent_layer(layer) {
                    zeros(&[tokens, d]).map(Some)
                } else {
                    Ok(None)
                }
            })
            .collect::<PgResult<Vec<_>>>()?;

        Ok(Self {
            layer_x: (0..config.num_layers)
                .map(|_| zeros(&[tokens, d]))
                .collect::<PgResult<_>>()?,
            x0: zeros(&[tokens, d])?,
            x_final: zeros(&[tokens, d])?,
            skips: (0..config.num_encoder_layers())
                .map(|_| zeros(&[tokens, d]))
                .collect::<PgResult<_>>()?,
            x_post_embed: zeros(&[tokens, d])?,
            x_post_norm: zeros(&[tokens, d])?,
            saved_layers,
            recurrent_pass1_layers,
            recurrent_mid_x,
            forward_generation: Cell::new(0),
        })
    }

    pub fn new_for_plan(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        let config = plan.run_spec.model.to_model_config();
        let save_layer_mask = if matches!(
            plan.run_spec.train.backend,
            crate::TrainBackend::CudaSingle | crate::TrainBackend::CudaDistributed
        ) {
            gpu_saved_layer_mask(&config)
        } else {
            None
        };
        let lean_bf16_direct_layers = gpu_lean_bf16_saved_layer_cache_enabled(
            &config,
            plan.run_spec.model.attention_backend,
            plan.run_spec.model.compute_precision,
        );
        Self::new_with_saved_layer_mask_and_options(
            &config,
            tokens,
            stream,
            save_layer_mask,
            lean_bf16_direct_layers,
        )
    }
}

#[cfg(feature = "cuda")]
impl GpuForwardCache {
    fn begin_forward_generation(&self) -> u64 {
        let next = self.forward_generation.get().wrapping_add(1);
        self.forward_generation.set(next);
        next
    }
}

#[cfg(feature = "cuda")]
fn gpu_saved_layer_mask(config: &ModelConfig) -> Option<Vec<bool>> {
    let raw = std::env::var("PG_GPU_SAVE_LAYER_ACTS").ok()?;
    gpu_saved_layer_mask_for_mode(config, &raw)
}

#[cfg(feature = "cuda")]
fn gpu_lean_bf16_saved_layer_cache_enabled(
    config: &ModelConfig,
    attention_backend: AttentionBackend,
    compute_precision: ModelComputePrecision,
) -> bool {
    attention_backend == AttentionBackend::CudnnSdpaBf16
        && compute_precision == ModelComputePrecision::Bf16TensorCore
        && gpu_direct_saved_activations_enabled()
        && config.xsa_last_n >= config.num_layers
        && !matches!(
            std::env::var("PG_GPU_BF16_PRIMARY_FORWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && !matches!(
            std::env::var("PG_GPU_BF16_BACKWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && !matches!(
            std::env::var("PG_GPU_LEAN_BF16_SAVED_ACTS")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && !gpu_q_lora_full_f32_saved_acts_enabled()
}

#[cfg(feature = "cuda")]
fn gpu_env_enabled(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

#[cfg(feature = "cuda")]
fn gpu_q_lora_full_f32_saved_acts_enabled() -> bool {
    gpu_env_enabled("PG_GPU_Q_LORA_FULL_F32_SAVED_ACTS", false)
}

#[cfg(feature = "cuda")]
fn gpu_shape_trace_enabled() -> bool {
    gpu_env_enabled("PG_GPU_SHAPE_TRACE", false)
}

#[cfg(feature = "cuda")]
fn gpu_cuda_graph_capture_debug_enabled() -> bool {
    gpu_env_enabled("PG_CUDA_GRAPH_CAPTURE_DEBUG", false)
}

#[cfg(feature = "cuda")]
fn gpu_cuda_graph_disable_cudnn_sdpa_enabled() -> bool {
    gpu_env_enabled("PG_CUDA_GRAPH_DISABLE_CUDNN_SDPA", false)
}

#[cfg(feature = "cuda")]
fn gpu_bigram_embedding_merge_enabled() -> bool {
    gpu_env_enabled("PG_GPU_BIGRAM_EMBED_MERGE", false)
}

#[cfg(feature = "cuda")]
fn check_cuda_graph_capture_stage(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    stage: &str,
) -> PgResult<()> {
    if !gpu_cuda_graph_capture_debug_enabled() || !gpu_cuda_backward_graph_enabled() {
        return Ok(());
    }
    let status = stream.capture_status().map_err(|e| {
        PgError::InvalidOp(format!(
            "cuda graph capture status query failed after {stage}: {e:?}"
        ))
    })?;
    if status == cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED {
        return Err(PgError::InvalidOp(format!(
            "cuda graph capture invalidated after {stage}"
        )));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn gpu_overwrite_bank_grads_enabled() -> bool {
    gpu_env_enabled("PG_GPU_OVERWRITE_BANK_GRADS", false)
}

#[cfg(feature = "cuda")]
fn gpu_lean_forward_cache_enabled() -> bool {
    gpu_env_enabled("PG_GPU_LEAN_FORWARD_CACHE", false)
}

#[cfg(feature = "cuda")]
fn gpu_bank_grad_dw_beta(first_layer_contribution: bool) -> f32 {
    if gpu_overwrite_bank_grads_enabled() && first_layer_contribution {
        0.0
    } else {
        1.0
    }
}

#[cfg(feature = "cuda")]
fn gpu_skip_recurrent_pass1_bank_grads_enabled() -> bool {
    gpu_env_enabled("PG_GPU_SKIP_RECURRENT_PASS1_BANK_GRADS", false)
}

#[cfg(feature = "cuda")]
fn gpu_skip_recurrent_bank_grads_enabled() -> bool {
    gpu_env_enabled("PG_GPU_SKIP_RECURRENT_BANK_GRADS", false)
}

#[cfg(feature = "cuda")]
fn gpu_recurrent_pass1_straight_through_enabled() -> bool {
    gpu_env_enabled("PG_GPU_RECURRENT_PASS1_STRAIGHT_THROUGH", false)
}

#[cfg(feature = "cuda")]
fn gpu_recurrent_all_straight_through_enabled() -> bool {
    gpu_env_enabled("PG_GPU_RECURRENT_ALL_STRAIGHT_THROUGH", false)
}

#[cfg(feature = "cuda")]
fn gpu_recurrent_straight_through_layers() -> usize {
    std::env::var("PG_GPU_RECURRENT_STRAIGHT_THROUGH_LAYERS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(0)
}

#[cfg(feature = "cuda")]
fn gpu_skip_bank_dw(dw_beta: f32) -> bool {
    dw_beta.is_nan()
}

#[cfg(feature = "cuda")]
fn gpu_bf16_backward_chain_requested() -> bool {
    gpu_env_enabled("PG_GPU_BF16_BACKWARD_CHAIN", false)
}

#[cfg(feature = "cuda")]
fn gpu_bf16_backward_chain_strict() -> bool {
    gpu_bf16_backward_chain_requested()
        && gpu_env_enabled("PG_GPU_BF16_BACKWARD_CHAIN_STRICT", true)
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bf16BackwardChainQkvNormResidReducer {
    DirectCompact,
    SplitCompact,
    ChunkedCompact,
}

#[cfg(feature = "cuda")]
fn gpu_bf16_backward_chain_qkv_norm_resid_reducer() -> Bf16BackwardChainQkvNormResidReducer {
    if let Ok(raw) = std::env::var("PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER") {
        match raw.to_ascii_lowercase().as_str() {
            "direct" | "direct_compact" | "one_pass" | "one_pass_compact" => {
                return Bf16BackwardChainQkvNormResidReducer::DirectCompact;
            }
            "split" | "split_compact" | "split_reduce" => {
                return Bf16BackwardChainQkvNormResidReducer::SplitCompact;
            }
            "chunked" | "chunked_compact" | "chunked_reduce" => {
                return Bf16BackwardChainQkvNormResidReducer::ChunkedCompact;
            }
            other => {
                eprintln!(
                    "warning: unknown PG_GPU_BF16_BACKWARD_CHAIN_QKV_NORM_RESID_REDUCER={other:?}; using direct_compact"
                );
            }
        }
    }
    if !gpu_bf16_backward_chain_requested() {
        return Bf16BackwardChainQkvNormResidReducer::DirectCompact;
    }
    if gpu_env_enabled("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", false) {
        Bf16BackwardChainQkvNormResidReducer::SplitCompact
    } else if gpu_env_enabled("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD", false) {
        Bf16BackwardChainQkvNormResidReducer::ChunkedCompact
    } else {
        // The split reducer is correctness-clean, but H100 record-shaped A/B
        // regressed qkv_norm_resid from ~19ms to ~47ms. The direct compact
        // reducer keeps the BF16 QKV tail and compact gate gradient without
        // the strided second-pass reduction, so it is the default chain target.
        Bf16BackwardChainQkvNormResidReducer::DirectCompact
    }
}

#[cfg(feature = "cuda")]
fn gpu_recompute_residual_mix_norm_inputs_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_RECOMPUTE_RESIDUAL_MIX_NORM_INPUTS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_bf16_norm_grad_path_env_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_BF16_NORM_GRAD_PATH")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_bf16_qkv_dx_output_env_enabled() -> bool {
    gpu_bf16_backward_chain_requested() || gpu_env_enabled("PG_GPU_BF16_QKV_DX_OUTPUT", false)
}

#[cfg(feature = "cuda")]
fn gpu_chunked_q_gain_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_CHUNKED_Q_GAIN_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_fused_qkv_rope_prepack_forward_enabled() -> bool {
    gpu_env_enabled("PG_GPU_FUSED_QKV_ROPE_PREPACK_FWD", false)
}

#[cfg(feature = "cuda")]
fn gpu_q_gain_backward_chunk_tokens() -> usize {
    std::env::var("PG_GPU_Q_GAIN_BWD_CHUNK_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        // The scratch arena is sized for the historical 256-token chunking.
        // Larger chunks reduce atomics/reduction buckets without requiring
        // more memory; smaller chunks need a larger scratch allocation.
        .filter(|tokens| *tokens >= 256)
        .unwrap_or(256)
}

#[cfg(feature = "cuda")]
fn gpu_combined_qkv_rope_tail_backward_enabled() -> bool {
    gpu_env_enabled("PG_GPU_COMBINED_QKV_ROPE_TAIL_BWD", false)
}

#[cfg(feature = "cuda")]
fn gpu_chunked_residual_mix_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_CHUNKED_RESIDUAL_MIX_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_skip_residual_mix_grad_enabled() -> bool {
    gpu_env_enabled("PG_GPU_SKIP_RESIDUAL_MIX_GRAD", false)
}

#[cfg(feature = "cuda")]
fn gpu_split_residual_mix_grad_enabled() -> bool {
    // Experimental only: v43 H100 record-shaped A/B regressed
    // qkv_norm_resid from ~19ms to ~47ms because the second-pass reduction is
    // memory-strided. Keep this opt-in until a coalesced reducer replaces it.
    matches!(
        std::env::var("PG_GPU_SPLIT_RESIDUAL_MIX_GRAD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_chunked_qkv_norm_resid_backward_enabled() -> bool {
    if gpu_bf16_backward_chain_requested() {
        return gpu_bf16_backward_chain_qkv_norm_resid_reducer()
            == Bf16BackwardChainQkvNormResidReducer::ChunkedCompact;
    }
    gpu_env_enabled("PG_GPU_CHUNKED_QKV_NORM_RESID_BWD", false)
}

#[cfg(feature = "cuda")]
fn gpu_split_qkv_norm_resid_backward_enabled() -> bool {
    if gpu_bf16_backward_chain_requested() {
        return gpu_bf16_backward_chain_qkv_norm_resid_reducer()
            == Bf16BackwardChainQkvNormResidReducer::SplitCompact;
    }
    gpu_env_enabled("PG_GPU_SPLIT_QKV_NORM_RESID_BWD", false)
}

#[cfg(feature = "cuda")]
fn gpu_qkv_norm_resid_backward_rows_per_chunk() -> usize {
    std::env::var("PG_GPU_QKV_NORM_RESID_BWD_ROWS_PER_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        // The shared scratch arena is sized for 256-row chunks. Larger chunks
        // are safe and reduce chunk count; smaller chunks need more scratch.
        .filter(|rows| *rows >= 256)
        .unwrap_or(1024)
}

#[cfg(feature = "cuda")]
fn gpu_residual_scale_reduce_enabled() -> bool {
    !matches!(
        std::env::var("PG_GPU_RESIDUAL_SCALE_REDUCE")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "cuda")]
fn gpu_chunked_residual_scale_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_CHUNKED_RESIDUAL_SCALE_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_residual_scale_backward_rows_per_chunk() -> usize {
    std::env::var("PG_GPU_RESIDUAL_SCALE_BWD_ROWS_PER_CHUNK")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|rows| *rows >= 64)
        .unwrap_or(256)
}

#[cfg(feature = "cuda")]
fn gpu_bf16_mlp_down_dx_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_BF16_MLP_DOWN_DX")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_compact_attn_gate_grad_input_enabled() -> bool {
    gpu_bf16_backward_chain_requested()
        || gpu_env_enabled("PG_GPU_COMPACT_ATTN_GATE_GRAD_INPUT", false)
}

#[cfg(feature = "cuda")]
fn gpu_overlap_linear_backward_gemms_enabled() -> bool {
    if gpu_cuda_backward_graph_enabled() && !gpu_graph_side_gemm_capture_enabled() {
        return false;
    }
    matches!(
        std::env::var("PG_GPU_OVERLAP_LINEAR_BWD_GEMMS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LinearBackwardOverlapRole {
    Generic,
    MlpDown,
    MlpUp,
    Qkv,
    AttnOut,
}

#[cfg(feature = "cuda")]
impl LinearBackwardOverlapRole {
    fn env_name(self) -> Option<&'static str> {
        match self {
            Self::Generic => None,
            Self::MlpDown => Some("PG_GPU_OVERLAP_MLP_DOWN_BWD_GEMMS"),
            Self::MlpUp => Some("PG_GPU_OVERLAP_MLP_UP_BWD_GEMMS"),
            Self::Qkv => Some("PG_GPU_OVERLAP_QKV_BWD_GEMMS"),
            Self::AttnOut => Some("PG_GPU_OVERLAP_ATTN_OUT_BWD_GEMMS"),
        }
    }
}

#[cfg(feature = "cuda")]
fn gpu_overlap_linear_backward_gemms_enabled_for_role(role: LinearBackwardOverlapRole) -> bool {
    if gpu_cuda_backward_graph_enabled() && !gpu_graph_side_gemm_capture_enabled() {
        return false;
    }
    if gpu_overlap_linear_backward_gemms_enabled() {
        return true;
    }
    let Some(name) = role.env_name() else {
        return false;
    };
    matches!(
        std::env::var(name)
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_defer_linear_backward_weight_gemms_enabled_for_role(
    role: LinearBackwardOverlapRole,
) -> bool {
    if gpu_cuda_backward_graph_enabled() && !gpu_graph_side_gemm_capture_enabled() {
        return false;
    }
    match role {
        LinearBackwardOverlapRole::MlpDown => {
            gpu_env_enabled("PG_GPU_DEFER_LINEAR_BACKWARD_WEIGHT_GEMMS", false)
                || gpu_env_enabled("PG_GPU_DEFER_MLP_DOWN_BWD_DW", false)
        }
        LinearBackwardOverlapRole::MlpUp => {
            gpu_env_enabled("PG_GPU_DEFER_LINEAR_BACKWARD_WEIGHT_GEMMS", false)
                || gpu_env_enabled("PG_GPU_DEFER_MLP_UP_BWD_DW", false)
        }
        LinearBackwardOverlapRole::Qkv => gpu_env_enabled("PG_GPU_DEFER_QKV_BWD_DW", false),
        LinearBackwardOverlapRole::AttnOut => {
            gpu_env_enabled("PG_GPU_DEFER_LINEAR_BACKWARD_WEIGHT_GEMMS", false)
                || gpu_env_enabled("PG_GPU_DEFER_ATTN_OUT_BWD_DW", false)
        }
        LinearBackwardOverlapRole::Generic => false,
    }
}

#[cfg(feature = "cuda")]
fn gpu_graph_side_gemm_capture_enabled() -> bool {
    gpu_cuda_backward_graph_enabled() && gpu_env_enabled("PG_GPU_GRAPH_SIDE_GEMM_CAPTURE", false)
}

#[cfg(feature = "cuda")]
fn gpu_cuda_backward_graph_enabled() -> bool {
    gpu_env_enabled("PG_CUDA_BACKWARD_GRAPH", false)
        && gpu_env_enabled("PG_CUDA_BACKWARD_GRAPH_STRICT", false)
}

#[cfg(feature = "cuda")]
fn gpu_any_overlap_linear_backward_gemms_enabled() -> bool {
    gpu_overlap_linear_backward_gemms_enabled()
        || [
            LinearBackwardOverlapRole::MlpDown,
            LinearBackwardOverlapRole::MlpUp,
            LinearBackwardOverlapRole::Qkv,
            LinearBackwardOverlapRole::AttnOut,
        ]
        .into_iter()
        .any(gpu_overlap_linear_backward_gemms_enabled_for_role)
        || [
            LinearBackwardOverlapRole::MlpDown,
            LinearBackwardOverlapRole::MlpUp,
            LinearBackwardOverlapRole::Qkv,
            LinearBackwardOverlapRole::AttnOut,
        ]
        .into_iter()
        .any(gpu_defer_linear_backward_weight_gemms_enabled_for_role)
}

#[cfg(feature = "cuda")]
fn gpu_vec2_mlp_act_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_VEC2_MLP_ACT_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_vec4_mlp_act_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_VEC4_MLP_ACT_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_fast_mlp_act_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_FAST_MLP_ACT_BWD")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_sparse_xsa_warphead_forward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SPARSE_XSA_WARPHEAD_FWD")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_sparse_xsa_warphead_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SPARSE_XSA_WARPHEAD_BWD")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_sparse_xsa_grouped_kv_backward_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_SPARSE_XSA_GROUPED_KV_BWD")
            .unwrap_or_else(|_| "0".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_bf16_attention_projection_output_env_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_BF16_ATTN_PROJ_OUTPUT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_saved_layer_mask_for_mode(config: &ModelConfig, raw: &str) -> Option<Vec<bool>> {
    let mode = raw.to_ascii_lowercase();
    if matches!(mode.as_str(), "0" | "false" | "no" | "off") {
        return None;
    }

    let mut mask = match mode.as_str() {
        // Save only non-checkpointed, non-recurrent edge blocks. This keeps the
        // default memory profile bounded while still removing edge recompute.
        "1" | "true" | "yes" | "on" | "checkpoint" | "edges" => checkpoint_layers(config)
            .into_iter()
            .enumerate()
            .map(|(layer, checkpoint)| !checkpoint && !config.is_recurrent_layer(layer))
            .collect::<Vec<_>>(),
        // Save only layers that execute twice. This is the lowest-memory mode
        // that targets the worst recompute multiplier in the recurrent stack.
        "recurrent" => (0..config.num_layers)
            .map(|layer| config.is_recurrent_layer(layer))
            .collect::<Vec<_>>(),
        // Save the layers the default checkpoint policy would otherwise
        // recompute, including recurrent layers. This is the practical
        // record-shaped timing mode before considering save-all.
        "inner" | "checkpointed" => checkpoint_layers(config),
        // Save every layer. Recurrent layers allocate an additional pass-1
        // activation cache plus the pass-1 output boundary so their two-pass
        // backward can avoid full block recompute.
        "all" => (0..config.num_layers).map(|_| true).collect::<Vec<_>>(),
        other => {
            eprintln!(
                "PG_GPU_SAVE_LAYER_ACTS={other:?} is not recognized; using no saved layer activations"
            );
            return None;
        }
    };

    // Avoid allocating empty saved-layer vectors for tiny or fully recurrent
    // configurations.
    if mask.iter().any(|&save| save) {
        Some(std::mem::take(&mut mask))
    } else {
        None
    }
}

#[cfg(feature = "cuda")]
fn gpu_backward_stage_timing_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_BACKWARD_STAGE_TIMING")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
fn gpu_direct_saved_activations_enabled() -> bool {
    matches!(
        std::env::var("PG_GPU_DIRECT_SAVED_ACTS")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBackwardStageTiming {
    pub forward_ms: f64,
    pub forward_embed_ms: f64,
    pub forward_encoder_ms: f64,
    pub forward_encoder_layer_max_ms: f64,
    pub forward_decoder_ms: f64,
    pub forward_decoder_layer_max_ms: f64,
    pub forward_logits_ms: f64,
    pub forward_block_pre_attn_ms: f64,
    pub forward_block_attention_ms: f64,
    pub forward_block_post_attn_ms: f64,
    pub forward_block_mlp_ms: f64,
    pub backward_block_recompute_ms: f64,
    pub backward_block_mlp_ms: f64,
    pub backward_block_mlp_residual_ms: f64,
    pub backward_block_mlp_down_ms: f64,
    pub backward_block_mlp_act_ms: f64,
    pub backward_block_mlp_up_ms: f64,
    pub backward_block_mlp_norm_ms: f64,
    pub backward_block_attn_out_ms: f64,
    pub backward_block_attn_out_residual_ms: f64,
    pub backward_block_attn_out_proj_ms: f64,
    pub backward_block_attn_out_gate_xsa_ms: f64,
    pub backward_block_attention_ms: f64,
    pub backward_block_attention_sdpa_ms: f64,
    pub backward_block_attention_xsa_accum_ms: f64,
    pub backward_block_qkv_ms: f64,
    pub backward_block_qkv_rope_ms: f64,
    pub backward_block_qkv_proj_ms: f64,
    pub backward_block_qkv_ve_ms: f64,
    pub backward_block_qkv_norm_resid_ms: f64,
    pub output_ms: f64,
    pub decoder_ms: f64,
    pub encoder_ms: f64,
    pub tail_ms: f64,
}

#[cfg(feature = "cuda")]
fn record_stage_event(stream: &Arc<CudaStream>) -> PgResult<Option<cudarc::driver::CudaEvent>> {
    record_stage_event_if(stream, true)
}

#[cfg(feature = "cuda")]
fn record_stage_event_if(
    stream: &Arc<CudaStream>,
    enabled: bool,
) -> PgResult<Option<cudarc::driver::CudaEvent>> {
    if !enabled || !gpu_backward_stage_timing_enabled() {
        return Ok(None);
    }
    stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map(Some)
        .map_err(|e| pg_core::PgError::InvalidOp(format!("cuda stage event record failed: {e:?}")))
}

#[cfg(feature = "cuda")]
fn finish_stage_event(
    stream: &Arc<CudaStream>,
    start: Option<cudarc::driver::CudaEvent>,
    slot: &mut f64,
) -> PgResult<()> {
    let Some(start) = start else {
        return Ok(());
    };
    let end = stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda stage event record failed: {e:?}"))
        })?;
    *slot += start.elapsed_ms(&end).map_err(|e| {
        pg_core::PgError::InvalidOp(format!("cuda stage event elapsed failed: {e:?}"))
    })? as f64;
    Ok(())
}

#[cfg(feature = "cuda")]
fn finish_stage_event_optional(
    stream: &Arc<CudaStream>,
    start: Option<cudarc::driver::CudaEvent>,
    slot: Option<&mut f64>,
) -> PgResult<()> {
    let Some(slot) = slot else {
        return Ok(());
    };
    finish_stage_event(stream, start, slot)
}

#[cfg(feature = "cuda")]
fn finish_stage_event_optional_max(
    stream: &Arc<CudaStream>,
    start: Option<cudarc::driver::CudaEvent>,
    slot: Option<&mut f64>,
) -> PgResult<()> {
    let Some(slot) = slot else {
        return Ok(());
    };
    let Some(start) = start else {
        return Ok(());
    };
    let end = stream
        .record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| {
            pg_core::PgError::InvalidOp(format!("cuda stage event record failed: {e:?}"))
        })?;
    let elapsed_ms = start.elapsed_ms(&end).map_err(|e| {
        pg_core::PgError::InvalidOp(format!("cuda stage event elapsed failed: {e:?}"))
    })? as f64;
    *slot = (*slot).max(elapsed_ms);
    Ok(())
}

/// Block-local recompute cache for GPU backward.
#[cfg(feature = "cuda")]
struct GpuBlockBackwardCache {
    q_pre_norm: GpuTensor,
    k_pre_norm: GpuTensor,
    q_post_rope: GpuTensor,
    attn_stats: GpuTensor,
    x_after_attn: GpuTensor,
    pass1_out: GpuTensor,
    grad_mid: GpuTensor,
    grad_x_after_attn: GpuTensor,
    grad_mlp_out: GpuTensor,
    grad_mlp_act: GpuTensor,
    grad_mlp_up: GpuTensor,
    grad_mlp_norm: GpuTensor,
    grad_x_pre_mlp_norm: GpuTensor,
    grad_x_in: GpuTensor,
    grad_proj_out: GpuTensor,
    grad_attn_result: GpuTensor,
    grad_raw: GpuTensor,
    grad_attn_out: GpuTensor,
    grad_v_xsa: GpuTensor,
    grad_q_post_gain: GpuTensor,
    grad_k_attn: GpuTensor,
    grad_v_projection: GpuTensor,
    grad_q_post_gain_bf16: GpuTensor,
    grad_k_attn_bf16: GpuTensor,
    grad_v_projection_bf16: GpuTensor,
    grad_q_post_rope: GpuTensor,
    grad_q_proj: GpuTensor,
    grad_k_proj: GpuTensor,
    grad_q_proj_bf16: GpuTensor,
    grad_k_proj_bf16: GpuTensor,
    q_gain_reduce_scratch: GpuTensor,
    residual_mix_reduce_scratch: GpuTensor,
    residual_mix_norm_stats: GpuTensor,
    grad_qkv_proj: GpuTensor,
    grad_attn_norm_bf16: GpuTensor,
    grad_qkv_weight: GpuTensor,
    grad_attn_norm: GpuTensor,
    grad_attn_norm_k: GpuTensor,
    grad_attn_norm_v: GpuTensor,
    grad_projected: GpuTensor,
    grad_ve_embed_out: GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuBlockBackwardCache {
    fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        let d = config.model_dim;
        let h = config.num_heads;
        let hkv = config.num_kv_heads;
        let hd = config.head_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;
        let ve_dim = config.ve_dim.max(1);
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);
        Ok(Self {
            q_pre_norm: zeros(&[tokens, h, hd])?,
            k_pre_norm: zeros(&[tokens, hkv, hd])?,
            q_post_rope: zeros(&[tokens, h, hd])?,
            attn_stats: zeros(&[tokens, h])?,
            x_after_attn: zeros(&[tokens, d])?,
            pass1_out: zeros(&[tokens, d])?,
            grad_mid: zeros(&[tokens, d])?,
            grad_x_after_attn: zeros(&[tokens, d])?,
            grad_mlp_out: zeros(&[tokens, d])?,
            grad_mlp_act: zeros(&[tokens, mlp])?,
            grad_mlp_up: zeros(&[tokens, mlp])?,
            grad_mlp_norm: zeros(&[tokens, d])?,
            grad_x_pre_mlp_norm: zeros(&[tokens, d])?,
            grad_x_in: zeros(&[tokens, d])?,
            grad_proj_out: zeros(&[tokens, d])?,
            grad_attn_result: zeros(&[tokens, h, hd])?,
            grad_raw: zeros(&[tokens, h, hd])?,
            grad_attn_out: zeros(&[tokens, h, hd])?,
            grad_v_xsa: zeros(&[tokens, hkv, hd])?,
            grad_q_post_gain: zeros(&[tokens, h, hd])?,
            grad_k_attn: zeros(&[tokens, hkv, hd])?,
            grad_v_projection: zeros(&[tokens, hkv, hd])?,
            grad_q_post_gain_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens, h, hd],
                DType::BF16,
            )?,
            grad_k_attn_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens, hkv, hd],
                DType::BF16,
            )?,
            grad_v_projection_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens, hkv, hd],
                DType::BF16,
            )?,
            grad_q_post_rope: zeros(&[tokens, h, hd])?,
            grad_q_proj: zeros(&[tokens, h, hd])?,
            grad_k_proj: zeros(&[tokens, hkv, hd])?,
            grad_q_proj_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, h, hd], DType::BF16)?,
            grad_k_proj_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[tokens, hkv, hd],
                DType::BF16,
            )?,
            q_gain_reduce_scratch: zeros(&[h, tokens.div_ceil(256)])?,
            residual_mix_reduce_scratch: zeros(&[2 * d, tokens.div_ceil(256)])?,
            residual_mix_norm_stats: zeros(&[tokens, 2])?,
            grad_qkv_proj: zeros(&[tokens, d + 2 * kv])?,
            grad_attn_norm_bf16: GpuTensor::zeros_gpu(stream.clone(), &[tokens, d], DType::BF16)?,
            grad_qkv_weight: zeros(&[d + 2 * kv, d])?,
            grad_attn_norm: zeros(&[tokens, d])?,
            grad_attn_norm_k: zeros(&[tokens, d])?,
            grad_attn_norm_v: zeros(&[tokens, d])?,
            grad_projected: zeros(&[tokens, kv])?,
            grad_ve_embed_out: zeros(&[tokens, ve_dim])?,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuOutputCeBackend {
    FullLogits,
    ChunkedBf16Cache,
    TiledRepeatedGemm,
    FusedExactWmma,
}

#[cfg(feature = "cuda")]
fn output_ce_backend_env_override() -> Option<OutputCeBackend> {
    let raw = std::env::var("PG_GPU_OUTPUT_CE_BACKEND").ok()?;
    match raw.to_ascii_lowercase().as_str() {
        "chunked_bf16_cache" | "chunked" | "cache" => Some(OutputCeBackend::ChunkedBf16Cache),
        "tiled_repeated_gemm" | "tiled" | "tiled_ce" => Some(OutputCeBackend::TiledRepeatedGemm),
        "fused_exact_wmma" | "fused_exact" | "fused" => Some(OutputCeBackend::FusedExactWmma),
        "auto" | "" => None,
        other => {
            eprintln!("warning: unknown PG_GPU_OUTPUT_CE_BACKEND={other:?}; ignoring");
            None
        }
    }
}

#[cfg(feature = "cuda")]
fn legacy_output_ce_backend_env() -> Option<OutputCeBackend> {
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
        std::env::var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    ) {
        return Some(OutputCeBackend::ChunkedBf16Cache);
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
    None
}

#[cfg(feature = "cuda")]
fn bf16_output_projection_enabled_env() -> bool {
    !matches!(
        std::env::var("PG_GPU_BF16_OUTPUT_GEMM")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "cuda")]
fn bf16_output_backward_enabled_env() -> bool {
    !matches!(
        std::env::var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM")
            .unwrap_or_else(|_| "1".to_string())
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "no" | "off"
    )
}

#[cfg(feature = "cuda")]
fn selected_output_ce_backend_for_config(
    config: &ModelConfig,
    compute_precision: ModelComputePrecision,
    spec_backend: Option<OutputCeBackend>,
) -> GpuOutputCeBackend {
    if compute_precision != ModelComputePrecision::Bf16TensorCore {
        return GpuOutputCeBackend::FullLogits;
    }
    if !bf16_output_projection_enabled_env() || !bf16_output_backward_enabled_env() {
        return GpuOutputCeBackend::FullLogits;
    }
    // Explicit runtime toggles are used for H100 A/B experiments and must be
    // able to override the record spec's conservative default. Prefer the new
    // single enum env, then legacy boolean envs, then the TOML default.
    let backend = output_ce_backend_env_override()
        .or_else(legacy_output_ce_backend_env)
        .or(spec_backend);
    match backend {
        Some(OutputCeBackend::ChunkedBf16Cache) => GpuOutputCeBackend::ChunkedBf16Cache,
        Some(OutputCeBackend::TiledRepeatedGemm)
            if config.vocab_size % output_ce_tile_vocab_for_config(config) == 0 =>
        {
            GpuOutputCeBackend::TiledRepeatedGemm
        }
        Some(OutputCeBackend::FusedExactWmma)
            if config.vocab_size % output_ce_tile_vocab_for_config(config) == 0 =>
        {
            GpuOutputCeBackend::FusedExactWmma
        }
        _ => GpuOutputCeBackend::FullLogits,
    }
}

#[cfg(feature = "cuda")]
fn output_ce_tile_vocab_for_config(config: &ModelConfig) -> usize {
    std::env::var("PG_GPU_OUTPUT_CE_TILE_VOCAB")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&tile| tile > 0 && tile <= config.vocab_size)
        .unwrap_or(512)
        .min(config.vocab_size)
}

#[cfg(feature = "cuda")]
fn output_ce_chunk_tokens_for_config(config: &ModelConfig, tokens: usize) -> usize {
    std::env::var("PG_GPU_OUTPUT_CE_CHUNK_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&chunk| chunk > 0)
        .unwrap_or(8192)
        .min(tokens)
        .max(1)
        .min(config.train_seq_len * 4)
}

#[cfg(feature = "cuda")]
fn gpu_bf16_logits_eligible_for_config(
    _config: &ModelConfig,
    compute_precision: ModelComputePrecision,
) -> bool {
    compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_LOGITS")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
}

/// Persistent backward recompute state reused across training steps.
#[cfg(feature = "cuda")]
pub struct GpuBackwardState {
    cache: GpuForwardCache,
    block_cache: GpuBlockBackwardCache,
    pub stage_timing: GpuBackwardStageTiming,
    losses: GpuTensor,
    loss_sum: GpuTensor,
    grad_logits: GpuTensor,
    grad_logits_bf16: GpuTensor,
    output_logits_tile: GpuTensor,
    output_grad_tile_bf16: GpuTensor,
    output_row_max: GpuTensor,
    output_row_sum: GpuTensor,
    output_target_logit: GpuTensor,
    grad_output_x: GpuTensor,
    grad_output_pre_norm: GpuTensor,
    grad_ping: GpuTensor,
    grad_pong: GpuTensor,
    grad_x0: GpuTensor,
    grad_encoder_skips: Vec<GpuTensor>,
    grad_x_post_skip: GpuTensor,
    grad_x_smear: GpuTensor,
    grad_x_prev: GpuTensor,
    grad_x_post_norm: GpuTensor,
    grad_x_post_embed: GpuTensor,
    grad_bigram_proj_out: GpuTensor,
    grad_bigram_out: GpuTensor,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct OutputCeTileScratch<'a> {
    logits_tile: &'a GpuTensor,
    grad_tile_bf16: &'a GpuTensor,
    row_max: &'a GpuTensor,
    row_sum: &'a GpuTensor,
    target_logit: &'a GpuTensor,
}

#[cfg(feature = "cuda")]
impl GpuBackwardState {
    pub fn new(config: &ModelConfig, tokens: usize, stream: Arc<CudaStream>) -> PgResult<Self> {
        let cache = GpuForwardCache::new(config, tokens, stream.clone())?;
        Self::new_with_cache(config, tokens, stream, cache, true, None)
    }

    fn new_with_cache(
        config: &ModelConfig,
        tokens: usize,
        stream: Arc<CudaStream>,
        cache: GpuForwardCache,
        materialize_logits: bool,
        output_ce_backend: Option<OutputCeBackend>,
    ) -> PgResult<Self> {
        let d = config.model_dim;
        let bigram_dim = config.bigram_dim.max(1);
        let output_tile_vocab = output_ce_tile_vocab_for_config(config);
        let selected_output_ce = selected_output_ce_backend_for_config(
            config,
            ModelComputePrecision::Bf16TensorCore,
            output_ce_backend,
        );
        let chunked_output_ce =
            !materialize_logits && selected_output_ce == GpuOutputCeBackend::ChunkedBf16Cache;
        let output_tile_tokens = if materialize_logits {
            1
        } else if chunked_output_ce {
            output_ce_chunk_tokens_for_config(config, tokens)
        } else {
            tokens
        };
        let output_tile_vocab_alloc = if materialize_logits {
            1
        } else if chunked_output_ce {
            config.vocab_size
        } else {
            output_tile_vocab
        };
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);
        Ok(Self {
            cache,
            block_cache: GpuBlockBackwardCache::new(config, tokens, stream.clone())?,
            stage_timing: GpuBackwardStageTiming::default(),
            losses: zeros(&[tokens])?,
            loss_sum: zeros(&[1])?,
            grad_logits: if materialize_logits {
                zeros(&[tokens, config.vocab_size])?
            } else {
                zeros(&[1])?
            },
            grad_logits_bf16: if materialize_logits {
                GpuTensor::zeros_gpu(stream.clone(), &[tokens, config.vocab_size], DType::BF16)?
            } else {
                GpuTensor::zeros_gpu(stream.clone(), &[1], DType::BF16)?
            },
            output_logits_tile: if chunked_output_ce {
                GpuTensor::zeros_gpu(
                    stream.clone(),
                    &[output_tile_tokens, output_tile_vocab_alloc],
                    DType::BF16,
                )?
            } else {
                zeros(&[output_tile_tokens, output_tile_vocab_alloc])?
            },
            output_grad_tile_bf16: GpuTensor::zeros_gpu(
                stream.clone(),
                &[output_tile_tokens, output_tile_vocab_alloc],
                DType::BF16,
            )?,
            output_row_max: zeros(&[output_tile_tokens])?,
            output_row_sum: zeros(&[output_tile_tokens])?,
            output_target_logit: zeros(&[output_tile_tokens])?,
            grad_output_x: zeros(&[tokens, d])?,
            grad_output_pre_norm: zeros(&[tokens, d])?,
            grad_ping: zeros(&[tokens, d])?,
            grad_pong: zeros(&[tokens, d])?,
            grad_x0: zeros(&[tokens, d])?,
            grad_encoder_skips: (0..config.num_skip_weights())
                .map(|_| zeros(&[tokens, d]))
                .collect::<PgResult<_>>()?,
            grad_x_post_skip: zeros(&[tokens, d])?,
            grad_x_smear: zeros(&[tokens, d])?,
            grad_x_prev: zeros(&[tokens, d])?,
            grad_x_post_norm: zeros(&[tokens, d])?,
            grad_x_post_embed: zeros(&[tokens, d])?,
            grad_bigram_proj_out: zeros(&[tokens, d])?,
            grad_bigram_out: zeros(&[tokens, bigram_dim])?,
        })
    }

    pub fn new_for_plan(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        Self::new_for_plan_with_lm_head_lora(plan, tokens, stream, false)
    }

    pub fn new_for_plan_for_ttt(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        Self::new_for_plan_with_lm_head_lora(
            plan,
            tokens,
            stream,
            plan.eval_plan.ttt_lora_targets.lm_head,
        )
    }

    fn new_for_plan_with_lm_head_lora(
        plan: &ExecutionPlan,
        tokens: usize,
        stream: Arc<CudaStream>,
        lm_head_lora: bool,
    ) -> PgResult<Self> {
        let config = plan.run_spec.model.to_model_config();
        let cache = GpuForwardCache::new_for_plan(plan, tokens, stream.clone())?;
        let materialize_logits = lm_head_lora
            || matches!(
                selected_output_ce_backend_for_config(
                    &config,
                    plan.run_spec.model.compute_precision,
                    Some(plan.run_spec.model.output_ce_backend),
                ),
                GpuOutputCeBackend::FullLogits
            );
        Self::new_with_cache(
            &config,
            tokens,
            stream,
            cache,
            materialize_logits,
            Some(plan.run_spec.model.output_ce_backend),
        )
    }
}

/// Training step phases on GPU.
///
/// Single step timeline on 8×H100:
///
/// ```text
/// |-- Data load (async memcpy) --|
/// |-- Forward pass (compute stream) --|
/// |-- Backward pass (compute stream) --|
/// |-- Phase 1: Async RS all banks (NCCL stream) --|
/// |-- Phase 2: AllReduce scalars + AdamW step (overlapped) --|
/// |-- Phase 3: Wait RS → NS5 → Async AG (per bank, pipelined) --|
/// |-- EMA update (compute stream) --|
/// ```
///
/// Key optimizations:
/// - Activation checkpointing layers 3-8 (saves ~40% memory)
/// - Production record gaps are explicit in `ModelSpec` / `TrainSpec`.
/// - Current distributed path prioritizes correctness over overlap.

/// Bank shapes for the Muon optimizer.
pub fn bank_shapes(config: &ModelConfig) -> Vec<[usize; 3]> {
    let n = config.num_layers;
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    vec![
        [2 * n, d, d],  // qo_bank
        [2 * n, kv, d], // kv_bank
        [n, mlp, d],    // mlp_up_bank
        [n, d, mlp],    // mlp_down_bank
    ]
}

/// Activation checkpointing config.
/// Layers 3-8 (0-indexed) recompute forward during backward.
/// Layers 0-2 and 9-10 save full activations.
pub fn checkpoint_layers(config: &ModelConfig) -> Vec<bool> {
    (0..config.num_layers).map(|i| i >= 3 && i <= 8).collect()
}

/// Estimate peak GPU memory for training (bytes).
pub fn estimate_memory(config: &ModelConfig, batch_tokens: usize) -> usize {
    let d = config.model_dim;
    let kv = config.kv_dim();
    let mlp = config.mlp_dim;
    let n = config.num_layers;
    let vocab = config.vocab_size;

    // Parameters (BF16 = 2 bytes)
    let param_bytes = (
        2 * n * d * d     // qo_bank
        + 2 * n * kv * d   // kv_bank
        + n * mlp * d       // mlp_up
        + n * d * mlp       // mlp_down
        + vocab * d
        // tok_emb
    ) * 2;

    // Gradients (same size as params)
    let grad_bytes = param_bytes;

    // Optimizer state: Muon momentum (same as banks) + AdamW m/v (2× scalars)
    let muon_state = param_bytes; // momentum buffer same size as banks
    let adamw_state = vocab * d * 2 * 2 * 4; // m + v for embeddings, F32

    // NS5 workspace: for each bank [B,M,N], need a_buf [B,rows,rows], aa, b_buf, new_x [B,rows,cols]
    // rows = min(M,N), cols = max(M,N)
    let ns5_workspace: usize = bank_shapes(config)
        .iter()
        .map(|s| {
            let (b, m, n_dim) = (s[0], s[1], s[2]);
            let (rows, cols) = if m > n_dim { (n_dim, m) } else { (m, n_dim) };
            (3 * b * rows * rows + b * rows * cols) * 2 // BF16
        })
        .sum();

    // Activations (BF16)
    let bt = batch_tokens;
    let act_per_layer = bt * d * 2 // x + attn_norm
        + bt * d * 2       // mlp_norm + proj_out
        + bt * config.num_heads as usize * config.head_dim * 2 // q, attn_out
        + bt * kv * 2       // k, v
        + bt * mlp * 2      // mlp_up, mlp_act
        + bt * d; // mlp_out
    let act_bytes = act_per_layer * 2 * 2; // BF16, keep ~2 layers live

    // Logits
    let logit_bytes = bt * vocab * 4; // F32 for numerical stability

    param_bytes + grad_bytes + muon_state + adamw_state + ns5_workspace + act_bytes + logit_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_estimate() {
        let config = ModelConfig::sota();
        let mem = estimate_memory(&config, config.train_seq_len);
        let gb = mem as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("Estimated peak GPU memory: {:.2} GB (per device)", gb);
        // Should fit in H100 80GB
        assert!(gb < 80.0, "exceeds H100 memory: {:.2} GB", gb);
    }

    #[test]
    fn test_checkpoint_layers() {
        let config = ModelConfig::sota();
        let ckpt = checkpoint_layers(&config);
        assert_eq!(ckpt.len(), 11);
        assert!(!ckpt[0]); // layer 0: not checkpointed
        assert!(!ckpt[2]); // layer 2: not checkpointed
        assert!(ckpt[3]); // layer 3: checkpointed
        assert!(ckpt[8]); // layer 8: checkpointed
        assert!(!ckpt[9]); // layer 9: not checkpointed
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_saved_layer_masks() {
        let mut config = ModelConfig::sota();
        config.recurrence_enabled = true;
        config.recurrence_start_layer = 4;
        config.recurrence_repeat_layers = 2;

        let recurrent =
            gpu_saved_layer_mask_for_mode(&config, "recurrent").expect("recurrent mask");
        assert_eq!(
            recurrent
                .iter()
                .enumerate()
                .filter_map(|(idx, save)| save.then_some(idx))
                .collect::<Vec<_>>(),
            vec![4, 5]
        );

        let inner = gpu_saved_layer_mask_for_mode(&config, "inner").expect("inner mask");
        assert_eq!(
            inner
                .iter()
                .enumerate()
                .filter_map(|(idx, save)| save.then_some(idx))
                .collect::<Vec<_>>(),
            vec![3, 4, 5, 6, 7, 8]
        );
    }

    #[test]
    fn test_bank_shapes() {
        let config = ModelConfig::sota();
        let shapes = bank_shapes(&config);
        assert_eq!(shapes.len(), 4);
        assert_eq!(shapes[0], [22, 512, 512]); // qo_bank
        assert_eq!(shapes[1], [22, 256, 512]); // kv_bank
        assert_eq!(shapes[2], [11, 1536, 512]); // mlp_up
        assert_eq!(shapes[3], [11, 512, 1536]); // mlp_down
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn bf16_qkv_freshness_rejects_stale_step_and_layer() {
        let freshness = Bf16QkvFreshness::new();
        freshness.mark(
            Bf16QkvProducer::FusedNormQkvRopeGain,
            7,
            3,
            98_304,
            2_048,
            8,
            4,
            64,
        );
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    7,
                    3,
                    98_304,
                    2_048,
                    8,
                    4,
                    64,
                )
                .is_ok()
        );
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    8,
                    3,
                    98_304,
                    2_048,
                    8,
                    4,
                    64,
                )
                .is_err()
        );
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    7,
                    4,
                    98_304,
                    2_048,
                    8,
                    4,
                    64,
                )
                .is_err()
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn bf16_qkv_freshness_rejects_never_marked_buffer() {
        // This is the exact failure mode the consumer-side `require()` is meant
        // to catch: a misconfigured run with prepacked attention enabled but the
        // fused producer skipped (or producer was never actually run for this
        // layer this step). Without the consumer-side gate, cuDNN would silently
        // read uninitialized scratch as Q/K/V.
        let freshness = Bf16QkvFreshness::new();
        let result = freshness.require(
            Bf16QkvProducer::FusedNormQkvRopeGain,
            1,
            0,
            98_304,
            2_048,
            8,
            4,
            64,
        );
        assert!(result.is_err(), "require on un-marked freshness must fail");
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("stale prepacked BF16 QKV"),
            "error message should describe stale BF16 QKV; got: {err}"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn bf16_qkv_freshness_rejects_shape_change_mid_step() {
        // If shape (tokens / heads / kv heads / head_dim) changes between mark
        // and require within the same step, the buffers are no longer valid
        // for the new shape. This catches cases like a runtime config change
        // or a dynamic-batch path producing different shapes per call.
        let freshness = Bf16QkvFreshness::new();
        freshness.mark(
            Bf16QkvProducer::FusedNormQkvRopeGain,
            5,
            2,
            65_536,
            1_024,
            16,
            4,
            64,
        );
        // Same step, same layer, same producer, but different head count.
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    5,
                    2,
                    65_536,
                    1_024,
                    32,
                    4,
                    64,
                )
                .is_err(),
            "different num_heads must reject"
        );
        // Different head_dim.
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    5,
                    2,
                    65_536,
                    1_024,
                    16,
                    4,
                    128,
                )
                .is_err(),
            "different head_dim must reject"
        );
        // Different num_kv_heads.
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    5,
                    2,
                    65_536,
                    1_024,
                    16,
                    8,
                    64,
                )
                .is_err(),
            "different num_kv_heads must reject"
        );
        // Different seq_len.
        assert!(
            freshness
                .require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    5,
                    2,
                    65_536,
                    2_048,
                    16,
                    4,
                    64,
                )
                .is_err(),
            "different seq_len must reject"
        );
    }
}

#[cfg(feature = "cuda")]
pub struct GpuQProjectionLora {
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
    pub input_dim: usize,
    pub output_dim: usize,
    pub slots: usize,
    /// A matrices are stored row-major as [rank, input_dim].
    pub a: Vec<GpuTensor>,
    /// B matrices are stored row-major as [output_dim, rank].
    pub b: Vec<GpuTensor>,
    pub grad_a: Vec<GpuTensor>,
    pub grad_b: Vec<GpuTensor>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct GpuQProjectionLoraHostState {
    pub rank: usize,
    pub alpha: f32,
    pub a: Vec<Vec<u8>>,
    pub b: Vec<Vec<u8>>,
}

#[cfg(feature = "cuda")]
impl GpuQProjectionLora {
    fn new_slots(
        stream: Arc<CudaStream>,
        slots: usize,
        input_dim: usize,
        output_dim: usize,
        rank: usize,
        alpha: f32,
        salt: u64,
    ) -> PgResult<Self> {
        if rank == 0 || rank > input_dim {
            return Err(pg_core::PgError::InvalidOp(format!(
                "LoRA rank must be in 1..={input_dim}, got {rank}"
            )));
        }
        let scale = alpha / rank as f32;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        let mut a = Vec::with_capacity(slots);
        let mut b = Vec::with_capacity(slots);
        let mut grad_a = Vec::with_capacity(slots);
        let mut grad_b = Vec::with_capacity(slots);
        for slot in 0..slots {
            let mut a_host = vec![0.0f32; rank * input_dim];
            // Warm-start A deterministically and keep B at zero. This matches the
            // frontier LoRA-TTT convention: first update moves B, later updates
            // can move both factors without perturbing score-before-update logits.
            for r in 0..rank {
                for col in 0..input_dim {
                    let x = salt
                        ^ (slot as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ (r as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                        ^ (col as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
                    let centered = ((x >> 40) as f32 / 16_777_216.0) - 0.5;
                    a_host[r * input_dim + col] = centered * 0.01;
                }
            }
            a.push(GpuTensor::from_host_data_gpu(
                stream.clone(),
                bytemuck::cast_slice(&a_host),
                &[rank, input_dim],
                DType::F32,
            )?);
            b.push(zeros(&[output_dim, rank])?);
            grad_a.push(zeros(&[rank, input_dim])?);
            grad_b.push(zeros(&[output_dim, rank])?);
        }

        Ok(Self {
            rank,
            alpha,
            scale,
            input_dim,
            output_dim,
            slots,
            a,
            b,
            grad_a,
            grad_b,
        })
    }

    fn new_q(
        config: &ModelConfig,
        stream: Arc<CudaStream>,
        rank: usize,
        alpha: f32,
    ) -> PgResult<Self> {
        Self::new_slots(
            stream,
            config.num_layers,
            config.model_dim,
            config.model_dim,
            rank,
            alpha,
            0x51_4c_4f_52_41,
        )
    }
}

#[cfg(feature = "cuda")]
pub struct GpuModel {
    pub config: ModelConfig,
    pub attention_backend: AttentionBackend,
    pub compute_precision: ModelComputePrecision,
    pub output_ce_backend: OutputCeBackend,
    pub weights: GpuWeights,
    pub gemm: pg_kernels::gemm::GemmEngine,
    pub side_gemm: Option<pg_kernels::gemm::GemmEngine>,
    pub side_gemm_main_to_side: Option<cudarc::driver::CudaEvent>,
    pub side_gemm_side_to_main: Option<cudarc::driver::CudaEvent>,
    side_gemm_deferred_weight_pending: Cell<bool>,
    recurrence_active: Cell<bool>,
    pub kernels: pg_kernels::gpu_kernels::GpuKernels,
    pub cuda_cpp_attention: Option<pg_kernels::flash_attn::CudaCppAttention>,
    pub cudnn_frontend_attention: Option<pg_kernels::flash_attn::CudnnFrontendAttention>,
    pub fused_output_ce: Option<pg_kernels::output_ce::FusedOutputCe>,
    pub q_lora: Option<GpuQProjectionLora>,
    pub k_lora: Option<GpuQProjectionLora>,
    pub v_lora: Option<GpuQProjectionLora>,
    pub o_lora: Option<GpuQProjectionLora>,
    pub mlp_lora: Option<GpuQProjectionLora>,
    pub lm_head_lora: Option<GpuQProjectionLora>,
    pub _ctx: Arc<CudaContext>,
}

#[cfg(feature = "cuda")]
impl GpuModel {
    pub fn from_cpu_reference(
        cpu: &GptModel,
        plan: &ExecutionPlan,
        ctx: Arc<CudaContext>,
        stream: Arc<CudaStream>,
    ) -> PgResult<Self> {
        plan.validate_model_config(&cpu.config)?;
        let weights = GpuWeights::from_cpu(cpu, stream.clone())?;
        let gemm = pg_kernels::gemm::GemmEngine::new(stream.clone())?;
        let (side_gemm, side_gemm_main_to_side, side_gemm_side_to_main) =
            if gpu_any_overlap_linear_backward_gemms_enabled() {
                let side_stream = gemm.stream().context().new_stream().map_err(|e| {
                    PgError::InvalidOp(format!("side GEMM stream init failed: {e:?}"))
                })?;
                let side_gemm = pg_kernels::gemm::GemmEngine::new(side_stream.clone())?;
                let event_flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING);
                let main_to_side = gemm
                    .stream()
                    .context()
                    .new_event(event_flags)
                    .map_err(|e| {
                        PgError::InvalidOp(format!("side GEMM main->side event init failed: {e:?}"))
                    })?;
                let side_to_main = gemm
                    .stream()
                    .context()
                    .new_event(event_flags)
                    .map_err(|e| {
                        PgError::InvalidOp(format!("side GEMM side->main event init failed: {e:?}"))
                    })?;
                (Some(side_gemm), Some(main_to_side), Some(side_to_main))
            } else {
                (None, None, None)
            };
        let kernels = pg_kernels::gpu_kernels::GpuKernels::new(ctx.clone(), stream)?;
        let cuda_cpp_attention =
            pg_kernels::flash_attn::CudaCppAttention::new(gemm.stream().clone()).ok();
        let cudnn_frontend_attention =
            pg_kernels::flash_attn::CudnnFrontendAttention::new(gemm.stream().clone()).ok();
        let fused_output_ce = pg_kernels::output_ce::FusedOutputCe::new(gemm.stream().clone()).ok();
        Ok(Self {
            config: cpu.config.clone(),
            attention_backend: plan.run_spec.model.attention_backend,
            compute_precision: plan.run_spec.model.compute_precision,
            output_ce_backend: plan.run_spec.model.output_ce_backend,
            weights,
            gemm,
            side_gemm,
            side_gemm_main_to_side,
            side_gemm_side_to_main,
            side_gemm_deferred_weight_pending: Cell::new(false),
            recurrence_active: Cell::new(cpu.config.recurrence_enabled),
            kernels,
            cuda_cpp_attention,
            cudnn_frontend_attention,
            fused_output_ce,
            q_lora: None,
            k_lora: None,
            v_lora: None,
            o_lora: None,
            mlp_lora: None,
            lm_head_lora: None,
            _ctx: ctx,
        })
    }

    pub fn sync_from_cpu_reference(
        &mut self,
        cpu: &GptModel,
        plan: &ExecutionPlan,
    ) -> PgResult<()> {
        plan.validate_model_config(&cpu.config)?;
        self.attention_backend = plan.run_spec.model.attention_backend;
        self.compute_precision = plan.run_spec.model.compute_precision;
        self.output_ce_backend = plan.run_spec.model.output_ce_backend;
        self.recurrence_active.set(cpu.config.recurrence_enabled);
        self.weights.sync_from_cpu(cpu)
    }

    pub fn sync_to_cpu_reference(&self, cpu: &mut GptModel, plan: &ExecutionPlan) -> PgResult<()> {
        plan.validate_model_config(&cpu.config)?;
        self.weights.sync_to_cpu(cpu)
    }

    pub fn refresh_bf16_shadows(&self) -> PgResult<()> {
        if self.compute_precision != ModelComputePrecision::Bf16TensorCore {
            return Ok(());
        }
        use pg_kernels::gpu_kernels::CudaPtr;
        let stream = self.gemm.stream();
        let convert = |src: &GpuTensor, dst: &GpuTensor| -> PgResult<()> {
            self.kernels.f32_to_bf16(
                CudaPtr(src.cu_ptr(stream)?),
                CudaPtr(dst.cu_ptr(stream)?),
                src.numel() as u32,
            )
        };
        convert(&self.weights.qo_bank, &self.weights.qo_bank_bf16)?;
        convert(&self.weights.kv_bank, &self.weights.kv_bank_bf16)?;
        self.kernels.pack_qkv_weights(
            CudaPtr(self.weights.qo_bank.cu_ptr(stream)?),
            CudaPtr(self.weights.kv_bank.cu_ptr(stream)?),
            CudaPtr(self.weights.qkv_bank.cu_ptr(stream)?),
            self.config.num_layers as u32,
            self.config.model_dim as u32,
            self.config.kv_dim() as u32,
        )?;
        convert(&self.weights.qkv_bank, &self.weights.qkv_bank_bf16)?;
        convert(&self.weights.mlp_up_bank, &self.weights.mlp_up_bank_bf16)?;
        convert(
            &self.weights.mlp_down_bank,
            &self.weights.mlp_down_bank_bf16,
        )?;
        self.kernels.f32_to_bf16(
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb_bf16.cu_ptr(stream)?),
            self.weights.tok_emb.numel() as u32,
        )
    }

    pub fn refresh_bf16_non_bank_shadows_after_sharded_bank_update(&self) -> PgResult<()> {
        if self.compute_precision != ModelComputePrecision::Bf16TensorCore {
            return Ok(());
        }
        use pg_kernels::gpu_kernels::CudaPtr;
        let stream = self.gemm.stream();
        self.kernels.pack_qkv_weights_bf16(
            CudaPtr(self.weights.qo_bank_bf16.cu_ptr(stream)?),
            CudaPtr(self.weights.kv_bank_bf16.cu_ptr(stream)?),
            CudaPtr(self.weights.qkv_bank_bf16.cu_ptr(stream)?),
            self.config.num_layers as u32,
            self.config.model_dim as u32,
            self.config.kv_dim() as u32,
        )?;
        self.kernels.f32_to_bf16(
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb_bf16.cu_ptr(stream)?),
            self.weights.tok_emb.numel() as u32,
        )
    }

    pub fn refresh_bf16_qkv_shadow_after_sharded_bank_update(&self) -> PgResult<()> {
        if self.compute_precision != ModelComputePrecision::Bf16TensorCore {
            return Ok(());
        }
        use pg_kernels::gpu_kernels::CudaPtr;
        let stream = self.gemm.stream();
        self.kernels.pack_qkv_weights_bf16(
            CudaPtr(self.weights.qo_bank_bf16.cu_ptr(stream)?),
            CudaPtr(self.weights.kv_bank_bf16.cu_ptr(stream)?),
            CudaPtr(self.weights.qkv_bank_bf16.cu_ptr(stream)?),
            self.config.num_layers as u32,
            self.config.model_dim as u32,
            self.config.kv_dim() as u32,
        )
    }

    fn use_bf16_forward_gemm(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && !matches!(
                std::env::var("PG_GPU_BF16_FORWARD_GEMM")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_primary_forward_gemm(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && !matches!(
                std::env::var("PG_GPU_BF16_PRIMARY_FORWARD_GEMM")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_backward_gemm(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && !matches!(
                std::env::var("PG_GPU_BF16_BACKWARD_GEMM")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_output_gemm(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && !matches!(
                std::env::var("PG_GPU_BF16_OUTPUT_GEMM")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_output_backward_gemm(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && !matches!(
                std::env::var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_logits(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && self.use_bf16_output_gemm()
            && self.use_bf16_output_backward_gemm()
            && !self.use_tiled_output_ce()
            && !self.use_chunked_bf16_output_ce_cache()
            && !matches!(
                std::env::var("PG_GPU_BF16_LOGITS")
                    .unwrap_or_else(|_| "0".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_final_norm_bf16_side_output(&self) -> bool {
        self.use_bf16_output_gemm()
            && matches!(
                std::env::var("PG_GPU_FINAL_NORM_BF16_OUTPUT")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_qkv_dx_beta_accum(&self) -> bool {
        matches!(
            std::env::var("PG_GPU_QKV_DX_BETA_ACCUM")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    }

    fn use_bf16_qkv_dx_output(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && self.use_bf16_backward_gemm()
            && gpu_bf16_qkv_dx_output_env_enabled()
    }

    fn use_fused_qkv_projection(&self) -> bool {
        self.compute_precision == ModelComputePrecision::Bf16TensorCore
            && (gpu_bf16_backward_chain_requested()
                || matches!(
                    std::env::var("PG_GPU_FUSED_QKV_PROJ")
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                        .as_str(),
                    "1" | "true" | "yes" | "on"
                ))
    }

    fn use_fused_qk_rope_gain_backward(&self) -> bool {
        self.config.rope_dims > 0
            && !matches!(
                std::env::var("PG_GPU_FUSED_QK_ROPE_GAIN_BWD")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_fused_qk_rope_gain_forward(&self) -> bool {
        !matches!(
            std::env::var("PG_GPU_FUSED_QK_ROPE_GAIN_FWD")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
    }

    fn use_fused_ce_loss_bwd(&self) -> bool {
        self.use_bf16_output_backward_gemm()
            && !matches!(
                std::env::var("PG_GPU_FUSED_CE_LOSS_BWD")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn selected_output_ce_backend(&self) -> GpuOutputCeBackend {
        selected_output_ce_backend_for_config(
            &self.config,
            self.compute_precision,
            Some(self.output_ce_backend),
        )
    }

    fn use_fused_exact_output_ce(&self) -> bool {
        self.use_bf16_output_gemm()
            && self.use_bf16_output_backward_gemm()
            && self.selected_output_ce_backend() == GpuOutputCeBackend::FusedExactWmma
    }

    fn use_tiled_output_ce(&self) -> bool {
        self.use_bf16_output_gemm()
            && self.use_bf16_output_backward_gemm()
            && self.selected_output_ce_backend() == GpuOutputCeBackend::TiledRepeatedGemm
    }

    fn use_chunked_bf16_output_ce_cache(&self) -> bool {
        self.use_bf16_output_gemm()
            && self.use_bf16_output_backward_gemm()
            && self.selected_output_ce_backend() == GpuOutputCeBackend::ChunkedBf16Cache
    }

    fn use_output_ce_no_full_logits(&self) -> bool {
        self.lm_head_lora.is_none()
            && (self.use_tiled_output_ce()
                || self.use_chunked_bf16_output_ce_cache()
                || self.use_fused_exact_output_ce())
    }

    fn use_fused_residual_mix_norm(&self) -> bool {
        !matches!(
            std::env::var("PG_GPU_FUSED_RESIDUAL_MIX_NORM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
    }

    fn use_fused_mlp_activation_bf16(&self) -> bool {
        self.use_bf16_forward_gemm()
            && !matches!(
                std::env::var("PG_GPU_FUSED_MLP_ACT_BF16")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_mlp_up_output(&self) -> bool {
        self.use_bf16_primary_forward_gemm()
            && matches!(
                std::env::var("PG_GPU_BF16_MLP_UP_OUTPUT")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_bf16_norm_side_outputs(&self) -> bool {
        self.use_bf16_primary_forward_gemm()
            && matches!(
                std::env::var("PG_GPU_BF16_NORM_SIDE_OUTPUTS")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_bf16_norm_grad_path(&self) -> bool {
        self.use_bf16_backward_gemm()
            && matches!(
                std::env::var("PG_GPU_BF16_NORM_GRAD_PATH")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_bf16_residual_projection_output(&self) -> bool {
        self.use_bf16_primary_forward_gemm()
            && self.use_bf16_backward_gemm()
            && self.mlp_lora.is_none()
            && matches!(
                std::env::var("PG_GPU_BF16_RESIDUAL_PROJ_OUTPUT")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_bf16_attention_projection_output(&self) -> bool {
        self.use_bf16_primary_forward_gemm()
            && self.use_bf16_backward_gemm()
            && self.o_lora.is_none()
            && matches!(
                std::env::var("PG_GPU_BF16_ATTN_PROJ_OUTPUT")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_bf16_attention_backward_tail(&self) -> bool {
        self.attention_backend == AttentionBackend::CudnnSdpaBf16
            && self.use_bf16_backward_gemm()
            && self.use_fused_qk_rope_gain_backward()
            && (gpu_bf16_backward_chain_requested()
                || matches!(
                    std::env::var("PG_GPU_BF16_ATTN_BACKWARD_TAIL")
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                        .as_str(),
                    "1" | "true" | "yes" | "on"
                ))
    }

    fn use_bf16_attention_tail_qkv_pack(&self) -> bool {
        self.use_bf16_attention_backward_tail()
            && self.use_fused_qkv_projection()
            && self.use_bf16_qkv_dx_output()
            && !self.any_qkv_lora_enabled()
            && self.config.ve_layers.is_empty()
            && (gpu_bf16_backward_chain_requested()
                || matches!(
                    std::env::var("PG_GPU_BF16_ATTN_TAIL_QKV_PACK")
                        .unwrap_or_default()
                        .to_ascii_lowercase()
                        .as_str(),
                    "1" | "true" | "yes" | "on"
                ))
    }

    fn use_bf16_attention_tail_direct_qkv_pack(&self) -> bool {
        self.use_bf16_attention_tail_qkv_pack()
            && !matches!(
                std::env::var("PG_GPU_BF16_ATTN_TAIL_DIRECT_QKV_PACK")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_bf16_attention_backward_bhsd_do(&self) -> bool {
        self.use_bf16_attention_tail_direct_qkv_pack()
            && self.config.sparse_attn_gate_enabled
            && self.config.xsa_last_n > 0
            && !matches!(
                std::env::var("PG_GPU_BF16_ATTN_BACKWARD_BHSD_DO")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_fused_attention_residual_from_base(&self) -> bool {
        !matches!(
            std::env::var("PG_GPU_FUSED_ATTN_RESIDUAL_FROM_BASE")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
    }

    fn use_fused_parallel_attn_residual_rms_norm(&self, layer: usize) -> bool {
        self.parallel_residual_enabled_for_layer(layer)
            && self.use_fused_attention_residual_from_base()
            && !matches!(
                std::env::var("PG_GPU_FUSED_PARALLEL_ATTN_RESID_RMS")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_cudnn_saved_bf16_attention(&self) -> bool {
        self.attention_backend == AttentionBackend::CudnnSdpaBf16
            && !matches!(
                std::env::var("PG_GPU_CUDNN_SAVED_BF16_ATTN")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn requested_cudnn_prepacked_bf16_attention(&self) -> bool {
        matches!(
            std::env::var("PG_GPU_CUDNN_PREPACKED_BF16_ATTN")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    }

    fn debug_poison_cudnn_prepacked_bf16_attention(&self) -> bool {
        self.use_cudnn_prepacked_bf16_attention()
            && matches!(
                std::env::var("PG_GPU_CUDNN_PREPACKED_BF16_POISON")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn validate_cudnn_prepacked_bf16_attention(&self) -> PgResult<()> {
        if self.requested_cudnn_prepacked_bf16_attention() && !self.use_fused_qk_rope_gain_forward()
        {
            return Err(pg_core::PgError::InvalidOp(
                "PG_GPU_CUDNN_PREPACKED_BF16_ATTN requires PG_GPU_FUSED_QK_ROPE_GAIN_FWD; otherwise the prepacked BF16 Q/K buffers have no fresh producer".into(),
            ));
        }
        Ok(())
    }

    fn use_cudnn_prepacked_bf16_attention(&self) -> bool {
        self.use_cudnn_saved_bf16_attention()
            && self.requested_cudnn_prepacked_bf16_attention()
            && self.use_fused_qk_rope_gain_forward()
    }

    fn use_bf16_sparse_xsa_forward(&self) -> bool {
        self.use_cudnn_prepacked_bf16_attention()
            && self.use_bf16_attention_projection_output()
            && self.use_skip_f32_attention_saved_acts()
            && !matches!(
                std::env::var("PG_GPU_BF16_SPARSE_XSA_FWD")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn use_skip_f32_attention_saved_acts(&self) -> bool {
        self.use_cudnn_saved_bf16_attention()
            && gpu_direct_saved_activations_enabled()
            && !self.any_lora_enabled()
            // The BF16 direct path covers XSA-all frontier blocks. Gated
            // variants still keep their F32 gate inputs/values, but do not
            // need full F32 q/k/v/attention copies once XSA spans all layers.
            && self.config.xsa_last_n >= self.config.num_layers
            && !matches!(
                std::env::var("PG_GPU_SKIP_F32_ATTN_SAVED_ACTS")
                    .unwrap_or_else(|_| "1".to_string())
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
    }

    fn output_projection_forward(
        &self,
        x_in: &GpuTensor,
        x_in_bf16: &GpuTensor,
        logits: &GpuTensor,
        tokens: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let stream = self.gemm.stream();
        if self.use_bf16_logits() && logits.dtype() != DType::BF16 {
            return Err(PgError::InvalidOp(
                "PG_GPU_BF16_LOGITS=1 requires BF16 logits storage; allocate GPU activations through GpuActivations::new_for_plan".into(),
            ));
        }
        if self.use_bf16_output_gemm() {
            if logits.dtype() == DType::BF16 && !self.use_bf16_logits() {
                return Err(PgError::InvalidOp(
                    "BF16 output logits require PG_GPU_BF16_LOGITS=1 with BF16 output forward/backward GEMMs".into(),
                ));
            }
            if !self.use_final_norm_bf16_side_output() {
                self.kernels.f32_to_bf16(
                    CudaPtr(x_in.cu_ptr(stream)?),
                    CudaPtr(x_in_bf16.cu_ptr(stream)?),
                    (tokens * d) as u32,
                )?;
            }
            unsafe {
                if logits.dtype() == DType::BF16 {
                    self.gemm.matmul_bf16_bt(
                        x_in_bf16.cu_ptr(stream)?,
                        self.weights.tok_emb_bf16.cu_ptr(stream)?,
                        logits.cu_ptr(stream)?,
                        tokens,
                        vocab,
                        d,
                        1.0,
                        0.0,
                    )?;
                } else {
                    self.gemm.matmul_bf16_bt_to_f32(
                        x_in_bf16.cu_ptr(stream)?,
                        self.weights.tok_emb_bf16.cu_ptr(stream)?,
                        logits.cu_ptr(stream)?,
                        tokens,
                        vocab,
                        d,
                        1.0,
                        0.0,
                    )?;
                }
            }
        } else {
            if logits.dtype() != DType::F32 {
                return Err(PgError::InvalidOp(
                    "F32 output projection path cannot write BF16 logits".into(),
                ));
            }
            unsafe {
                self.gemm.matmul_f32(
                    x_in.cu_ptr(stream)?,
                    self.weights.tok_emb.cu_ptr(stream)?,
                    logits.cu_ptr(stream)?,
                    tokens,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
            }
        }
        Ok(())
    }

    fn linear_forward(
        &self,
        input: &GpuTensor,
        input_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> PgResult<()> {
        self.linear_forward_impl(
            input,
            input_bf16,
            weight,
            weight_bf16,
            output,
            tokens,
            out_dim,
            in_dim,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_forward_bf16_input_ready(
        &self,
        input: &GpuTensor,
        input_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> PgResult<()> {
        self.linear_forward_impl(
            input,
            input_bf16,
            weight,
            weight_bf16,
            output,
            tokens,
            out_dim,
            in_dim,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_forward_bf16_output_ready(
        &self,
        input_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        output_bf16: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> PgResult<()> {
        let stream = self.gemm.stream();
        unsafe {
            self.gemm.matmul_bf16_bt(
                input_bf16.cu_ptr(stream)?,
                weight_bf16.cu_ptr(stream)?,
                output_bf16.cu_ptr(stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                0.0,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_forward_f32(
        &self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> PgResult<()> {
        let stream = self.gemm.stream();
        unsafe {
            self.gemm.matmul_f32(
                input.cu_ptr(stream)?,
                weight.cu_ptr(stream)?,
                output.cu_ptr(stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                0.0,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_forward_impl(
        &self,
        input: &GpuTensor,
        input_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        convert_bf16_input: bool,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        if self.use_bf16_forward_gemm() {
            if convert_bf16_input {
                self.kernels.f32_to_bf16(
                    CudaPtr(input.cu_ptr(stream)?),
                    CudaPtr(input_bf16.cu_ptr(stream)?),
                    (tokens * in_dim) as u32,
                )?;
            }
            unsafe {
                self.gemm.matmul_bf16_bt_to_f32(
                    input_bf16.cu_ptr(stream)?,
                    weight_bf16.cu_ptr(stream)?,
                    output.cu_ptr(stream)?,
                    tokens,
                    out_dim,
                    in_dim,
                    1.0,
                    0.0,
                )?;
            }
        } else {
            unsafe {
                self.gemm.matmul_f32(
                    input.cu_ptr(stream)?,
                    weight.cu_ptr(stream)?,
                    output.cu_ptr(stream)?,
                    tokens,
                    out_dim,
                    in_dim,
                    1.0,
                    0.0,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    fn linear_backward_input_and_weight_x_bf16_ready(
        &self,
        dy: &GpuTensor,
        dy_bf16: &GpuTensor,
        x: &GpuTensor,
        x_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> PgResult<()> {
        self.linear_backward_input_and_weight_impl(
            dy,
            dy_bf16,
            x,
            x_bf16,
            weight,
            weight_bf16,
            dx,
            dw,
            tokens,
            out_dim,
            in_dim,
            false,
            0.0,
            1.0,
        )
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    fn linear_backward_input_and_weight_x_bf16_ready_with_dx_beta(
        &self,
        dy: &GpuTensor,
        dy_bf16: &GpuTensor,
        x: &GpuTensor,
        x_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
    ) -> PgResult<()> {
        self.linear_backward_input_and_weight_impl(
            dy,
            dy_bf16,
            x,
            x_bf16,
            weight,
            weight_bf16,
            dx,
            dw,
            tokens,
            out_dim,
            in_dim,
            false,
            dx_beta,
            1.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_backward_input_and_weight_x_bf16_ready_with_betas(
        &self,
        dy: &GpuTensor,
        dy_bf16: &GpuTensor,
        x: &GpuTensor,
        x_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
    ) -> PgResult<()> {
        self.linear_backward_input_and_weight_impl(
            dy,
            dy_bf16,
            x,
            x_bf16,
            weight,
            weight_bf16,
            dx,
            dw,
            tokens,
            out_dim,
            in_dim,
            false,
            dx_beta,
            dw_beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_backward_input_and_weight_impl(
        &self,
        dy: &GpuTensor,
        dy_bf16: &GpuTensor,
        x: &GpuTensor,
        x_bf16: &GpuTensor,
        weight: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        convert_x_bf16: bool,
        dx_beta: f32,
        dw_beta: f32,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        let skip_dw = gpu_skip_bank_dw(dw_beta);
        if self.use_bf16_backward_gemm() {
            self.kernels.f32_to_bf16(
                CudaPtr(dy.cu_ptr(stream)?),
                CudaPtr(dy_bf16.cu_ptr(stream)?),
                (tokens * out_dim) as u32,
            )?;
            if convert_x_bf16 {
                self.kernels.f32_to_bf16(
                    CudaPtr(x.cu_ptr(stream)?),
                    CudaPtr(x_bf16.cu_ptr(stream)?),
                    (tokens * in_dim) as u32,
                )?;
            }
            if !skip_dw
                && self.try_overlap_linear_backward_gemms_to_f32(
                    dy_bf16,
                    x_bf16,
                    weight_bf16,
                    dx,
                    dw,
                    tokens,
                    out_dim,
                    in_dim,
                    dx_beta,
                    dw_beta,
                    LinearBackwardOverlapRole::Generic,
                )?
            {
                return Ok(());
            }
            unsafe {
                self.gemm.linear_backward_input_bf16_to_f32(
                    dy_bf16.cu_ptr(stream)?,
                    weight_bf16.cu_ptr(stream)?,
                    dx.cu_ptr(stream)?,
                    tokens,
                    out_dim,
                    in_dim,
                    1.0,
                    dx_beta,
                )?;
                if skip_dw {
                    return Ok(());
                }
                self.gemm.linear_backward_weight_bf16_to_f32(
                    dy_bf16.cu_ptr(stream)?,
                    x_bf16.cu_ptr(stream)?,
                    dw.cu_ptr(stream)?,
                    tokens,
                    out_dim,
                    in_dim,
                    1.0,
                    dw_beta,
                )?;
            }
        } else {
            unsafe {
                self.gemm.linear_backward_input_f32(
                    dy.cu_ptr(stream)?,
                    weight.cu_ptr(stream)?,
                    dx.cu_ptr(stream)?,
                    tokens,
                    out_dim,
                    in_dim,
                    1.0,
                    dx_beta,
                )?;
                if skip_dw {
                    return Ok(());
                }
                self.gemm.linear_backward_weight_f32(
                    dy.cu_ptr(stream)?,
                    x.cu_ptr(stream)?,
                    dw.cu_ptr(stream)?,
                    tokens,
                    out_dim,
                    in_dim,
                    1.0,
                    dw_beta,
                )?;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_defer_linear_backward_weight_gemm_to_side_f32(
        &self,
        dy_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
        role: LinearBackwardOverlapRole,
    ) -> PgResult<bool> {
        if !gpu_defer_linear_backward_weight_gemms_enabled_for_role(role) {
            return Ok(false);
        }
        let Some(side_gemm) = self.side_gemm.as_ref() else {
            return Ok(false);
        };
        let Some(main_to_side) = self.side_gemm_main_to_side.as_ref() else {
            return Ok(false);
        };
        let Some(side_to_main) = self.side_gemm_side_to_main.as_ref() else {
            return Ok(false);
        };

        let main_stream = self.gemm.stream();
        let side_stream = side_gemm.stream();
        main_to_side.record(main_stream).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM deferred dW: main->side event record failed: {e:?}"
            ))
        })?;
        side_stream.wait(main_to_side).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM deferred dW: side wait for inputs failed: {e:?}"
            ))
        })?;

        unsafe {
            side_gemm.linear_backward_weight_bf16_to_f32(
                dy_bf16.cu_ptr(side_stream)?,
                x_bf16.cu_ptr(side_stream)?,
                dw.cu_ptr(side_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dw_beta,
            )?;
            side_to_main.record(side_stream).map_err(|e| {
                PgError::InvalidOp(format!(
                    "side GEMM deferred dW: side->main event record failed: {e:?}"
                ))
            })?;
            self.gemm.linear_backward_input_bf16_to_f32(
                dy_bf16.cu_ptr(main_stream)?,
                weight_bf16.cu_ptr(main_stream)?,
                dx.cu_ptr(main_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dx_beta,
            )?;
        }
        self.side_gemm_deferred_weight_pending.set(true);
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_defer_linear_backward_weight_gemm_to_side_bf16(
        &self,
        dy_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx_bf16: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
        role: LinearBackwardOverlapRole,
    ) -> PgResult<bool> {
        if !gpu_defer_linear_backward_weight_gemms_enabled_for_role(role) {
            return Ok(false);
        }
        let Some(side_gemm) = self.side_gemm.as_ref() else {
            return Ok(false);
        };
        let Some(main_to_side) = self.side_gemm_main_to_side.as_ref() else {
            return Ok(false);
        };
        let Some(side_to_main) = self.side_gemm_side_to_main.as_ref() else {
            return Ok(false);
        };

        let main_stream = self.gemm.stream();
        let side_stream = side_gemm.stream();
        main_to_side.record(main_stream).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM deferred dW: main->side event record failed: {e:?}"
            ))
        })?;
        side_stream.wait(main_to_side).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM deferred dW: side wait for inputs failed: {e:?}"
            ))
        })?;

        unsafe {
            side_gemm.linear_backward_weight_bf16_to_f32(
                dy_bf16.cu_ptr(side_stream)?,
                x_bf16.cu_ptr(side_stream)?,
                dw.cu_ptr(side_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dw_beta,
            )?;
            side_to_main.record(side_stream).map_err(|e| {
                PgError::InvalidOp(format!(
                    "side GEMM deferred dW: side->main event record failed: {e:?}"
                ))
            })?;
            self.gemm.linear_backward_input_bf16_to_bf16(
                dy_bf16.cu_ptr(main_stream)?,
                weight_bf16.cu_ptr(main_stream)?,
                dx_bf16.cu_ptr(main_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dx_beta,
            )?;
        }
        self.side_gemm_deferred_weight_pending.set(true);
        Ok(true)
    }

    fn wait_deferred_side_weight_gemms(&self) -> PgResult<()> {
        if !self.side_gemm_deferred_weight_pending.replace(false) {
            return Ok(());
        }
        let Some(side_to_main) = self.side_gemm_side_to_main.as_ref() else {
            return Ok(());
        };
        self.gemm.stream().wait(side_to_main).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM deferred dW: main wait for pending weights failed: {e:?}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn try_overlap_linear_backward_gemms_to_f32(
        &self,
        dy_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
        role: LinearBackwardOverlapRole,
    ) -> PgResult<bool> {
        if self.side_gemm_deferred_weight_pending.get() {
            return Ok(false);
        }
        if !gpu_overlap_linear_backward_gemms_enabled_for_role(role) {
            return Ok(false);
        }
        let Some(side_gemm) = self.side_gemm.as_ref() else {
            return Ok(false);
        };
        let Some(main_to_side) = self.side_gemm_main_to_side.as_ref() else {
            return Ok(false);
        };
        let Some(side_to_main) = self.side_gemm_side_to_main.as_ref() else {
            return Ok(false);
        };

        let main_stream = self.gemm.stream();
        let side_stream = side_gemm.stream();
        main_to_side.record(main_stream).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM overlap: main->side event record failed: {e:?}"
            ))
        })?;
        side_stream.wait(main_to_side).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM overlap: side wait for inputs failed: {e:?}"
            ))
        })?;

        unsafe {
            side_gemm.linear_backward_input_bf16_to_f32(
                dy_bf16.cu_ptr(side_stream)?,
                weight_bf16.cu_ptr(side_stream)?,
                dx.cu_ptr(side_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dx_beta,
            )?;
            side_to_main.record(side_stream).map_err(|e| {
                PgError::InvalidOp(format!(
                    "side GEMM overlap: side->main event record failed: {e:?}"
                ))
            })?;
            self.gemm.linear_backward_weight_bf16_to_f32(
                dy_bf16.cu_ptr(main_stream)?,
                x_bf16.cu_ptr(main_stream)?,
                dw.cu_ptr(main_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dw_beta,
            )?;
        }
        main_stream.wait(side_to_main).map_err(|e| {
            PgError::InvalidOp(format!("side GEMM overlap: main wait for dx failed: {e:?}"))
        })?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_overlap_linear_backward_gemms_to_bf16(
        &self,
        dy_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx_bf16: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
        role: LinearBackwardOverlapRole,
    ) -> PgResult<bool> {
        if self.side_gemm_deferred_weight_pending.get() {
            return Ok(false);
        }
        if !gpu_overlap_linear_backward_gemms_enabled_for_role(role) {
            return Ok(false);
        }
        let Some(side_gemm) = self.side_gemm.as_ref() else {
            return Ok(false);
        };
        let Some(main_to_side) = self.side_gemm_main_to_side.as_ref() else {
            return Ok(false);
        };
        let Some(side_to_main) = self.side_gemm_side_to_main.as_ref() else {
            return Ok(false);
        };

        let main_stream = self.gemm.stream();
        let side_stream = side_gemm.stream();
        main_to_side.record(main_stream).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM overlap: main->side event record failed: {e:?}"
            ))
        })?;
        side_stream.wait(main_to_side).map_err(|e| {
            PgError::InvalidOp(format!(
                "side GEMM overlap: side wait for inputs failed: {e:?}"
            ))
        })?;

        unsafe {
            side_gemm.linear_backward_input_bf16_to_bf16(
                dy_bf16.cu_ptr(side_stream)?,
                weight_bf16.cu_ptr(side_stream)?,
                dx_bf16.cu_ptr(side_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dx_beta,
            )?;
            side_to_main.record(side_stream).map_err(|e| {
                PgError::InvalidOp(format!(
                    "side GEMM overlap: side->main event record failed: {e:?}"
                ))
            })?;
            self.gemm.linear_backward_weight_bf16_to_f32(
                dy_bf16.cu_ptr(main_stream)?,
                x_bf16.cu_ptr(main_stream)?,
                dw.cu_ptr(main_stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dw_beta,
            )?;
        }
        main_stream.wait(side_to_main).map_err(|e| {
            PgError::InvalidOp(format!("side GEMM overlap: main wait for dx failed: {e:?}"))
        })?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_backward_input_and_weight_dy_x_bf16_ready_with_role(
        &self,
        dy_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
        role: LinearBackwardOverlapRole,
    ) -> PgResult<()> {
        if !self.use_bf16_backward_gemm() {
            return Err(PgError::InvalidOp(
                "bf16-ready backward helper requires BF16 backward GEMMs".into(),
            ));
        }
        let stream = self.gemm.stream();
        let skip_dw = gpu_skip_bank_dw(dw_beta);
        if !skip_dw
            && self.try_defer_linear_backward_weight_gemm_to_side_f32(
                dy_bf16,
                x_bf16,
                weight_bf16,
                dx,
                dw,
                tokens,
                out_dim,
                in_dim,
                dx_beta,
                dw_beta,
                role,
            )?
        {
            return Ok(());
        }
        if !skip_dw
            && self.try_overlap_linear_backward_gemms_to_f32(
                dy_bf16,
                x_bf16,
                weight_bf16,
                dx,
                dw,
                tokens,
                out_dim,
                in_dim,
                dx_beta,
                dw_beta,
                role,
            )?
        {
            return Ok(());
        }
        unsafe {
            self.gemm.linear_backward_input_bf16_to_f32(
                dy_bf16.cu_ptr(stream)?,
                weight_bf16.cu_ptr(stream)?,
                dx.cu_ptr(stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dx_beta,
            )?;
            if skip_dw {
                return Ok(());
            }
            self.gemm.linear_backward_weight_bf16_to_f32(
                dy_bf16.cu_ptr(stream)?,
                x_bf16.cu_ptr(stream)?,
                dw.cu_ptr(stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dw_beta,
            )?;
        }
        Ok(())
    }

    fn qkv_projection_forward(
        &self,
        layer: usize,
        buf: &mut GpuActivations,
        tokens: usize,
        attn_norm_bf16_ready: bool,
        q_pre_norm_save: Option<&GpuTensor>,
        k_pre_norm_save: Option<&GpuTensor>,
    ) -> PgResult<bool> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if !self.use_fused_qkv_projection() {
            return Ok(false);
        }

        let stream = self.gemm.stream();
        let d = self.config.model_dim;
        let kv = self.config.kv_dim();
        if !attn_norm_bf16_ready {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (tokens * d) as u32,
            )?;
        }
        let qkv_w = self.weights.qkv_bank.slice_first(layer)?;
        let qkv_w_bf16 = self.weights.qkv_bank_bf16.slice_first(layer)?;
        self.linear_forward_bf16_input_ready(
            &buf.attn_norm,
            &buf.x_in_bf16,
            &qkv_w,
            &qkv_w_bf16,
            &buf.qkv_out,
            tokens,
            d + 2 * kv,
            d,
        )?;
        if let (Some(q_save), Some(k_save)) = (q_pre_norm_save, k_pre_norm_save) {
            self.kernels.unpack_qkv_output_save_qk(
                CudaPtr(buf.qkv_out.cu_ptr(stream)?),
                CudaPtr(buf.q.cu_ptr(stream)?),
                CudaPtr(buf.k.cu_ptr(stream)?),
                CudaPtr(buf.v.cu_ptr(stream)?),
                CudaPtr(q_save.cu_ptr(stream)?),
                CudaPtr(k_save.cu_ptr(stream)?),
                tokens as u32,
                d as u32,
                kv as u32,
            )?;
        } else {
            self.kernels.unpack_qkv_output(
                CudaPtr(buf.qkv_out.cu_ptr(stream)?),
                CudaPtr(buf.q.cu_ptr(stream)?),
                CudaPtr(buf.k.cu_ptr(stream)?),
                CudaPtr(buf.v.cu_ptr(stream)?),
                tokens as u32,
                d as u32,
                kv as u32,
            )?;
        }
        Ok(true)
    }

    fn qkv_projection_forward_packed_only(
        &self,
        layer: usize,
        buf: &mut GpuActivations,
        tokens: usize,
        attn_norm_bf16_ready: bool,
    ) -> PgResult<bool> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if !self.use_fused_qkv_projection() {
            return Ok(false);
        }

        let stream = self.gemm.stream();
        let d = self.config.model_dim;
        let kv = self.config.kv_dim();
        if !attn_norm_bf16_ready {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (tokens * d) as u32,
            )?;
        }
        let qkv_w = self.weights.qkv_bank.slice_first(layer)?;
        let qkv_w_bf16 = self.weights.qkv_bank_bf16.slice_first(layer)?;
        self.linear_forward_bf16_input_ready(
            &buf.attn_norm,
            &buf.x_in_bf16,
            &qkv_w,
            &qkv_w_bf16,
            &buf.qkv_out,
            tokens,
            d + 2 * kv,
            d,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_projection_backward(
        &self,
        layer: usize,
        buf: &mut GpuActivations,
        block_cache: &GpuBlockBackwardCache,
        grads: &mut GpuGradBuffers,
        grad_q_proj: &GpuTensor,
        grad_k_proj: &GpuTensor,
        grad_v_projection: &GpuTensor,
        grad_attn_norm: &GpuTensor,
        attn_norm_override: Option<&GpuTensor>,
        attn_norm_bf16_override: Option<&GpuTensor>,
        tokens: usize,
        bank_dw_beta: f32,
    ) -> PgResult<bool> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if !self.use_fused_qkv_projection() {
            return Ok(false);
        }

        let stream = self.gemm.stream();
        let d = self.config.model_dim;
        let kv = self.config.kv_dim();
        let n = self.config.num_layers;
        let attn_norm_handle = attn_norm_override
            .cloned()
            .unwrap_or_else(|| buf.attn_norm.clone());
        let attn_norm = &attn_norm_handle;

        let attn_norm_bf16 = if let Some(attn_norm_bf16) = attn_norm_bf16_override {
            attn_norm_bf16
        } else {
            self.kernels.f32_to_bf16(
                CudaPtr(attn_norm.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (tokens * d) as u32,
            )?;
            &buf.x_in_bf16
        };

        let qkv_w = self.weights.qkv_bank.slice_first(layer)?;
        let qkv_w_bf16 = self.weights.qkv_bank_bf16.slice_first(layer)?;
        if self.use_bf16_backward_gemm() {
            self.kernels.pack_qkv_grads_bf16(
                CudaPtr(grad_q_proj.cu_ptr(stream)?),
                CudaPtr(grad_k_proj.cu_ptr(stream)?),
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(buf.qkv_aux_bf16.cu_ptr(stream)?),
                tokens as u32,
                d as u32,
                kv as u32,
            )?;
            if self.use_bf16_qkv_dx_output() {
                self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready_with_role(
                    &buf.qkv_aux_bf16,
                    attn_norm_bf16,
                    &qkv_w_bf16,
                    &block_cache.grad_attn_norm_bf16,
                    &block_cache.grad_qkv_weight,
                    tokens,
                    d + 2 * kv,
                    d,
                    0.0,
                    bank_dw_beta,
                    LinearBackwardOverlapRole::Qkv,
                )?;
                if self.any_qkv_lora_enabled() {
                    self.kernels.bf16_to_f32(
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_attn_norm.cu_ptr(stream)?),
                        (tokens * d) as u32,
                    )?;
                }
            } else {
                self.linear_backward_input_and_weight_dy_x_bf16_ready_with_role(
                    &buf.qkv_aux_bf16,
                    attn_norm_bf16,
                    &qkv_w_bf16,
                    grad_attn_norm,
                    &block_cache.grad_qkv_weight,
                    tokens,
                    d + 2 * kv,
                    d,
                    0.0,
                    bank_dw_beta,
                    LinearBackwardOverlapRole::Qkv,
                )?;
            }
        } else {
            self.kernels.pack_qkv_grads(
                CudaPtr(grad_q_proj.cu_ptr(stream)?),
                CudaPtr(grad_k_proj.cu_ptr(stream)?),
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(block_cache.grad_qkv_proj.cu_ptr(stream)?),
                tokens as u32,
                d as u32,
                kv as u32,
            )?;
            self.linear_backward_input_and_weight_x_bf16_ready_with_betas(
                &block_cache.grad_qkv_proj,
                &buf.qkv_aux_bf16,
                attn_norm,
                attn_norm_bf16,
                &qkv_w,
                &qkv_w_bf16,
                grad_attn_norm,
                &block_cache.grad_qkv_weight,
                tokens,
                d + 2 * kv,
                d,
                0.0,
                bank_dw_beta,
            )?;
        }
        if !gpu_skip_bank_dw(bank_dw_beta) {
            self.kernels.unpack_qkv_weight_grad(
                CudaPtr(block_cache.grad_qkv_weight.cu_ptr(stream)?),
                CudaPtr(grads.qo_bank.cu_ptr(stream)?),
                CudaPtr(grads.kv_bank.cu_ptr(stream)?),
                layer as u32,
                n as u32,
                d as u32,
                kv as u32,
            )?;
        }
        self.backward_qkv_loras(
            layer,
            attn_norm,
            grad_q_proj,
            grad_k_proj,
            grad_v_projection,
            grad_attn_norm,
            buf,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn qkv_projection_backward_from_tail_bf16(
        &self,
        layer: usize,
        buf: &mut GpuActivations,
        block_cache: &GpuBlockBackwardCache,
        grads: &mut GpuGradBuffers,
        grad_q_proj_bf16: &GpuTensor,
        grad_k_proj_bf16: &GpuTensor,
        grad_v_projection_bf16: &GpuTensor,
        grad_v_xsa: &GpuTensor,
        add_v_xsa: bool,
        qkv_aux_already_packed: bool,
        attn_norm_override: Option<&GpuTensor>,
        attn_norm_bf16_override: Option<&GpuTensor>,
        tokens: usize,
        bank_dw_beta: f32,
    ) -> PgResult<bool> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if !self.use_bf16_attention_tail_qkv_pack() {
            return Ok(false);
        }

        let stream = self.gemm.stream();
        let d = self.config.model_dim;
        let kv = self.config.kv_dim();
        let n = self.config.num_layers;
        let attn_norm = attn_norm_override.unwrap_or(&buf.attn_norm);
        let attn_norm_bf16 = if let Some(attn_norm_bf16) = attn_norm_bf16_override {
            attn_norm_bf16
        } else {
            self.kernels.f32_to_bf16(
                CudaPtr(attn_norm.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (tokens * d) as u32,
            )?;
            &buf.x_in_bf16
        };

        if !qkv_aux_already_packed {
            self.kernels.pack_qkv_grads_tail_bf16(
                CudaPtr(grad_q_proj_bf16.cu_ptr(stream)?),
                CudaPtr(grad_k_proj_bf16.cu_ptr(stream)?),
                CudaPtr(grad_v_projection_bf16.cu_ptr(stream)?),
                CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                CudaPtr(buf.qkv_aux_bf16.cu_ptr(stream)?),
                tokens as u32,
                d as u32,
                kv as u32,
                add_v_xsa,
            )?;
        }

        let qkv_w_bf16 = self.weights.qkv_bank_bf16.slice_first(layer)?;
        self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready_with_role(
            &buf.qkv_aux_bf16,
            attn_norm_bf16,
            &qkv_w_bf16,
            &block_cache.grad_attn_norm_bf16,
            &block_cache.grad_qkv_weight,
            tokens,
            d + 2 * kv,
            d,
            0.0,
            bank_dw_beta,
            LinearBackwardOverlapRole::Qkv,
        )?;
        if !gpu_skip_bank_dw(bank_dw_beta) {
            self.kernels.unpack_qkv_weight_grad(
                CudaPtr(block_cache.grad_qkv_weight.cu_ptr(stream)?),
                CudaPtr(grads.qo_bank.cu_ptr(stream)?),
                CudaPtr(grads.kv_bank.cu_ptr(stream)?),
                layer as u32,
                n as u32,
                d as u32,
                kv as u32,
            )?;
        }
        Ok(true)
    }

    pub fn enable_q_lora(&mut self, rank: usize, alpha: f32) -> PgResult<()> {
        let stream = self.gemm.stream().clone();
        self.q_lora = Some(GpuQProjectionLora::new_q(
            &self.config,
            stream,
            rank,
            alpha,
        )?);
        Ok(())
    }

    pub fn enable_ttt_lora_targets(
        &mut self,
        targets: &crate::spec::TttLoraTargetsSpec,
        rank: usize,
        alpha: f32,
    ) -> PgResult<()> {
        let stream = self.gemm.stream().clone();
        let d = self.config.model_dim;
        let kv = self.config.kv_dim();
        let n = self.config.num_layers;
        self.q_lora = if targets.q {
            Some(GpuQProjectionLora::new_slots(
                stream.clone(),
                n,
                d,
                d,
                rank,
                alpha,
                0x51_5f_4c_4f_52_41,
            )?)
        } else {
            None
        };
        self.k_lora = if targets.k {
            Some(GpuQProjectionLora::new_slots(
                stream.clone(),
                n,
                d,
                kv,
                rank,
                alpha,
                0x4b_5f_4c_4f_52_41,
            )?)
        } else {
            None
        };
        self.v_lora = if targets.v {
            Some(GpuQProjectionLora::new_slots(
                stream.clone(),
                n,
                d,
                kv,
                rank,
                alpha,
                0x56_5f_4c_4f_52_41,
            )?)
        } else {
            None
        };
        self.o_lora = if targets.o {
            Some(GpuQProjectionLora::new_slots(
                stream.clone(),
                n,
                d,
                d,
                rank,
                alpha,
                0x4f_5f_4c_4f_52_41,
            )?)
        } else {
            None
        };
        self.mlp_lora = if targets.mlp {
            Some(GpuQProjectionLora::new_slots(
                stream.clone(),
                n,
                d,
                d,
                rank,
                alpha,
                0x4d_4c_50_5f_4c_4f_52_41,
            )?)
        } else {
            None
        };
        self.lm_head_lora = if targets.lm_head {
            Some(GpuQProjectionLora::new_slots(
                stream,
                1,
                d,
                self.config.vocab_size,
                rank,
                alpha,
                0x4c_4d_48_45_41_44,
            )?)
        } else {
            None
        };
        self.zero_q_lora_grads()?;
        Ok(())
    }

    pub fn q_lora_enabled(&self) -> bool {
        self.q_lora.is_some()
    }

    fn any_lora_enabled(&self) -> bool {
        self.q_lora.is_some()
            || self.k_lora.is_some()
            || self.v_lora.is_some()
            || self.o_lora.is_some()
            || self.mlp_lora.is_some()
            || self.lm_head_lora.is_some()
    }

    fn any_qkv_lora_enabled(&self) -> bool {
        self.q_lora.is_some() || self.k_lora.is_some() || self.v_lora.is_some()
    }

    fn for_each_lora_adapter<F>(&self, mut f: F) -> PgResult<()>
    where
        F: FnMut(&GpuQProjectionLora) -> PgResult<()>,
    {
        for adapter in [
            self.q_lora.as_ref(),
            self.k_lora.as_ref(),
            self.v_lora.as_ref(),
            self.o_lora.as_ref(),
            self.mlp_lora.as_ref(),
            self.lm_head_lora.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            f(adapter)?;
        }
        Ok(())
    }

    pub fn q_lora_state_to_host(&self) -> PgResult<GpuQProjectionLoraHostState> {
        if !self.any_lora_enabled() {
            return Err(pg_core::PgError::InvalidOp(
                "q_lora_state_to_host requires enabled LoRA".into(),
            ));
        }
        let mut rank = 0usize;
        let mut alpha = 0.0f32;
        let mut a = Vec::new();
        let mut b = Vec::new();
        self.for_each_lora_adapter(|lora| {
            if rank == 0 {
                rank = lora.rank;
                alpha = lora.alpha;
            }
            for slot in 0..lora.slots {
                a.push(lora.a[slot].to_host_bytes()?);
                b.push(lora.b[slot].to_host_bytes()?);
            }
            Ok(())
        })?;
        Ok(GpuQProjectionLoraHostState { rank, alpha, a, b })
    }

    pub fn copy_q_lora_state_from_host(
        &mut self,
        state: &GpuQProjectionLoraHostState,
    ) -> PgResult<()> {
        if !self.any_lora_enabled() {
            return Err(pg_core::PgError::InvalidOp(
                "copy_q_lora_state_from_host requires enabled LoRA".into(),
            ));
        }
        let expected_slots = [
            self.q_lora.as_ref(),
            self.k_lora.as_ref(),
            self.v_lora.as_ref(),
            self.o_lora.as_ref(),
            self.mlp_lora.as_ref(),
            self.lm_head_lora.as_ref(),
        ]
        .into_iter()
        .flatten()
        .map(|lora| lora.slots)
        .sum::<usize>();
        if state.a.len() != expected_slots || state.b.len() != expected_slots {
            return Err(pg_core::PgError::InvalidOp(format!(
                "LoRA state slot count mismatch: got A={} B={} expected {}",
                state.a.len(),
                state.b.len(),
                expected_slots
            )));
        }
        let mut offset = 0usize;
        let mut copy_adapter = |name: &str,
                                lora: Option<&mut GpuQProjectionLora>|
         -> PgResult<()> {
            if let Some(lora) = lora {
                if lora.rank != state.rank || (lora.alpha - state.alpha).abs() > f32::EPSILON {
                    return Err(pg_core::PgError::InvalidOp(format!(
                        "LoRA {name} state shape mismatch: model rank/alpha={} / {:.6}, state rank/alpha={} / {:.6}",
                        lora.rank, lora.alpha, state.rank, state.alpha
                    )));
                }
                for slot in 0..lora.slots {
                    lora.a[slot].copy_from_host_bytes(&state.a[offset])?;
                    lora.b[slot].copy_from_host_bytes(&state.b[offset])?;
                    offset += 1;
                }
            }
            Ok(())
        };
        copy_adapter("q", self.q_lora.as_mut())?;
        copy_adapter("k", self.k_lora.as_mut())?;
        copy_adapter("v", self.v_lora.as_mut())?;
        copy_adapter("o", self.o_lora.as_mut())?;
        copy_adapter("mlp", self.mlp_lora.as_mut())?;
        copy_adapter("lm_head", self.lm_head_lora.as_mut())?;
        self.zero_q_lora_grads()?;
        Ok(())
    }

    pub fn zero_q_lora_grads(&self) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        self.for_each_lora_adapter(|lora| {
            for slot in 0..lora.slots {
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_a[slot].cu_ptr(stream)?),
                    0.0,
                    lora.grad_a[slot].numel() as u32,
                )?;
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_b[slot].cu_ptr(stream)?),
                    0.0,
                    lora.grad_b[slot].numel() as u32,
                )?;
            }
            Ok(())
        })
    }

    pub fn q_lora_grad_numel(&self) -> PgResult<usize> {
        let mut total = 0usize;
        self.for_each_lora_adapter(|lora| {
            for slot in 0..lora.slots {
                total += lora.grad_a[slot].numel();
                total += lora.grad_b[slot].numel();
            }
            Ok(())
        })?;
        if total == 0 {
            return Err(pg_core::PgError::InvalidOp(
                "q_lora_grad_numel requires enabled LoRA".into(),
            ));
        }
        Ok(total)
    }

    pub fn pack_q_lora_grads(&self, packed: &GpuTensor) -> PgResult<()> {
        let expected = self.q_lora_grad_numel()?;
        if packed.numel() != expected {
            return Err(pg_core::PgError::ShapeMismatch {
                expected: vec![expected],
                got: packed.shape().to_vec(),
            });
        }
        let mut offset = 0usize;
        self.for_each_lora_adapter(|lora| {
            for slot in 0..lora.slots {
                let a_len = lora.grad_a[slot].numel();
                let dst = packed.slice_range(offset, offset + a_len)?;
                self.copy_tensor_flat_f32(&lora.grad_a[slot], &dst)?;
                offset += a_len;

                let b_len = lora.grad_b[slot].numel();
                let dst = packed.slice_range(offset, offset + b_len)?;
                self.copy_tensor_flat_f32(&lora.grad_b[slot], &dst)?;
                offset += b_len;
            }
            Ok(())
        })
    }

    pub fn unpack_q_lora_grads(&self, packed: &GpuTensor) -> PgResult<()> {
        let expected = self.q_lora_grad_numel()?;
        if packed.numel() != expected {
            return Err(pg_core::PgError::ShapeMismatch {
                expected: vec![expected],
                got: packed.shape().to_vec(),
            });
        }
        let mut offset = 0usize;
        self.for_each_lora_adapter(|lora| {
            for slot in 0..lora.slots {
                let a_len = lora.grad_a[slot].numel();
                let src = packed.slice_range(offset, offset + a_len)?;
                self.copy_tensor_flat_f32(&src, &lora.grad_a[slot])?;
                offset += a_len;

                let b_len = lora.grad_b[slot].numel();
                let src = packed.slice_range(offset, offset + b_len)?;
                self.copy_tensor_flat_f32(&src, &lora.grad_b[slot])?;
                offset += b_len;
            }
            Ok(())
        })
    }

    pub fn scale_q_lora_grads(&self, alpha: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        self.for_each_lora_adapter(|lora| {
            for slot in 0..lora.slots {
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_a[slot].cu_ptr(stream)?),
                    alpha,
                    lora.grad_a[slot].numel() as u32,
                )?;
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_b[slot].cu_ptr(stream)?),
                    alpha,
                    lora.grad_b[slot].numel() as u32,
                )?;
            }
            Ok(())
        })
    }

    pub fn reset_q_lora_b(&self) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        self.for_each_lora_adapter(|lora| {
            for b in &lora.b {
                self.kernels
                    .scale_inplace(CudaPtr(b.cu_ptr(stream)?), 0.0, b.numel() as u32)?;
            }
            Ok(())
        })?;
        self.zero_q_lora_grads()?;
        Ok(())
    }

    pub fn step_q_lora_sgd(&self, lr: f32, weight_decay: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        self.for_each_lora_adapter(|lora| {
            for slot in 0..lora.slots {
                self.kernels.decay_sgd_step(
                    CudaPtr(lora.a[slot].cu_ptr(stream)?),
                    CudaPtr(lora.grad_a[slot].cu_ptr(stream)?),
                    lr,
                    weight_decay,
                    lora.a[slot].numel() as u32,
                )?;
                self.kernels.decay_sgd_step(
                    CudaPtr(lora.b[slot].cu_ptr(stream)?),
                    CudaPtr(lora.grad_b[slot].cu_ptr(stream)?),
                    lr,
                    weight_decay,
                    lora.b[slot].numel() as u32,
                )?;
            }
            Ok(())
        })?;
        self.zero_q_lora_grads()?;
        Ok(())
    }

    fn ln_scale_factor(&self, layer: usize) -> f32 {
        if self.config.ln_scale {
            1.0 / ((layer + 1) as f32).sqrt()
        } else {
            1.0
        }
    }

    fn parallel_residual_enabled_for_layer(&self, layer: usize) -> bool {
        self.config.parallel_residual_enabled_for_layer(layer)
    }

    fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.recurrence_active.get() && self.config.is_recurrent_layer(layer)
    }

    fn recurrent_layer_uses_straight_through_backward(&self, layer: usize) -> bool {
        if !self.is_recurrent_layer(layer) {
            return false;
        }
        if gpu_recurrent_all_straight_through_enabled() {
            return true;
        }
        let st_layers =
            gpu_recurrent_straight_through_layers().min(self.config.recurrence_repeat_layers);
        st_layers > 0
            && layer >= self.config.recurrence_start_layer
            && layer < self.config.recurrence_start_layer + st_layers
    }

    pub fn set_recurrence_active(&self, active: bool) {
        self.recurrence_active
            .set(active && self.config.recurrence_enabled);
    }

    pub fn recurrence_active(&self) -> bool {
        self.recurrence_active.get()
    }

    fn run_attention_forward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        stats: Option<u64>,
        saved_bf16_bhsd: Option<(u64, u64, u64, u64)>,
        bf16_output_only: bool,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if self.attention_backend == AttentionBackend::CudnnSdpaBf16
            && !(gpu_cuda_backward_graph_enabled() && gpu_cuda_graph_disable_cudnn_sdpa_enabled())
        {
            let Some(cudnn_attention) = &self.cudnn_frontend_attention else {
                return Err(pg_core::PgError::InvalidOp(
                    "attention_backend=cudnn_sdpa_bf16 was selected, but the cuDNN frontend SDPA backend was not compiled into this build".into(),
                ));
            };
            if let (Some(stats), Some((q_bf16, k_bf16, v_bf16, out_bf16))) =
                (stats, saved_bf16_bhsd)
            {
                if self.use_cudnn_prepacked_bf16_attention() {
                    if bf16_output_only {
                        return cudnn_attention.forward_with_stats_prepacked_bf16_only(
                            q_bf16,
                            k_bf16,
                            v_bf16,
                            stats,
                            out_bf16,
                            tokens,
                            seq_len,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        );
                    }
                    return cudnn_attention.forward_with_stats_prepacked_bf16(
                        q_bf16,
                        k_bf16,
                        v_bf16,
                        out,
                        stats,
                        out_bf16,
                        tokens,
                        seq_len,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    );
                }
                return cudnn_attention.forward_with_stats_saved_bf16(
                    q,
                    k,
                    v,
                    out,
                    stats,
                    q_bf16,
                    k_bf16,
                    v_bf16,
                    out_bf16,
                    tokens,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
            if let Some(stats) = stats {
                return cudnn_attention.forward_with_stats(
                    q,
                    k,
                    v,
                    out,
                    stats,
                    tokens,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
            return cudnn_attention.forward(
                q,
                k,
                v,
                out,
                tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        if std::env::var("PG_USE_CPP_NAIVE_ATTENTION")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
            && seq_len == tokens
        {
            if let Some(cuda_cpp_attention) = &self.cuda_cpp_attention {
                return cuda_cpp_attention.forward(
                    q,
                    k,
                    v,
                    out,
                    tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
        }

        self.kernels.causal_attention_online_fwd(
            CudaPtr(q),
            CudaPtr(k),
            CudaPtr(v),
            CudaPtr(out),
            tokens as u32,
            seq_len as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_backward(
        &self,
        q: u64,
        k: u64,
        v: u64,
        out: u64,
        grad_out: u64,
        grad_q: u64,
        grad_k: u64,
        grad_v: u64,
        tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        stats: Option<u64>,
        saved_bf16_bhsd: Option<(u64, u64, u64, u64)>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if self.attention_backend == AttentionBackend::CudnnSdpaBf16
            && !(gpu_cuda_backward_graph_enabled() && gpu_cuda_graph_disable_cudnn_sdpa_enabled())
        {
            let Some(cudnn_attention) = &self.cudnn_frontend_attention else {
                return Err(pg_core::PgError::InvalidOp(
                    "attention_backend=cudnn_sdpa_bf16 was selected, but the cuDNN frontend SDPA backend was not compiled into this build".into(),
                ));
            };
            if let (Some(stats), Some((q_bf16, k_bf16, v_bf16, out_bf16))) =
                (stats, saved_bf16_bhsd)
            {
                return cudnn_attention.backward_with_saved_bf16_stats(
                    q_bf16,
                    k_bf16,
                    v_bf16,
                    out_bf16,
                    grad_out,
                    grad_q,
                    grad_k,
                    grad_v,
                    stats,
                    tokens,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
            if let Some(stats) = stats {
                return cudnn_attention.backward_with_stats(
                    q,
                    k,
                    v,
                    out,
                    grad_out,
                    grad_q,
                    grad_k,
                    grad_v,
                    stats,
                    tokens,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                );
            }
            return cudnn_attention.backward(
                q,
                k,
                v,
                out,
                grad_out,
                grad_q,
                grad_k,
                grad_v,
                tokens,
                seq_len,
                num_heads,
                num_kv_heads,
                head_dim,
            );
        }

        self.kernels.causal_attention_online_bwd(
            CudaPtr(q),
            CudaPtr(k),
            CudaPtr(v),
            CudaPtr(grad_out),
            CudaPtr(grad_q),
            CudaPtr(grad_k),
            CudaPtr(grad_v),
            tokens as u32,
            seq_len as u32,
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_backward_bf16_grads(
        &self,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out_bf16: u64,
        grad_out: u64,
        grad_q_bf16: u64,
        grad_k_bf16: u64,
        grad_v_bf16: u64,
        tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        stats: u64,
    ) -> PgResult<()> {
        let Some(cudnn_attention) = &self.cudnn_frontend_attention else {
            return Err(pg_core::PgError::InvalidOp(
                "attention_backend=cudnn_sdpa_bf16 was selected, but the cuDNN frontend SDPA backend was not compiled into this build".into(),
            ));
        };
        if gpu_cuda_backward_graph_enabled() && gpu_cuda_graph_disable_cudnn_sdpa_enabled() {
            return Err(pg_core::PgError::InvalidOp(
                "PG_CUDA_GRAPH_DISABLE_CUDNN_SDPA cannot be combined with the BF16 cuDNN SDPA backward-tail path".into(),
            ));
        }
        cudnn_attention.backward_with_saved_bf16_stats_bf16_grads(
            q_bf16,
            k_bf16,
            v_bf16,
            out_bf16,
            grad_out,
            grad_q_bf16,
            grad_k_bf16,
            grad_v_bf16,
            stats,
            tokens,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention_backward_bf16_bhsd_do_bf16_grads(
        &self,
        q_bf16: u64,
        k_bf16: u64,
        v_bf16: u64,
        out_bf16: u64,
        grad_out_bhsd_bf16: u64,
        grad_q_bf16: u64,
        grad_k_bf16: u64,
        grad_v_bf16: u64,
        tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        stats: u64,
    ) -> PgResult<()> {
        let Some(cudnn_attention) = &self.cudnn_frontend_attention else {
            return Err(pg_core::PgError::InvalidOp(
                "attention_backend=cudnn_sdpa_bf16 was selected, but the cuDNN frontend SDPA backend was not compiled into this build".into(),
            ));
        };
        if gpu_cuda_backward_graph_enabled() && gpu_cuda_graph_disable_cudnn_sdpa_enabled() {
            return Err(pg_core::PgError::InvalidOp(
                "PG_CUDA_GRAPH_DISABLE_CUDNN_SDPA cannot be combined with the BF16 cuDNN SDPA backward-tail path".into(),
            ));
        }
        cudnn_attention.backward_with_saved_bf16_stats_bhsd_do_bf16_grads(
            q_bf16,
            k_bf16,
            v_bf16,
            out_bf16,
            grad_out_bhsd_bf16,
            grad_q_bf16,
            grad_k_bf16,
            grad_v_bf16,
            stats,
            tokens,
            seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
        )
    }

    #[track_caller]
    fn copy_tensor(&self, src: &GpuTensor, dst: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if src.shape() != dst.shape() {
            if gpu_shape_trace_enabled() {
                let caller = std::panic::Location::caller();
                eprintln!(
                    "gpu_shape_trace copy_tensor mismatch at {}:{} src_shape={:?} dst_shape={:?} src_dtype={:?} dst_dtype={:?}",
                    caller.file(),
                    caller.line(),
                    src.shape(),
                    dst.shape(),
                    src.dtype(),
                    dst.dtype(),
                );
            }
            return Err(pg_core::PgError::ShapeMismatch {
                expected: src.shape().to_vec(),
                got: dst.shape().to_vec(),
            });
        }

        let stream = self.gemm.stream();
        self.kernels.copy_fwd(
            CudaPtr(src.cu_ptr(stream)?),
            CudaPtr(dst.cu_ptr(stream)?),
            src.numel() as u32,
        )
    }

    #[track_caller]
    fn copy_bf16_tensor(&self, src: &GpuTensor, dst: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if src.shape() != dst.shape() {
            if gpu_shape_trace_enabled() {
                let caller = std::panic::Location::caller();
                eprintln!(
                    "gpu_shape_trace copy_bf16_tensor mismatch at {}:{} src_shape={:?} dst_shape={:?} src_dtype={:?} dst_dtype={:?}",
                    caller.file(),
                    caller.line(),
                    src.shape(),
                    dst.shape(),
                    src.dtype(),
                    dst.dtype(),
                );
            }
            return Err(pg_core::PgError::ShapeMismatch {
                expected: src.shape().to_vec(),
                got: dst.shape().to_vec(),
            });
        }
        if src.dtype() != DType::BF16 || dst.dtype() != DType::BF16 {
            return Err(PgError::InvalidOp(
                "copy_bf16_tensor requires bf16 source and destination".into(),
            ));
        }

        let stream = self.gemm.stream();
        self.kernels.copy_u16_fwd(
            CudaPtr(src.cu_ptr(stream)?),
            CudaPtr(dst.cu_ptr(stream)?),
            src.numel() as u32,
        )
    }

    #[track_caller]
    fn copy_tensor_flat_f32(&self, src: &GpuTensor, dst: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if src.dtype() != DType::F32 || dst.dtype() != DType::F32 {
            return Err(PgError::InvalidOp(format!(
                "copy_tensor_flat_f32 requires F32 tensors, got src={:?} dst={:?}",
                src.dtype(),
                dst.dtype()
            )));
        }
        if src.numel() != dst.numel() {
            if gpu_shape_trace_enabled() {
                let caller = std::panic::Location::caller();
                eprintln!(
                    "gpu_shape_trace copy_tensor_flat_f32 mismatch at {}:{} src_shape={:?} dst_shape={:?} src_numel={} dst_numel={}",
                    caller.file(),
                    caller.line(),
                    src.shape(),
                    dst.shape(),
                    src.numel(),
                    dst.numel(),
                );
            }
            return Err(pg_core::PgError::ShapeMismatch {
                expected: vec![src.numel()],
                got: vec![dst.numel()],
            });
        }

        let stream = self.gemm.stream();
        self.kernels.copy_fwd(
            CudaPtr(src.cu_ptr(stream)?),
            CudaPtr(dst.cu_ptr(stream)?),
            src.numel() as u32,
        )
    }

    #[track_caller]
    fn add_inplace(&self, dst: &GpuTensor, src: &GpuTensor, alpha: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if dst.shape() != src.shape() {
            if gpu_shape_trace_enabled() {
                let caller = std::panic::Location::caller();
                eprintln!(
                    "gpu_shape_trace add_inplace mismatch at {}:{} dst_shape={:?} src_shape={:?} dst_dtype={:?} src_dtype={:?}",
                    caller.file(),
                    caller.line(),
                    dst.shape(),
                    src.shape(),
                    dst.dtype(),
                    src.dtype(),
                );
            }
            return Err(pg_core::PgError::ShapeMismatch {
                expected: dst.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }

        let stream = self.gemm.stream();
        self.kernels.add_scaled_fwd(
            CudaPtr(dst.cu_ptr(stream)?),
            CudaPtr(src.cu_ptr(stream)?),
            alpha,
            dst.numel() as u32,
        )
    }

    fn apply_linear_lora_forward(
        &self,
        lora: &GpuQProjectionLora,
        slot: usize,
        input: &GpuTensor,
        output: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        if slot >= lora.slots {
            return Err(PgError::InvalidOp(format!(
                "LoRA slot {slot} out of range for {} slots",
                lora.slots
            )));
        }
        let stream = self.gemm.stream();
        let t = input.shape()[0];
        let r = lora.rank;
        let lora_tmp = &buf.lora_tmp;
        unsafe {
            self.gemm.matmul_f32(
                input.cu_ptr(stream)?,
                lora.a[slot].cu_ptr(stream)?,
                lora_tmp.cu_ptr(stream)?,
                t,
                r,
                lora.input_dim,
                1.0,
                0.0,
            )?;
            self.gemm.matmul_f32(
                lora_tmp.cu_ptr(stream)?,
                lora.b[slot].cu_ptr(stream)?,
                output.cu_ptr(stream)?,
                t,
                lora.output_dim,
                r,
                lora.scale,
                1.0,
            )?;
        }
        Ok(())
    }

    fn apply_qkv_loras_forward(&self, layer: usize, buf: &mut GpuActivations) -> PgResult<()> {
        let attn_norm = buf.attn_norm.clone();
        if let Some(lora) = &self.q_lora {
            let q = buf.q.clone();
            self.apply_linear_lora_forward(lora, layer, &attn_norm, &q, buf)?;
        }
        if let Some(lora) = &self.k_lora {
            let k = buf.k.clone();
            self.apply_linear_lora_forward(lora, layer, &attn_norm, &k, buf)?;
        }
        if let Some(lora) = &self.v_lora {
            let v = buf.v.clone();
            self.apply_linear_lora_forward(lora, layer, &attn_norm, &v, buf)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_backward_input_bf16_and_weight_dy_x_bf16_ready_with_role(
        &self,
        dy_bf16: &GpuTensor,
        x_bf16: &GpuTensor,
        weight_bf16: &GpuTensor,
        dx_bf16: &GpuTensor,
        dw: &GpuTensor,
        tokens: usize,
        out_dim: usize,
        in_dim: usize,
        dx_beta: f32,
        dw_beta: f32,
        role: LinearBackwardOverlapRole,
    ) -> PgResult<()> {
        if !self.use_bf16_backward_gemm() {
            return Err(PgError::InvalidOp(
                "bf16-output backward helper requires BF16 backward GEMMs".into(),
            ));
        }
        let stream = self.gemm.stream();
        let skip_dw = gpu_skip_bank_dw(dw_beta);
        if !skip_dw
            && self.try_defer_linear_backward_weight_gemm_to_side_bf16(
                dy_bf16,
                x_bf16,
                weight_bf16,
                dx_bf16,
                dw,
                tokens,
                out_dim,
                in_dim,
                dx_beta,
                dw_beta,
                role,
            )?
        {
            return Ok(());
        }
        if !skip_dw
            && self.try_overlap_linear_backward_gemms_to_bf16(
                dy_bf16,
                x_bf16,
                weight_bf16,
                dx_bf16,
                dw,
                tokens,
                out_dim,
                in_dim,
                dx_beta,
                dw_beta,
                role,
            )?
        {
            return Ok(());
        }
        unsafe {
            self.gemm.linear_backward_input_bf16_to_bf16(
                dy_bf16.cu_ptr(stream)?,
                weight_bf16.cu_ptr(stream)?,
                dx_bf16.cu_ptr(stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dx_beta,
            )?;
            if skip_dw {
                return Ok(());
            }
            self.gemm.linear_backward_weight_bf16_to_f32(
                dy_bf16.cu_ptr(stream)?,
                x_bf16.cu_ptr(stream)?,
                dw.cu_ptr(stream)?,
                tokens,
                out_dim,
                in_dim,
                1.0,
                dw_beta,
            )?;
        }
        Ok(())
    }

    fn backward_linear_lora(
        &self,
        lora: &GpuQProjectionLora,
        slot: usize,
        input: &GpuTensor,
        grad_output: &GpuTensor,
        grad_input_accum: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        if slot >= lora.slots {
            return Err(PgError::InvalidOp(format!(
                "LoRA slot {slot} out of range for {} slots",
                lora.slots
            )));
        }
        let stream = self.gemm.stream();
        let t = input.shape()[0];
        let r = lora.rank;
        let lora_tmp = &buf.lora_tmp;
        let lora_grad_tmp = &buf.lora_grad_tmp;
        let lora_delta = &buf.lora_delta;
        unsafe {
            self.gemm.matmul_f32(
                input.cu_ptr(stream)?,
                lora.a[slot].cu_ptr(stream)?,
                lora_tmp.cu_ptr(stream)?,
                t,
                r,
                lora.input_dim,
                1.0,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                grad_output.cu_ptr(stream)?,
                lora_tmp.cu_ptr(stream)?,
                lora.grad_b[slot].cu_ptr(stream)?,
                t,
                lora.output_dim,
                r,
                lora.scale,
                1.0,
            )?;
            self.gemm.linear_backward_input_f32(
                grad_output.cu_ptr(stream)?,
                lora.b[slot].cu_ptr(stream)?,
                lora_grad_tmp.cu_ptr(stream)?,
                t,
                lora.output_dim,
                r,
                lora.scale,
                0.0,
            )?;
            self.gemm.linear_backward_weight_f32(
                lora_grad_tmp.cu_ptr(stream)?,
                input.cu_ptr(stream)?,
                lora.grad_a[slot].cu_ptr(stream)?,
                t,
                r,
                lora.input_dim,
                1.0,
                1.0,
            )?;
            self.gemm.linear_backward_input_f32(
                lora_grad_tmp.cu_ptr(stream)?,
                lora.a[slot].cu_ptr(stream)?,
                lora_delta.cu_ptr(stream)?,
                t,
                r,
                lora.input_dim,
                1.0,
                0.0,
            )?;
        }
        self.add_inplace(grad_input_accum, lora_delta, 1.0)
    }

    fn backward_qkv_loras(
        &self,
        layer: usize,
        attn_norm: &GpuTensor,
        grad_q_proj: &GpuTensor,
        grad_k_proj: &GpuTensor,
        grad_v_proj: &GpuTensor,
        grad_attn_norm: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        if let Some(lora) = &self.q_lora {
            self.backward_linear_lora(lora, layer, attn_norm, grad_q_proj, grad_attn_norm, buf)?;
        }
        if let Some(lora) = &self.k_lora {
            self.backward_linear_lora(lora, layer, attn_norm, grad_k_proj, grad_attn_norm, buf)?;
        }
        if let Some(lora) = &self.v_lora {
            self.backward_linear_lora(lora, layer, attn_norm, grad_v_proj, grad_attn_norm, buf)?;
        }
        Ok(())
    }

    fn zero_tensor(&self, tensor: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        self.kernels.scale_inplace(
            CudaPtr(tensor.cu_ptr(self.gemm.stream())?),
            0.0,
            tensor.numel() as u32,
        )
    }

    fn mean_losses_only_with_sum_scratch(
        &self,
        losses: &GpuTensor,
        loss_sum: &GpuTensor,
        tokens: usize,
    ) -> PgResult<f32> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if tokens == 0 {
            return Ok(0.0);
        }
        let stream = self.gemm.stream();
        self.kernels.loss_window_sum(
            CudaPtr(losses.cu_ptr(stream)?),
            CudaPtr(loss_sum.cu_ptr(stream)?),
            0,
            tokens as u32,
        )?;
        let values = decode_f32_host_bytes(&loss_sum.to_host_bytes()?)?;
        values
            .first()
            .copied()
            .map(|sum| sum / tokens as f32)
            .ok_or_else(|| PgError::InvalidOp("loss_sum scratch download was empty".into()))
    }

    pub fn cross_entropy_losses(
        &self,
        logits: &GpuTensor,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tokens: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        if logits.dtype() == DType::BF16 {
            self.kernels.cross_entropy_fwd_bf16_logits(
                CudaPtr(logits.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(losses.cu_ptr(stream)?),
                self.config.vocab_size as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                tokens as u32,
            )
        } else {
            self.kernels.cross_entropy_fwd(
                CudaPtr(logits.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(losses.cu_ptr(stream)?),
                self.config.vocab_size as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                tokens as u32,
            )
        }
    }

    fn apply_qk_norm_rope_gain_forward(
        &self,
        layer: usize,
        q: &GpuTensor,
        k: &GpuTensor,
        q_post_rope: Option<&GpuTensor>,
        q_bhsd_bf16: Option<&GpuTensor>,
        k_bhsd_bf16: Option<&GpuTensor>,
        tokens: usize,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let rope_dims = self.config.rope_dims;

        if self.use_fused_qk_rope_gain_forward() {
            let q_post_rope_ptr = q_post_rope
                .map(|tensor| tensor.cu_ptr(stream))
                .transpose()?
                .map(CudaPtr);
            if let (Some(q_bhsd_bf16), Some(k_bhsd_bf16)) = (q_bhsd_bf16, k_bhsd_bf16) {
                self.kernels.q_gain_rope_qk_norm_fwd_bf16_bhsd(
                    CudaPtr(q.cu_ptr(stream)?),
                    q_post_rope_ptr,
                    CudaPtr(q_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                    rope_dims as u32,
                    (tokens * h) as u32,
                    1e-6,
                )?;
                self.kernels.rope_qk_norm_fwd_bf16_bhsd(
                    CudaPtr(k.cu_ptr(stream)?),
                    CudaPtr(k_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    hkv as u32,
                    hd as u32,
                    rope_dims as u32,
                    (tokens * hkv) as u32,
                    1e-6,
                )?;
            } else {
                self.kernels.q_gain_rope_qk_norm_fwd(
                    CudaPtr(q.cu_ptr(stream)?),
                    q_post_rope_ptr,
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                    rope_dims as u32,
                    (tokens * h) as u32,
                    1e-6,
                )?;
                self.kernels.rope_qk_norm_fwd(
                    CudaPtr(k.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    hkv as u32,
                    hd as u32,
                    rope_dims as u32,
                    (tokens * hkv) as u32,
                    1e-6,
                )?;
            }
            return Ok(());
        }

        self.kernels.qk_norm_fwd(
            CudaPtr(q.cu_ptr(stream)?),
            hd as u32,
            (tokens * h) as u32,
            1e-6,
        )?;
        self.kernels.qk_norm_fwd(
            CudaPtr(k.cu_ptr(stream)?),
            hd as u32,
            (tokens * hkv) as u32,
            1e-6,
        )?;
        if rope_dims > 0 {
            self.kernels.partial_rope_fwd(
                CudaPtr(q.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                h as u32,
                hd as u32,
                rope_dims as u32,
                (tokens * h) as u32,
            )?;
            self.kernels.partial_rope_fwd(
                CudaPtr(k.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                rope_dims as u32,
                (tokens * hkv) as u32,
            )?;
        }
        if let Some(q_post_rope) = q_post_rope {
            self.copy_tensor(q, q_post_rope)?;
        }
        self.kernels.q_gain_fwd(
            CudaPtr(q.cu_ptr(stream)?),
            CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
            h as u32,
            hd as u32,
            (tokens * h) as u32,
        )?;
        Ok(())
    }

    fn block_recompute_for_backward(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        layer_x: &GpuTensor,
        x0: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuBlockBackwardCache,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        let n = self.config.num_layers;
        let stream = self.gemm.stream();

        self.copy_tensor(layer_x, &buf.x)?;
        if self.use_fused_residual_mix_norm() {
            self.kernels.residual_mix_rms_norm_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(x0.cu_ptr(stream)?),
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
                (t * d) as u32,
            )?;
        } else {
            self.kernels.residual_mix_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(x0.cu_ptr(stream)?),
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;

            self.kernels.rms_norm_forward(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
            )?;
        }

        let save_qk_during_qkv = !self.any_qkv_lora_enabled();
        let qkv_projected = self.qkv_projection_forward(
            layer,
            buf,
            t,
            false,
            save_qk_during_qkv.then_some(&cache.q_pre_norm),
            save_qk_during_qkv.then_some(&cache.k_pre_norm),
        )?;
        let qk_pre_norm_saved_by_qkv = qkv_projected && save_qk_during_qkv;
        if qkv_projected {
            // Hot path handled by a single packed QKV GEMM.
        } else if self.use_bf16_primary_forward_gemm() {
            let q_w = self.weights.qo_bank.slice_first(layer)?;
            let q_w_bf16 = self.weights.qo_bank_bf16.slice_first(layer)?;
            let k_w = self.weights.kv_bank.slice_first(layer)?;
            let k_w_bf16 = self.weights.kv_bank_bf16.slice_first(layer)?;
            let v_w = self.weights.kv_bank.slice_first(n + layer)?;
            let v_w_bf16 = self.weights.kv_bank_bf16.slice_first(n + layer)?;
            self.kernels.f32_to_bf16(
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (t * d) as u32,
            )?;
            self.linear_forward_bf16_input_ready(
                &buf.attn_norm,
                &buf.x_in_bf16,
                &q_w,
                &q_w_bf16,
                &buf.q,
                t,
                d,
                d,
            )?;
            self.linear_forward_bf16_input_ready(
                &buf.attn_norm,
                &buf.x_in_bf16,
                &k_w,
                &k_w_bf16,
                &buf.k,
                t,
                kv,
                d,
            )?;
            self.linear_forward_bf16_input_ready(
                &buf.attn_norm,
                &buf.x_in_bf16,
                &v_w,
                &v_w_bf16,
                &buf.v,
                t,
                kv,
                d,
            )?;
        } else {
            let q_w = self.weights.qo_bank.slice_first(layer)?;
            let k_w = self.weights.kv_bank.slice_first(layer)?;
            let v_w = self.weights.kv_bank.slice_first(n + layer)?;
            self.linear_forward_f32(&buf.attn_norm, &q_w, &buf.q, t, d, d)?;
            self.linear_forward_f32(&buf.attn_norm, &k_w, &buf.k, t, kv, d)?;
            self.linear_forward_f32(&buf.attn_norm, &v_w, &buf.v, t, kv, d)?;
        }
        self.apply_qkv_loras_forward(layer, buf)?;
        if !qk_pre_norm_saved_by_qkv {
            self.copy_tensor(&buf.q, &cache.q_pre_norm)?;
            self.copy_tensor(&buf.k, &cache.k_pre_norm)?;
        }

        if let Some(ve_idx) = self.config.ve_layers.iter().position(|&l| l == layer) {
            self.kernels.embedding_gather_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_embed.cu_ptr(stream)?),
                CudaPtr(buf.ve_embed_out.cu_ptr(stream)?),
                self.config.ve_dim as u32,
                t as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.ve_embed_out.cu_ptr(stream)?,
                    self.weights.ve_proj.cu_ptr(stream)?,
                    buf.ve_out.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_by_param_product_fwd(
                CudaPtr(buf.v.cu_ptr(stream)?),
                CudaPtr(buf.ve_out.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_scale_param.cu_ptr(stream)?),
                0,
                CudaPtr(self.weights.ve_layer_scales.cu_ptr(stream)?),
                ve_idx as u32,
                1.0,
                (t * kv) as u32,
            )?;
        }

        self.apply_qk_norm_rope_gain_forward(
            layer,
            &buf.q,
            &buf.k,
            Some(&cache.q_post_rope),
            None,
            None,
            t,
            runtime_seq_len,
        )?;
        self.run_attention_forward(
            buf.q.cu_ptr(stream)?,
            buf.k.cu_ptr(stream)?,
            buf.v.cu_ptr(stream)?,
            buf.attn_out.cu_ptr(stream)?,
            t,
            h,
            hkv,
            hd,
            runtime_seq_len,
            Some(cache.attn_stats.cu_ptr(stream)?),
            None,
            false,
        )?;

        let attn_src = if layer >= n.saturating_sub(self.config.xsa_last_n) {
            self.kernels.xsa_fwd(
                CudaPtr(buf.attn_out.cu_ptr(stream)?),
                CudaPtr(buf.v.cu_ptr(stream)?),
                CudaPtr(buf.xsa_out.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hkv as u32,
                hd as u32,
            )?;
            &buf.xsa_out
        } else {
            &buf.attn_out
        };
        let attn_src = if self.config.attn_out_gate_enabled {
            self.kernels.attn_out_gate_fwd(
                CudaPtr(attn_src.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_biases[layer].cu_ptr(stream)?),
                CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.attn_out_gate_width as u32,
            )?;
            &buf.attn_gated
        } else if self.config.sparse_attn_gate_enabled {
            self.kernels.sparse_attn_gate_fwd(
                CudaPtr(attn_src.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.sparse_attn_gate_width as u32,
                self.config.sparse_attn_gate_scale,
            )?;
            &buf.attn_gated
        } else {
            attn_src
        };

        let o_w = self.weights.qo_bank.slice_first(n + layer)?;
        let o_w_bf16 = self.weights.qo_bank_bf16.slice_first(n + layer)?;
        self.linear_forward(
            attn_src,
            &buf.x_in_bf16,
            &o_w,
            &o_w_bf16,
            &buf.proj_out,
            t,
            d,
            d,
        )?;
        if let Some(lora) = &self.o_lora {
            let attn_norm = buf.attn_norm.clone();
            let proj_out = buf.proj_out.clone();
            self.apply_linear_lora_forward(lora, layer, &attn_norm, &proj_out, buf)?;
        }

        if self.use_fused_attention_residual_from_base() {
            self.kernels.residual_add_scale_from_base_fwd(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.proj_out.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                CudaPtr(buf.x.cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        } else {
            self.copy_tensor(&buf.x_in, &buf.x)?;
            self.kernels.residual_add_scale_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.proj_out.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        }
        if self.parallel_residual_enabled_for_layer(layer) {
            self.copy_tensor(&buf.x_in, &cache.x_after_attn)?;
        } else {
            self.copy_tensor(&buf.x, &cache.x_after_attn)?;
        }

        self.kernels.rms_norm_forward(
            if self.parallel_residual_enabled_for_layer(layer) {
                CudaPtr(buf.x_in.cu_ptr(stream)?)
            } else {
                CudaPtr(buf.x.cu_ptr(stream)?)
            },
            CudaPtr(buf.mlp_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            self.ln_scale_factor(layer),
            1e-6,
        )?;

        let up_w = self.weights.mlp_up_bank.slice_first(layer)?;
        let up_w_bf16 = self.weights.mlp_up_bank_bf16.slice_first(layer)?;
        let down_w = self.weights.mlp_down_bank.slice_first(layer)?;
        let down_w_bf16 = self.weights.mlp_down_bank_bf16.slice_first(layer)?;
        self.linear_forward(
            &buf.mlp_norm,
            &buf.x_in_bf16,
            &up_w,
            &up_w_bf16,
            &buf.mlp_up,
            t,
            mlp,
            d,
        )?;
        if self.use_fused_mlp_activation_bf16() {
            self.kernels.leaky_relu_sq_forward_bf16(
                CudaPtr(buf.mlp_up.cu_ptr(stream)?),
                CudaPtr(buf.mlp_act.cu_ptr(stream)?),
                CudaPtr(buf.wide_bf16.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
            self.linear_forward_bf16_input_ready(
                &buf.mlp_act,
                &buf.wide_bf16,
                &down_w,
                &down_w_bf16,
                &buf.mlp_out,
                t,
                d,
                mlp,
            )?;
        } else {
            self.kernels.leaky_relu_sq_forward(
                CudaPtr(buf.mlp_up.cu_ptr(stream)?),
                CudaPtr(buf.mlp_act.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
            self.linear_forward(
                &buf.mlp_act,
                &buf.wide_bf16,
                &down_w,
                &down_w_bf16,
                &buf.mlp_out,
                t,
                d,
                mlp,
            )?;
        }
        if let Some(lora) = &self.mlp_lora {
            let mlp_norm = buf.mlp_norm.clone();
            let mlp_out = buf.mlp_out.clone();
            self.apply_linear_lora_forward(lora, layer, &mlp_norm, &mlp_out, buf)?;
        }
        self.kernels.residual_add_scale_fwd(
            CudaPtr(buf.x.cu_ptr(stream)?),
            CudaPtr(buf.mlp_out.cu_ptr(stream)?),
            CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
            d as u32,
            (t * d) as u32,
        )?;

        Ok(())
    }

    fn restore_saved_block_for_backward(
        &self,
        saved: &GpuLayerForwardCache,
        buf: &mut GpuActivations,
        block_cache: &mut GpuBlockBackwardCache,
    ) -> PgResult<()> {
        if saved.lean_bf16_direct {
            return Err(PgError::InvalidOp(
                "lean BF16 saved layer cache requires direct saved-activation backward; restore/recompute fallback requested a skipped F32 field".into(),
            ));
        }
        self.copy_tensor(&saved.x_in, &buf.x_in)?;
        self.copy_tensor(&saved.attn_norm, &buf.attn_norm)?;
        self.copy_tensor(&saved.q_pre_norm, &block_cache.q_pre_norm)?;
        self.copy_tensor(&saved.k_pre_norm, &block_cache.k_pre_norm)?;
        self.copy_tensor(&saved.q_post_rope, &block_cache.q_post_rope)?;
        self.copy_tensor(&saved.q, &buf.q)?;
        self.copy_tensor(&saved.k, &buf.k)?;
        self.copy_tensor(&saved.v, &buf.v)?;
        self.copy_tensor(&saved.ve_embed_out, &buf.ve_embed_out)?;
        self.copy_tensor(&saved.ve_out, &buf.ve_out)?;
        self.copy_tensor(&saved.attn_out, &buf.attn_out)?;
        self.copy_tensor(&saved.xsa_out, &buf.xsa_out)?;
        if self.config.attn_out_gate_enabled || self.config.sparse_attn_gate_enabled {
            self.copy_tensor(&saved.attn_gated, &buf.attn_gated)?;
            self.copy_tensor(&saved.attn_gate_values, &buf.attn_gate_values)?;
        }
        self.copy_tensor(&saved.proj_out, &buf.proj_out)?;
        self.copy_tensor(&saved.x_after_attn, &block_cache.x_after_attn)?;
        self.copy_tensor(&saved.mlp_norm, &buf.mlp_norm)?;
        self.copy_tensor(&saved.mlp_up, &buf.mlp_up)?;
        self.copy_tensor(&saved.mlp_act, &buf.mlp_act)?;
        self.copy_tensor(&saved.mlp_out, &buf.mlp_out)?;
        Ok(())
    }

    fn block_backward_single_into(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        layer_x: &GpuTensor,
        x0: &GpuTensor,
        buf: &mut GpuActivations,
        block_cache: &mut GpuBlockBackwardCache,
        grad_x: &GpuTensor,
        grad_x0: &GpuTensor,
        grad_x_out: &GpuTensor,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
        saved: Option<&GpuLayerForwardCache>,
        forward_generation: u64,
        bank_dw_beta: f32,
        mut stage_timing: Option<&mut GpuBackwardStageTiming>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let stream = self.gemm.stream();
        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let direct_saved = saved.is_some()
            && gpu_direct_saved_activations_enabled()
            // The LoRA backward path consumes forward LoRA scratch that is not
            // part of the saved block cache. Fall back to the established
            // restore/recompute path when LoRA is active.
            && !self.any_lora_enabled();
        if direct_saved {
            // Read-only saved activations are consumed directly below.
        } else if let Some(saved) = saved {
            self.restore_saved_block_for_backward(saved, buf, block_cache)?;
        } else {
            self.block_recompute_for_backward(
                layer,
                input_ids,
                layer_x,
                x0,
                buf,
                block_cache,
                runtime_seq_len,
            )?;
        }
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_recompute_ms),
        )?;
        check_cuda_graph_capture_stage(
            stream,
            &format!("block_{layer}_restore_or_recompute_for_backward"),
        )?;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        let n = self.config.num_layers;
        let saved_direct = saved.filter(|_| direct_saved);
        let saved_bf16_direct = saved_direct
            .filter(|_| self.use_bf16_primary_forward_gemm() && self.use_bf16_backward_gemm());
        let recompute_residual_mix_norm_inputs =
            saved_bf16_direct.is_some() && gpu_recompute_residual_mix_norm_inputs_enabled();
        macro_rules! act {
            ($field:ident) => {
                if let Some(saved) = saved_direct {
                    &saved.$field
                } else {
                    &buf.$field
                }
            };
        }
        macro_rules! act_bf16 {
            ($saved_field:ident, $scratch:expr) => {
                if let Some(saved) = saved_bf16_direct {
                    &saved.$saved_field
                } else {
                    $scratch
                }
            };
        }
        macro_rules! block_act {
            ($field:ident) => {
                if let Some(saved) = saved_direct {
                    &saved.$field
                } else {
                    &block_cache.$field
                }
            };
        }

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let grad_x_after_attn = &block_cache.grad_x_after_attn;
        let grad_mlp_out = &block_cache.grad_mlp_out;
        let grad_mlp_out_bf16_storage = buf.x_aux_bf16.clone();
        let grad_mlp_out_bf16 = &grad_mlp_out_bf16_storage;
        let residual_scale_rows_per_chunk = gpu_residual_scale_backward_rows_per_chunk();
        let mlp_residual_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if !gpu_residual_scale_reduce_enabled() {
            self.zero_tensor(grad_x_after_attn)?;
        }
        if let Some(saved) = saved_bf16_direct
            .filter(|saved| saved.lean_bf16_direct && self.use_bf16_residual_projection_output())
        {
            if gpu_residual_scale_reduce_enabled() && gpu_chunked_residual_scale_backward_enabled()
            {
                self.kernels.residual_add_scale_bwd_from_bf16_only_chunked(
                    CudaPtr(saved.mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_x.cu_ptr(stream)?),
                    CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(block_cache.residual_mix_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_mlp_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                    residual_scale_rows_per_chunk as u32,
                )?;
            } else {
                self.kernels.residual_add_scale_bwd_from_bf16_only(
                    CudaPtr(saved.mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_x.cu_ptr(stream)?),
                    CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grads.block_mlp_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
        } else if self.use_bf16_backward_gemm() {
            if gpu_residual_scale_reduce_enabled() && gpu_chunked_residual_scale_backward_enabled()
            {
                self.kernels.residual_add_scale_bwd_bf16_only_chunked(
                    CudaPtr(act!(mlp_out).cu_ptr(stream)?),
                    CudaPtr(grad_x.cu_ptr(stream)?),
                    CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(block_cache.residual_mix_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_mlp_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                    residual_scale_rows_per_chunk as u32,
                )?;
            } else {
                self.kernels.residual_add_scale_bwd_bf16_only(
                    CudaPtr(act!(mlp_out).cu_ptr(stream)?),
                    CudaPtr(grad_x.cu_ptr(stream)?),
                    CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grads.block_mlp_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
        } else {
            self.kernels.residual_add_scale_bwd(
                CudaPtr(act!(mlp_out).cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                CudaPtr(grad_mlp_out.cu_ptr(stream)?),
                CudaPtr(grads.block_mlp_scale[layer].cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        }
        finish_stage_event_optional(
            stream,
            mlp_residual_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_mlp_residual_ms),
        )?;

        let grad_mlp_act = &block_cache.grad_mlp_act;
        let grad_mlp_act_bf16 = &buf.wide_bf16;
        let mlp_down_w = self.weights.mlp_down_bank.slice_first(layer)?;
        let mlp_down_w_bf16 = self.weights.mlp_down_bank_bf16.slice_first(layer)?;
        let use_bf16_mlp_down_dx = self.use_bf16_backward_gemm()
            && gpu_bf16_mlp_down_dx_enabled()
            && saved_bf16_direct
                .map(|saved| saved.lean_bf16_direct)
                .unwrap_or(false);
        let mlp_down_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if use_bf16_mlp_down_dx {
            self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready_with_role(
                grad_mlp_out_bf16,
                act_bf16!(mlp_act_bf16, &buf.wide_bf16),
                &mlp_down_w_bf16,
                grad_mlp_act_bf16,
                &grads.mlp_down_bank.slice_first(layer)?,
                t,
                d,
                mlp,
                0.0,
                bank_dw_beta,
                LinearBackwardOverlapRole::MlpDown,
            )?;
        } else if self.use_bf16_backward_gemm() {
            self.linear_backward_input_and_weight_dy_x_bf16_ready_with_role(
                grad_mlp_out_bf16,
                act_bf16!(mlp_act_bf16, &buf.wide_bf16),
                &mlp_down_w_bf16,
                grad_mlp_act,
                &grads.mlp_down_bank.slice_first(layer)?,
                t,
                d,
                mlp,
                0.0,
                bank_dw_beta,
                LinearBackwardOverlapRole::MlpDown,
            )?;
        } else {
            self.linear_backward_input_and_weight_impl(
                grad_mlp_out,
                grad_mlp_out_bf16,
                act!(mlp_act),
                act_bf16!(mlp_act_bf16, &buf.wide_bf16),
                &mlp_down_w,
                &mlp_down_w_bf16,
                grad_mlp_act,
                &grads.mlp_down_bank.slice_first(layer)?,
                t,
                d,
                mlp,
                saved_bf16_direct.is_none(),
                0.0,
                bank_dw_beta,
            )?;
        }
        finish_stage_event_optional(
            stream,
            mlp_down_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_mlp_down_ms),
        )?;

        let grad_mlp_up = &block_cache.grad_mlp_up;
        let grad_mlp_up_bf16 = &buf.wide_bf16;
        let mlp_act_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.use_bf16_backward_gemm() {
            if use_bf16_mlp_down_dx {
                if gpu_vec4_mlp_act_backward_enabled() {
                    self.kernels
                        .leaky_relu_sq_backward_x_go_bf16_only_vec4_fast(
                            CudaPtr(act_bf16!(mlp_up_bf16, &buf.wide_bf16).cu_ptr(stream)?),
                            CudaPtr(grad_mlp_act_bf16.cu_ptr(stream)?),
                            CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                            (t * mlp) as u32,
                        )?;
                } else if gpu_fast_mlp_act_backward_enabled() {
                    self.kernels.leaky_relu_sq_backward_x_go_bf16_only_fast(
                        CudaPtr(act_bf16!(mlp_up_bf16, &buf.wide_bf16).cu_ptr(stream)?),
                        CudaPtr(grad_mlp_act_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                } else {
                    self.kernels.leaky_relu_sq_backward_x_go_bf16_only(
                        CudaPtr(act_bf16!(mlp_up_bf16, &buf.wide_bf16).cu_ptr(stream)?),
                        CudaPtr(grad_mlp_act_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                }
            } else if let Some(saved) = saved_bf16_direct.filter(|saved| saved.lean_bf16_direct) {
                if gpu_vec4_mlp_act_backward_enabled() {
                    self.kernels.leaky_relu_sq_backward_x_bf16_only_vec4_fast(
                        CudaPtr(saved.mlp_up_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                } else if gpu_fast_mlp_act_backward_enabled() {
                    self.kernels.leaky_relu_sq_backward_x_bf16_only_fast(
                        CudaPtr(saved.mlp_up_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                } else if gpu_vec2_mlp_act_backward_enabled() {
                    self.kernels.leaky_relu_sq_backward_x_bf16_only_vec2(
                        CudaPtr(saved.mlp_up_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                } else {
                    self.kernels.leaky_relu_sq_backward_x_bf16_only(
                        CudaPtr(saved.mlp_up_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                        CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                }
            } else {
                self.kernels.leaky_relu_sq_backward_bf16(
                    CudaPtr(act!(mlp_up).cu_ptr(stream)?),
                    CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_up.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                    (t * mlp) as u32,
                )?;
            }
        } else {
            self.kernels.leaky_relu_sq_backward(
                CudaPtr(act!(mlp_up).cu_ptr(stream)?),
                CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                CudaPtr(grad_mlp_up.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
        }
        finish_stage_event_optional(
            stream,
            mlp_act_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_mlp_act_ms),
        )?;

        let grad_mlp_norm = &block_cache.grad_mlp_norm;
        let grad_mlp_norm_bf16_storage = buf.x_aux_bf16.clone();
        let grad_mlp_norm_bf16 = &grad_mlp_norm_bf16_storage;
        let mlp_up_w = self.weights.mlp_up_bank.slice_first(layer)?;
        let mlp_up_w_bf16 = self.weights.mlp_up_bank_bf16.slice_first(layer)?;
        // MLP-down dW may still be reading buf.x_aux_bf16; MLP-up dX writes it.
        self.wait_deferred_side_weight_gemms()?;
        let mlp_up_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.use_bf16_norm_grad_path() {
            self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready_with_role(
                grad_mlp_up_bf16,
                act_bf16!(mlp_norm_bf16, &buf.x_in_bf16),
                &mlp_up_w_bf16,
                grad_mlp_norm_bf16,
                &grads.mlp_up_bank.slice_first(layer)?,
                t,
                mlp,
                d,
                0.0,
                bank_dw_beta,
                LinearBackwardOverlapRole::MlpUp,
            )?;
        } else if self.use_bf16_backward_gemm() {
            self.linear_backward_input_and_weight_dy_x_bf16_ready_with_role(
                grad_mlp_up_bf16,
                act_bf16!(mlp_norm_bf16, &buf.x_in_bf16),
                &mlp_up_w_bf16,
                grad_mlp_norm,
                &grads.mlp_up_bank.slice_first(layer)?,
                t,
                mlp,
                d,
                0.0,
                bank_dw_beta,
                LinearBackwardOverlapRole::MlpUp,
            )?;
        } else {
            self.linear_backward_input_and_weight_impl(
                grad_mlp_up,
                &buf.wide_bf16,
                act!(mlp_norm),
                act_bf16!(mlp_norm_bf16, &buf.x_in_bf16),
                &mlp_up_w,
                &mlp_up_w_bf16,
                grad_mlp_norm,
                &grads.mlp_up_bank.slice_first(layer)?,
                t,
                mlp,
                d,
                saved_bf16_direct.is_none(),
                0.0,
                bank_dw_beta,
            )?;
        }
        finish_stage_event_optional(
            stream,
            mlp_up_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_mlp_up_ms),
        )?;

        let grad_x_pre_mlp_norm = &block_cache.grad_x_pre_mlp_norm;
        if let Some(lora) = &self.mlp_lora {
            if self.use_bf16_backward_gemm() {
                self.kernels.bf16_to_f32(
                    CudaPtr(grad_mlp_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_out.cu_ptr(stream)?),
                    grad_mlp_out.numel() as u32,
                )?;
            }
            let mlp_norm_for_lora = act!(mlp_norm).clone();
            let grad_mlp_norm_for_lora = grad_mlp_norm.clone();
            self.backward_linear_lora(
                lora,
                layer,
                &mlp_norm_for_lora,
                grad_mlp_out,
                &grad_mlp_norm_for_lora,
                buf,
            )?;
        }
        let mlp_norm_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.parallel_residual_enabled_for_layer(layer)
            && recompute_residual_mix_norm_inputs
            && self.use_bf16_norm_grad_path()
        {
            self.kernels
                .rms_norm_backward_accum_residual_mix_input_go_bf16(
                    CudaPtr(layer_x.cu_ptr(stream)?),
                    CudaPtr(x0.cu_ptr(stream)?),
                    CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                    CudaPtr(grad_mlp_norm_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                    0.0,
                )?;
        } else if self.parallel_residual_enabled_for_layer(layer)
            && recompute_residual_mix_norm_inputs
        {
            self.kernels.rms_norm_backward_accum_residual_mix_input(
                CudaPtr(layer_x.cu_ptr(stream)?),
                CudaPtr(x0.cu_ptr(stream)?),
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                CudaPtr(grad_mlp_norm.cu_ptr(stream)?),
                CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
                0.0,
            )?;
        } else if self.use_bf16_norm_grad_path() {
            let mlp_norm_input = if self.parallel_residual_enabled_for_layer(layer) {
                act!(x_in)
            } else {
                block_act!(x_after_attn)
            };
            self.kernels.rms_norm_backward_go_bf16(
                CudaPtr(mlp_norm_input.cu_ptr(stream)?),
                CudaPtr(grad_mlp_norm_bf16.cu_ptr(stream)?),
                CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
            )?;
        } else {
            let mlp_norm_input = if self.parallel_residual_enabled_for_layer(layer) {
                act!(x_in)
            } else {
                block_act!(x_after_attn)
            };
            self.kernels.rms_norm_backward(
                CudaPtr(mlp_norm_input.cu_ptr(stream)?),
                CudaPtr(grad_mlp_norm.cu_ptr(stream)?),
                CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
            )?;
        }
        if !self.parallel_residual_enabled_for_layer(layer) {
            self.add_inplace(grad_x_after_attn, grad_x_pre_mlp_norm, 1.0)?;
        }
        finish_stage_event_optional(
            stream,
            mlp_norm_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_mlp_norm_ms),
        )?;
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_mlp_ms),
        )?;
        check_cuda_graph_capture_stage(stream, &format!("block_{layer}_mlp_backward"))?;

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let grad_x_in = &block_cache.grad_x_in;
        let grad_proj_out = &block_cache.grad_proj_out;
        let grad_proj_out_bf16_storage = buf.x_aux_bf16.clone();
        let grad_proj_out_bf16 = &grad_proj_out_bf16_storage;
        let attn_out_residual_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let fuse_attn_residual_add =
            self.parallel_residual_enabled_for_layer(layer) && gpu_residual_scale_reduce_enabled();
        let mut attn_residual_add_fused = false;
        if !gpu_residual_scale_reduce_enabled() {
            self.zero_tensor(grad_x_in)?;
        }
        if let Some(saved) = saved_bf16_direct
            .filter(|saved| saved.lean_bf16_direct && self.use_bf16_attention_projection_output())
        {
            if gpu_residual_scale_reduce_enabled() && gpu_chunked_residual_scale_backward_enabled()
            {
                if fuse_attn_residual_add {
                    self.kernels
                        .residual_add_scale_bwd_from_bf16_only_chunked_add(
                            CudaPtr(saved.proj_out_bf16.cu_ptr(stream)?),
                            CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                            CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
                            CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                            CudaPtr(grad_x_in.cu_ptr(stream)?),
                            CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                            CudaPtr(block_cache.residual_mix_reduce_scratch.cu_ptr(stream)?),
                            CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                            d as u32,
                            (t * d) as u32,
                            residual_scale_rows_per_chunk as u32,
                        )?;
                    attn_residual_add_fused = true;
                } else {
                    self.kernels.residual_add_scale_bwd_from_bf16_only_chunked(
                        CudaPtr(saved.proj_out_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                        CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                        CudaPtr(block_cache.residual_mix_reduce_scratch.cu_ptr(stream)?),
                        CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                        d as u32,
                        (t * d) as u32,
                        residual_scale_rows_per_chunk as u32,
                    )?;
                }
            } else if fuse_attn_residual_add {
                self.kernels.residual_add_scale_bwd_from_bf16_only_add(
                    CudaPtr(saved.proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(grad_x_pre_mlp_norm.cu_ptr(stream)?),
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
                attn_residual_add_fused = true;
            } else {
                self.kernels.residual_add_scale_bwd_from_bf16_only(
                    CudaPtr(saved.proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
        } else if self.use_bf16_backward_gemm() {
            if gpu_residual_scale_reduce_enabled() && gpu_chunked_residual_scale_backward_enabled()
            {
                self.kernels.residual_add_scale_bwd_bf16_only_chunked(
                    CudaPtr(act!(proj_out).cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(block_cache.residual_mix_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                    residual_scale_rows_per_chunk as u32,
                )?;
            } else {
                self.kernels.residual_add_scale_bwd_bf16_only(
                    CudaPtr(act!(proj_out).cu_ptr(stream)?),
                    CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
        } else {
            self.kernels.residual_add_scale_bwd(
                CudaPtr(act!(proj_out).cu_ptr(stream)?),
                CudaPtr(grad_x_after_attn.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                CudaPtr(grad_x_in.cu_ptr(stream)?),
                CudaPtr(grad_proj_out.cu_ptr(stream)?),
                CudaPtr(grads.block_attn_scale[layer].cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        }
        if self.parallel_residual_enabled_for_layer(layer) && !attn_residual_add_fused {
            self.add_inplace(grad_x_in, grad_x_pre_mlp_norm, 1.0)?;
        }
        finish_stage_event_optional(
            stream,
            attn_out_residual_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_attn_out_residual_ms),
        )?;
        let grad_attn_result = &block_cache.grad_attn_result;
        let o_w = self.weights.qo_bank.slice_first(n + layer)?;
        let o_w_bf16 = self.weights.qo_bank_bf16.slice_first(n + layer)?;
        let attn_weight_input =
            if self.config.attn_out_gate_enabled || self.config.sparse_attn_gate_enabled {
                act!(attn_gated)
            } else if layer >= n.saturating_sub(self.config.xsa_last_n) {
                act!(xsa_out)
            } else {
                act!(attn_out)
            };
        let attn_out_proj_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.use_bf16_backward_gemm() {
            self.linear_backward_input_and_weight_dy_x_bf16_ready_with_role(
                grad_proj_out_bf16,
                act_bf16!(attn_weight_input_bf16, &buf.x_in_bf16),
                &o_w_bf16,
                grad_attn_result,
                &grads.qo_bank.slice_first(n + layer)?,
                t,
                d,
                d,
                0.0,
                bank_dw_beta,
                LinearBackwardOverlapRole::AttnOut,
            )?;
        } else {
            self.linear_backward_input_and_weight_impl(
                grad_proj_out,
                &buf.x_aux_bf16,
                attn_weight_input,
                act_bf16!(attn_weight_input_bf16, &buf.x_in_bf16),
                &o_w,
                &o_w_bf16,
                grad_attn_result,
                &grads.qo_bank.slice_first(n + layer)?,
                t,
                d,
                d,
                saved_bf16_direct.is_none(),
                0.0,
                bank_dw_beta,
            )?;
        }
        finish_stage_event_optional(
            stream,
            attn_out_proj_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_attn_out_proj_ms),
        )?;

        let grad_attn_out = &block_cache.grad_attn_out;
        let grad_v_xsa = &block_cache.grad_v_xsa;
        // Consumer-side freshness gate (backward): if forward used the prepacked
        // BF16 producer for this layer, the saved BF16 Q/K/V buffers must still
        // belong to THIS step and layer. Catches stale-cache reuse if backward
        // is somehow invoked without a fresh forward, or if the backward layer
        // order desyncs from the forward layer order.
        if self.use_cudnn_prepacked_bf16_attention() {
            if let Some(saved) = saved_direct {
                saved.bf16_qkv_freshness.require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    forward_generation,
                    layer,
                    t,
                    runtime_seq_len,
                    h,
                    hkv,
                    hd,
                )?;
            }
        }
        let saved_attention_bf16 = if self.use_cudnn_saved_bf16_attention() {
            if let Some(saved) = saved_direct {
                Some((
                    saved.q_bhsd_bf16.cu_ptr(stream)?,
                    saved.k_bhsd_bf16.cu_ptr(stream)?,
                    saved.v_bhsd_bf16.cu_ptr(stream)?,
                    saved.attn_out_bhsd_bf16.cu_ptr(stream)?,
                ))
            } else {
                None
            }
        } else {
            None
        };
        let is_xsa_layer = layer >= n.saturating_sub(self.config.xsa_last_n);
        let mut fused_sparse_xsa_bwd = false;
        let use_bf16_attention_backward_bhsd_do = self.use_bf16_attention_backward_bhsd_do()
            && is_xsa_layer
            && self.use_skip_f32_attention_saved_acts();
        let compact_attn_gate_grad_input = gpu_compact_attn_gate_grad_input_enabled()
            && self.config.sparse_attn_gate_enabled
            && !self.config.attn_out_gate_enabled
            && is_xsa_layer
            && self.use_skip_f32_attention_saved_acts()
            && saved_attention_bf16.is_some()
            && self.use_fused_qkv_projection()
            && self.use_bf16_qkv_dx_output()
            && !self.any_lora_enabled()
            && gpu_sparse_xsa_warphead_backward_enabled()
            && !gpu_sparse_xsa_grouped_kv_backward_enabled();
        let compact_attn_gate_grad_width = self.config.sparse_attn_gate_width;
        let mut grad_attn_out_bhsd_bf16_ptr: Option<u64> = None;
        let attn_out_gate_xsa_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let grad_attn_pre_gate = if self.config.attn_out_gate_enabled {
            let raw_src = if layer >= n.saturating_sub(self.config.xsa_last_n) {
                act!(xsa_out)
            } else {
                act!(attn_out)
            };
            let grad_raw = &block_cache.grad_raw;
            self.kernels.scale_inplace(
                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                0.0,
                (t * d) as u32,
            )?;
            self.kernels.attn_out_gate_bwd(
                CudaPtr(raw_src.cu_ptr(stream)?),
                CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                CudaPtr(grad_attn_result.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(grad_raw.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                CudaPtr(grads.block_attn_gate_weight[layer].cu_ptr(stream)?),
                CudaPtr(grads.block_attn_gate_bias[layer].cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.attn_out_gate_width as u32,
            )?;
            grad_raw
        } else if self.config.sparse_attn_gate_enabled {
            if compact_attn_gate_grad_input {
                self.kernels.scale_inplace(
                    CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                    0.0,
                    (t * compact_attn_gate_grad_width) as u32,
                )?;
            } else {
                self.kernels.scale_inplace(
                    CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                    0.0,
                    (t * d) as u32,
                )?;
            }
            if is_xsa_layer && self.use_skip_f32_attention_saved_acts() {
                if let Some((_, _, saved_v_bhsd, saved_attn_out_bhsd)) = saved_attention_bf16 {
                    if t % runtime_seq_len != 0 {
                        return Err(PgError::InvalidOp(format!(
                            "saved BF16 SparseAttnGate+XSA backward requires tokens ({t}) to be divisible by seq_len ({runtime_seq_len})"
                        )));
                    }
                    self.zero_tensor(grad_v_xsa)?;
                    if gpu_sparse_xsa_grouped_kv_backward_enabled()
                        && use_bf16_attention_backward_bhsd_do
                    {
                        let do_bhsd_bf16 = buf.qkv_aux_bf16.cu_ptr(stream)?;
                        self.kernels
                            .sparse_attn_gate_xsa_bwd_bf16_bhsd_grouped_kv_do_bf16_two_pass(
                                CudaPtr(saved_attn_out_bhsd),
                                CudaPtr(saved_v_bhsd),
                                CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                CudaPtr(
                                    self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?,
                                ),
                                CudaPtr(do_bhsd_bf16),
                                CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                                CudaPtr(grads.block_sparse_attn_gate_weight[layer].cu_ptr(stream)?),
                                (t / runtime_seq_len) as u32,
                                runtime_seq_len as u32,
                                h as u32,
                                hkv as u32,
                                hd as u32,
                                d as u32,
                                self.config.sparse_attn_gate_width as u32,
                                self.config.sparse_attn_gate_scale,
                            )?;
                        grad_attn_out_bhsd_bf16_ptr = Some(do_bhsd_bf16);
                    } else if gpu_sparse_xsa_grouped_kv_backward_enabled()
                        && compact_attn_gate_grad_input
                    {
                        self.kernels
                            .sparse_attn_gate_xsa_bwd_bf16_bhsd_grouped_kv_two_pass_compact_grad_input(
                                CudaPtr(saved_attn_out_bhsd),
                                CudaPtr(saved_v_bhsd),
                                CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                CudaPtr(
                                    self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?,
                                ),
                                CudaPtr(grad_attn_out.cu_ptr(stream)?),
                                CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                                CudaPtr(grads.block_sparse_attn_gate_weight[layer].cu_ptr(stream)?),
                                (t / runtime_seq_len) as u32,
                                runtime_seq_len as u32,
                                h as u32,
                                hkv as u32,
                                hd as u32,
                                d as u32,
                                compact_attn_gate_grad_width as u32,
                                self.config.sparse_attn_gate_scale,
                            )?;
                    } else if gpu_sparse_xsa_grouped_kv_backward_enabled()
                        && !compact_attn_gate_grad_input
                    {
                        self.kernels
                            .sparse_attn_gate_xsa_bwd_bf16_bhsd_grouped_kv_two_pass(
                                CudaPtr(saved_attn_out_bhsd),
                                CudaPtr(saved_v_bhsd),
                                CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                CudaPtr(
                                    self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?,
                                ),
                                CudaPtr(grad_attn_out.cu_ptr(stream)?),
                                CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                                CudaPtr(grads.block_sparse_attn_gate_weight[layer].cu_ptr(stream)?),
                                (t / runtime_seq_len) as u32,
                                runtime_seq_len as u32,
                                h as u32,
                                hkv as u32,
                                hd as u32,
                                d as u32,
                                self.config.sparse_attn_gate_width as u32,
                                self.config.sparse_attn_gate_scale,
                            )?;
                    } else if gpu_sparse_xsa_warphead_backward_enabled() {
                        if use_bf16_attention_backward_bhsd_do {
                            let do_bhsd_bf16 = buf.qkv_aux_bf16.cu_ptr(stream)?;
                            if compact_attn_gate_grad_input {
                                self.kernels
                                    .sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads_do_bf16_two_pass_compact_grad_input(
                                        CudaPtr(saved_attn_out_bhsd),
                                        CudaPtr(saved_v_bhsd),
                                        CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                        CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                        CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                        CudaPtr(
                                            self.weights.sparse_attn_gate_weights[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        CudaPtr(do_bhsd_bf16),
                                        CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                        CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                                        CudaPtr(
                                            grads.block_sparse_attn_gate_weight[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        (t / runtime_seq_len) as u32,
                                        runtime_seq_len as u32,
                                        h as u32,
                                        hkv as u32,
                                        hd as u32,
                                        d as u32,
                                        compact_attn_gate_grad_width as u32,
                                        self.config.sparse_attn_gate_scale,
                                    )?;
                            } else {
                                self.kernels
                                    .sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads_do_bf16_two_pass(
                                        CudaPtr(saved_attn_out_bhsd),
                                        CudaPtr(saved_v_bhsd),
                                        CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                        CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                        CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                        CudaPtr(
                                            self.weights.sparse_attn_gate_weights[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        CudaPtr(do_bhsd_bf16),
                                        CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                        CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                                        CudaPtr(
                                            grads.block_sparse_attn_gate_weight[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        (t / runtime_seq_len) as u32,
                                        runtime_seq_len as u32,
                                        h as u32,
                                        hkv as u32,
                                        hd as u32,
                                        d as u32,
                                        self.config.sparse_attn_gate_width as u32,
                                        self.config.sparse_attn_gate_scale,
                                    )?;
                            }
                            grad_attn_out_bhsd_bf16_ptr = Some(do_bhsd_bf16);
                        } else {
                            if compact_attn_gate_grad_input {
                                self.kernels
                                    .sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads_two_pass_compact_grad_input(
                                        CudaPtr(saved_attn_out_bhsd),
                                        CudaPtr(saved_v_bhsd),
                                        CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                        CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                        CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                        CudaPtr(
                                            self.weights.sparse_attn_gate_weights[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        CudaPtr(grad_attn_out.cu_ptr(stream)?),
                                        CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                        CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                                        CudaPtr(
                                            grads.block_sparse_attn_gate_weight[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        (t / runtime_seq_len) as u32,
                                        runtime_seq_len as u32,
                                        h as u32,
                                        hkv as u32,
                                        hd as u32,
                                        d as u32,
                                        compact_attn_gate_grad_width as u32,
                                        self.config.sparse_attn_gate_scale,
                                    )?;
                            } else {
                                self.kernels
                                    .sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads_two_pass(
                                        CudaPtr(saved_attn_out_bhsd),
                                        CudaPtr(saved_v_bhsd),
                                        CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                        CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                        CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                        CudaPtr(
                                            self.weights.sparse_attn_gate_weights[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        CudaPtr(grad_attn_out.cu_ptr(stream)?),
                                        CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                        CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                                        CudaPtr(
                                            grads.block_sparse_attn_gate_weight[layer]
                                                .cu_ptr(stream)?,
                                        ),
                                        (t / runtime_seq_len) as u32,
                                        runtime_seq_len as u32,
                                        h as u32,
                                        hkv as u32,
                                        hd as u32,
                                        d as u32,
                                        self.config.sparse_attn_gate_width as u32,
                                        self.config.sparse_attn_gate_scale,
                                    )?;
                            }
                        }
                    } else {
                        if use_bf16_attention_backward_bhsd_do {
                            let do_bhsd_bf16 = buf.qkv_aux_bf16.cu_ptr(stream)?;
                            self.kernels
                                .sparse_attn_gate_xsa_bwd_bf16_bhsd_do_bf16_two_pass(
                                    CudaPtr(saved_attn_out_bhsd),
                                    CudaPtr(saved_v_bhsd),
                                    CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                    CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                    CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                    CudaPtr(
                                        self.weights.sparse_attn_gate_weights[layer]
                                            .cu_ptr(stream)?,
                                    ),
                                    CudaPtr(do_bhsd_bf16),
                                    CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                    CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                                    CudaPtr(
                                        grads.block_sparse_attn_gate_weight[layer]
                                            .cu_ptr(stream)?,
                                    ),
                                    (t / runtime_seq_len) as u32,
                                    runtime_seq_len as u32,
                                    h as u32,
                                    hkv as u32,
                                    hd as u32,
                                    d as u32,
                                    self.config.sparse_attn_gate_width as u32,
                                    self.config.sparse_attn_gate_scale,
                                )?;
                            grad_attn_out_bhsd_bf16_ptr = Some(do_bhsd_bf16);
                        } else {
                            self.kernels.sparse_attn_gate_xsa_bwd_bf16_bhsd_two_pass(
                                CudaPtr(saved_attn_out_bhsd),
                                CudaPtr(saved_v_bhsd),
                                CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                                CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                                CudaPtr(grad_attn_result.cu_ptr(stream)?),
                                CudaPtr(
                                    self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?,
                                ),
                                CudaPtr(grad_attn_out.cu_ptr(stream)?),
                                CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                                CudaPtr(grads.block_sparse_attn_gate_weight[layer].cu_ptr(stream)?),
                                (t / runtime_seq_len) as u32,
                                runtime_seq_len as u32,
                                h as u32,
                                hkv as u32,
                                hd as u32,
                                d as u32,
                                self.config.sparse_attn_gate_width as u32,
                                self.config.sparse_attn_gate_scale,
                            )?;
                        }
                    }
                    fused_sparse_xsa_bwd = true;
                    grad_attn_out
                } else {
                    let raw_src = act!(xsa_out);
                    let grad_raw = &block_cache.grad_raw;
                    self.kernels.sparse_attn_gate_bwd_two_pass(
                        CudaPtr(raw_src.cu_ptr(stream)?),
                        CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                        CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                        CudaPtr(grad_attn_result.cu_ptr(stream)?),
                        CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                        CudaPtr(grad_raw.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                        CudaPtr(grads.block_sparse_attn_gate_weight[layer].cu_ptr(stream)?),
                        t as u32,
                        h as u32,
                        hd as u32,
                        d as u32,
                        self.config.sparse_attn_gate_width as u32,
                        self.config.sparse_attn_gate_scale,
                    )?;
                    grad_raw
                }
            } else {
                let raw_src = if is_xsa_layer {
                    act!(xsa_out)
                } else {
                    act!(attn_out)
                };
                let grad_raw = &block_cache.grad_raw;
                self.kernels.sparse_attn_gate_bwd_two_pass(
                    CudaPtr(raw_src.cu_ptr(stream)?),
                    CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                    CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                    CudaPtr(grad_attn_result.cu_ptr(stream)?),
                    CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                    CudaPtr(grad_raw.cu_ptr(stream)?),
                    CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                    CudaPtr(grads.block_sparse_attn_gate_weight[layer].cu_ptr(stream)?),
                    t as u32,
                    h as u32,
                    hd as u32,
                    d as u32,
                    self.config.sparse_attn_gate_width as u32,
                    self.config.sparse_attn_gate_scale,
                )?;
                grad_raw
            }
        } else {
            grad_attn_result
        };

        if is_xsa_layer {
            if !fused_sparse_xsa_bwd {
                self.zero_tensor(grad_v_xsa)?;
            }
            if self.use_skip_f32_attention_saved_acts() {
                if let Some((_, _, saved_v_bhsd, saved_attn_out_bhsd)) = saved_attention_bf16 {
                    if fused_sparse_xsa_bwd {
                        // The fused SparseAttnGate+XSA path has already produced
                        // grad_attn_out and accumulated the XSA contribution to grad_v_xsa.
                    } else if t % runtime_seq_len != 0 {
                        return Err(PgError::InvalidOp(format!(
                            "saved BF16 XSA backward requires tokens ({t}) to be divisible by seq_len ({runtime_seq_len})"
                        )));
                    } else {
                        self.kernels.xsa_bwd_bf16_bhsd(
                            CudaPtr(saved_attn_out_bhsd),
                            CudaPtr(saved_v_bhsd),
                            CudaPtr(grad_attn_pre_gate.cu_ptr(stream)?),
                            CudaPtr(grad_attn_out.cu_ptr(stream)?),
                            CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                            (t / runtime_seq_len) as u32,
                            runtime_seq_len as u32,
                            h as u32,
                            hkv as u32,
                            hd as u32,
                        )?;
                    }
                } else {
                    self.kernels.xsa_bwd(
                        CudaPtr(act!(attn_out).cu_ptr(stream)?),
                        CudaPtr(act!(v).cu_ptr(stream)?),
                        CudaPtr(grad_attn_pre_gate.cu_ptr(stream)?),
                        CudaPtr(grad_attn_out.cu_ptr(stream)?),
                        CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                        t as u32,
                        h as u32,
                        hkv as u32,
                        hd as u32,
                    )?;
                }
            } else {
                self.kernels.xsa_bwd(
                    CudaPtr(act!(attn_out).cu_ptr(stream)?),
                    CudaPtr(act!(v).cu_ptr(stream)?),
                    CudaPtr(grad_attn_pre_gate.cu_ptr(stream)?),
                    CudaPtr(grad_attn_out.cu_ptr(stream)?),
                    CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                    t as u32,
                    h as u32,
                    hkv as u32,
                    hd as u32,
                )?;
            }
        } else {
            self.copy_tensor(&grad_attn_pre_gate, &grad_attn_out)?;
        }
        finish_stage_event_optional(
            stream,
            attn_out_gate_xsa_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_attn_out_gate_xsa_ms),
        )?;
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_attn_out_ms),
        )?;
        check_cuda_graph_capture_stage(stream, &format!("block_{layer}_attn_out_backward"))?;

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let grad_q_post_gain = &block_cache.grad_q_post_gain;
        let grad_k_attn = &block_cache.grad_k_attn;
        let grad_v_projection = &block_cache.grad_v_projection;
        let grad_q_post_gain_bf16 = &block_cache.grad_q_post_gain_bf16;
        let grad_k_attn_bf16 = &block_cache.grad_k_attn_bf16;
        let grad_v_projection_bf16 = &block_cache.grad_v_projection_bf16;
        let attn_stats = if let Some(saved) = saved {
            Some(saved.attn_stats.cu_ptr(stream)?)
        } else {
            Some(block_cache.attn_stats.cu_ptr(stream)?)
        };
        let use_bf16_attention_backward_tail = self.use_bf16_attention_backward_tail()
            && saved_attention_bf16.is_some()
            && attn_stats.is_some();
        let use_bf16_attention_tail_qkv_pack =
            use_bf16_attention_backward_tail && self.use_bf16_attention_tail_qkv_pack();
        let use_bf16_attention_tail_direct_qkv_pack =
            use_bf16_attention_tail_qkv_pack && self.use_bf16_attention_tail_direct_qkv_pack();
        let add_v_xsa = layer >= n.saturating_sub(self.config.xsa_last_n);
        if gpu_bf16_backward_chain_strict() {
            if !self.use_cudnn_prepacked_bf16_attention() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires PG_GPU_CUDNN_PREPACKED_BF16_ATTN with the fused QK/RoPE/gain forward producer; refusing saved-BF16 bridge mode".into(),
                ));
            }
            if !self.use_skip_f32_attention_saved_acts() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires direct BF16 saved attention activations with F32 attention saves skipped; refusing F32 saved-attention fallback".into(),
                ));
            }
            if self.any_lora_enabled() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires LoRA-TTT adapters disabled; LoRA backward still depends on F32 adapter paths".into(),
                ));
            }
            if !self.config.ve_layers.is_empty() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires ve_layers disabled; VE backward still consumes the F32 V gradient path".into(),
                ));
            }
            if saved_attention_bf16.is_none() || attn_stats.is_none() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires direct saved BF16 cuDNN attention tensors and stats for every backward layer; refusing F32 attention-backward fallback".into(),
                ));
            }
            if !use_bf16_attention_backward_tail {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires BF16 cuDNN SDPA backward outputs; refusing F32 dQ/dK/dV fallback".into(),
                ));
            }
            if !use_bf16_attention_tail_qkv_pack {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires BF16 tail QKV packing into the fused QKV backward GEMM; refusing non-packed QKV backward fallback".into(),
                ));
            }
            if !use_bf16_attention_tail_direct_qkv_pack {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires direct BF16 Q/K/V gradient packing through QK/RoPE/gain; refusing intermediate F32 QKV-gradient materialization".into(),
                ));
            }
            if recompute_residual_mix_norm_inputs {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires saved residual/norm inputs; recompute would re-enter the F32 QKV norm/resid tail".into(),
                ));
            }
            if gpu_split_residual_mix_grad_enabled() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN refuses PG_GPU_SPLIT_RESIDUAL_MIX_GRAD; the validated BF16 chain uses the compact/split QKV norm-resid reducer instead".into(),
                ));
            }
            if self.config.sparse_attn_gate_enabled && !gpu_compact_attn_gate_grad_input_enabled() {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires compact SparseAttnGate grad input so the QKV norm/resid reducer does not add a full [tokens, d] F32 gate gradient".into(),
                ));
            }
            if self.config.sparse_attn_gate_enabled
                && add_v_xsa
                && grad_attn_out_bhsd_bf16_ptr.is_none()
            {
                return Err(pg_core::PgError::InvalidOp(
                    "PG_GPU_BF16_BACKWARD_CHAIN requires SparseAttnGate/XSA backward to produce BF16 [B,H,S,D] dO for cuDNN SDPA backward; refusing F32 dO bridge".into(),
                ));
            }
        }
        let attention_sdpa_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if use_bf16_attention_backward_tail {
            let (q_bf16, k_bf16, v_bf16, out_bf16) =
                saved_attention_bf16.expect("checked by use_bf16_attention_backward_tail");
            if let Some(grad_out_bhsd_bf16) = grad_attn_out_bhsd_bf16_ptr {
                self.run_attention_backward_bf16_bhsd_do_bf16_grads(
                    q_bf16,
                    k_bf16,
                    v_bf16,
                    out_bf16,
                    grad_out_bhsd_bf16,
                    grad_q_post_gain_bf16.cu_ptr(stream)?,
                    grad_k_attn_bf16.cu_ptr(stream)?,
                    grad_v_projection_bf16.cu_ptr(stream)?,
                    t,
                    h,
                    hkv,
                    hd,
                    runtime_seq_len,
                    attn_stats.expect("checked by use_bf16_attention_backward_tail"),
                )?;
            } else {
                self.run_attention_backward_bf16_grads(
                    q_bf16,
                    k_bf16,
                    v_bf16,
                    out_bf16,
                    grad_attn_out.cu_ptr(stream)?,
                    grad_q_post_gain_bf16.cu_ptr(stream)?,
                    grad_k_attn_bf16.cu_ptr(stream)?,
                    grad_v_projection_bf16.cu_ptr(stream)?,
                    t,
                    h,
                    hkv,
                    hd,
                    runtime_seq_len,
                    attn_stats.expect("checked by use_bf16_attention_backward_tail"),
                )?;
            }
            if !use_bf16_attention_tail_qkv_pack {
                self.kernels.bf16_to_f32(
                    CudaPtr(grad_v_projection_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_v_projection.cu_ptr(stream)?),
                    (t * kv) as u32,
                )?;
            }
        } else {
            self.run_attention_backward(
                act!(q).cu_ptr(stream)?,
                act!(k).cu_ptr(stream)?,
                act!(v).cu_ptr(stream)?,
                act!(attn_out).cu_ptr(stream)?,
                grad_attn_out.cu_ptr(stream)?,
                grad_q_post_gain.cu_ptr(stream)?,
                grad_k_attn.cu_ptr(stream)?,
                grad_v_projection.cu_ptr(stream)?,
                t,
                h,
                hkv,
                hd,
                runtime_seq_len,
                attn_stats,
                saved_attention_bf16,
            )?;
        }
        finish_stage_event_optional(
            stream,
            attention_sdpa_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_attention_sdpa_ms),
        )?;
        if add_v_xsa && !use_bf16_attention_tail_qkv_pack {
            let attention_xsa_accum_start = record_stage_event_if(stream, stage_timing.is_some())?;
            self.add_inplace(&grad_v_projection, &grad_v_xsa, 1.0)?;
            finish_stage_event_optional(
                stream,
                attention_xsa_accum_start,
                stage_timing
                    .as_deref_mut()
                    .map(|timing| &mut timing.backward_block_attention_xsa_accum_ms),
            )?;
        }
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_attention_ms),
        )?;
        check_cuda_graph_capture_stage(stream, &format!("block_{layer}_sdpa_backward"))?;

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;

        let grad_q_proj = &block_cache.grad_q_proj;
        let grad_k_proj = &block_cache.grad_k_proj;
        let grad_q_proj_bf16 = &block_cache.grad_q_proj_bf16;
        let grad_k_proj_bf16 = &block_cache.grad_k_proj_bf16;
        let qkv_rope_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let q_gain_chunk_tokens = gpu_q_gain_backward_chunk_tokens();
        if use_bf16_attention_tail_direct_qkv_pack {
            let q_gain_chunks = t.div_ceil(q_gain_chunk_tokens).max(1);
            self.zero_tensor(&block_cache.q_gain_reduce_scratch)?;
            if gpu_combined_qkv_rope_tail_backward_enabled() {
                self.kernels.qkv_rope_qk_norm_bwd_chunked_go_bf16_pack(
                    CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                    CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                    CudaPtr(grad_q_post_gain_bf16.cu_ptr(stream)?),
                    CudaPtr(block_act!(k_pre_norm).cu_ptr(stream)?),
                    CudaPtr(grad_k_attn_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_v_projection_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    CudaPtr(buf.qkv_aux_bf16.cu_ptr(stream)?),
                    CudaPtr(block_cache.q_gain_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hkv as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    t as u32,
                    q_gain_chunk_tokens as u32,
                    q_gain_chunks as u32,
                    d as u32,
                    kv as u32,
                    add_v_xsa,
                    1e-6,
                )?;
            } else {
                self.kernels
                    .q_gain_rope_qk_norm_bwd_chunked_go_bf16_pack_q(
                        CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                        CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                        CudaPtr(grad_q_post_gain_bf16.cu_ptr(stream)?),
                        CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                        CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                        CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                        CudaPtr(buf.qkv_aux_bf16.cu_ptr(stream)?),
                        CudaPtr(block_cache.q_gain_reduce_scratch.cu_ptr(stream)?),
                        CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                        runtime_seq_len as u32,
                        h as u32,
                        hd as u32,
                        self.config.rope_dims as u32,
                        (t * h) as u32,
                        q_gain_chunk_tokens as u32,
                        q_gain_chunks as u32,
                        (d + 2 * kv) as u32,
                        1e-6,
                    )?;
                self.kernels.rope_qk_norm_bwd_go_bf16_pack_kv(
                    CudaPtr(block_act!(k_pre_norm).cu_ptr(stream)?),
                    CudaPtr(grad_k_attn_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_v_projection_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_v_xsa.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    CudaPtr(buf.qkv_aux_bf16.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    hkv as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    (t * hkv) as u32,
                    d as u32,
                    kv as u32,
                    add_v_xsa,
                    1e-6,
                )?;
            }
        } else if use_bf16_attention_tail_qkv_pack {
            let q_gain_chunks = t.div_ceil(q_gain_chunk_tokens).max(1);
            self.zero_tensor(&block_cache.q_gain_reduce_scratch)?;
            self.kernels
                .q_gain_rope_qk_norm_bwd_chunked_go_bf16_out_bf16(
                    CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                    CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                    CudaPtr(grad_q_post_gain_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    CudaPtr(grad_q_proj_bf16.cu_ptr(stream)?),
                    CudaPtr(block_cache.q_gain_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    (t * h) as u32,
                    q_gain_chunk_tokens as u32,
                    q_gain_chunks as u32,
                    1e-6,
                )?;
            self.kernels.rope_qk_norm_bwd_go_bf16_out_bf16(
                CudaPtr(block_act!(k_pre_norm).cu_ptr(stream)?),
                CudaPtr(grad_k_attn_bf16.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                CudaPtr(grad_k_proj_bf16.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * hkv) as u32,
                1e-6,
            )?;
        } else if use_bf16_attention_backward_tail {
            let q_gain_chunks = t.div_ceil(q_gain_chunk_tokens).max(1);
            self.zero_tensor(&block_cache.q_gain_reduce_scratch)?;
            self.kernels.q_gain_rope_qk_norm_bwd_chunked_go_bf16(
                CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                CudaPtr(grad_q_post_gain_bf16.cu_ptr(stream)?),
                CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                CudaPtr(grad_q_proj.cu_ptr(stream)?),
                CudaPtr(block_cache.q_gain_reduce_scratch.cu_ptr(stream)?),
                CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                runtime_seq_len as u32,
                h as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * h) as u32,
                q_gain_chunk_tokens as u32,
                q_gain_chunks as u32,
                1e-6,
            )?;
            self.kernels.rope_qk_norm_bwd_go_bf16(
                CudaPtr(block_act!(k_pre_norm).cu_ptr(stream)?),
                CudaPtr(grad_k_attn_bf16.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                CudaPtr(grad_k_proj.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * hkv) as u32,
                1e-6,
            )?;
        } else if self.use_fused_qk_rope_gain_backward() {
            if gpu_chunked_q_gain_backward_enabled() {
                let q_gain_chunks = t.div_ceil(q_gain_chunk_tokens).max(1);
                self.zero_tensor(&block_cache.q_gain_reduce_scratch)?;
                self.kernels.q_gain_rope_qk_norm_bwd_chunked(
                    CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                    CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                    CudaPtr(grad_q_post_gain.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    CudaPtr(grad_q_proj.cu_ptr(stream)?),
                    CudaPtr(block_cache.q_gain_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    (t * h) as u32,
                    q_gain_chunk_tokens as u32,
                    q_gain_chunks as u32,
                    1e-6,
                )?;
            } else {
                self.kernels.q_gain_rope_qk_norm_bwd(
                    CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                    CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                    CudaPtr(grad_q_post_gain.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    CudaPtr(grad_q_proj.cu_ptr(stream)?),
                    CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    (t * h) as u32,
                    1e-6,
                )?;
            }
            self.kernels.rope_qk_norm_bwd(
                CudaPtr(block_act!(k_pre_norm).cu_ptr(stream)?),
                CudaPtr(grad_k_attn.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                CudaPtr(grad_k_proj.cu_ptr(stream)?),
                runtime_seq_len as u32,
                hkv as u32,
                hd as u32,
                self.config.rope_dims as u32,
                (t * hkv) as u32,
                1e-6,
            )?;
        } else {
            let grad_q_post_rope = &block_cache.grad_q_post_rope;
            self.kernels.q_gain_bwd(
                CudaPtr(block_act!(q_post_rope).cu_ptr(stream)?),
                CudaPtr(grad_q_post_gain.cu_ptr(stream)?),
                CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                CudaPtr(grad_q_post_rope.cu_ptr(stream)?),
                CudaPtr(grads.block_q_gain[layer].cu_ptr(stream)?),
                h as u32,
                hd as u32,
                (t * h) as u32,
            )?;
            if self.config.rope_dims > 0 {
                self.kernels.partial_rope_bwd(
                    CudaPtr(grad_q_post_rope.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    (t * h) as u32,
                )?;
                self.kernels.partial_rope_bwd(
                    CudaPtr(grad_k_attn.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    runtime_seq_len as u32,
                    hkv as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    (t * hkv) as u32,
                )?;
            }
            self.kernels.qk_norm_bwd(
                CudaPtr(block_act!(q_pre_norm).cu_ptr(stream)?),
                CudaPtr(grad_q_post_rope.cu_ptr(stream)?),
                CudaPtr(grad_q_proj.cu_ptr(stream)?),
                hd as u32,
                (t * h) as u32,
                1e-6,
            )?;
            self.kernels.qk_norm_bwd(
                CudaPtr(block_act!(k_pre_norm).cu_ptr(stream)?),
                CudaPtr(grad_k_attn.cu_ptr(stream)?),
                CudaPtr(grad_k_proj.cu_ptr(stream)?),
                hd as u32,
                (t * hkv) as u32,
                1e-6,
            )?;
        }
        finish_stage_event_optional(
            stream,
            qkv_rope_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_qkv_rope_ms),
        )?;

        let grad_attn_norm = &block_cache.grad_attn_norm;
        if !use_bf16_attention_tail_qkv_pack {
            self.wait_deferred_side_weight_gemms()?;
        }
        let qkv_proj_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let qkv_fused = if use_bf16_attention_tail_qkv_pack {
            self.qkv_projection_backward_from_tail_bf16(
                layer,
                buf,
                block_cache,
                grads,
                grad_q_proj_bf16,
                grad_k_proj_bf16,
                grad_v_projection_bf16,
                grad_v_xsa,
                add_v_xsa,
                use_bf16_attention_tail_direct_qkv_pack,
                saved_direct.map(|saved| &saved.attn_norm),
                saved_bf16_direct.map(|saved| &saved.attn_norm_bf16),
                t,
                bank_dw_beta,
            )?
        } else {
            self.qkv_projection_backward(
                layer,
                buf,
                block_cache,
                grads,
                grad_q_proj,
                grad_k_proj,
                grad_v_projection,
                grad_attn_norm,
                saved_direct.map(|saved| &saved.attn_norm),
                saved_bf16_direct.map(|saved| &saved.attn_norm_bf16),
                t,
                bank_dw_beta,
            )?
        };
        if use_bf16_attention_tail_qkv_pack && !qkv_fused {
            return Err(pg_core::PgError::InvalidOp(
                "BF16 attention tail QKV pack requires fused QKV projection backward".into(),
            ));
        }
        if !qkv_fused {
            let q_w = self.weights.qo_bank.slice_first(layer)?;
            let q_w_bf16 = self.weights.qo_bank_bf16.slice_first(layer)?;
            let attn_norm_bf16 = act_bf16!(attn_norm_bf16, &buf.x_in_bf16);
            if saved_bf16_direct.is_none() && self.use_bf16_backward_gemm() {
                self.kernels.f32_to_bf16(
                    CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
            }
            self.linear_backward_input_and_weight_x_bf16_ready_with_betas(
                grad_q_proj,
                &buf.x_aux_bf16,
                act!(attn_norm),
                attn_norm_bf16,
                &q_w,
                &q_w_bf16,
                grad_attn_norm,
                &grads.qo_bank.slice_first(layer)?,
                t,
                d,
                d,
                0.0,
                bank_dw_beta,
            )?;

            let k_w = self.weights.kv_bank.slice_first(layer)?;
            let k_w_bf16 = self.weights.kv_bank_bf16.slice_first(layer)?;
            if self.use_qkv_dx_beta_accum() {
                self.linear_backward_input_and_weight_x_bf16_ready_with_betas(
                    grad_k_proj,
                    &buf.x_aux_bf16,
                    act!(attn_norm),
                    attn_norm_bf16,
                    &k_w,
                    &k_w_bf16,
                    grad_attn_norm,
                    &grads.kv_bank.slice_first(layer)?,
                    t,
                    kv,
                    d,
                    1.0,
                    bank_dw_beta,
                )?;
            } else {
                let grad_attn_norm_k = &block_cache.grad_attn_norm_k;
                self.linear_backward_input_and_weight_x_bf16_ready_with_betas(
                    grad_k_proj,
                    &buf.x_aux_bf16,
                    act!(attn_norm),
                    attn_norm_bf16,
                    &k_w,
                    &k_w_bf16,
                    grad_attn_norm_k,
                    &grads.kv_bank.slice_first(layer)?,
                    t,
                    kv,
                    d,
                    0.0,
                    bank_dw_beta,
                )?;
                self.add_inplace(&grad_attn_norm, grad_attn_norm_k, 1.0)?;
            }
        }

        if let Some(ve_idx) = self.config.ve_layers.iter().position(|&l| l == layer) {
            let qkv_ve_start = record_stage_event_if(stream, stage_timing.is_some())?;
            self.kernels.dot_accumulate_by_param(
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(act!(ve_out).cu_ptr(stream)?),
                CudaPtr(grads.ve_scale.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_layer_scales.cu_ptr(stream)?),
                ve_idx as u32,
                1.0,
                (t * kv) as u32,
            )?;
            self.kernels.dot_accumulate_by_param(
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(act!(ve_out).cu_ptr(stream)?),
                CudaPtr(grads.ve_layer_scales.slice_first(ve_idx)?.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_scale_param.cu_ptr(stream)?),
                0,
                1.0,
                (t * kv) as u32,
            )?;

            let grad_projected = &block_cache.grad_projected;
            self.zero_tensor(grad_projected)?;
            self.kernels.add_scaled_by_param_product_fwd(
                CudaPtr(grad_projected.cu_ptr(stream)?),
                CudaPtr(grad_v_projection.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_scale_param.cu_ptr(stream)?),
                0,
                CudaPtr(self.weights.ve_layer_scales.cu_ptr(stream)?),
                ve_idx as u32,
                1.0,
                (t * kv) as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_weight_f32(
                    grad_projected.cu_ptr(stream)?,
                    act!(ve_embed_out).cu_ptr(stream)?,
                    grads.ve_proj.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    1.0,
                )?;
            }
            let grad_ve_embed_out = &block_cache.grad_ve_embed_out;
            unsafe {
                self.gemm.linear_backward_input_f32(
                    grad_projected.cu_ptr(stream)?,
                    self.weights.ve_proj.cu_ptr(stream)?,
                    grad_ve_embed_out.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.embedding_gather_bwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(grad_ve_embed_out.cu_ptr(stream)?),
                CudaPtr(grads.ve_embed.cu_ptr(stream)?),
                self.config.ve_dim as u32,
                t as u32,
            )?;
            finish_stage_event_optional(
                stream,
                qkv_ve_start,
                stage_timing
                    .as_deref_mut()
                    .map(|timing| &mut timing.backward_block_qkv_ve_ms),
            )?;
        }

        if !qkv_fused {
            let v_w = self.weights.kv_bank.slice_first(n + layer)?;
            let v_w_bf16 = self.weights.kv_bank_bf16.slice_first(n + layer)?;
            if self.use_qkv_dx_beta_accum() {
                self.linear_backward_input_and_weight_x_bf16_ready_with_betas(
                    grad_v_projection,
                    &buf.x_aux_bf16,
                    act!(attn_norm),
                    act_bf16!(attn_norm_bf16, &buf.x_in_bf16),
                    &v_w,
                    &v_w_bf16,
                    grad_attn_norm,
                    &grads.kv_bank.slice_first(n + layer)?,
                    t,
                    kv,
                    d,
                    1.0,
                    bank_dw_beta,
                )?;
            } else {
                let grad_attn_norm_v = &block_cache.grad_attn_norm_v;
                self.linear_backward_input_and_weight_x_bf16_ready_with_betas(
                    grad_v_projection,
                    &buf.x_aux_bf16,
                    act!(attn_norm),
                    act_bf16!(attn_norm_bf16, &buf.x_in_bf16),
                    &v_w,
                    &v_w_bf16,
                    grad_attn_norm_v,
                    &grads.kv_bank.slice_first(n + layer)?,
                    t,
                    kv,
                    d,
                    0.0,
                    bank_dw_beta,
                )?;
                self.add_inplace(&grad_attn_norm, grad_attn_norm_v, 1.0)?;
            }
            let attn_norm_for_lora = act!(attn_norm).clone();
            self.backward_qkv_loras(
                layer,
                &attn_norm_for_lora,
                &grad_q_proj,
                &grad_k_proj,
                &grad_v_projection,
                &grad_attn_norm,
                buf,
            )?;
        }
        if let Some(lora) = &self.o_lora {
            if self.use_bf16_backward_gemm() {
                self.kernels.bf16_to_f32(
                    CudaPtr(grad_proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_proj_out.cu_ptr(stream)?),
                    grad_proj_out.numel() as u32,
                )?;
            }
            let attn_norm_for_lora = act!(attn_norm).clone();
            let grad_attn_norm_for_lora = grad_attn_norm.clone();
            self.backward_linear_lora(
                lora,
                layer,
                &attn_norm_for_lora,
                grad_proj_out,
                &grad_attn_norm_for_lora,
                buf,
            )?;
        }
        finish_stage_event_optional(
            stream,
            qkv_proj_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_qkv_proj_ms),
        )?;
        check_cuda_graph_capture_stage(stream, &format!("block_{layer}_qkv_projection_backward"))?;
        let qkv_norm_resid_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let qkv_dx_bf16_for_norm_tail =
            qkv_fused && self.use_bf16_qkv_dx_output() && !self.any_qkv_lora_enabled();
        let gate_extra_grad =
            self.config.attn_out_gate_enabled || self.config.sparse_attn_gate_enabled;
        if gate_extra_grad && !qkv_dx_bf16_for_norm_tail {
            self.add_inplace(&grad_attn_norm, &buf.attn_gate_grad_input, 1.0)?;
        }

        if qkv_dx_bf16_for_norm_tail
            && !recompute_residual_mix_norm_inputs
            && !gpu_split_residual_mix_grad_enabled()
            && (gpu_chunked_qkv_norm_resid_backward_enabled()
                || !gpu_chunked_residual_mix_backward_enabled())
        {
            if gpu_split_qkv_norm_resid_backward_enabled()
                && gate_extra_grad
                && compact_attn_gate_grad_input
            {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_compact_split_reduce(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(block_cache.residual_mix_norm_stats.cu_ptr(stream)?),
                        CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        compact_attn_gate_grad_width as u32,
                        gpu_qkv_norm_resid_backward_rows_per_chunk() as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            } else if gpu_split_qkv_norm_resid_backward_enabled()
                && gate_extra_grad
                && !compact_attn_gate_grad_input
            {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_split_reduce(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(block_cache.residual_mix_norm_stats.cu_ptr(stream)?),
                        CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        gpu_qkv_norm_resid_backward_rows_per_chunk() as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            } else if gpu_chunked_qkv_norm_resid_backward_enabled() {
                let residual_mix_reduce_scratch = &block_cache.residual_mix_reduce_scratch;
                self.zero_tensor(residual_mix_reduce_scratch)?;
                let chunk_rows = gpu_qkv_norm_resid_backward_rows_per_chunk();
                let num_chunks = t.div_ceil(chunk_rows);
                if gate_extra_grad && compact_attn_gate_grad_input {
                    self.kernels
                        .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_chunked_compact(
                            CudaPtr(act!(x_in).cu_ptr(stream)?),
                            CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                            CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                            CudaPtr(layer_x.cu_ptr(stream)?),
                            CudaPtr(x0.cu_ptr(stream)?),
                            CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                            CudaPtr(grad_x_in.cu_ptr(stream)?),
                            CudaPtr(grad_x_out.cu_ptr(stream)?),
                            CudaPtr(grad_x0.cu_ptr(stream)?),
                            CudaPtr(residual_mix_reduce_scratch.cu_ptr(stream)?),
                            CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                            t as u32,
                            d as u32,
                            compact_attn_gate_grad_width as u32,
                            num_chunks as u32,
                            chunk_rows as u32,
                            self.ln_scale_factor(layer),
                            1e-6,
                            1.0,
                        )?;
                } else if gate_extra_grad {
                    self.kernels
                        .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_chunked(
                            CudaPtr(act!(x_in).cu_ptr(stream)?),
                            CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                            CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                            CudaPtr(layer_x.cu_ptr(stream)?),
                            CudaPtr(x0.cu_ptr(stream)?),
                            CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                            CudaPtr(grad_x_in.cu_ptr(stream)?),
                            CudaPtr(grad_x_out.cu_ptr(stream)?),
                            CudaPtr(grad_x0.cu_ptr(stream)?),
                            CudaPtr(residual_mix_reduce_scratch.cu_ptr(stream)?),
                            CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                            t as u32,
                            d as u32,
                            num_chunks as u32,
                            chunk_rows as u32,
                            self.ln_scale_factor(layer),
                            1e-6,
                            1.0,
                        )?;
                } else {
                    self.kernels
                        .rms_norm_backward_accum_residual_mix_bwd_go_bf16_chunked(
                            CudaPtr(act!(x_in).cu_ptr(stream)?),
                            CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                            CudaPtr(layer_x.cu_ptr(stream)?),
                            CudaPtr(x0.cu_ptr(stream)?),
                            CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                            CudaPtr(grad_x_in.cu_ptr(stream)?),
                            CudaPtr(grad_x_out.cu_ptr(stream)?),
                            CudaPtr(grad_x0.cu_ptr(stream)?),
                            CudaPtr(residual_mix_reduce_scratch.cu_ptr(stream)?),
                            CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                            t as u32,
                            d as u32,
                            num_chunks as u32,
                            chunk_rows as u32,
                            self.ln_scale_factor(layer),
                            1e-6,
                            1.0,
                        )?;
                }
            } else if gate_extra_grad
                && compact_attn_gate_grad_input
                && gpu_skip_residual_mix_grad_enabled()
            {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_compact_no_mix_grad(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(block_cache.residual_mix_norm_stats.cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        compact_attn_gate_grad_width as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            } else if gate_extra_grad && compact_attn_gate_grad_input {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_compact(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input_compact.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        compact_attn_gate_grad_width as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            } else if gate_extra_grad && gpu_skip_residual_mix_grad_enabled() {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add_no_mix_grad(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(block_cache.residual_mix_norm_stats.cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            } else if gate_extra_grad {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16_add(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            } else {
                self.kernels
                    .rms_norm_backward_accum_residual_mix_bwd_go_bf16(
                        CudaPtr(act!(x_in).cu_ptr(stream)?),
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(layer_x.cu_ptr(stream)?),
                        CudaPtr(x0.cu_ptr(stream)?),
                        CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                        CudaPtr(grad_x_in.cu_ptr(stream)?),
                        CudaPtr(grad_x_out.cu_ptr(stream)?),
                        CudaPtr(grad_x0.cu_ptr(stream)?),
                        CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                        t as u32,
                        d as u32,
                        self.ln_scale_factor(layer),
                        1e-6,
                        1.0,
                    )?;
            }
        } else if recompute_residual_mix_norm_inputs {
            self.kernels
                .rms_norm_backward_accum_residual_mix_bwd_recompute(
                    CudaPtr(grad_attn_norm.cu_ptr(stream)?),
                    CudaPtr(layer_x.cu_ptr(stream)?),
                    CudaPtr(x0.cu_ptr(stream)?),
                    CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_x_out.cu_ptr(stream)?),
                    CudaPtr(grad_x0.cu_ptr(stream)?),
                    CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                    1.0,
                )?;
        } else if gpu_split_residual_mix_grad_enabled() {
            self.kernels
                .rms_norm_backward_accum_residual_mix_bwd_split_reduce(
                    CudaPtr(act!(x_in).cu_ptr(stream)?),
                    CudaPtr(grad_attn_norm.cu_ptr(stream)?),
                    CudaPtr(layer_x.cu_ptr(stream)?),
                    CudaPtr(x0.cu_ptr(stream)?),
                    CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_x_out.cu_ptr(stream)?),
                    CudaPtr(grad_x0.cu_ptr(stream)?),
                    CudaPtr(block_cache.residual_mix_norm_stats.cu_ptr(stream)?),
                    CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    256,
                    self.ln_scale_factor(layer),
                    1e-6,
                    1.0,
                )?;
        } else if gpu_chunked_residual_mix_backward_enabled() {
            let residual_mix_reduce_scratch = &block_cache.residual_mix_reduce_scratch;
            self.zero_tensor(residual_mix_reduce_scratch)?;
            let chunk_rows = 256usize;
            let num_chunks = t.div_ceil(chunk_rows);
            self.kernels
                .rms_norm_backward_accum_residual_mix_bwd_chunked(
                    CudaPtr(act!(x_in).cu_ptr(stream)?),
                    CudaPtr(grad_attn_norm.cu_ptr(stream)?),
                    CudaPtr(layer_x.cu_ptr(stream)?),
                    CudaPtr(x0.cu_ptr(stream)?),
                    CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                    CudaPtr(grad_x_in.cu_ptr(stream)?),
                    CudaPtr(grad_x_out.cu_ptr(stream)?),
                    CudaPtr(grad_x0.cu_ptr(stream)?),
                    CudaPtr(residual_mix_reduce_scratch.cu_ptr(stream)?),
                    CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    num_chunks as u32,
                    chunk_rows as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                    1.0,
                )?;
        } else {
            self.kernels.rms_norm_backward_accum_residual_mix_bwd(
                CudaPtr(act!(x_in).cu_ptr(stream)?),
                CudaPtr(grad_attn_norm.cu_ptr(stream)?),
                CudaPtr(layer_x.cu_ptr(stream)?),
                CudaPtr(x0.cu_ptr(stream)?),
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                CudaPtr(grad_x_in.cu_ptr(stream)?),
                CudaPtr(grad_x_out.cu_ptr(stream)?),
                CudaPtr(grad_x0.cu_ptr(stream)?),
                CudaPtr(grads.block_resid_mix[layer].cu_ptr(stream)?),
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
                1.0,
            )?;
        }
        finish_stage_event_optional(
            stream,
            qkv_norm_resid_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_qkv_norm_resid_ms),
        )?;
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_qkv_ms),
        )?;
        check_cuda_graph_capture_stage(stream, &format!("block_{layer}_qkv_norm_resid_backward"))?;

        self.wait_deferred_side_weight_gemms()?;
        Ok(())
    }

    fn block_backward_into(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        layer_x: &GpuTensor,
        x0: &GpuTensor,
        buf: &mut GpuActivations,
        block_cache: &mut GpuBlockBackwardCache,
        grad_x: &GpuTensor,
        grad_x0: &GpuTensor,
        grad_x_out: &GpuTensor,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
        saved: Option<&GpuLayerForwardCache>,
        recurrent_pass1_saved: Option<&GpuLayerForwardCache>,
        recurrent_mid_x: Option<&GpuTensor>,
        forward_generation: u64,
        mut stage_timing: Option<&mut GpuBackwardStageTiming>,
    ) -> PgResult<()> {
        let first_bank_contribution_beta = gpu_bank_grad_dw_beta(true);
        let accum_bank_contribution_beta = gpu_bank_grad_dw_beta(false);
        let recurrent_bank_beta = if gpu_skip_recurrent_bank_grads_enabled() {
            f32::NAN
        } else {
            first_bank_contribution_beta
        };
        let recurrent_pass1_bank_beta = if gpu_skip_recurrent_bank_grads_enabled()
            || gpu_skip_recurrent_pass1_bank_grads_enabled()
        {
            f32::NAN
        } else {
            accum_bank_contribution_beta
        };
        if self.is_recurrent_layer(layer) {
            if self.recurrent_layer_uses_straight_through_backward(layer) {
                self.copy_tensor(grad_x, grad_x_out)?;
                return Ok(());
            }
            if let (Some(pass2_saved), Some(mid_x)) = (saved, recurrent_mid_x) {
                let grad_mid = block_cache.grad_mid.clone();
                self.block_backward_single_into(
                    layer,
                    input_ids,
                    mid_x,
                    x0,
                    buf,
                    block_cache,
                    grad_x,
                    grad_x0,
                    &grad_mid,
                    grads,
                    runtime_seq_len,
                    Some(pass2_saved),
                    forward_generation,
                    recurrent_bank_beta,
                    stage_timing.as_deref_mut(),
                )?;
                if gpu_recurrent_pass1_straight_through_enabled() {
                    self.copy_tensor(&grad_mid, grad_x_out)?;
                    return Ok(());
                }
                let pass1_saved = recurrent_pass1_saved.ok_or_else(|| {
                    PgError::InvalidOp(format!(
                        "missing recurrent pass-1 saved activations for layer {layer}"
                    ))
                })?;
                return self.block_backward_single_into(
                    layer,
                    input_ids,
                    layer_x,
                    x0,
                    buf,
                    block_cache,
                    &grad_mid,
                    grad_x0,
                    grad_x_out,
                    grads,
                    runtime_seq_len,
                    Some(pass1_saved),
                    forward_generation,
                    recurrent_pass1_bank_beta,
                    stage_timing.as_deref_mut(),
                );
            }
            self.block_recompute_for_backward(
                layer,
                input_ids,
                layer_x,
                x0,
                buf,
                block_cache,
                runtime_seq_len,
            )?;
            self.copy_tensor(&buf.x, &block_cache.pass1_out)?;
            let pass1_out = block_cache.pass1_out.clone();
            let grad_mid = block_cache.grad_mid.clone();
            self.block_backward_single_into(
                layer,
                input_ids,
                &pass1_out,
                x0,
                buf,
                block_cache,
                grad_x,
                grad_x0,
                &grad_mid,
                grads,
                runtime_seq_len,
                None,
                forward_generation,
                recurrent_bank_beta,
                stage_timing.as_deref_mut(),
            )?;
            if gpu_recurrent_pass1_straight_through_enabled() {
                self.copy_tensor(&grad_mid, grad_x_out)?;
                return Ok(());
            }
            return self.block_backward_single_into(
                layer,
                input_ids,
                layer_x,
                x0,
                buf,
                block_cache,
                &grad_mid,
                grad_x0,
                grad_x_out,
                grads,
                runtime_seq_len,
                None,
                forward_generation,
                recurrent_pass1_bank_beta,
                stage_timing.as_deref_mut(),
            );
        }

        self.block_backward_single_into(
            layer,
            input_ids,
            layer_x,
            x0,
            buf,
            block_cache,
            grad_x,
            grad_x0,
            grad_x_out,
            grads,
            runtime_seq_len,
            saved,
            forward_generation,
            first_bank_contribution_beta,
            stage_timing,
        )
    }

    fn block_forward_once(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
        forward_generation: u64,
        save: Option<&GpuLayerForwardCache>,
        save_backward_activations: bool,
        mut stage_timing: Option<&mut GpuBackwardStageTiming>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let h = self.config.num_heads;
        let hkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let kv = self.config.kv_dim();
        let mlp = self.config.mlp_dim;
        let n = self.config.num_layers;
        let stream = self.gemm.stream();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);
        let attn_norm = CudaPtr(buf.attn_norm.cu_ptr(stream)?);
        let mlp_norm = CudaPtr(buf.mlp_norm.cu_ptr(stream)?);
        let v = CudaPtr(buf.v.cu_ptr(stream)?);
        let attn_out = CudaPtr(buf.attn_out.cu_ptr(stream)?);
        let xsa_out = CudaPtr(buf.xsa_out.cu_ptr(stream)?);
        let proj_out = CudaPtr(buf.proj_out.cu_ptr(stream)?);
        let mlp_up = CudaPtr(buf.mlp_up.cu_ptr(stream)?);
        let mlp_act = CudaPtr(buf.mlp_act.cu_ptr(stream)?);
        let mlp_out = CudaPtr(buf.mlp_out.cu_ptr(stream)?);
        let skip_f32_attention_saved = save.is_some() && self.use_skip_f32_attention_saved_acts();
        let lean_bf16_direct_saved = skip_f32_attention_saved
            && self.use_bf16_primary_forward_gemm()
            && self.use_bf16_backward_gemm();

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let fused_residual_mix_norm = self.use_fused_residual_mix_norm();
        let mut attn_norm_bf16_ready = false;
        if fused_residual_mix_norm && self.use_bf16_norm_side_outputs() {
            self.kernels.residual_mix_rms_norm_fwd_bf16(
                x,
                x0,
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                x_in,
                attn_norm,
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
                (t * d) as u32,
            )?;
            attn_norm_bf16_ready = true;
        } else if fused_residual_mix_norm {
            self.kernels.residual_mix_rms_norm_fwd(
                x,
                x0,
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                x_in,
                attn_norm,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
                (t * d) as u32,
            )?;
        } else {
            self.kernels.residual_mix_fwd(
                x,
                x0,
                CudaPtr(self.weights.resid_mix[layer].cu_ptr(stream)?),
                x_in,
                d as u32,
                (t * d) as u32,
            )?;
        }
        if let Some(save) = save.filter(|_| {
            save_backward_activations
                && !(lean_bf16_direct_saved && gpu_recompute_residual_mix_norm_inputs_enabled())
        }) {
            self.copy_tensor(&buf.x_in, &save.x_in)?;
        }

        if !fused_residual_mix_norm && self.use_bf16_norm_side_outputs() {
            self.kernels.rms_norm_forward_bf16(
                x_in,
                attn_norm,
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
            )?;
            attn_norm_bf16_ready = true;
        } else if !fused_residual_mix_norm {
            self.kernels.rms_norm_forward(
                x_in,
                attn_norm,
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
            )?;
        }
        if let Some(save) = save.filter(|_| !lean_bf16_direct_saved) {
            self.copy_tensor(&buf.attn_norm, &save.attn_norm)?;
        }

        let save_qk_during_qkv = save.filter(|_| !self.any_qkv_lora_enabled());
        let fused_qkv_rope_prepack_forward = gpu_fused_qkv_rope_prepack_forward_enabled()
            && self.use_fused_qkv_projection()
            && self.use_cudnn_prepacked_bf16_attention()
            && save_qk_during_qkv.is_some()
            && !self.any_qkv_lora_enabled()
            && self.config.ve_layers.is_empty()
            && self.config.rope_dims > 0
            && runtime_seq_len > 0
            && t % runtime_seq_len == 0;
        let is_xsa_layer = layer >= n.saturating_sub(self.config.xsa_last_n);
        let forward_only_bf16_prepack = fused_qkv_rope_prepack_forward
            && !save_backward_activations
            && is_xsa_layer
            && self.config.sparse_attn_gate_enabled
            && !self.config.attn_out_gate_enabled
            && self.use_cudnn_saved_bf16_attention()
            && self.use_bf16_sparse_xsa_forward();
        let qkv_projected = if fused_qkv_rope_prepack_forward {
            self.qkv_projection_forward_packed_only(layer, buf, t, attn_norm_bf16_ready)?
        } else {
            self.qkv_projection_forward(
                layer,
                buf,
                t,
                attn_norm_bf16_ready,
                save_qk_during_qkv.map(|cache| &cache.q_pre_norm),
                save_qk_during_qkv.map(|cache| &cache.k_pre_norm),
            )?
        };
        let qk_pre_norm_saved_by_qkv = qkv_projected && save_qk_during_qkv.is_some();
        if qkv_projected {
            // Hot path handled by a single packed QKV GEMM.
            if let Some(save) = save.filter(|_| save_backward_activations) {
                self.copy_bf16_tensor(&buf.x_in_bf16, &save.attn_norm_bf16)?;
            }
        } else if self.use_bf16_primary_forward_gemm() {
            let q_w = self.weights.qo_bank.slice_first(layer)?;
            let q_w_bf16 = self.weights.qo_bank_bf16.slice_first(layer)?;
            let k_w = self.weights.kv_bank.slice_first(layer)?;
            let k_w_bf16 = self.weights.kv_bank_bf16.slice_first(layer)?;
            let v_w = self.weights.kv_bank.slice_first(n + layer)?;
            let v_w_bf16 = self.weights.kv_bank_bf16.slice_first(n + layer)?;
            if !attn_norm_bf16_ready {
                self.kernels.f32_to_bf16(
                    CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
            }
            if let Some(save) = save.filter(|_| save_backward_activations) {
                self.copy_bf16_tensor(&buf.x_in_bf16, &save.attn_norm_bf16)?;
            }
            self.linear_forward_bf16_input_ready(
                &buf.attn_norm,
                &buf.x_in_bf16,
                &q_w,
                &q_w_bf16,
                &buf.q,
                t,
                d,
                d,
            )?;
            self.linear_forward_bf16_input_ready(
                &buf.attn_norm,
                &buf.x_in_bf16,
                &k_w,
                &k_w_bf16,
                &buf.k,
                t,
                kv,
                d,
            )?;
            self.linear_forward_bf16_input_ready(
                &buf.attn_norm,
                &buf.x_in_bf16,
                &v_w,
                &v_w_bf16,
                &buf.v,
                t,
                kv,
                d,
            )?;
        } else {
            let q_w = self.weights.qo_bank.slice_first(layer)?;
            let k_w = self.weights.kv_bank.slice_first(layer)?;
            let v_w = self.weights.kv_bank.slice_first(n + layer)?;
            self.linear_forward_f32(&buf.attn_norm, &q_w, &buf.q, t, d, d)?;
            self.linear_forward_f32(&buf.attn_norm, &k_w, &buf.k, t, kv, d)?;
            self.linear_forward_f32(&buf.attn_norm, &v_w, &buf.v, t, kv, d)?;
        }
        self.apply_qkv_loras_forward(layer, buf)?;
        if let Some(save) = save.filter(|_| save_backward_activations && !qk_pre_norm_saved_by_qkv)
        {
            self.copy_tensor(&buf.q, &save.q_pre_norm)?;
            self.copy_tensor(&buf.k, &save.k_pre_norm)?;
        }

        if let Some(ve_idx) = self.config.ve_layers.iter().position(|&l| l == layer) {
            self.kernels.embedding_gather_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_embed.cu_ptr(stream)?),
                CudaPtr(buf.ve_embed_out.cu_ptr(stream)?),
                self.config.ve_dim as u32,
                t as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.ve_embed_out.cu_ptr(stream)?,
                    self.weights.ve_proj.cu_ptr(stream)?,
                    buf.ve_out.cu_ptr(stream)?,
                    t,
                    kv,
                    self.config.ve_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_by_param_product_fwd(
                v,
                CudaPtr(buf.ve_out.cu_ptr(stream)?),
                CudaPtr(self.weights.ve_scale_param.cu_ptr(stream)?),
                0,
                CudaPtr(self.weights.ve_layer_scales.cu_ptr(stream)?),
                ve_idx as u32,
                1.0,
                (t * kv) as u32,
            )?;
            if let Some(save) = save.filter(|_| save_backward_activations) {
                self.copy_tensor(&buf.ve_embed_out, &save.ve_embed_out)?;
                self.copy_tensor(&buf.ve_out, &save.ve_out)?;
            }
        }

        self.validate_cudnn_prepacked_bf16_attention()?;
        let prepack_bf16_attention = self.use_cudnn_prepacked_bf16_attention()
            && save.is_some()
            && runtime_seq_len > 0
            && t % runtime_seq_len == 0;
        if prepack_bf16_attention && self.debug_poison_cudnn_prepacked_bf16_attention() {
            let save = save.expect("prepacked BF16 attention requires saved BF16 buffers");
            const BF16_QNAN: u16 = 0x7fc0;
            self.kernels.fill_u16(
                CudaPtr(save.q_bhsd_bf16.cu_ptr(stream)?),
                BF16_QNAN,
                save.q_bhsd_bf16.numel() as u32,
            )?;
            self.kernels.fill_u16(
                CudaPtr(save.k_bhsd_bf16.cu_ptr(stream)?),
                BF16_QNAN,
                save.k_bhsd_bf16.numel() as u32,
            )?;
            self.kernels.fill_u16(
                CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                BF16_QNAN,
                save.v_bhsd_bf16.numel() as u32,
            )?;
        }
        if fused_qkv_rope_prepack_forward {
            let save = save.expect("prepacked BF16 attention requires saved BF16 buffers");
            if forward_only_bf16_prepack {
                self.kernels.unpack_qkv_rope_gain_prepack_fwd_bf16_only(
                    CudaPtr(buf.qkv_out.cu_ptr(stream)?),
                    CudaPtr(save.q_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(save.k_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    t as u32,
                    runtime_seq_len as u32,
                    d as u32,
                    kv as u32,
                    h as u32,
                    hkv as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    1e-6,
                )?;
            } else {
                self.kernels.unpack_qkv_rope_gain_prepack_fwd(
                    CudaPtr(buf.qkv_out.cu_ptr(stream)?),
                    CudaPtr(buf.q.cu_ptr(stream)?),
                    CudaPtr(buf.k.cu_ptr(stream)?),
                    CudaPtr(buf.v.cu_ptr(stream)?),
                    CudaPtr(save.q_pre_norm.cu_ptr(stream)?),
                    CudaPtr(save.k_pre_norm.cu_ptr(stream)?),
                    CudaPtr(save.q_post_rope.cu_ptr(stream)?),
                    CudaPtr(save.q_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(save.k_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.q_gains[layer].cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_cos.cu_ptr(stream)?),
                    CudaPtr(self.weights.rope_sin.cu_ptr(stream)?),
                    t as u32,
                    runtime_seq_len as u32,
                    d as u32,
                    kv as u32,
                    h as u32,
                    hkv as u32,
                    hd as u32,
                    self.config.rope_dims as u32,
                    1e-6,
                )?;
            }
            save.bf16_qkv_freshness.mark(
                Bf16QkvProducer::FusedNormQkvRopeGain,
                forward_generation,
                layer,
                t,
                runtime_seq_len,
                h,
                hkv,
                hd,
            );
        } else {
            self.apply_qk_norm_rope_gain_forward(
                layer,
                &buf.q,
                &buf.k,
                save.map(|cache| &cache.q_post_rope),
                if prepack_bf16_attention {
                    save.map(|cache| &cache.q_bhsd_bf16)
                } else {
                    None
                },
                if prepack_bf16_attention {
                    save.map(|cache| &cache.k_bhsd_bf16)
                } else {
                    None
                },
                t,
                runtime_seq_len,
            )?;
            if prepack_bf16_attention {
                let save = save.expect("prepacked BF16 attention requires saved BF16 buffers");
                self.kernels.bthd_to_bhsd_bf16(
                    CudaPtr(buf.v.cu_ptr(stream)?),
                    CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                    (t / runtime_seq_len) as u32,
                    runtime_seq_len as u32,
                    hkv as u32,
                    hd as u32,
                )?;
                save.bf16_qkv_freshness.mark(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    forward_generation,
                    layer,
                    t,
                    runtime_seq_len,
                    h,
                    hkv,
                    hd,
                );
                // Consumer-side `require()` runs at the cuDNN prepacked-attention call site
                // (forward) and at the backward attention consumer. This is the canonical
                // freshness check; an immediate `require()` here would be a tautological
                // self-test of `mark()` rather than a stale-buffer guard. See test
                // `bf16_qkv_freshness_rejects_stale_step_and_layer` for the contract.
            }
        }
        if let Some(save) = save.filter(|_| save_backward_activations && !skip_f32_attention_saved)
        {
            self.copy_tensor(&buf.q, &save.q)?;
            self.copy_tensor(&buf.k, &save.k)?;
            self.copy_tensor(&buf.v, &save.v)?;
        }
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_block_pre_attn_ms),
        )?;

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let attn_stats = if let Some(save) = save {
            Some(save.attn_stats.cu_ptr(stream)?)
        } else {
            None
        };
        let saved_attention_bf16 = if self.use_cudnn_saved_bf16_attention() {
            if let Some(save) = save {
                Some((
                    save.q_bhsd_bf16.cu_ptr(stream)?,
                    save.k_bhsd_bf16.cu_ptr(stream)?,
                    save.v_bhsd_bf16.cu_ptr(stream)?,
                    save.attn_out_bhsd_bf16.cu_ptr(stream)?,
                ))
            } else {
                None
            }
        } else {
            None
        };
        // Consumer-side freshness gate: when the prepacked BF16 attention path is in
        // effect, the BF16 Q/K/V buffers must have been just produced by the fused
        // norm+QKV+RoPE+gain forward for THIS layer/step/shape. If the producer was
        // skipped (e.g., a misconfigured gate combination), this fires before cuDNN
        // can read garbage. The non-prepacked saved-BF16 path uses the buffers as
        // cuDNN write sinks, so freshness is not required there.
        if prepack_bf16_attention {
            if let Some(save) = save {
                save.bf16_qkv_freshness.require(
                    Bf16QkvProducer::FusedNormQkvRopeGain,
                    forward_generation,
                    layer,
                    t,
                    runtime_seq_len,
                    h,
                    hkv,
                    hd,
                )?;
            }
        }
        let bf16_attention_projection_output =
            self.use_bf16_attention_projection_output() && lean_bf16_direct_saved && save.is_some();
        let saved_attn_weight_input_bf16_direct = save
            .filter(|_| bf16_attention_projection_output)
            .map(|cache| cache.attn_weight_input_bf16.clone());
        let attn_weight_input_bf16 = saved_attn_weight_input_bf16_direct
            .as_ref()
            .unwrap_or(&buf.x_in_bf16);
        let bf16_sparse_xsa_forward = is_xsa_layer
            && self.config.sparse_attn_gate_enabled
            && !self.config.attn_out_gate_enabled
            && self.use_bf16_sparse_xsa_forward()
            && prepack_bf16_attention
            && saved_attention_bf16.is_some()
            && runtime_seq_len > 0
            && t % runtime_seq_len == 0;
        self.run_attention_forward(
            buf.q.cu_ptr(stream)?,
            buf.k.cu_ptr(stream)?,
            buf.v.cu_ptr(stream)?,
            buf.attn_out.cu_ptr(stream)?,
            t,
            h,
            hkv,
            hd,
            runtime_seq_len,
            attn_stats,
            saved_attention_bf16,
            bf16_sparse_xsa_forward,
        )?;
        if let Some(save) = save.filter(|_| !skip_f32_attention_saved) {
            self.copy_tensor(&buf.attn_out, &save.attn_out)?;
        }
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_block_attention_ms),
        )?;

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let mut fused_sparse_xsa_forward = false;
        let attn_src_tensor = if is_xsa_layer
            && self.config.sparse_attn_gate_enabled
            && !self.config.attn_out_gate_enabled
        {
            if bf16_sparse_xsa_forward {
                let save =
                    save.expect("BF16 SparseAttnGate+XSA forward requires saved BF16 attention");
                if gpu_sparse_xsa_warphead_forward_enabled() {
                    self.kernels.sparse_attn_gate_xsa_fwd_bf16_bhsd_warpheads(
                        CudaPtr(save.attn_out_bhsd_bf16.cu_ptr(stream)?),
                        CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                        CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                        CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                        CudaPtr(attn_weight_input_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                        (t / runtime_seq_len) as u32,
                        runtime_seq_len as u32,
                        h as u32,
                        hkv as u32,
                        hd as u32,
                        d as u32,
                        self.config.sparse_attn_gate_width as u32,
                        self.config.sparse_attn_gate_scale,
                    )?;
                } else {
                    self.kernels.sparse_attn_gate_xsa_fwd_bf16_bhsd(
                        CudaPtr(save.attn_out_bhsd_bf16.cu_ptr(stream)?),
                        CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                        CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                        CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                        CudaPtr(attn_weight_input_bf16.cu_ptr(stream)?),
                        CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                        (t / runtime_seq_len) as u32,
                        runtime_seq_len as u32,
                        h as u32,
                        hkv as u32,
                        hd as u32,
                        d as u32,
                        self.config.sparse_attn_gate_width as u32,
                        self.config.sparse_attn_gate_scale,
                    )?;
                }
            } else {
                self.kernels.sparse_attn_gate_xsa_fwd(
                    attn_out,
                    v,
                    CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                    CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                    CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                    CudaPtr(attn_weight_input_bf16.cu_ptr(stream)?),
                    CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                    t as u32,
                    h as u32,
                    hkv as u32,
                    hd as u32,
                    d as u32,
                    self.config.sparse_attn_gate_width as u32,
                    self.config.sparse_attn_gate_scale,
                )?;
            }
            if let Some(save) = save.filter(|_| save_backward_activations) {
                self.copy_tensor(&buf.attn_gated, &save.attn_gated)?;
                self.copy_tensor(&buf.attn_gate_values, &save.attn_gate_values)?;
            }
            fused_sparse_xsa_forward = true;
            &buf.attn_gated
        } else if is_xsa_layer {
            self.kernels.xsa_fwd(
                attn_out, v, xsa_out, t as u32, h as u32, hkv as u32, hd as u32,
            )?;
            if let Some(save) =
                save.filter(|_| save_backward_activations && !lean_bf16_direct_saved)
            {
                self.copy_tensor(&buf.xsa_out, &save.xsa_out)?;
            }
            &buf.xsa_out
        } else {
            &buf.attn_out
        };
        let attn_src_tensor = if fused_sparse_xsa_forward {
            attn_src_tensor
        } else if self.config.attn_out_gate_enabled {
            self.kernels.attn_out_gate_fwd(
                CudaPtr(attn_src_tensor.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(self.weights.attn_gate_biases[layer].cu_ptr(stream)?),
                CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.attn_out_gate_width as u32,
            )?;
            if let Some(save) = save.filter(|_| save_backward_activations) {
                self.copy_tensor(&buf.attn_gated, &save.attn_gated)?;
                self.copy_tensor(&buf.attn_gate_values, &save.attn_gate_values)?;
            }
            &buf.attn_gated
        } else if self.config.sparse_attn_gate_enabled {
            self.kernels.sparse_attn_gate_fwd(
                CudaPtr(attn_src_tensor.cu_ptr(stream)?),
                CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                CudaPtr(buf.attn_gate_values.cu_ptr(stream)?),
                t as u32,
                h as u32,
                hd as u32,
                d as u32,
                self.config.sparse_attn_gate_width as u32,
                self.config.sparse_attn_gate_scale,
            )?;
            if let Some(save) = save.filter(|_| save_backward_activations) {
                self.copy_tensor(&buf.attn_gated, &save.attn_gated)?;
                self.copy_tensor(&buf.attn_gate_values, &save.attn_gate_values)?;
            }
            &buf.attn_gated
        } else {
            attn_src_tensor
        };

        let o_w = self.weights.qo_bank.slice_first(n + layer)?;
        let o_w_bf16 = self.weights.qo_bank_bf16.slice_first(n + layer)?;
        let direct_bf16_attention_output_projection_input = bf16_attention_projection_output
            && self.use_cudnn_saved_bf16_attention()
            && runtime_seq_len > 0
            && t % runtime_seq_len == 0
            && !fused_sparse_xsa_forward
            && !is_xsa_layer
            && !self.config.attn_out_gate_enabled
            && !self.config.sparse_attn_gate_enabled;
        if bf16_attention_projection_output {
            let save = save.expect("checked save exists for bf16 attention projection output");
            if direct_bf16_attention_output_projection_input {
                self.kernels.bhsd_to_bthd_bf16(
                    CudaPtr(save.attn_out_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(attn_weight_input_bf16.cu_ptr(stream)?),
                    (t / runtime_seq_len) as u32,
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                )?;
                self.linear_forward_bf16_output_ready(
                    attn_weight_input_bf16,
                    &o_w_bf16,
                    &save.proj_out_bf16,
                    t,
                    d,
                    d,
                )?;
            } else if fused_sparse_xsa_forward {
                self.linear_forward_bf16_output_ready(
                    attn_weight_input_bf16,
                    &o_w_bf16,
                    &save.proj_out_bf16,
                    t,
                    d,
                    d,
                )?;
            } else {
                self.kernels.f32_to_bf16(
                    CudaPtr(attn_src_tensor.cu_ptr(stream)?),
                    CudaPtr(attn_weight_input_bf16.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
                self.linear_forward_bf16_output_ready(
                    attn_weight_input_bf16,
                    &o_w_bf16,
                    &save.proj_out_bf16,
                    t,
                    d,
                    d,
                )?;
            }
        } else if self.use_bf16_primary_forward_gemm() {
            if fused_sparse_xsa_forward {
                self.linear_forward_bf16_input_ready(
                    attn_src_tensor,
                    &buf.x_in_bf16,
                    &o_w,
                    &o_w_bf16,
                    &buf.proj_out,
                    t,
                    d,
                    d,
                )?;
            } else {
                self.linear_forward(
                    attn_src_tensor,
                    &buf.x_in_bf16,
                    &o_w,
                    &o_w_bf16,
                    &buf.proj_out,
                    t,
                    d,
                    d,
                )?;
            }
            if let Some(save) = save {
                self.copy_bf16_tensor(&buf.x_in_bf16, &save.attn_weight_input_bf16)?;
            }
        } else {
            self.linear_forward_f32(attn_src_tensor, &o_w, &buf.proj_out, t, d, d)?;
        }
        if let Some(lora) = &self.o_lora {
            let attn_norm = buf.attn_norm.clone();
            let proj_out = buf.proj_out.clone();
            self.apply_linear_lora_forward(lora, layer, &attn_norm, &proj_out, buf)?;
        }
        if let Some(save) =
            save.filter(|_| save_backward_activations && !bf16_attention_projection_output)
        {
            self.copy_tensor(&buf.proj_out, &save.proj_out)?;
        }

        let fused_parallel_attn_resid_norm = self.use_fused_parallel_attn_residual_rms_norm(layer);
        let mut mlp_norm_bf16_ready = false;
        let saved_mlp_norm_bf16_direct = save
            .filter(|_| lean_bf16_direct_saved && self.use_bf16_norm_side_outputs())
            .map(|cache| cache.mlp_norm_bf16.clone());
        let mlp_norm_bf16_output = saved_mlp_norm_bf16_direct
            .as_ref()
            .unwrap_or(&buf.x_in_bf16);
        if fused_parallel_attn_resid_norm && bf16_attention_projection_output {
            let save = save.expect("checked save exists for bf16 attention projection output");
            self.kernels
                .residual_add_scale_from_base_rms_norm_fwd_bf16_proj(
                    x_in,
                    CudaPtr(save.proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    x,
                    mlp_norm,
                    CudaPtr(mlp_norm_bf16_output.cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                )?;
            mlp_norm_bf16_ready = true;
        } else if fused_parallel_attn_resid_norm && self.use_bf16_norm_side_outputs() {
            self.kernels
                .residual_add_scale_from_base_rms_norm_fwd_bf16(
                    x_in,
                    proj_out,
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    x,
                    mlp_norm,
                    CudaPtr(mlp_norm_bf16_output.cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                )?;
            mlp_norm_bf16_ready = true;
        } else if fused_parallel_attn_resid_norm {
            self.kernels.residual_add_scale_from_base_rms_norm_fwd(
                x_in,
                proj_out,
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                x,
                mlp_norm,
                t as u32,
                d as u32,
                self.ln_scale_factor(layer),
                1e-6,
            )?;
        } else if self.use_fused_attention_residual_from_base() && bf16_attention_projection_output
        {
            let save = save.expect("checked save exists for bf16 attention projection output");
            self.kernels.residual_add_scale_from_base_bf16_proj_fwd(
                x_in,
                CudaPtr(save.proj_out_bf16.cu_ptr(stream)?),
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                x,
                d as u32,
                (t * d) as u32,
            )?;
        } else if self.use_fused_attention_residual_from_base() {
            self.kernels.residual_add_scale_from_base_fwd(
                x_in,
                proj_out,
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                x,
                d as u32,
                (t * d) as u32,
            )?;
        } else {
            self.kernels.copy_fwd(x_in, x, (t * d) as u32)?;
            self.kernels.residual_add_scale_fwd(
                x,
                proj_out,
                CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        }
        if let Some(save) = save.filter(|_| save_backward_activations) {
            if self.parallel_residual_enabled_for_layer(layer) && lean_bf16_direct_saved {
                // In parallel-residual mode the MLP norm input is exactly
                // x_in. The lean BF16 direct-saved path can reuse saved.x_in
                // during backward instead of storing a duplicate F32
                // x_after_attn activation.
            } else if self.parallel_residual_enabled_for_layer(layer) {
                self.copy_tensor(&buf.x_in, &save.x_after_attn)?;
            } else {
                self.copy_tensor(&buf.x, &save.x_after_attn)?;
            }
        }

        if !fused_parallel_attn_resid_norm {
            let mlp_norm_input = if self.parallel_residual_enabled_for_layer(layer) {
                x_in
            } else {
                x
            };
            if self.use_bf16_norm_side_outputs() {
                self.kernels.rms_norm_forward_bf16(
                    mlp_norm_input,
                    mlp_norm,
                    CudaPtr(mlp_norm_bf16_output.cu_ptr(stream)?),
                    t as u32,
                    d as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                )?;
                mlp_norm_bf16_ready = true;
            } else {
                self.kernels.rms_norm_forward(
                    mlp_norm_input,
                    mlp_norm,
                    t as u32,
                    d as u32,
                    self.ln_scale_factor(layer),
                    1e-6,
                )?;
            }
        }
        if let Some(save) = save.filter(|_| save_backward_activations && !lean_bf16_direct_saved) {
            self.copy_tensor(&buf.mlp_norm, &save.mlp_norm)?;
        }
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_block_post_attn_ms),
        )?;

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let up_w = self.weights.mlp_up_bank.slice_first(layer)?;
        let up_w_bf16 = self.weights.mlp_up_bank_bf16.slice_first(layer)?;
        let down_w = self.weights.mlp_down_bank.slice_first(layer)?;
        let down_w_bf16 = self.weights.mlp_down_bank_bf16.slice_first(layer)?;
        let bf16_mlp_up_output =
            self.use_bf16_mlp_up_output() && lean_bf16_direct_saved && save.is_some();
        if bf16_mlp_up_output {
            let save = save.expect("checked save exists for bf16 mlp-up output");
            if !mlp_norm_bf16_ready {
                self.kernels.f32_to_bf16(
                    CudaPtr(buf.mlp_norm.cu_ptr(stream)?),
                    CudaPtr(mlp_norm_bf16_output.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
            }
            self.linear_forward_bf16_output_ready(
                mlp_norm_bf16_output,
                &up_w_bf16,
                &save.mlp_up_bf16,
                t,
                mlp,
                d,
            )?;
            self.kernels.leaky_relu_sq_forward_x_bf16_only(
                CudaPtr(save.mlp_up_bf16.cu_ptr(stream)?),
                CudaPtr(buf.wide_bf16.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
        } else if self.use_bf16_primary_forward_gemm() {
            if mlp_norm_bf16_ready {
                self.linear_forward_bf16_input_ready(
                    &buf.mlp_norm,
                    mlp_norm_bf16_output,
                    &up_w,
                    &up_w_bf16,
                    &buf.mlp_up,
                    t,
                    mlp,
                    d,
                )?;
            } else {
                self.linear_forward(
                    &buf.mlp_norm,
                    mlp_norm_bf16_output,
                    &up_w,
                    &up_w_bf16,
                    &buf.mlp_up,
                    t,
                    mlp,
                    d,
                )?;
            }
            if let Some(save) = save.filter(|_| save_backward_activations) {
                if saved_mlp_norm_bf16_direct.is_none() {
                    self.copy_bf16_tensor(mlp_norm_bf16_output, &save.mlp_norm_bf16)?;
                }
            }
        } else {
            self.linear_forward_f32(&buf.mlp_norm, &up_w, &buf.mlp_up, t, mlp, d)?;
        }
        if let Some(save) = save.filter(|_| save_backward_activations && !lean_bf16_direct_saved) {
            self.copy_tensor(&buf.mlp_up, &save.mlp_up)?;
        }
        let fused_mlp_act_bf16 = self.use_fused_mlp_activation_bf16();
        if bf16_mlp_up_output {
            // Activation BF16 was produced directly from the BF16 up-projection.
        } else if fused_mlp_act_bf16 {
            self.kernels.leaky_relu_sq_forward_bf16(
                mlp_up,
                mlp_act,
                CudaPtr(buf.wide_bf16.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
        } else {
            self.kernels
                .leaky_relu_sq_forward(mlp_up, mlp_act, (t * mlp) as u32)?;
        }
        if let Some(save) = save
            .filter(|_| save_backward_activations && lean_bf16_direct_saved && !bf16_mlp_up_output)
        {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.mlp_up.cu_ptr(stream)?),
                CudaPtr(save.mlp_up_bf16.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
        }
        if let Some(save) = save.filter(|_| save_backward_activations && !lean_bf16_direct_saved) {
            self.copy_tensor(&buf.mlp_act, &save.mlp_act)?;
        }
        let bf16_residual_projection_output =
            self.use_bf16_residual_projection_output() && lean_bf16_direct_saved && save.is_some();
        if bf16_residual_projection_output {
            let save = save.expect("checked save exists for bf16 residual projection output");
            if fused_mlp_act_bf16 || bf16_mlp_up_output {
                self.linear_forward_bf16_output_ready(
                    &buf.wide_bf16,
                    &down_w_bf16,
                    &save.mlp_out_bf16,
                    t,
                    d,
                    mlp,
                )?;
            } else {
                if self.use_bf16_primary_forward_gemm() {
                    self.kernels.f32_to_bf16(
                        CudaPtr(buf.mlp_act.cu_ptr(stream)?),
                        CudaPtr(buf.wide_bf16.cu_ptr(stream)?),
                        (t * mlp) as u32,
                    )?;
                }
                self.linear_forward_bf16_output_ready(
                    &buf.wide_bf16,
                    &down_w_bf16,
                    &save.mlp_out_bf16,
                    t,
                    d,
                    mlp,
                )?;
            }
        } else if self.use_bf16_primary_forward_gemm() {
            if fused_mlp_act_bf16 || bf16_mlp_up_output {
                self.linear_forward_bf16_input_ready(
                    &buf.mlp_act,
                    &buf.wide_bf16,
                    &down_w,
                    &down_w_bf16,
                    &buf.mlp_out,
                    t,
                    d,
                    mlp,
                )?;
            } else {
                self.linear_forward(
                    &buf.mlp_act,
                    &buf.wide_bf16,
                    &down_w,
                    &down_w_bf16,
                    &buf.mlp_out,
                    t,
                    d,
                    mlp,
                )?;
            }
            if let Some(save) = save {
                self.copy_bf16_tensor(&buf.wide_bf16, &save.mlp_act_bf16)?;
            }
        } else {
            self.linear_forward_f32(&buf.mlp_act, &down_w, &buf.mlp_out, t, d, mlp)?;
        }
        if let Some(lora) = &self.mlp_lora {
            let mlp_norm = buf.mlp_norm.clone();
            let mlp_out = buf.mlp_out.clone();
            self.apply_linear_lora_forward(lora, layer, &mlp_norm, &mlp_out, buf)?;
        }
        if let Some(save) =
            save.filter(|_| save_backward_activations && !bf16_residual_projection_output)
        {
            self.copy_tensor(&buf.mlp_out, &save.mlp_out)?;
        }
        if bf16_residual_projection_output {
            let save = save.expect("checked save exists for bf16 residual projection output");
            self.kernels.residual_add_scale_bf16_proj_fwd(
                x,
                CudaPtr(save.mlp_out_bf16.cu_ptr(stream)?),
                CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        } else {
            self.kernels.residual_add_scale_fwd(
                x,
                mlp_out,
                CudaPtr(self.weights.mlp_scales[layer].cu_ptr(stream)?),
                d as u32,
                (t * d) as u32,
            )?;
        }
        finish_stage_event_optional(
            stream,
            substage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_block_mlp_ms),
        )?;

        Ok(())
    }

    fn block_forward(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
        mut stage_timing: Option<&mut GpuBackwardStageTiming>,
    ) -> PgResult<()> {
        self.block_forward_once(
            layer,
            input_ids,
            buf,
            runtime_seq_len,
            0,
            None,
            false,
            stage_timing.as_deref_mut(),
        )?;
        if self.is_recurrent_layer(layer) {
            self.block_forward_once(
                layer,
                input_ids,
                buf,
                runtime_seq_len,
                0,
                None,
                false,
                stage_timing.as_deref_mut(),
            )?;
        }
        Ok(())
    }

    fn block_forward_with_cache_slot(
        &self,
        layer: usize,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &GpuForwardCache,
        runtime_seq_len: usize,
        forward_generation: u64,
        mut stage_timing: Option<&mut GpuBackwardStageTiming>,
    ) -> PgResult<()> {
        if let Some(saved) = cache
            .saved_layers
            .get(layer)
            .and_then(|saved| saved.as_ref())
        {
            if self.is_recurrent_layer(layer) {
                let mid_x = cache
                    .recurrent_mid_x
                    .get(layer)
                    .and_then(|mid| mid.as_ref())
                    .ok_or_else(|| {
                        PgError::InvalidOp(format!(
                            "missing recurrent mid activation boundary for layer {layer}"
                        ))
                    })?;
                let pass1_save_backward_activations =
                    !gpu_recurrent_pass1_straight_through_enabled();
                let recurrent_pass1_cache = cache
                    .recurrent_pass1_layers
                    .get(layer)
                    .and_then(|saved| saved.as_ref());
                let pass1_saved = if pass1_save_backward_activations {
                    Some(recurrent_pass1_cache.ok_or_else(|| {
                        PgError::InvalidOp(format!(
                            "missing recurrent pass-1 saved activations for layer {layer}"
                        ))
                    })?)
                } else {
                    // Pass-1 straight-through does not need F32 activations for
                    // backward, but still benefits from the lightweight BF16
                    // cache: it enables the fused QKV/RoPE prepack path and
                    // cuDNN BF16 SDPA for the recurrent pass instead of falling
                    // back to the slower unpacked forward.
                    recurrent_pass1_cache
                };
                self.block_forward_once(
                    layer,
                    input_ids,
                    buf,
                    runtime_seq_len,
                    forward_generation,
                    pass1_saved,
                    pass1_save_backward_activations,
                    stage_timing.as_deref_mut(),
                )?;
                self.copy_tensor(&buf.x, mid_x)?;
                self.block_forward_once(
                    layer,
                    input_ids,
                    buf,
                    runtime_seq_len,
                    forward_generation,
                    Some(saved),
                    true,
                    stage_timing.as_deref_mut(),
                )?;
            } else {
                self.block_forward_once(
                    layer,
                    input_ids,
                    buf,
                    runtime_seq_len,
                    forward_generation,
                    Some(saved),
                    true,
                    stage_timing.as_deref_mut(),
                )?;
            }
        } else {
            self.block_forward(
                layer,
                input_ids,
                buf,
                runtime_seq_len,
                stage_timing.as_deref_mut(),
            )?;
        }
        Ok(())
    }

    pub fn forward(&self, input_ids: &GpuTensor, buf: &mut GpuActivations) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        self.forward_with_seq_len(input_ids, buf, t)
    }

    pub fn forward_with_seq_len(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        self.forward_with_seq_len_impl(input_ids, buf, runtime_seq_len, true)
    }

    pub fn forward_hidden_with_seq_len(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        // Used by tiled output-CE eval/TTT paths: they need the final hidden
        // state, then compute output projection and loss through CE scratch
        // tiles instead of a persistent [tokens, vocab] logits buffer.
        self.forward_with_seq_len_impl(input_ids, buf, runtime_seq_len, false)
    }

    fn forward_with_seq_len_impl(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        runtime_seq_len: usize,
        materialize_output_logits: bool,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let runtime_seq_len = runtime_seq_len.min(t).max(1);
        let d = self.config.model_dim;
        let stream = self.gemm.stream();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);

        if self.config.bigram_vocab_size > 0 && gpu_bigram_embedding_merge_enabled() {
            self.kernels.embedding_bigram_project_merge_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_proj.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_scale_param.cu_ptr(stream)?),
                x,
                d as u32,
                self.config.bigram_dim as u32,
                self.config.bigram_vocab_size as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
        } else {
            self.kernels.embedding_gather_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
                x,
                d as u32,
                t as u32,
            )?;
        }

        if self.config.bigram_vocab_size > 0 && !gpu_bigram_embedding_merge_enabled() {
            self.kernels.bigram_hash_embed_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_by_param_fwd(
                x,
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_scale_param.cu_ptr(stream)?),
                1.0,
                (t * d) as u32,
            )?;
        }

        self.kernels
            .rms_norm_forward(x, x_in, t as u32, d as u32, 1.0, 1e-6)?;
        if let Some(boundary) = self.config.smear_gate_boundary_token_id {
            self.kernels.smear_gate_fwd_boundary(
                x_in,
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
                x,
                t as u32,
                runtime_seq_len as u32,
                d as u32,
                boundary,
            )?;
        } else {
            self.kernels.smear_gate_fwd(
                x_in,
                CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
                x,
                t as u32,
                runtime_seq_len as u32,
                d as u32,
            )?;
        }
        self.kernels.copy_fwd(x, x0, (t * d) as u32)?;

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        for layer in 0..n_enc {
            self.block_forward(layer, input_ids, buf, runtime_seq_len, None)?;
            self.kernels.copy_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.encoder_skips[layer].cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }

        for i in 0..n_dec {
            if i < self.config.num_skip_weights() {
                let skip_idx = self.config.num_skip_weights() - 1 - i;
                self.kernels.residual_add_scale_fwd(
                    CudaPtr(buf.x.cu_ptr(stream)?),
                    CudaPtr(buf.encoder_skips[skip_idx].cu_ptr(stream)?),
                    CudaPtr(self.weights.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
            self.block_forward(n_enc + i, input_ids, buf, runtime_seq_len, None)?;
        }

        if self.use_final_norm_bf16_side_output() {
            self.kernels.rms_norm_forward_bf16(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
        } else {
            self.kernels.rms_norm_forward(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
        }
        if materialize_output_logits || self.lm_head_lora.is_some() {
            let required_logits = t * self.config.vocab_size;
            if buf.logits.numel() < required_logits {
                return Err(PgError::ShapeMismatch {
                    expected: vec![t, self.config.vocab_size],
                    got: buf.logits.shape().to_vec(),
                });
            }
            self.output_projection_forward(&buf.x_in, &buf.x_in_bf16, &buf.logits, t)?;
            if let Some(lora) = &self.lm_head_lora {
                let x_in = buf.x_in.clone();
                let logits = buf.logits.clone();
                self.apply_linear_lora_forward(lora, 0, &x_in, &logits, buf)?;
            }
        }
        Ok(())
    }

    pub fn forward_with_cache(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuForwardCache,
    ) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        self.forward_with_cache_seq_len(input_ids, buf, cache, t)
    }

    pub fn forward_with_cache_seq_len(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuForwardCache,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        self.forward_with_cache_seq_len_timed(input_ids, buf, cache, runtime_seq_len, None)
    }

    fn forward_with_cache_seq_len_timed(
        &self,
        input_ids: &GpuTensor,
        buf: &mut GpuActivations,
        cache: &mut GpuForwardCache,
        runtime_seq_len: usize,
        mut stage_timing: Option<&mut GpuBackwardStageTiming>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let runtime_seq_len = runtime_seq_len.min(t).max(1);
        let d = self.config.model_dim;
        let stream = self.gemm.stream();
        let time_forward_substages = stage_timing.is_some();
        let forward_generation = cache.begin_forward_generation();
        let lean_forward_cache = gpu_lean_forward_cache_enabled();

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);

        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        if self.config.bigram_vocab_size > 0 && gpu_bigram_embedding_merge_enabled() {
            self.kernels.embedding_bigram_project_merge_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_proj.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_scale_param.cu_ptr(stream)?),
                x,
                d as u32,
                self.config.bigram_dim as u32,
                self.config.bigram_vocab_size as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
        } else {
            self.kernels.embedding_gather_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
                x,
                d as u32,
                t as u32,
            )?;
        }

        if self.config.bigram_vocab_size > 0 && !gpu_bigram_embedding_merge_enabled() {
            self.kernels.bigram_hash_embed_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.add_scaled_by_param_fwd(
                x,
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_scale_param.cu_ptr(stream)?),
                1.0,
                (t * d) as u32,
            )?;
        }

        self.copy_tensor(&buf.x, &cache.x_post_embed)?;

        self.kernels
            .rms_norm_forward(x, x_in, t as u32, d as u32, 1.0, 1e-6)?;
        self.copy_tensor(&buf.x_in, &cache.x_post_norm)?;

        if let Some(boundary) = self.config.smear_gate_boundary_token_id {
            self.kernels.smear_gate_fwd_boundary(
                x_in,
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
                x,
                t as u32,
                runtime_seq_len as u32,
                d as u32,
                boundary,
            )?;
        } else {
            self.kernels.smear_gate_fwd(
                x_in,
                CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
                x,
                t as u32,
                runtime_seq_len as u32,
                d as u32,
            )?;
        }
        self.kernels.copy_fwd(x, x0, (t * d) as u32)?;
        if !lean_forward_cache {
            self.copy_tensor(&buf.x, &cache.x0)?;
        }
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_embed_ms),
        )?;
        check_cuda_graph_capture_stage(stream, "forward_embed")?;

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        for layer in 0..n_enc {
            let layer_start = record_stage_event_if(stream, time_forward_substages)?;
            if !lean_forward_cache {
                self.copy_tensor(&buf.x, &cache.layer_x[layer])?;
            }
            self.block_forward_with_cache_slot(
                layer,
                input_ids,
                buf,
                cache,
                runtime_seq_len,
                forward_generation,
                stage_timing.as_deref_mut(),
            )?;
            self.kernels.copy_fwd(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.encoder_skips[layer].cu_ptr(stream)?),
                (t * d) as u32,
            )?;
            if !lean_forward_cache {
                self.copy_tensor(&buf.x, &cache.skips[layer])?;
            }
            finish_stage_event_optional_max(
                stream,
                layer_start,
                stage_timing
                    .as_deref_mut()
                    .map(|timing| &mut timing.forward_encoder_layer_max_ms),
            )?;
            check_cuda_graph_capture_stage(stream, &format!("forward_encoder_layer_{layer}"))?;
        }
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_encoder_ms),
        )?;

        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        for i in 0..n_dec {
            let layer_start = record_stage_event_if(stream, time_forward_substages)?;
            if i < self.config.num_skip_weights() {
                let skip_idx = self.config.num_skip_weights() - 1 - i;
                self.kernels.residual_add_scale_fwd(
                    CudaPtr(buf.x.cu_ptr(stream)?),
                    CudaPtr(buf.encoder_skips[skip_idx].cu_ptr(stream)?),
                    CudaPtr(self.weights.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
            }
            self.copy_tensor(&buf.x, &cache.layer_x[n_enc + i])?;
            let layer = n_enc + i;
            self.block_forward_with_cache_slot(
                layer,
                input_ids,
                buf,
                cache,
                runtime_seq_len,
                forward_generation,
                stage_timing.as_deref_mut(),
            )?;
            finish_stage_event_optional_max(
                stream,
                layer_start,
                stage_timing
                    .as_deref_mut()
                    .map(|timing| &mut timing.forward_decoder_layer_max_ms),
            )?;
            check_cuda_graph_capture_stage(stream, &format!("forward_decoder_layer_{layer}"))?;
        }
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_decoder_ms),
        )?;

        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        if !lean_forward_cache {
            self.copy_tensor(&buf.x, &cache.x_final)?;
        }

        if self.use_final_norm_bf16_side_output() {
            self.kernels.rms_norm_forward_bf16(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
        } else {
            self.kernels.rms_norm_forward(
                CudaPtr(buf.x.cu_ptr(stream)?),
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
        }
        if !self.use_output_ce_no_full_logits() || self.lm_head_lora.is_some() {
            self.output_projection_forward(&buf.x_in, &buf.x_in_bf16, &buf.logits, t)?;
            if let Some(lora) = &self.lm_head_lora {
                let x_in = buf.x_in.clone();
                let logits = buf.logits.clone();
                self.apply_linear_lora_forward(lora, 0, &x_in, &logits, buf)?;
            }
        }
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_logits_ms),
        )?;
        check_cuda_graph_capture_stage(stream, "forward_logits")?;
        Ok(())
    }

    pub fn backward_output_loss_only(
        &self,
        cache: &GpuForwardCache,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<GpuTensor> {
        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let stream = self.gemm.stream();

        let grad_logits = GpuTensor::zeros_gpu(stream.clone(), &[t, vocab], DType::F32)?;
        let grad_logits_bf16 = GpuTensor::zeros_gpu(stream.clone(), &[t, vocab], DType::BF16)?;
        let grad_x = GpuTensor::zeros_gpu(stream.clone(), &[t, d], DType::F32)?;
        let grad_pre_norm = GpuTensor::zeros_gpu(stream.clone(), &[t, d], DType::F32)?;

        self.backward_output_loss_only_into(
            cache,
            &cache.x_final,
            buf,
            targets,
            grads,
            None,
            &grad_logits,
            &grad_logits_bf16,
            &grad_x,
            &grad_pre_norm,
            None,
        )?;

        Ok(grad_pre_norm)
    }

    fn backward_output_loss_only_into(
        &self,
        _cache: &GpuForwardCache,
        final_norm_input: &GpuTensor,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        grads: &mut GpuGradBuffers,
        losses: Option<&GpuTensor>,
        grad_logits: &GpuTensor,
        grad_logits_bf16: &GpuTensor,
        grad_x: &GpuTensor,
        grad_pre_norm: &GpuTensor,
        tiled_output: Option<OutputCeTileScratch<'_>>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let stream = self.gemm.stream();

        if self.use_tiled_output_ce() && losses.is_none() {
            return Err(PgError::InvalidOp(
                "PG_GPU_TILED_OUTPUT_CE requires output-loss scratch because output projection is fused into loss/backward".into(),
            ));
        }
        if self.use_fused_exact_output_ce() && losses.is_none() {
            return Err(PgError::InvalidOp(
                "fused_exact_wmma output CE requires output-loss scratch because output projection is fused into loss/backward".into(),
            ));
        }

        if let Some(lora) = &self.lm_head_lora {
            if let Some(losses) = losses {
                self.cross_entropy_losses(&buf.logits, targets, losses, t)?;
            }
            self.kernels.cross_entropy_bwd(
                CudaPtr(buf.logits.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(grad_logits.cu_ptr(stream)?),
                vocab as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                1.0 / t as f32,
                t as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_input_f32(
                    grad_logits.cu_ptr(stream)?,
                    self.weights.tok_emb.cu_ptr(stream)?,
                    grad_x.cu_ptr(stream)?,
                    t,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_f32(
                    grad_logits.cu_ptr(stream)?,
                    buf.x_in.cu_ptr(stream)?,
                    grads.tok_emb.cu_ptr(stream)?,
                    t,
                    vocab,
                    d,
                    1.0,
                    1.0,
                )?;
            }
            let x_in = buf.x_in.clone();
            let grad_x_accum = grad_x.clone();
            self.backward_linear_lora(lora, 0, &x_in, grad_logits, &grad_x_accum, buf)?;
            self.kernels.rms_norm_backward(
                CudaPtr(final_norm_input.cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(grad_pre_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
            return Ok(());
        }

        if self.use_chunked_bf16_output_ce_cache() {
            let tiled = tiled_output.ok_or_else(|| {
                PgError::InvalidOp("PG_GPU_CHUNKED_OUTPUT_CE_CACHE requires chunk scratch".into())
            })?;
            if let Some(losses) = losses {
                self.backward_output_chunked_cached_ce_into(
                    buf, targets, losses, grads, grad_x, tiled,
                )?;
            } else {
                self.backward_output_chunked_cached_ce_no_loss_into(
                    buf, targets, grads, grad_x, tiled,
                )?;
            }
            self.kernels.rms_norm_backward(
                CudaPtr(final_norm_input.cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(grad_pre_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
            return Ok(());
        }

        if self.use_fused_exact_output_ce() {
            let losses = losses.ok_or_else(|| {
                PgError::InvalidOp("fused_exact_wmma output CE requires loss storage".into())
            })?;
            let tiled = tiled_output.ok_or_else(|| {
                PgError::InvalidOp("fused_exact_wmma output CE requires tile scratch".into())
            })?;
            self.backward_output_fused_exact_ce_into(buf, targets, losses, grads, grad_x, tiled)?;
            self.kernels.rms_norm_backward(
                CudaPtr(final_norm_input.cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(grad_pre_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
            return Ok(());
        }

        if self.use_tiled_output_ce() {
            let losses = losses.ok_or_else(|| {
                PgError::InvalidOp("PG_GPU_TILED_OUTPUT_CE requires loss storage".into())
            })?;
            let tiled = tiled_output.ok_or_else(|| {
                PgError::InvalidOp("PG_GPU_TILED_OUTPUT_CE requires tile scratch".into())
            })?;
            self.backward_output_tiled_ce_into(buf, targets, losses, grads, grad_x, tiled)?;
            self.kernels.rms_norm_backward(
                CudaPtr(final_norm_input.cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(grad_pre_norm.cu_ptr(stream)?),
                t as u32,
                d as u32,
                1.0,
                1e-6,
            )?;
            return Ok(());
        }

        if self.use_bf16_output_backward_gemm() {
            if buf.logits.dtype() == DType::BF16 {
                if let Some(losses) = losses.filter(|_| self.use_fused_ce_loss_bwd()) {
                    self.kernels.cross_entropy_loss_bwd_bf16_logits(
                        CudaPtr(buf.logits.cu_ptr(stream)?),
                        CudaPtr(targets.cu_ptr(stream)?),
                        CudaPtr(losses.cu_ptr(stream)?),
                        CudaPtr(grad_logits_bf16.cu_ptr(stream)?),
                        vocab as u32,
                        self.config.logit_softcap_pos,
                        self.config.logit_softcap_neg,
                        1.0 / t as f32,
                        t as u32,
                    )?;
                } else {
                    self.kernels.cross_entropy_bwd_bf16_logits(
                        CudaPtr(buf.logits.cu_ptr(stream)?),
                        CudaPtr(targets.cu_ptr(stream)?),
                        CudaPtr(grad_logits_bf16.cu_ptr(stream)?),
                        vocab as u32,
                        self.config.logit_softcap_pos,
                        self.config.logit_softcap_neg,
                        1.0 / t as f32,
                        t as u32,
                    )?;
                }
            } else {
                if let Some(losses) = losses.filter(|_| self.use_fused_ce_loss_bwd()) {
                    self.kernels.cross_entropy_loss_bwd_bf16(
                        CudaPtr(buf.logits.cu_ptr(stream)?),
                        CudaPtr(targets.cu_ptr(stream)?),
                        CudaPtr(losses.cu_ptr(stream)?),
                        CudaPtr(grad_logits_bf16.cu_ptr(stream)?),
                        vocab as u32,
                        self.config.logit_softcap_pos,
                        self.config.logit_softcap_neg,
                        1.0 / t as f32,
                        t as u32,
                    )?;
                } else {
                    self.kernels.cross_entropy_bwd_bf16(
                        CudaPtr(buf.logits.cu_ptr(stream)?),
                        CudaPtr(targets.cu_ptr(stream)?),
                        CudaPtr(grad_logits_bf16.cu_ptr(stream)?),
                        vocab as u32,
                        self.config.logit_softcap_pos,
                        self.config.logit_softcap_neg,
                        1.0 / t as f32,
                        t as u32,
                    )?;
                }
            }
            if !self.use_bf16_output_gemm() {
                self.kernels.f32_to_bf16(
                    CudaPtr(buf.x_in.cu_ptr(stream)?),
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
            }
            unsafe {
                self.gemm.linear_backward_input_bf16_to_f32(
                    grad_logits_bf16.cu_ptr(stream)?,
                    self.weights.tok_emb_bf16.cu_ptr(stream)?,
                    grad_x.cu_ptr(stream)?,
                    t,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_bf16_to_f32(
                    grad_logits_bf16.cu_ptr(stream)?,
                    buf.x_in_bf16.cu_ptr(stream)?,
                    grads.tok_emb.cu_ptr(stream)?,
                    t,
                    vocab,
                    d,
                    1.0,
                    1.0,
                )?;
            }
        } else {
            self.kernels.cross_entropy_bwd(
                CudaPtr(buf.logits.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(grad_logits.cu_ptr(stream)?),
                vocab as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                1.0 / t as f32,
                t as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_input_f32(
                    grad_logits.cu_ptr(stream)?,
                    self.weights.tok_emb.cu_ptr(stream)?,
                    grad_x.cu_ptr(stream)?,
                    t,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_f32(
                    grad_logits.cu_ptr(stream)?,
                    buf.x_in.cu_ptr(stream)?,
                    grads.tok_emb.cu_ptr(stream)?,
                    t,
                    vocab,
                    d,
                    1.0,
                    1.0,
                )?;
            }
        }

        self.kernels.rms_norm_backward(
            CudaPtr(final_norm_input.cu_ptr(stream)?),
            CudaPtr(grad_x.cu_ptr(stream)?),
            CudaPtr(grad_pre_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;

        Ok(())
    }

    fn backward_output_chunked_cached_ce_no_loss_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        grads: &mut GpuGradBuffers,
        grad_x: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let chunk_tokens = output_ce_chunk_tokens_for_config(&self.config, t);
        let stream = self.gemm.stream();
        if tiled.logits_tile.dtype() != DType::BF16 {
            return Err(PgError::InvalidOp(
                "PG_GPU_CHUNKED_OUTPUT_CE_CACHE requires BF16 chunk logits scratch".into(),
            ));
        }
        let tile_shape = tiled.logits_tile.shape();
        if tile_shape.len() != 2 || tile_shape[0] < chunk_tokens.min(t) || tile_shape[1] != vocab {
            return Err(PgError::InvalidOp(format!(
                "PG_GPU_CHUNKED_OUTPUT_CE_CACHE scratch shape mismatch: expected at least [{}, {}], got {:?}",
                chunk_tokens.min(t),
                vocab,
                tile_shape
            )));
        }
        if !self.use_final_norm_bf16_side_output() {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }

        for chunk_start in (0..t).step_by(chunk_tokens) {
            let chunk_end = (chunk_start + chunk_tokens).min(t);
            let chunk = chunk_end - chunk_start;
            let hidden_chunk = buf.x_in_bf16.slice_range(chunk_start, chunk_end)?;
            let targets_chunk = targets.slice_range(chunk_start, chunk_end)?;
            let logits_chunk = tiled.logits_tile.slice_range(0, chunk)?;
            unsafe {
                self.gemm.matmul_bf16_bt(
                    hidden_chunk.cu_ptr(stream)?,
                    self.weights.tok_emb_bf16.cu_ptr(stream)?,
                    logits_chunk.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            let grad_x_chunk = grad_x.slice_range(chunk_start, chunk_end)?;
            let grad_chunk = tiled.grad_tile_bf16.slice_range(0, chunk)?;
            self.kernels.cross_entropy_bwd_bf16_logits(
                CudaPtr(logits_chunk.cu_ptr(stream)?),
                CudaPtr(targets_chunk.cu_ptr(stream)?),
                CudaPtr(grad_chunk.cu_ptr(stream)?),
                vocab as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                1.0 / t as f32,
                chunk as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_input_bf16_to_f32(
                    grad_chunk.cu_ptr(stream)?,
                    self.weights.tok_emb_bf16.cu_ptr(stream)?,
                    grad_x_chunk.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_bf16_to_f32(
                    grad_chunk.cu_ptr(stream)?,
                    hidden_chunk.cu_ptr(stream)?,
                    grads.tok_emb.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    1.0,
                )?;
            }
        }
        Ok(())
    }

    fn backward_output_chunked_cached_ce_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        losses: &GpuTensor,
        grads: &mut GpuGradBuffers,
        grad_x: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let chunk_tokens = output_ce_chunk_tokens_for_config(&self.config, t);
        let stream = self.gemm.stream();
        if tiled.logits_tile.dtype() != DType::BF16 {
            return Err(PgError::InvalidOp(
                "PG_GPU_CHUNKED_OUTPUT_CE_CACHE requires BF16 chunk logits scratch".into(),
            ));
        }
        let tile_shape = tiled.logits_tile.shape();
        if tile_shape.len() != 2 || tile_shape[0] < chunk_tokens.min(t) || tile_shape[1] != vocab {
            return Err(PgError::InvalidOp(format!(
                "PG_GPU_CHUNKED_OUTPUT_CE_CACHE scratch shape mismatch: expected at least [{}, {}], got {:?}",
                chunk_tokens.min(t),
                vocab,
                tile_shape
            )));
        }
        if !self.use_final_norm_bf16_side_output() {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }

        for chunk_start in (0..t).step_by(chunk_tokens) {
            let chunk_end = (chunk_start + chunk_tokens).min(t);
            let chunk = chunk_end - chunk_start;
            let hidden_chunk = buf.x_in_bf16.slice_range(chunk_start, chunk_end)?;
            let targets_chunk = targets.slice_range(chunk_start, chunk_end)?;
            let losses_chunk = losses.slice_range(chunk_start, chunk_end)?;
            let logits_chunk = tiled.logits_tile.slice_range(0, chunk)?;
            unsafe {
                self.gemm.matmul_bf16_bt(
                    hidden_chunk.cu_ptr(stream)?,
                    self.weights.tok_emb_bf16.cu_ptr(stream)?,
                    logits_chunk.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            let grad_x_chunk = grad_x.slice_range(chunk_start, chunk_end)?;
            let grad_chunk = tiled.grad_tile_bf16.slice_range(0, chunk)?;
            self.kernels.cross_entropy_loss_bwd_bf16_logits(
                CudaPtr(logits_chunk.cu_ptr(stream)?),
                CudaPtr(targets_chunk.cu_ptr(stream)?),
                CudaPtr(losses_chunk.cu_ptr(stream)?),
                CudaPtr(grad_chunk.cu_ptr(stream)?),
                vocab as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                1.0 / t as f32,
                chunk as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_input_bf16_to_f32(
                    grad_chunk.cu_ptr(stream)?,
                    self.weights.tok_emb_bf16.cu_ptr(stream)?,
                    grad_x_chunk.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_bf16_to_f32(
                    grad_chunk.cu_ptr(stream)?,
                    hidden_chunk.cu_ptr(stream)?,
                    grads.tok_emb.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    1.0,
                )?;
            }
        }
        Ok(())
    }

    fn output_chunked_cached_ce_forward_losses_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let chunk_tokens = output_ce_chunk_tokens_for_config(&self.config, t);
        let stream = self.gemm.stream();
        if !self.use_final_norm_bf16_side_output() {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }

        for chunk_start in (0..t).step_by(chunk_tokens) {
            let chunk_end = (chunk_start + chunk_tokens).min(t);
            let chunk = chunk_end - chunk_start;
            let hidden_chunk = buf.x_in_bf16.slice_range(chunk_start, chunk_end)?;
            let targets_chunk = targets.slice_range(chunk_start, chunk_end)?;
            let losses_chunk = losses.slice_range(chunk_start, chunk_end)?;
            let logits_chunk = tiled.logits_tile.slice_range(0, chunk)?;
            unsafe {
                self.gemm.matmul_bf16_bt(
                    hidden_chunk.cu_ptr(stream)?,
                    self.weights.tok_emb_bf16.cu_ptr(stream)?,
                    logits_chunk.cu_ptr(stream)?,
                    chunk,
                    vocab,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.cross_entropy_fwd_bf16_logits(
                CudaPtr(logits_chunk.cu_ptr(stream)?),
                CudaPtr(targets_chunk.cu_ptr(stream)?),
                CudaPtr(losses_chunk.cu_ptr(stream)?),
                vocab as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                chunk as u32,
            )?;
        }
        Ok(())
    }

    fn backward_output_tiled_ce_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        losses: &GpuTensor,
        grads: &mut GpuGradBuffers,
        grad_x: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        self.output_tiled_ce_forward_losses_into(
            buf,
            targets,
            losses,
            OutputCeTileScratch {
                logits_tile: tiled.logits_tile,
                grad_tile_bf16: tiled.grad_tile_bf16,
                row_max: tiled.row_max,
                row_sum: tiled.row_sum,
                target_logit: tiled.target_logit,
            },
        )?;
        self.output_tiled_ce_backward_from_stats_into(buf, targets, grads, grad_x, tiled)
    }

    fn backward_output_fused_exact_ce_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        losses: &GpuTensor,
        grads: &mut GpuGradBuffers,
        grad_x: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        self.output_fused_exact_ce_forward_losses_into(buf, targets, losses, tiled)?;
        self.output_fused_exact_ce_backward_from_stats_into(buf, targets, grads, grad_x, tiled)
    }

    fn output_fused_exact_ce_forward_losses_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let tile = output_ce_tile_vocab_for_config(&self.config);
        if vocab % tile != 0 {
            return Err(PgError::InvalidOp(format!(
                "fused_exact_wmma output CE requires vocab_size divisible by tile; vocab={vocab} tile={tile}"
            )));
        }
        if tiled.logits_tile.dtype() != DType::F32 || tiled.grad_tile_bf16.dtype() != DType::BF16 {
            return Err(PgError::InvalidOp(
                "fused_exact_wmma output CE requires F32 logits scratch and BF16 grad scratch"
                    .into(),
            ));
        }
        if !self.use_final_norm_bf16_side_output() {
            let stream = self.gemm.stream();
            self.kernels.f32_to_bf16(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }
        let Some(fused) = self.fused_output_ce.as_ref() else {
            return Err(PgError::InvalidOp(
                "fused_exact_wmma output CE was requested but pg-kernels was not built with has_fused_output_ce".into(),
            ));
        };
        let stream = self.gemm.stream();
        fused.stats_bf16(
            buf.x_in_bf16.cu_ptr(stream)?,
            self.weights.tok_emb_bf16.cu_ptr(stream)?,
            targets.cu_ptr(stream)?,
            tiled.row_max.cu_ptr(stream)?,
            tiled.row_sum.cu_ptr(stream)?,
            tiled.target_logit.cu_ptr(stream)?,
            losses.cu_ptr(stream)?,
            tiled.logits_tile.cu_ptr(stream)?,
            t,
            vocab,
            d,
            tile,
            self.config.logit_softcap_pos,
        )
    }

    fn output_fused_exact_ce_backward_from_stats_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        grads: &mut GpuGradBuffers,
        grad_x: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let tile = output_ce_tile_vocab_for_config(&self.config);
        if vocab % tile != 0 {
            return Err(PgError::InvalidOp(format!(
                "fused_exact_wmma output CE requires vocab_size divisible by tile; vocab={vocab} tile={tile}"
            )));
        }
        let Some(fused) = self.fused_output_ce.as_ref() else {
            return Err(PgError::InvalidOp(
                "fused_exact_wmma output CE was requested but pg-kernels was not built with has_fused_output_ce".into(),
            ));
        };
        let stream = self.gemm.stream();
        fused.backward_bf16(
            buf.x_in_bf16.cu_ptr(stream)?,
            self.weights.tok_emb_bf16.cu_ptr(stream)?,
            targets.cu_ptr(stream)?,
            tiled.row_max.cu_ptr(stream)?,
            tiled.row_sum.cu_ptr(stream)?,
            grad_x.cu_ptr(stream)?,
            grads.tok_emb.cu_ptr(stream)?,
            tiled.logits_tile.cu_ptr(stream)?,
            tiled.grad_tile_bf16.cu_ptr(stream)?,
            t,
            vocab,
            d,
            tile,
            self.config.logit_softcap_pos,
            1.0 / t as f32,
        )
    }

    fn output_tiled_ce_forward_losses_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let tile = output_ce_tile_vocab_for_config(&self.config);
        if vocab % tile != 0 {
            return Err(PgError::InvalidOp(format!(
                "tiled output CE requires vocab_size divisible by tile; vocab={vocab} tile={tile}"
            )));
        }
        let stream = self.gemm.stream();
        if !self.use_final_norm_bf16_side_output() {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.x_in.cu_ptr(stream)?),
                CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                (t * d) as u32,
            )?;
        }

        self.kernels.output_ce_stats_init(
            CudaPtr(tiled.row_max.cu_ptr(stream)?),
            CudaPtr(tiled.row_sum.cu_ptr(stream)?),
            CudaPtr(tiled.target_logit.cu_ptr(stream)?),
            t as u32,
        )?;

        for vocab_start in (0..vocab).step_by(tile) {
            let weight_tile = self
                .weights
                .tok_emb_bf16
                .slice_range(vocab_start, vocab_start + tile)?;
            unsafe {
                self.gemm.matmul_bf16_bt_to_f32(
                    buf.x_in_bf16.cu_ptr(stream)?,
                    weight_tile.cu_ptr(stream)?,
                    tiled.logits_tile.cu_ptr(stream)?,
                    t,
                    tile,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.output_ce_tile_stats_update(
                CudaPtr(tiled.logits_tile.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(tiled.row_max.cu_ptr(stream)?),
                CudaPtr(tiled.row_sum.cu_ptr(stream)?),
                CudaPtr(tiled.target_logit.cu_ptr(stream)?),
                vocab_start as u32,
                tile as u32,
                tile as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                t as u32,
            )?;
        }

        self.kernels.output_ce_finalize_loss(
            CudaPtr(tiled.row_max.cu_ptr(stream)?),
            CudaPtr(tiled.row_sum.cu_ptr(stream)?),
            CudaPtr(tiled.target_logit.cu_ptr(stream)?),
            CudaPtr(losses.cu_ptr(stream)?),
            t as u32,
        )?;
        Ok(())
    }

    fn output_tiled_ce_backward_from_stats_into(
        &self,
        buf: &mut GpuActivations,
        targets: &GpuTensor,
        grads: &mut GpuGradBuffers,
        grad_x: &GpuTensor,
        tiled: OutputCeTileScratch<'_>,
    ) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = targets.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let vocab = self.config.vocab_size;
        let tile = output_ce_tile_vocab_for_config(&self.config);
        if vocab % tile != 0 {
            return Err(PgError::InvalidOp(format!(
                "tiled output CE requires vocab_size divisible by tile; vocab={vocab} tile={tile}"
            )));
        }
        let stream = self.gemm.stream();
        for (tile_idx, vocab_start) in (0..vocab).step_by(tile).enumerate() {
            let weight_tile = self
                .weights
                .tok_emb_bf16
                .slice_range(vocab_start, vocab_start + tile)?;
            let grad_weight_tile = grads.tok_emb.slice_range(vocab_start, vocab_start + tile)?;
            unsafe {
                self.gemm.matmul_bf16_bt_to_f32(
                    buf.x_in_bf16.cu_ptr(stream)?,
                    weight_tile.cu_ptr(stream)?,
                    tiled.logits_tile.cu_ptr(stream)?,
                    t,
                    tile,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.output_ce_tile_grad_bf16(
                CudaPtr(tiled.logits_tile.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(tiled.row_max.cu_ptr(stream)?),
                CudaPtr(tiled.row_sum.cu_ptr(stream)?),
                CudaPtr(tiled.grad_tile_bf16.cu_ptr(stream)?),
                vocab_start as u32,
                tile as u32,
                tile as u32,
                self.config.logit_softcap_pos,
                self.config.logit_softcap_neg,
                1.0 / t as f32,
                t as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_input_bf16_to_f32(
                    tiled.grad_tile_bf16.cu_ptr(stream)?,
                    weight_tile.cu_ptr(stream)?,
                    grad_x.cu_ptr(stream)?,
                    t,
                    tile,
                    d,
                    1.0,
                    if tile_idx == 0 { 0.0 } else { 1.0 },
                )?;
                self.gemm.linear_backward_weight_bf16_to_f32(
                    tiled.grad_tile_bf16.cu_ptr(stream)?,
                    buf.x_in_bf16.cu_ptr(stream)?,
                    grad_weight_tile.cu_ptr(stream)?,
                    t,
                    tile,
                    d,
                    1.0,
                    1.0,
                )?;
            }
        }
        Ok(())
    }

    pub fn uses_tiled_output_ce(&self) -> bool {
        self.use_tiled_output_ce()
    }

    pub fn uses_chunked_bf16_output_ce_cache(&self) -> bool {
        self.use_chunked_bf16_output_ce_cache()
    }

    pub fn uses_fused_exact_output_ce(&self) -> bool {
        self.use_fused_exact_output_ce()
    }

    pub fn cross_entropy_losses_with_state(
        &self,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tokens: usize,
    ) -> PgResult<()> {
        if self.lm_head_lora.is_some() {
            let target_tokens = targets.shape().iter().product::<usize>();
            if target_tokens != tokens {
                return Err(PgError::InvalidOp(format!(
                    "cross_entropy_losses_with_state token mismatch: targets={target_tokens} tokens={tokens}"
                )));
            }
            return self.cross_entropy_losses(&buf.logits, targets, losses, tokens);
        }
        if self.use_chunked_bf16_output_ce_cache() {
            let target_tokens = targets.shape().iter().product::<usize>();
            if target_tokens != tokens {
                return Err(PgError::InvalidOp(format!(
                    "cross_entropy_losses_with_state token mismatch: targets={target_tokens} tokens={tokens}"
                )));
            }
            self.output_chunked_cached_ce_forward_losses_into(
                buf,
                targets,
                losses,
                OutputCeTileScratch {
                    logits_tile: &state.output_logits_tile,
                    grad_tile_bf16: &state.output_grad_tile_bf16,
                    row_max: &state.output_row_max,
                    row_sum: &state.output_row_sum,
                    target_logit: &state.output_target_logit,
                },
            )
        } else if self.use_fused_exact_output_ce() {
            let target_tokens = targets.shape().iter().product::<usize>();
            if target_tokens != tokens {
                return Err(PgError::InvalidOp(format!(
                    "cross_entropy_losses_with_state token mismatch: targets={target_tokens} tokens={tokens}"
                )));
            }
            self.output_fused_exact_ce_forward_losses_into(
                buf,
                targets,
                losses,
                OutputCeTileScratch {
                    logits_tile: &state.output_logits_tile,
                    grad_tile_bf16: &state.output_grad_tile_bf16,
                    row_max: &state.output_row_max,
                    row_sum: &state.output_row_sum,
                    target_logit: &state.output_target_logit,
                },
            )
        } else if self.use_tiled_output_ce() {
            let target_tokens = targets.shape().iter().product::<usize>();
            if target_tokens != tokens {
                return Err(PgError::InvalidOp(format!(
                    "cross_entropy_losses_with_state token mismatch: targets={target_tokens} tokens={tokens}"
                )));
            }
            self.output_tiled_ce_forward_losses_into(
                buf,
                targets,
                losses,
                OutputCeTileScratch {
                    logits_tile: &state.output_logits_tile,
                    grad_tile_bf16: &state.output_grad_tile_bf16,
                    row_max: &state.output_row_max,
                    row_sum: &state.output_row_sum,
                    target_logit: &state.output_target_logit,
                },
            )
        } else {
            self.cross_entropy_losses(&buf.logits, targets, losses, tokens)
        }
    }

    pub fn backward_with_state(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<f32> {
        let t = input_ids.shape().iter().product::<usize>();
        self.backward_with_state_seq_len_loss_mode(input_ids, targets, buf, state, grads, true, t)
    }

    pub fn backward_with_state_no_loss(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<()> {
        let t = input_ids.shape().iter().product::<usize>();
        self.backward_with_state_seq_len_loss_mode(input_ids, targets, buf, state, grads, false, t)
            .map(|_| ())
    }

    pub fn backward_with_state_seq_len(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
    ) -> PgResult<f32> {
        self.backward_with_state_seq_len_loss_mode(
            input_ids,
            targets,
            buf,
            state,
            grads,
            true,
            runtime_seq_len,
        )
    }

    pub fn backward_with_state_seq_len_no_loss(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
    ) -> PgResult<()> {
        self.backward_with_state_seq_len_loss_mode(
            input_ids,
            targets,
            buf,
            state,
            grads,
            false,
            runtime_seq_len,
        )
        .map(|_| ())
    }

    pub fn backward_with_state_seq_len_no_loss_observed(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        runtime_seq_len: usize,
        observer: &mut dyn GpuBackwardLayerObserver,
    ) -> PgResult<()> {
        self.backward_with_state_seq_len_loss_mode_observed(
            input_ids,
            targets,
            buf,
            state,
            grads,
            false,
            runtime_seq_len,
            Some(observer),
        )
        .map(|_| ())
    }

    fn backward_with_state_seq_len_loss_mode(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        compute_loss: bool,
        runtime_seq_len: usize,
    ) -> PgResult<f32> {
        self.backward_with_state_seq_len_loss_mode_observed(
            input_ids,
            targets,
            buf,
            state,
            grads,
            compute_loss,
            runtime_seq_len,
            None,
        )
    }

    fn backward_with_state_seq_len_loss_mode_observed(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        grads: &mut GpuGradBuffers,
        compute_loss: bool,
        runtime_seq_len: usize,
        mut observer: Option<&mut dyn GpuBackwardLayerObserver>,
    ) -> PgResult<f32> {
        use pg_kernels::gpu_kernels::CudaPtr;

        let t = input_ids.shape().iter().product::<usize>();
        let d = self.config.model_dim;
        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        let n_skip = self.config.num_skip_weights();
        let stream = self.gemm.stream();
        let runtime_seq_len = runtime_seq_len.min(t).max(1);
        state.stage_timing = GpuBackwardStageTiming::default();

        let stage_start = record_stage_event(stream)?;
        let cache = &mut state.cache;
        let stage_timing = &mut state.stage_timing;
        self.forward_with_cache_seq_len_timed(
            input_ids,
            buf,
            cache,
            runtime_seq_len,
            Some(stage_timing),
        )?;
        // Capture the forward_generation that the just-completed forward stamped
        // into the saved BF16 freshness state. Backward consumer-side `require()`
        // calls verify the saved BF16 buffers belong to this exact step.
        let forward_generation = state.cache.forward_generation.get();
        let lean_forward_cache = gpu_lean_forward_cache_enabled();
        finish_stage_event(stream, stage_start, &mut state.stage_timing.forward_ms)?;
        check_cuda_graph_capture_stage(stream, "forward_with_cache")?;

        let stage_start = record_stage_event(stream)?;
        let tiled_output_loss = self.use_tiled_output_ce()
            || self.use_chunked_bf16_output_ce_cache()
            || self.use_fused_exact_output_ce();
        let fused_output_loss = compute_loss && self.use_fused_ce_loss_bwd();
        let output_losses = if compute_loss && (tiled_output_loss || fused_output_loss) {
            Some(&state.losses)
        } else if !compute_loss && (self.use_tiled_output_ce() || self.use_fused_exact_output_ce())
        {
            // The record-shaped graph path drops the scalar loss, but the
            // tiled/fused exact output kernels still use the persistent loss
            // buffer as per-row CE scratch. Keep the chunked BF16 cache on its
            // no-loss path; only the opt-in fused/tiled experiments pay this.
            Some(&state.losses)
        } else {
            None
        };
        let loss = if compute_loss && output_losses.is_none() {
            self.cross_entropy_losses(&buf.logits, targets, &state.losses, t)?;
            self.mean_losses_only_with_sum_scratch(&state.losses, &state.loss_sum, t)?
        } else {
            0.0
        };

        let final_norm_input = if lean_forward_cache {
            buf.x.clone()
        } else {
            state.cache.x_final.clone()
        };
        self.backward_output_loss_only_into(
            &state.cache,
            &final_norm_input,
            buf,
            targets,
            grads,
            output_losses,
            &state.grad_logits,
            &state.grad_logits_bf16,
            &state.grad_output_x,
            &state.grad_output_pre_norm,
            Some(OutputCeTileScratch {
                logits_tile: &state.output_logits_tile,
                grad_tile_bf16: &state.output_grad_tile_bf16,
                row_max: &state.output_row_max,
                row_sum: &state.output_row_sum,
                target_logit: &state.output_target_logit,
            }),
        )?;
        let loss = if compute_loss && output_losses.is_some() {
            self.mean_losses_only_with_sum_scratch(&state.losses, &state.loss_sum, t)?
        } else {
            loss
        };
        self.copy_tensor(&state.grad_output_pre_norm, &state.grad_ping)?;
        self.zero_tensor(&state.grad_x0)?;
        finish_stage_event(stream, stage_start, &mut state.stage_timing.output_ms)?;
        check_cuda_graph_capture_stage(stream, "output_loss_backward")?;

        let mut grad_x = state.grad_ping.clone();
        let mut next_grad_x = state.grad_pong.clone();
        let backward_x0 = if lean_forward_cache {
            buf.x0.clone()
        } else {
            state.cache.x0.clone()
        };

        let stage_start = record_stage_event(stream)?;
        for i in (0..n_dec).rev() {
            let bi = n_enc + i;
            if grad_x.overlaps_storage_region(&next_grad_x) {
                return Err(PgError::InvalidOp(format!(
                    "decoder backward gradient ping-pong alias before layer {bi}"
                )));
            }
            let saved = state
                .cache
                .saved_layers
                .get(bi)
                .and_then(|saved| saved.as_ref());
            let recurrent_pass1_saved = state
                .cache
                .recurrent_pass1_layers
                .get(bi)
                .and_then(|saved| saved.as_ref());
            let recurrent_mid_x = state
                .cache
                .recurrent_mid_x
                .get(bi)
                .and_then(|mid| mid.as_ref());
            self.block_backward_into(
                bi,
                input_ids,
                &state.cache.layer_x[bi],
                &backward_x0,
                buf,
                &mut state.block_cache,
                &grad_x,
                &state.grad_x0,
                &next_grad_x,
                grads,
                runtime_seq_len,
                saved,
                recurrent_pass1_saved,
                recurrent_mid_x,
                forward_generation,
                Some(&mut state.stage_timing),
            )?;
            if let Some(observer) = observer.as_deref_mut() {
                observer.after_layer(self, bi, grads)?;
            }
            check_cuda_graph_capture_stage(stream, &format!("decoder_layer_{bi}_backward"))?;
            std::mem::swap(&mut grad_x, &mut next_grad_x);

            if i < n_skip {
                let enc_layer = n_enc - 1 - i;
                let skip_src = if lean_forward_cache {
                    buf.encoder_skips[enc_layer].clone()
                } else {
                    state.cache.skips[enc_layer].clone()
                };
                self.kernels.residual_add_scale_bwd_assign(
                    CudaPtr(skip_src.cu_ptr(stream)?),
                    CudaPtr(grad_x.cu_ptr(stream)?),
                    CudaPtr(self.weights.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    CudaPtr(state.grad_x_post_skip.cu_ptr(stream)?),
                    CudaPtr(state.grad_encoder_skips[enc_layer].cu_ptr(stream)?),
                    CudaPtr(grads.skip_weights.slice_first(i)?.cu_ptr(stream)?),
                    d as u32,
                    (t * d) as u32,
                )?;
                grad_x = state.grad_x_post_skip.clone();
                // `grad_x_post_skip` is a third scratch buffer outside the ping/pong pair.
                // After rebinding `grad_x` to it, make the next block write back into a
                // real ping/pong buffer; otherwise consecutive decoder skips can alias the
                // input and output gradients for the following block.
                next_grad_x = state.grad_ping.clone();
                check_cuda_graph_capture_stage(stream, &format!("decoder_layer_{bi}_skip_bwd"))?;
            }
        }
        finish_stage_event(stream, stage_start, &mut state.stage_timing.decoder_ms)?;
        check_cuda_graph_capture_stage(stream, "decoder_backward")?;

        let stage_start = record_stage_event(stream)?;
        for i in (0..n_enc).rev() {
            if i < n_skip {
                self.add_inplace(&grad_x, &state.grad_encoder_skips[i], 1.0)?;
            }
            if grad_x.overlaps_storage_region(&next_grad_x) {
                return Err(PgError::InvalidOp(format!(
                    "encoder backward gradient ping-pong alias before layer {i}"
                )));
            }
            let saved = state
                .cache
                .saved_layers
                .get(i)
                .and_then(|saved| saved.as_ref());
            let recurrent_pass1_saved = state
                .cache
                .recurrent_pass1_layers
                .get(i)
                .and_then(|saved| saved.as_ref());
            let recurrent_mid_x = state
                .cache
                .recurrent_mid_x
                .get(i)
                .and_then(|mid| mid.as_ref());
            let layer_x = if lean_forward_cache {
                if i == 0 {
                    backward_x0.clone()
                } else {
                    buf.encoder_skips[i - 1].clone()
                }
            } else {
                state.cache.layer_x[i].clone()
            };
            self.block_backward_into(
                i,
                input_ids,
                &layer_x,
                &backward_x0,
                buf,
                &mut state.block_cache,
                &grad_x,
                &state.grad_x0,
                &next_grad_x,
                grads,
                runtime_seq_len,
                saved,
                recurrent_pass1_saved,
                recurrent_mid_x,
                forward_generation,
                Some(&mut state.stage_timing),
            )?;
            if let Some(observer) = observer.as_deref_mut() {
                observer.after_layer(self, i, grads)?;
            }
            check_cuda_graph_capture_stage(stream, &format!("encoder_layer_{i}_backward"))?;
            std::mem::swap(&mut grad_x, &mut next_grad_x);
        }
        finish_stage_event(stream, stage_start, &mut state.stage_timing.encoder_ms)?;
        check_cuda_graph_capture_stage(stream, "encoder_backward")?;

        let stage_start = record_stage_event(stream)?;
        self.add_inplace(&grad_x, &state.grad_x0, 1.0)?;

        if let Some(boundary) = self.config.smear_gate_boundary_token_id {
            self.kernels.smear_gate_bwd_boundary(
                CudaPtr(state.cache.x_post_norm.cu_ptr(stream)?),
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(state.grad_x_smear.cu_ptr(stream)?),
                CudaPtr(state.grad_x_prev.cu_ptr(stream)?),
                CudaPtr(grads.smear_gate.cu_ptr(stream)?),
                t as u32,
                runtime_seq_len as u32,
                d as u32,
                boundary,
            )?;
        } else {
            self.kernels.smear_gate_bwd(
                CudaPtr(state.cache.x_post_norm.cu_ptr(stream)?),
                CudaPtr(self.weights.smear_gate.cu_ptr(stream)?),
                CudaPtr(grad_x.cu_ptr(stream)?),
                CudaPtr(state.grad_x_smear.cu_ptr(stream)?),
                CudaPtr(state.grad_x_prev.cu_ptr(stream)?),
                CudaPtr(grads.smear_gate.cu_ptr(stream)?),
                t as u32,
                runtime_seq_len as u32,
                d as u32,
            )?;
        }

        self.copy_tensor(&state.grad_x_smear, &state.grad_x_post_norm)?;
        if t > 1 {
            let dst = state.grad_x_post_norm.slice_range(0, t - 1)?;
            let src = state.grad_x_prev.slice_range(1, t)?;
            self.add_inplace(&dst, &src, 1.0)?;
        }

        self.kernels.rms_norm_backward(
            CudaPtr(state.cache.x_post_embed.cu_ptr(stream)?),
            CudaPtr(state.grad_x_post_norm.cu_ptr(stream)?),
            CudaPtr(state.grad_x_post_embed.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;

        if self.config.bigram_vocab_size > 0 {
            self.kernels.bigram_hash_embed_fwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_out.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
            unsafe {
                self.gemm.matmul_f32(
                    buf.bigram_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    buf.bigram_proj_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.dot_accumulate(
                CudaPtr(state.grad_x_post_embed.cu_ptr(stream)?),
                CudaPtr(buf.bigram_proj_out.cu_ptr(stream)?),
                CudaPtr(grads.bigram_scale.cu_ptr(stream)?),
                1.0,
                (t * d) as u32,
            )?;

            self.zero_tensor(&state.grad_bigram_proj_out)?;
            self.kernels.add_scaled_by_param_fwd(
                CudaPtr(state.grad_bigram_proj_out.cu_ptr(stream)?),
                CudaPtr(state.grad_x_post_embed.cu_ptr(stream)?),
                CudaPtr(self.weights.bigram_scale_param.cu_ptr(stream)?),
                1.0,
                (t * d) as u32,
            )?;
            unsafe {
                self.gemm.linear_backward_weight_f32(
                    state.grad_bigram_proj_out.cu_ptr(stream)?,
                    buf.bigram_out.cu_ptr(stream)?,
                    grads.bigram_proj.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    1.0,
                )?;
            }
            unsafe {
                self.gemm.linear_backward_input_f32(
                    state.grad_bigram_proj_out.cu_ptr(stream)?,
                    self.weights.bigram_proj.cu_ptr(stream)?,
                    state.grad_bigram_out.cu_ptr(stream)?,
                    t,
                    d,
                    self.config.bigram_dim,
                    1.0,
                    0.0,
                )?;
            }
            self.kernels.bigram_hash_embed_bwd(
                CudaPtr(input_ids.cu_ptr(stream)?),
                CudaPtr(state.grad_bigram_out.cu_ptr(stream)?),
                CudaPtr(grads.bigram_embed.cu_ptr(stream)?),
                self.config.bigram_vocab_size as u32,
                self.config.bigram_dim as u32,
                t as u32,
                runtime_seq_len as u32,
            )?;
        }

        self.kernels.embedding_gather_bwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(state.grad_x_post_embed.cu_ptr(stream)?),
            CudaPtr(grads.tok_emb.cu_ptr(stream)?),
            d as u32,
            t as u32,
        )?;
        finish_stage_event(stream, stage_start, &mut state.stage_timing.tail_ms)?;
        check_cuda_graph_capture_stage(stream, "tail_backward")?;

        Ok(loss)
    }

    pub fn backward(
        &self,
        input_ids: &GpuTensor,
        targets: &GpuTensor,
        buf: &mut GpuActivations,
        grads: &mut GpuGradBuffers,
    ) -> PgResult<f32> {
        let t = input_ids.shape().iter().product::<usize>();
        let stream = self.gemm.stream().clone();
        let mut state = GpuBackwardState::new(&self.config, t, stream)?;
        self.backward_with_state(input_ids, targets, buf, &mut state, grads)
    }
}
