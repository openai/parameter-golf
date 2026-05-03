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
use crate::{AttentionBackend, ExecutionPlan, GptModel, ModelComputePrecision};

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
            let bytes = tensor.to_host_bytes()?;
            Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
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
        let config = plan.run_spec.model.to_model_config();
        let materialize_logits = !gpu_output_ce_no_full_logits_eligible_for_config(
            &config,
            plan.run_spec.model.compute_precision,
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
        let save_x_after_attn = !lean_bf16_direct || !config.parallel_residual;
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
            .map(|save| {
                if save {
                    GpuLayerForwardCache::new_with_options(
                        config,
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
    matches!(
        std::env::var("PG_GPU_BF16_QKV_DX_OUTPUT")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
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
fn gpu_chunked_bf16_output_ce_cache_eligible_for_config(
    _config: &ModelConfig,
    compute_precision: ModelComputePrecision,
) -> bool {
    compute_precision == ModelComputePrecision::Bf16TensorCore
        && !matches!(
            std::env::var("PG_GPU_BF16_OUTPUT_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && !matches!(
            std::env::var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && matches!(
            std::env::var("PG_GPU_CHUNKED_OUTPUT_CE_CACHE")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

#[cfg(feature = "cuda")]
fn gpu_tiled_output_ce_eligible_for_config(
    config: &ModelConfig,
    compute_precision: ModelComputePrecision,
) -> bool {
    compute_precision == ModelComputePrecision::Bf16TensorCore
        && config.vocab_size % output_ce_tile_vocab_for_config(config) == 0
        && !gpu_chunked_bf16_output_ce_cache_eligible_for_config(config, compute_precision)
        && !matches!(
            std::env::var("PG_GPU_BF16_OUTPUT_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && !matches!(
            std::env::var("PG_GPU_BF16_OUTPUT_BACKWARD_GEMM")
                .unwrap_or_else(|_| "1".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
        && matches!(
            std::env::var("PG_GPU_TILED_OUTPUT_CE")
                .unwrap_or_else(|_| "0".to_string())
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
}

#[cfg(feature = "cuda")]
fn gpu_output_ce_no_full_logits_eligible_for_config(
    config: &ModelConfig,
    compute_precision: ModelComputePrecision,
) -> bool {
    gpu_chunked_bf16_output_ce_cache_eligible_for_config(config, compute_precision)
        || gpu_tiled_output_ce_eligible_for_config(config, compute_precision)
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
        Self::new_with_cache(config, tokens, stream, cache, true)
    }

    fn new_with_cache(
        config: &ModelConfig,
        tokens: usize,
        stream: Arc<CudaStream>,
        cache: GpuForwardCache,
        materialize_logits: bool,
    ) -> PgResult<Self> {
        let d = config.model_dim;
        let bigram_dim = config.bigram_dim.max(1);
        let output_tile_vocab = output_ce_tile_vocab_for_config(config);
        let chunked_output_ce = !materialize_logits
            && gpu_chunked_bf16_output_ce_cache_eligible_for_config(
                config,
                ModelComputePrecision::Bf16TensorCore,
            );
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
        let config = plan.run_spec.model.to_model_config();
        let cache = GpuForwardCache::new_for_plan(plan, tokens, stream.clone())?;
        let materialize_logits = !gpu_output_ce_no_full_logits_eligible_for_config(
            &config,
            plan.run_spec.model.compute_precision,
        );
        Self::new_with_cache(&config, tokens, stream, cache, materialize_logits)
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
    /// A matrices are stored row-major as [rank, model_dim].
    pub a: Vec<GpuTensor>,
    /// B matrices are stored row-major as [model_dim, rank].
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
    fn new(
        config: &ModelConfig,
        stream: Arc<CudaStream>,
        rank: usize,
        alpha: f32,
    ) -> PgResult<Self> {
        if rank == 0 || rank > config.model_dim {
            return Err(pg_core::PgError::InvalidOp(format!(
                "LoRA rank must be in 1..={}, got {rank}",
                config.model_dim
            )));
        }
        let d = config.model_dim;
        let scale = alpha / rank as f32;
        let zeros = |shape: &[usize]| GpuTensor::zeros_gpu(stream.clone(), shape, DType::F32);

        let mut a = Vec::with_capacity(config.num_layers);
        let mut b = Vec::with_capacity(config.num_layers);
        let mut grad_a = Vec::with_capacity(config.num_layers);
        let mut grad_b = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let mut a_host = vec![0.0f32; rank * d];
            // Warm-start A deterministically and keep B at zero. This matches the
            // frontier LoRA-TTT convention: first update moves B, later updates
            // can move both factors without perturbing score-before-update logits.
            for r in 0..rank {
                for col in 0..d {
                    let x = (layer as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        ^ (r as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                        ^ (col as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
                    let centered = ((x >> 40) as f32 / 16_777_216.0) - 0.5;
                    a_host[r * d + col] = centered * 0.01;
                }
            }
            a.push(GpuTensor::from_host_data_gpu(
                stream.clone(),
                bytemuck::cast_slice(&a_host),
                &[rank, d],
                DType::F32,
            )?);
            b.push(zeros(&[d, rank])?);
            grad_a.push(zeros(&[rank, d])?);
            grad_b.push(zeros(&[d, rank])?);
        }

        Ok(Self {
            rank,
            alpha,
            scale,
            a,
            b,
            grad_a,
            grad_b,
        })
    }
}

#[cfg(feature = "cuda")]
pub struct GpuModel {
    pub config: ModelConfig,
    pub attention_backend: AttentionBackend,
    pub compute_precision: ModelComputePrecision,
    pub weights: GpuWeights,
    pub gemm: pg_kernels::gemm::GemmEngine,
    pub kernels: pg_kernels::gpu_kernels::GpuKernels,
    pub cuda_cpp_attention: Option<pg_kernels::flash_attn::CudaCppAttention>,
    pub cudnn_frontend_attention: Option<pg_kernels::flash_attn::CudnnFrontendAttention>,
    pub q_lora: Option<GpuQProjectionLora>,
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
        let kernels = pg_kernels::gpu_kernels::GpuKernels::new(ctx.clone(), stream)?;
        let cuda_cpp_attention =
            pg_kernels::flash_attn::CudaCppAttention::new(gemm.stream().clone()).ok();
        let cudnn_frontend_attention =
            pg_kernels::flash_attn::CudnnFrontendAttention::new(gemm.stream().clone()).ok();
        Ok(Self {
            config: cpu.config.clone(),
            attention_backend: plan.run_spec.model.attention_backend,
            compute_precision: plan.run_spec.model.compute_precision,
            weights,
            gemm,
            kernels,
            cuda_cpp_attention,
            cudnn_frontend_attention,
            q_lora: None,
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
            && matches!(
                std::env::var("PG_GPU_FUSED_QKV_PROJ")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
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

    fn use_tiled_output_ce(&self) -> bool {
        self.use_bf16_output_gemm()
            && self.use_bf16_output_backward_gemm()
            && self.config.vocab_size % output_ce_tile_vocab_for_config(&self.config) == 0
            && !self.use_chunked_bf16_output_ce_cache()
            && matches!(
                std::env::var("PG_GPU_TILED_OUTPUT_CE")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_chunked_bf16_output_ce_cache(&self) -> bool {
        self.use_bf16_output_gemm()
            && self.use_bf16_output_backward_gemm()
            && gpu_chunked_bf16_output_ce_cache_eligible_for_config(
                &self.config,
                self.compute_precision,
            )
    }

    fn use_output_ce_no_full_logits(&self) -> bool {
        self.use_tiled_output_ce() || self.use_chunked_bf16_output_ce_cache()
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
            && matches!(
                std::env::var("PG_GPU_BF16_ATTN_BACKWARD_TAIL")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
    }

    fn use_bf16_attention_tail_qkv_pack(&self) -> bool {
        self.use_bf16_attention_backward_tail()
            && self.use_fused_qkv_projection()
            && self.use_bf16_qkv_dx_output()
            && self.q_lora.is_none()
            && self.config.ve_layers.is_empty()
            && matches!(
                std::env::var("PG_GPU_BF16_ATTN_TAIL_QKV_PACK")
                    .unwrap_or_default()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
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

    fn use_fused_parallel_attn_residual_rms_norm(&self) -> bool {
        self.parallel_residual_enabled()
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
            && self.q_lora.is_none()
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
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
    fn linear_backward_input_and_weight_dy_x_bf16_ready(
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
    ) -> PgResult<()> {
        if !self.use_bf16_backward_gemm() {
            return Err(PgError::InvalidOp(
                "bf16-ready backward helper requires BF16 backward GEMMs".into(),
            ));
        }
        let stream = self.gemm.stream();
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
        self.kernels.unpack_qkv_output(
            CudaPtr(buf.qkv_out.cu_ptr(stream)?),
            CudaPtr(buf.q.cu_ptr(stream)?),
            CudaPtr(buf.k.cu_ptr(stream)?),
            CudaPtr(buf.v.cu_ptr(stream)?),
            tokens as u32,
            d as u32,
            kv as u32,
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
    ) -> PgResult<bool> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if !self.use_fused_qkv_projection() {
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
                self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready(
                    &buf.qkv_aux_bf16,
                    attn_norm_bf16,
                    &qkv_w_bf16,
                    &block_cache.grad_attn_norm_bf16,
                    &block_cache.grad_qkv_weight,
                    tokens,
                    d + 2 * kv,
                    d,
                    0.0,
                    0.0,
                )?;
                if self.q_lora.is_some() {
                    self.kernels.bf16_to_f32(
                        CudaPtr(block_cache.grad_attn_norm_bf16.cu_ptr(stream)?),
                        CudaPtr(grad_attn_norm.cu_ptr(stream)?),
                        (tokens * d) as u32,
                    )?;
                }
            } else {
                self.linear_backward_input_and_weight_dy_x_bf16_ready(
                    &buf.qkv_aux_bf16,
                    attn_norm_bf16,
                    &qkv_w_bf16,
                    grad_attn_norm,
                    &block_cache.grad_qkv_weight,
                    tokens,
                    d + 2 * kv,
                    d,
                    0.0,
                    0.0,
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
                0.0,
            )?;
        }
        self.kernels.unpack_qkv_weight_grad(
            CudaPtr(block_cache.grad_qkv_weight.cu_ptr(stream)?),
            CudaPtr(grads.qo_bank.cu_ptr(stream)?),
            CudaPtr(grads.kv_bank.cu_ptr(stream)?),
            layer as u32,
            n as u32,
            d as u32,
            kv as u32,
        )?;
        self.backward_q_lora(layer, grad_q_proj, grad_attn_norm, buf)?;
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
        attn_norm_override: Option<&GpuTensor>,
        attn_norm_bf16_override: Option<&GpuTensor>,
        tokens: usize,
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

        let qkv_w_bf16 = self.weights.qkv_bank_bf16.slice_first(layer)?;
        self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready(
            &buf.qkv_aux_bf16,
            attn_norm_bf16,
            &qkv_w_bf16,
            &block_cache.grad_attn_norm_bf16,
            &block_cache.grad_qkv_weight,
            tokens,
            d + 2 * kv,
            d,
            0.0,
            0.0,
        )?;
        self.kernels.unpack_qkv_weight_grad(
            CudaPtr(block_cache.grad_qkv_weight.cu_ptr(stream)?),
            CudaPtr(grads.qo_bank.cu_ptr(stream)?),
            CudaPtr(grads.kv_bank.cu_ptr(stream)?),
            layer as u32,
            n as u32,
            d as u32,
            kv as u32,
        )?;
        Ok(true)
    }

    pub fn enable_q_lora(&mut self, rank: usize, alpha: f32) -> PgResult<()> {
        let stream = self.gemm.stream().clone();
        self.q_lora = Some(GpuQProjectionLora::new(&self.config, stream, rank, alpha)?);
        Ok(())
    }

    pub fn q_lora_enabled(&self) -> bool {
        self.q_lora.is_some()
    }

    pub fn q_lora_state_to_host(&self) -> PgResult<GpuQProjectionLoraHostState> {
        let lora = self.q_lora.as_ref().ok_or_else(|| {
            pg_core::PgError::InvalidOp("q_lora_state_to_host requires enabled LoRA".into())
        })?;
        let mut a = Vec::with_capacity(lora.a.len());
        let mut b = Vec::with_capacity(lora.b.len());
        for layer in 0..self.config.num_layers {
            a.push(lora.a[layer].to_host_bytes()?);
            b.push(lora.b[layer].to_host_bytes()?);
        }
        Ok(GpuQProjectionLoraHostState {
            rank: lora.rank,
            alpha: lora.alpha,
            a,
            b,
        })
    }

    pub fn copy_q_lora_state_from_host(
        &mut self,
        state: &GpuQProjectionLoraHostState,
    ) -> PgResult<()> {
        let lora = self.q_lora.as_mut().ok_or_else(|| {
            pg_core::PgError::InvalidOp("copy_q_lora_state_from_host requires enabled LoRA".into())
        })?;
        if lora.rank != state.rank || (lora.alpha - state.alpha).abs() > f32::EPSILON {
            return Err(pg_core::PgError::InvalidOp(format!(
                "LoRA state shape mismatch: model rank/alpha={} / {:.6}, state rank/alpha={} / {:.6}",
                lora.rank, lora.alpha, state.rank, state.alpha
            )));
        }
        if state.a.len() != self.config.num_layers || state.b.len() != self.config.num_layers {
            return Err(pg_core::PgError::InvalidOp(format!(
                "LoRA state layer count mismatch: got A={} B={} expected {}",
                state.a.len(),
                state.b.len(),
                self.config.num_layers
            )));
        }
        for layer in 0..self.config.num_layers {
            lora.a[layer].copy_from_host_bytes(&state.a[layer])?;
            lora.b[layer].copy_from_host_bytes(&state.b[layer])?;
        }
        self.zero_q_lora_grads()?;
        Ok(())
    }

    pub fn zero_q_lora_grads(&self) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for layer in 0..self.config.num_layers {
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_a[layer].cu_ptr(stream)?),
                    0.0,
                    lora.grad_a[layer].numel() as u32,
                )?;
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_b[layer].cu_ptr(stream)?),
                    0.0,
                    lora.grad_b[layer].numel() as u32,
                )?;
            }
        }
        Ok(())
    }

    pub fn q_lora_grad_numel(&self) -> PgResult<usize> {
        let lora = self.q_lora.as_ref().ok_or_else(|| {
            pg_core::PgError::InvalidOp("q_lora_grad_numel requires enabled LoRA".into())
        })?;
        let mut total = 0usize;
        for layer in 0..self.config.num_layers {
            total += lora.grad_a[layer].numel();
            total += lora.grad_b[layer].numel();
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
        let lora = self.q_lora.as_ref().ok_or_else(|| {
            pg_core::PgError::InvalidOp("pack_q_lora_grads requires enabled LoRA".into())
        })?;
        let mut offset = 0usize;
        for layer in 0..self.config.num_layers {
            let a_len = lora.grad_a[layer].numel();
            let dst = packed.slice_range(offset, offset + a_len)?;
            self.copy_tensor(&lora.grad_a[layer], &dst)?;
            offset += a_len;

            let b_len = lora.grad_b[layer].numel();
            let dst = packed.slice_range(offset, offset + b_len)?;
            self.copy_tensor(&lora.grad_b[layer], &dst)?;
            offset += b_len;
        }
        Ok(())
    }

    pub fn unpack_q_lora_grads(&self, packed: &GpuTensor) -> PgResult<()> {
        let expected = self.q_lora_grad_numel()?;
        if packed.numel() != expected {
            return Err(pg_core::PgError::ShapeMismatch {
                expected: vec![expected],
                got: packed.shape().to_vec(),
            });
        }
        let lora = self.q_lora.as_ref().ok_or_else(|| {
            pg_core::PgError::InvalidOp("unpack_q_lora_grads requires enabled LoRA".into())
        })?;
        let mut offset = 0usize;
        for layer in 0..self.config.num_layers {
            let a_len = lora.grad_a[layer].numel();
            let src = packed.slice_range(offset, offset + a_len)?;
            self.copy_tensor(&src, &lora.grad_a[layer])?;
            offset += a_len;

            let b_len = lora.grad_b[layer].numel();
            let src = packed.slice_range(offset, offset + b_len)?;
            self.copy_tensor(&src, &lora.grad_b[layer])?;
            offset += b_len;
        }
        Ok(())
    }

    pub fn scale_q_lora_grads(&self, alpha: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for layer in 0..self.config.num_layers {
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_a[layer].cu_ptr(stream)?),
                    alpha,
                    lora.grad_a[layer].numel() as u32,
                )?;
                self.kernels.scale_inplace(
                    CudaPtr(lora.grad_b[layer].cu_ptr(stream)?),
                    alpha,
                    lora.grad_b[layer].numel() as u32,
                )?;
            }
        }
        Ok(())
    }

    pub fn reset_q_lora_b(&self) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for b in &lora.b {
                self.kernels
                    .scale_inplace(CudaPtr(b.cu_ptr(stream)?), 0.0, b.numel() as u32)?;
            }
            self.zero_q_lora_grads()?;
        }
        Ok(())
    }

    pub fn step_q_lora_sgd(&self, lr: f32, weight_decay: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            for layer in 0..self.config.num_layers {
                self.kernels.decay_sgd_step(
                    CudaPtr(lora.a[layer].cu_ptr(stream)?),
                    CudaPtr(lora.grad_a[layer].cu_ptr(stream)?),
                    lr,
                    weight_decay,
                    lora.a[layer].numel() as u32,
                )?;
                self.kernels.decay_sgd_step(
                    CudaPtr(lora.b[layer].cu_ptr(stream)?),
                    CudaPtr(lora.grad_b[layer].cu_ptr(stream)?),
                    lr,
                    weight_decay,
                    lora.b[layer].numel() as u32,
                )?;
            }
            self.zero_q_lora_grads()?;
        }
        Ok(())
    }

    fn ln_scale_factor(&self, layer: usize) -> f32 {
        if self.config.ln_scale {
            1.0 / ((layer + 1) as f32).sqrt()
        } else {
            1.0
        }
    }

    fn parallel_residual_enabled(&self) -> bool {
        self.config.parallel_residual
    }

    fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.config.is_recurrent_layer(layer)
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

        if self.attention_backend == AttentionBackend::CudnnSdpaBf16 {
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

        if self.attention_backend == AttentionBackend::CudnnSdpaBf16 {
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

    fn copy_tensor(&self, src: &GpuTensor, dst: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if src.shape() != dst.shape() {
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

    fn copy_bf16_tensor(&self, src: &GpuTensor, dst: &GpuTensor) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if src.shape() != dst.shape() {
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

    fn add_inplace(&self, dst: &GpuTensor, src: &GpuTensor, alpha: f32) -> PgResult<()> {
        use pg_kernels::gpu_kernels::CudaPtr;

        if dst.shape() != src.shape() {
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

    fn apply_q_lora_forward(&self, layer: usize, buf: &mut GpuActivations) -> PgResult<()> {
        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            let t = buf.attn_norm.shape()[0];
            let d = self.config.model_dim;
            let r = lora.rank;
            unsafe {
                self.gemm.matmul_f32(
                    buf.attn_norm.cu_ptr(stream)?,
                    lora.a[layer].cu_ptr(stream)?,
                    buf.lora_tmp.cu_ptr(stream)?,
                    t,
                    r,
                    d,
                    1.0,
                    0.0,
                )?;
                self.gemm.matmul_f32(
                    buf.lora_tmp.cu_ptr(stream)?,
                    lora.b[layer].cu_ptr(stream)?,
                    buf.lora_delta.cu_ptr(stream)?,
                    t,
                    d,
                    r,
                    lora.scale,
                    0.0,
                )?;
            }
            let lora_delta_q =
                buf.lora_delta
                    .reshape(&[t, self.config.num_heads, self.config.head_dim])?;
            self.add_inplace(&buf.q, &lora_delta_q, 1.0)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn linear_backward_input_bf16_and_weight_dy_x_bf16_ready(
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
    ) -> PgResult<()> {
        if !self.use_bf16_backward_gemm() {
            return Err(PgError::InvalidOp(
                "bf16-output backward helper requires BF16 backward GEMMs".into(),
            ));
        }
        let stream = self.gemm.stream();
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

    fn backward_q_lora(
        &self,
        layer: usize,
        grad_q_proj: &GpuTensor,
        grad_attn_norm: &GpuTensor,
        buf: &mut GpuActivations,
    ) -> PgResult<()> {
        if let Some(lora) = &self.q_lora {
            let stream = self.gemm.stream();
            let t = buf.attn_norm.shape()[0];
            let d = self.config.model_dim;
            let r = lora.rank;
            unsafe {
                self.gemm.linear_backward_weight_f32(
                    grad_q_proj.cu_ptr(stream)?,
                    buf.lora_tmp.cu_ptr(stream)?,
                    lora.grad_b[layer].cu_ptr(stream)?,
                    t,
                    d,
                    r,
                    lora.scale,
                    1.0,
                )?;
                self.gemm.linear_backward_input_f32(
                    grad_q_proj.cu_ptr(stream)?,
                    lora.b[layer].cu_ptr(stream)?,
                    buf.lora_grad_tmp.cu_ptr(stream)?,
                    t,
                    d,
                    r,
                    lora.scale,
                    0.0,
                )?;
                self.gemm.linear_backward_weight_f32(
                    buf.lora_grad_tmp.cu_ptr(stream)?,
                    buf.attn_norm.cu_ptr(stream)?,
                    lora.grad_a[layer].cu_ptr(stream)?,
                    t,
                    r,
                    d,
                    1.0,
                    1.0,
                )?;
                self.gemm.linear_backward_input_f32(
                    buf.lora_grad_tmp.cu_ptr(stream)?,
                    lora.a[layer].cu_ptr(stream)?,
                    buf.lora_delta.cu_ptr(stream)?,
                    t,
                    r,
                    d,
                    1.0,
                    0.0,
                )?;
            }
            self.add_inplace(grad_attn_norm, &buf.lora_delta, 1.0)?;
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
        let bytes = loss_sum.to_host_bytes()?;
        let values = bytemuck::cast_slice::<u8, f32>(&bytes);
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
                self.config.logit_softcap,
                tokens as u32,
            )
        } else {
            self.kernels.cross_entropy_fwd(
                CudaPtr(logits.cu_ptr(stream)?),
                CudaPtr(targets.cu_ptr(stream)?),
                CudaPtr(losses.cu_ptr(stream)?),
                self.config.vocab_size as u32,
                self.config.logit_softcap,
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

        if self.qkv_projection_forward(layer, buf, t, false)? {
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
        self.apply_q_lora_forward(layer, buf)?;
        self.copy_tensor(&buf.q, &cache.q_pre_norm)?;
        self.copy_tensor(&buf.k, &cache.k_pre_norm)?;

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
        if self.parallel_residual_enabled() {
            self.copy_tensor(&buf.x_in, &cache.x_after_attn)?;
        } else {
            self.copy_tensor(&buf.x, &cache.x_after_attn)?;
        }

        self.kernels.rms_norm_forward(
            if self.parallel_residual_enabled() {
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
            && self.q_lora.is_none();
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
        let grad_mlp_out_bf16 = &buf.x_aux_bf16;
        let mlp_residual_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if !gpu_residual_scale_reduce_enabled() {
            self.zero_tensor(grad_x_after_attn)?;
        }
        if let Some(saved) = saved_bf16_direct
            .filter(|saved| saved.lean_bf16_direct && self.use_bf16_residual_projection_output())
        {
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
        } else if self.use_bf16_backward_gemm() {
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
        let mlp_down_w = self.weights.mlp_down_bank.slice_first(layer)?;
        let mlp_down_w_bf16 = self.weights.mlp_down_bank_bf16.slice_first(layer)?;
        let mlp_down_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.use_bf16_backward_gemm() {
            self.linear_backward_input_and_weight_dy_x_bf16_ready(
                grad_mlp_out_bf16,
                act_bf16!(mlp_act_bf16, &buf.wide_bf16),
                &mlp_down_w_bf16,
                grad_mlp_act,
                &grads.mlp_down_bank.slice_first(layer)?,
                t,
                d,
                mlp,
                0.0,
                1.0,
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
                1.0,
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
            if let Some(saved) = saved_bf16_direct.filter(|saved| saved.lean_bf16_direct) {
                self.kernels.leaky_relu_sq_backward_x_bf16_only(
                    CudaPtr(saved.mlp_up_bf16.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_act.cu_ptr(stream)?),
                    CudaPtr(grad_mlp_up_bf16.cu_ptr(stream)?),
                    (t * mlp) as u32,
                )?;
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
        let grad_mlp_norm_bf16 = &buf.x_aux_bf16;
        let mlp_up_w = self.weights.mlp_up_bank.slice_first(layer)?;
        let mlp_up_w_bf16 = self.weights.mlp_up_bank_bf16.slice_first(layer)?;
        let mlp_up_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.use_bf16_norm_grad_path() {
            self.linear_backward_input_bf16_and_weight_dy_x_bf16_ready(
                grad_mlp_up_bf16,
                act_bf16!(mlp_norm_bf16, &buf.x_in_bf16),
                &mlp_up_w_bf16,
                grad_mlp_norm_bf16,
                &grads.mlp_up_bank.slice_first(layer)?,
                t,
                mlp,
                d,
                0.0,
                1.0,
            )?;
        } else if self.use_bf16_backward_gemm() {
            self.linear_backward_input_and_weight_dy_x_bf16_ready(
                grad_mlp_up_bf16,
                act_bf16!(mlp_norm_bf16, &buf.x_in_bf16),
                &mlp_up_w_bf16,
                grad_mlp_norm,
                &grads.mlp_up_bank.slice_first(layer)?,
                t,
                mlp,
                d,
                0.0,
                1.0,
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
                1.0,
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
        let mlp_norm_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if self.parallel_residual_enabled()
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
        } else if self.parallel_residual_enabled() && recompute_residual_mix_norm_inputs {
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
            let mlp_norm_input = if self.parallel_residual_enabled() {
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
            let mlp_norm_input = if self.parallel_residual_enabled() {
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
        if !self.parallel_residual_enabled() {
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

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let grad_x_in = &block_cache.grad_x_in;
        let grad_proj_out = &block_cache.grad_proj_out;
        let grad_proj_out_bf16 = &buf.x_aux_bf16;
        let attn_out_residual_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if !gpu_residual_scale_reduce_enabled() {
            self.zero_tensor(grad_x_in)?;
        }
        if let Some(saved) = saved_bf16_direct
            .filter(|saved| saved.lean_bf16_direct && self.use_bf16_attention_projection_output())
        {
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
        } else if self.use_bf16_backward_gemm() {
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
        if self.parallel_residual_enabled() {
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
            self.linear_backward_input_and_weight_dy_x_bf16_ready(
                grad_proj_out_bf16,
                act_bf16!(attn_weight_input_bf16, &buf.x_in_bf16),
                &o_w_bf16,
                grad_attn_result,
                &grads.qo_bank.slice_first(n + layer)?,
                t,
                d,
                d,
                0.0,
                1.0,
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
                1.0,
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
            self.kernels.scale_inplace(
                CudaPtr(buf.attn_gate_grad_input.cu_ptr(stream)?),
                0.0,
                (t * d) as u32,
            )?;
            if is_xsa_layer && self.use_skip_f32_attention_saved_acts() {
                if let Some((_, _, saved_v_bhsd, saved_attn_out_bhsd)) = saved_attention_bf16 {
                    if t % runtime_seq_len != 0 {
                        return Err(PgError::InvalidOp(format!(
                            "saved BF16 SparseAttnGate+XSA backward requires tokens ({t}) to be divisible by seq_len ({runtime_seq_len})"
                        )));
                    }
                    self.zero_tensor(grad_v_xsa)?;
                    if gpu_sparse_xsa_warphead_backward_enabled() {
                        self.kernels
                            .sparse_attn_gate_xsa_bwd_bf16_bhsd_warpheads_two_pass(
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
                    } else {
                        self.kernels.sparse_attn_gate_xsa_bwd_bf16_bhsd_two_pass(
                            CudaPtr(saved_attn_out_bhsd),
                            CudaPtr(saved_v_bhsd),
                            CudaPtr(act!(attn_norm).cu_ptr(stream)?),
                            CudaPtr(act!(attn_gate_values).cu_ptr(stream)?),
                            CudaPtr(grad_attn_result.cu_ptr(stream)?),
                            CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
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
        let add_v_xsa = layer >= n.saturating_sub(self.config.xsa_last_n);
        let attention_sdpa_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if use_bf16_attention_backward_tail {
            let (q_bf16, k_bf16, v_bf16, out_bf16) =
                saved_attention_bf16.expect("checked by use_bf16_attention_backward_tail");
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

        let substage_start = record_stage_event_if(stream, stage_timing.is_some())?;

        let grad_q_proj = &block_cache.grad_q_proj;
        let grad_k_proj = &block_cache.grad_k_proj;
        let grad_q_proj_bf16 = &block_cache.grad_q_proj_bf16;
        let grad_k_proj_bf16 = &block_cache.grad_k_proj_bf16;
        let qkv_rope_start = record_stage_event_if(stream, stage_timing.is_some())?;
        if use_bf16_attention_tail_qkv_pack {
            let q_gain_chunks = t.div_ceil(256).max(1);
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
                    256,
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
            let q_gain_chunks = t.div_ceil(256).max(1);
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
                256,
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
                let q_gain_chunks = t.div_ceil(256).max(1);
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
                    256,
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
                saved_direct.map(|saved| &saved.attn_norm),
                saved_bf16_direct.map(|saved| &saved.attn_norm_bf16),
                t,
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
            self.linear_backward_input_and_weight_x_bf16_ready(
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
            )?;

            let k_w = self.weights.kv_bank.slice_first(layer)?;
            let k_w_bf16 = self.weights.kv_bank_bf16.slice_first(layer)?;
            if self.use_qkv_dx_beta_accum() {
                self.linear_backward_input_and_weight_x_bf16_ready_with_dx_beta(
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
                )?;
            } else {
                let grad_attn_norm_k = &block_cache.grad_attn_norm_k;
                self.linear_backward_input_and_weight_x_bf16_ready(
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
                self.linear_backward_input_and_weight_x_bf16_ready_with_dx_beta(
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
                )?;
            } else {
                let grad_attn_norm_v = &block_cache.grad_attn_norm_v;
                self.linear_backward_input_and_weight_x_bf16_ready(
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
                )?;
                self.add_inplace(&grad_attn_norm, grad_attn_norm_v, 1.0)?;
            }
            self.backward_q_lora(layer, &grad_q_proj, &grad_attn_norm, buf)?;
        }
        finish_stage_event_optional(
            stream,
            qkv_proj_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.backward_block_qkv_proj_ms),
        )?;
        let qkv_norm_resid_start = record_stage_event_if(stream, stage_timing.is_some())?;
        let qkv_dx_bf16_for_norm_tail =
            qkv_fused && self.use_bf16_qkv_dx_output() && self.q_lora.is_none();
        let gate_extra_grad =
            self.config.attn_out_gate_enabled || self.config.sparse_attn_gate_enabled;
        if gate_extra_grad && !qkv_dx_bf16_for_norm_tail {
            self.add_inplace(&grad_attn_norm, &buf.attn_gate_grad_input, 1.0)?;
        }

        if qkv_dx_bf16_for_norm_tail
            && !recompute_residual_mix_norm_inputs
            && !gpu_split_residual_mix_grad_enabled()
            && !gpu_chunked_residual_mix_backward_enabled()
        {
            if gate_extra_grad {
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
        if self.is_recurrent_layer(layer) {
            if let (Some(pass2_saved), Some(pass1_saved), Some(mid_x)) =
                (saved, recurrent_pass1_saved, recurrent_mid_x)
            {
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
                    stage_timing.as_deref_mut(),
                )?;
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
                stage_timing.as_deref_mut(),
            )?;
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
            !(lean_bf16_direct_saved && gpu_recompute_residual_mix_norm_inputs_enabled())
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

        if self.qkv_projection_forward(layer, buf, t, attn_norm_bf16_ready)? {
            // Hot path handled by a single packed QKV GEMM.
            if let Some(save) = save {
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
            if let Some(save) = save {
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
        self.apply_q_lora_forward(layer, buf)?;
        if let Some(save) = save {
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
            if let Some(save) = save {
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
        if let Some(save) = save.filter(|_| !skip_f32_attention_saved) {
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
        let is_xsa_layer = layer >= n.saturating_sub(self.config.xsa_last_n);
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
                self.kernels.sparse_attn_gate_xsa_fwd_bf16_bhsd(
                    CudaPtr(save.attn_out_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(save.v_bhsd_bf16.cu_ptr(stream)?),
                    CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                    CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                    CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
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
                self.kernels.sparse_attn_gate_xsa_fwd(
                    attn_out,
                    v,
                    CudaPtr(buf.attn_norm.cu_ptr(stream)?),
                    CudaPtr(self.weights.sparse_attn_gate_weights[layer].cu_ptr(stream)?),
                    CudaPtr(buf.attn_gated.cu_ptr(stream)?),
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
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
            if let Some(save) = save {
                self.copy_tensor(&buf.attn_gated, &save.attn_gated)?;
                self.copy_tensor(&buf.attn_gate_values, &save.attn_gate_values)?;
            }
            fused_sparse_xsa_forward = true;
            &buf.attn_gated
        } else if is_xsa_layer {
            self.kernels.xsa_fwd(
                attn_out, v, xsa_out, t as u32, h as u32, hkv as u32, hd as u32,
            )?;
            if let Some(save) = save.filter(|_| !lean_bf16_direct_saved) {
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
            if let Some(save) = save {
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
            if let Some(save) = save {
                self.copy_tensor(&buf.attn_gated, &save.attn_gated)?;
                self.copy_tensor(&buf.attn_gate_values, &save.attn_gate_values)?;
            }
            &buf.attn_gated
        } else {
            attn_src_tensor
        };

        let o_w = self.weights.qo_bank.slice_first(n + layer)?;
        let o_w_bf16 = self.weights.qo_bank_bf16.slice_first(n + layer)?;
        let bf16_attention_projection_output =
            self.use_bf16_attention_projection_output() && lean_bf16_direct_saved && save.is_some();
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
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                    (t / runtime_seq_len) as u32,
                    runtime_seq_len as u32,
                    h as u32,
                    hd as u32,
                )?;
                self.linear_forward_bf16_output_ready(
                    &buf.x_in_bf16,
                    &o_w_bf16,
                    &save.proj_out_bf16,
                    t,
                    d,
                    d,
                )?;
            } else if fused_sparse_xsa_forward {
                self.linear_forward_bf16_output_ready(
                    &buf.x_in_bf16,
                    &o_w_bf16,
                    &save.proj_out_bf16,
                    t,
                    d,
                    d,
                )?;
            } else {
                self.kernels.f32_to_bf16(
                    CudaPtr(attn_src_tensor.cu_ptr(stream)?),
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
                self.linear_forward_bf16_output_ready(
                    &buf.x_in_bf16,
                    &o_w_bf16,
                    &save.proj_out_bf16,
                    t,
                    d,
                    d,
                )?;
            }
            self.copy_bf16_tensor(&buf.x_in_bf16, &save.attn_weight_input_bf16)?;
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
        if let Some(save) = save.filter(|_| !bf16_attention_projection_output) {
            self.copy_tensor(&buf.proj_out, &save.proj_out)?;
        }

        let fused_parallel_attn_resid_norm = self.use_fused_parallel_attn_residual_rms_norm();
        let mut mlp_norm_bf16_ready = false;
        if fused_parallel_attn_resid_norm && bf16_attention_projection_output {
            let save = save.expect("checked save exists for bf16 attention projection output");
            self.kernels
                .residual_add_scale_from_base_rms_norm_fwd_bf16_proj(
                    x_in,
                    CudaPtr(save.proj_out_bf16.cu_ptr(stream)?),
                    CudaPtr(self.weights.attn_scales[layer].cu_ptr(stream)?),
                    x,
                    mlp_norm,
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
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
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
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
        if let Some(save) = save {
            if self.parallel_residual_enabled() && lean_bf16_direct_saved {
                // In parallel-residual mode the MLP norm input is exactly
                // x_in. The lean BF16 direct-saved path can reuse saved.x_in
                // during backward instead of storing a duplicate F32
                // x_after_attn activation.
            } else if self.parallel_residual_enabled() {
                self.copy_tensor(&buf.x_in, &save.x_after_attn)?;
            } else {
                self.copy_tensor(&buf.x, &save.x_after_attn)?;
            }
        }

        if !fused_parallel_attn_resid_norm {
            let mlp_norm_input = if self.parallel_residual_enabled() {
                x_in
            } else {
                x
            };
            if self.use_bf16_norm_side_outputs() {
                self.kernels.rms_norm_forward_bf16(
                    mlp_norm_input,
                    mlp_norm,
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
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
        if let Some(save) = save.filter(|_| !lean_bf16_direct_saved) {
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
                    CudaPtr(buf.x_in_bf16.cu_ptr(stream)?),
                    (t * d) as u32,
                )?;
            }
            self.copy_bf16_tensor(&buf.x_in_bf16, &save.mlp_norm_bf16)?;
            self.linear_forward_bf16_output_ready(
                &buf.x_in_bf16,
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
                    &buf.x_in_bf16,
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
                    &buf.x_in_bf16,
                    &up_w,
                    &up_w_bf16,
                    &buf.mlp_up,
                    t,
                    mlp,
                    d,
                )?;
            }
            if let Some(save) = save {
                self.copy_bf16_tensor(&buf.x_in_bf16, &save.mlp_norm_bf16)?;
            }
        } else {
            self.linear_forward_f32(&buf.mlp_norm, &up_w, &buf.mlp_up, t, mlp, d)?;
        }
        if let Some(save) = save.filter(|_| !lean_bf16_direct_saved) {
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
        if let Some(save) = save.filter(|_| lean_bf16_direct_saved && !bf16_mlp_up_output) {
            self.kernels.f32_to_bf16(
                CudaPtr(buf.mlp_up.cu_ptr(stream)?),
                CudaPtr(save.mlp_up_bf16.cu_ptr(stream)?),
                (t * mlp) as u32,
            )?;
        }
        if let Some(save) = save.filter(|_| !lean_bf16_direct_saved) {
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
        if let Some(save) = save.filter(|_| !bf16_residual_projection_output) {
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
                let pass1_saved = cache
                    .recurrent_pass1_layers
                    .get(layer)
                    .and_then(|saved| saved.as_ref())
                    .ok_or_else(|| {
                        PgError::InvalidOp(format!(
                            "missing recurrent pass-1 saved activations for layer {layer}"
                        ))
                    })?;
                let mid_x = cache
                    .recurrent_mid_x
                    .get(layer)
                    .and_then(|mid| mid.as_ref())
                    .ok_or_else(|| {
                        PgError::InvalidOp(format!(
                            "missing recurrent mid activation boundary for layer {layer}"
                        ))
                    })?;
                self.block_forward_once(
                    layer,
                    input_ids,
                    buf,
                    runtime_seq_len,
                    forward_generation,
                    Some(pass1_saved),
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

        self.kernels.embedding_gather_fwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            x,
            d as u32,
            t as u32,
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
        if materialize_output_logits {
            let required_logits = t * self.config.vocab_size;
            if buf.logits.numel() < required_logits {
                return Err(PgError::ShapeMismatch {
                    expected: vec![t, self.config.vocab_size],
                    got: buf.logits.shape().to_vec(),
                });
            }
            self.output_projection_forward(&buf.x_in, &buf.x_in_bf16, &buf.logits, t)?;
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

        let x = CudaPtr(buf.x.cu_ptr(stream)?);
        let x_in = CudaPtr(buf.x_in.cu_ptr(stream)?);
        let x0 = CudaPtr(buf.x0.cu_ptr(stream)?);

        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        self.kernels.embedding_gather_fwd(
            CudaPtr(input_ids.cu_ptr(stream)?),
            CudaPtr(self.weights.tok_emb.cu_ptr(stream)?),
            x,
            d as u32,
            t as u32,
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
        self.copy_tensor(&buf.x, &cache.x0)?;
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_embed_ms),
        )?;

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();
        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        for layer in 0..n_enc {
            let layer_start = record_stage_event_if(stream, time_forward_substages)?;
            self.copy_tensor(&buf.x, &cache.layer_x[layer])?;
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
            self.copy_tensor(&buf.x, &cache.skips[layer])?;
            finish_stage_event_optional_max(
                stream,
                layer_start,
                stage_timing
                    .as_deref_mut()
                    .map(|timing| &mut timing.forward_encoder_layer_max_ms),
            )?;
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
        }
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_decoder_ms),
        )?;

        let stage_start = record_stage_event_if(stream, time_forward_substages)?;
        self.copy_tensor(&buf.x, &cache.x_final)?;

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
        if !self.use_output_ce_no_full_logits() {
            self.output_projection_forward(&buf.x_in, &buf.x_in_bf16, &buf.logits, t)?;
        }
        finish_stage_event_optional(
            stream,
            stage_start,
            stage_timing
                .as_deref_mut()
                .map(|timing| &mut timing.forward_logits_ms),
        )?;
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
        cache: &GpuForwardCache,
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
                "PG_GPU_TILED_OUTPUT_CE requires compute_loss=true because output projection is fused into loss/backward".into(),
            ));
        }
        if self.use_chunked_bf16_output_ce_cache() && losses.is_none() {
            return Err(PgError::InvalidOp(
                "PG_GPU_CHUNKED_OUTPUT_CE_CACHE requires compute_loss=true because output projection is fused into loss/backward".into(),
            ));
        }

        if self.use_chunked_bf16_output_ce_cache() {
            let losses = losses.ok_or_else(|| {
                PgError::InvalidOp("PG_GPU_CHUNKED_OUTPUT_CE_CACHE requires loss storage".into())
            })?;
            let tiled = tiled_output.ok_or_else(|| {
                PgError::InvalidOp("PG_GPU_CHUNKED_OUTPUT_CE_CACHE requires chunk scratch".into())
            })?;
            self.backward_output_chunked_cached_ce_into(
                buf, targets, losses, grads, grad_x, tiled,
            )?;
            self.kernels.rms_norm_backward(
                CudaPtr(cache.x_final.cu_ptr(stream)?),
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
                CudaPtr(cache.x_final.cu_ptr(stream)?),
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
                        self.config.logit_softcap,
                        1.0 / t as f32,
                        t as u32,
                    )?;
                } else {
                    self.kernels.cross_entropy_bwd_bf16_logits(
                        CudaPtr(buf.logits.cu_ptr(stream)?),
                        CudaPtr(targets.cu_ptr(stream)?),
                        CudaPtr(grad_logits_bf16.cu_ptr(stream)?),
                        vocab as u32,
                        self.config.logit_softcap,
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
                        self.config.logit_softcap,
                        1.0 / t as f32,
                        t as u32,
                    )?;
                } else {
                    self.kernels.cross_entropy_bwd_bf16(
                        CudaPtr(buf.logits.cu_ptr(stream)?),
                        CudaPtr(targets.cu_ptr(stream)?),
                        CudaPtr(grad_logits_bf16.cu_ptr(stream)?),
                        vocab as u32,
                        self.config.logit_softcap,
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
                self.config.logit_softcap,
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
            CudaPtr(cache.x_final.cu_ptr(stream)?),
            CudaPtr(grad_x.cu_ptr(stream)?),
            CudaPtr(grad_pre_norm.cu_ptr(stream)?),
            t as u32,
            d as u32,
            1.0,
            1e-6,
        )?;

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
                self.config.logit_softcap,
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
                self.config.logit_softcap,
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
                self.config.logit_softcap,
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
                self.config.logit_softcap,
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

    pub fn cross_entropy_losses_with_state(
        &self,
        buf: &mut GpuActivations,
        state: &mut GpuBackwardState,
        targets: &GpuTensor,
        losses: &GpuTensor,
        tokens: usize,
    ) -> PgResult<()> {
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
        finish_stage_event(stream, stage_start, &mut state.stage_timing.forward_ms)?;

        let stage_start = record_stage_event(stream)?;
        let tiled_output_loss =
            self.use_tiled_output_ce() || self.use_chunked_bf16_output_ce_cache();
        let fused_output_loss = compute_loss && self.use_fused_ce_loss_bwd();
        let output_losses = if tiled_output_loss || fused_output_loss {
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

        self.backward_output_loss_only_into(
            &state.cache,
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
        for skip_grad in &state.grad_encoder_skips {
            self.zero_tensor(skip_grad)?;
        }
        finish_stage_event(stream, stage_start, &mut state.stage_timing.output_ms)?;

        let mut grad_x = state.grad_ping.clone();
        let mut next_grad_x = state.grad_pong.clone();

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
                &state.cache.x0,
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
            std::mem::swap(&mut grad_x, &mut next_grad_x);

            if i < n_skip {
                let enc_layer = n_enc - 1 - i;
                self.zero_tensor(&state.grad_x_post_skip)?;
                self.kernels.residual_add_scale_bwd(
                    CudaPtr(state.cache.skips[enc_layer].cu_ptr(stream)?),
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
            }
        }
        finish_stage_event(stream, stage_start, &mut state.stage_timing.decoder_ms)?;

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
            self.block_backward_into(
                i,
                input_ids,
                &state.cache.layer_x[i],
                &state.cache.x0,
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
            std::mem::swap(&mut grad_x, &mut next_grad_x);
        }
        finish_stage_event(stream, stage_start, &mut state.stage_timing.encoder_ms)?;

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
