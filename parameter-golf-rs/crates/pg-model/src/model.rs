/// GPT model with parameter banks and U-Net skip connections.
///
/// Architecture (from SOTA 1.1194 BPB):
///   tok_emb + bigram → RMSNorm → SmearGate
///   → U-Net (5 encoder + 6 decoder with skip connections)
///   → final_norm → tied linear → softcap(30)
///
/// Weights are organized into 4 contiguous 3D "parameter banks":
///   - qo_bank [22, 512, 512]: Q projections [0..11], O projections [11..22]
///   - kv_bank [22, 256, 512]: K projections [0..11], V projections [11..22]
///   - mlp_up_bank [11, 1536, 512]: MLP gate/up projections
///   - mlp_down_bank [11, 512, 1536]: MLP down projections
use crate::config::ModelConfig;
use crate::plan::ExecutionPlan;
use pg_core::PgResult;

/// Per-block learnable parameters (non-banked).
pub struct BlockParams {
    pub attn_scale: Vec<f32>,              // [model_dim]
    pub mlp_scale: Vec<f32>,               // [model_dim]
    pub resid_mix: Vec<f32>,               // [2, model_dim] — resid_mix[0] and resid_mix[1]
    pub q_gain: Vec<f32>,                  // [num_heads]
    pub attn_gate_weight: Vec<f32>,        // [num_heads, attn_out_gate_width]
    pub attn_gate_bias: Vec<f32>,          // [num_heads]
    pub sparse_attn_gate_weight: Vec<f32>, // [num_heads, sparse_attn_gate_width]
    pub ln_scale_factor: f32,              // 1/sqrt(layer_idx + 1) if ln_scale
    pub use_xsa: bool,
}

/// The complete GPT model state on CPU.
/// All parameter storage is explicit flat buffers.
pub struct GptModel {
    pub config: ModelConfig,

    // Token embedding [vocab_size, model_dim] — tied with output projection
    pub tok_emb: Vec<f32>,

    // BigramHash: embed [bigram_vocab_size, bigram_dim], proj [model_dim, bigram_dim], scale scalar
    pub bigram_embed: Vec<f32>,
    pub bigram_proj: Vec<f32>,
    pub bigram_scale: f32,

    // SmearGate: [model_dim]
    pub smear_gate: Vec<f32>,

    // U-Net skip weights: [num_skip_weights, model_dim]
    pub skip_weights: Vec<f32>,

    // Parameter banks (contiguous 3D)
    pub qo_bank: Vec<f32>,       // [2*num_layers, model_dim, model_dim]
    pub kv_bank: Vec<f32>,       // [2*num_layers, kv_dim, model_dim]
    pub mlp_up_bank: Vec<f32>,   // [num_layers, mlp_dim, model_dim]
    pub mlp_down_bank: Vec<f32>, // [num_layers, model_dim, mlp_dim]

    // Per-block parameters
    pub blocks: Vec<BlockParams>,

    // Value Embedding (shared): embed [vocab_size, ve_dim], proj [kv_dim, ve_dim]
    pub ve_embed: Vec<f32>,
    pub ve_proj: Vec<f32>,
    pub ve_scale: f32,
    pub ve_layer_scales: Vec<f32>, // one per VE layer

    // Precomputed RoPE tables
    pub rope_cos: Vec<f32>, // [seq_len, rope_dims/2]
    pub rope_sin: Vec<f32>, // [seq_len, rope_dims/2]
}

/// Intermediate activations for a single forward pass.
/// Pre-allocated to avoid per-step allocation.
pub struct ForwardBuffer {
    // Dimensions
    pub tokens: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub kv_dim: usize,
    pub mlp_dim: usize,

    // Scratch buffers
    pub x: Vec<f32>,               // [tokens, model_dim] — main hidden state
    pub x0: Vec<f32>,              // [tokens, model_dim] — residual stream anchor
    pub x_in: Vec<f32>,            // [tokens, model_dim]
    pub attn_norm_out: Vec<f32>,   // [tokens, model_dim]
    pub mlp_norm_out: Vec<f32>,    // [tokens, model_dim]
    pub q: Vec<f32>,               // [tokens, num_heads, head_dim]
    pub k: Vec<f32>,               // [tokens, num_kv_heads, head_dim]
    pub v: Vec<f32>,               // [tokens, num_kv_heads, head_dim]
    pub attn_out: Vec<f32>,        // [tokens, num_heads, head_dim]
    pub xsa_out: Vec<f32>,         // [tokens, num_heads, head_dim]
    pub attn_gated: Vec<f32>,      // [tokens, num_heads, head_dim]
    pub attn_gate: Vec<f32>,       // [tokens, num_heads]
    pub proj_out: Vec<f32>,        // [tokens, model_dim]
    pub mlp_up: Vec<f32>,          // [tokens, mlp_dim]
    pub mlp_act: Vec<f32>,         // [tokens, mlp_dim] (after activation)
    pub mlp_out: Vec<f32>,         // [tokens, model_dim]
    pub bigram_out: Vec<f32>,      // [tokens, bigram_dim]
    pub bigram_proj_out: Vec<f32>, // [tokens, model_dim]
    pub ve_out: Vec<f32>,          // [tokens, kv_dim]
    pub ve_embed_out: Vec<f32>,    // [tokens, ve_dim]
    pub logits: Vec<f32>,          // [tokens, vocab_size]

    // Skip connections stack (encoder outputs)
    pub skips: Vec<Vec<f32>>,

    // v0 cache for VRL
    pub v0: Option<Vec<f32>>, // [tokens, num_kv_heads, head_dim]

    // VE cache (shared computation)
    pub ve_cache: Option<Vec<f32>>, // [tokens, kv_dim]
}

impl ForwardBuffer {
    /// Resize tokens without reallocation. Panics if tokens > max capacity.
    /// All buffers were pre-allocated to max size, so only the token count changes.
    pub fn resize_tokens(&mut self, tokens: usize) {
        assert!(
            tokens * self.model_dim <= self.x.len(),
            "resize_tokens({}) exceeds buffer capacity (max {})",
            tokens,
            self.x.len() / self.model_dim,
        );
        self.tokens = tokens;
    }

    pub fn new(config: &ModelConfig, tokens: usize) -> Self {
        let d = config.model_dim;
        let h = config.num_heads;
        let hkv = config.num_kv_heads;
        let hd = config.head_dim;
        let kv_dim = config.kv_dim();
        let mlp = config.mlp_dim;

        Self {
            tokens,
            model_dim: d,
            num_heads: h,
            num_kv_heads: hkv,
            head_dim: hd,
            kv_dim,
            mlp_dim: mlp,

            x: vec![0.0; tokens * d],
            x0: vec![0.0; tokens * d],
            x_in: vec![0.0; tokens * d],
            attn_norm_out: vec![0.0; tokens * d],
            mlp_norm_out: vec![0.0; tokens * d],
            q: vec![0.0; tokens * h * hd],
            k: vec![0.0; tokens * hkv * hd],
            v: vec![0.0; tokens * hkv * hd],
            attn_out: vec![0.0; tokens * h * hd],
            xsa_out: vec![0.0; tokens * h * hd],
            attn_gated: vec![0.0; tokens * h * hd],
            attn_gate: vec![0.0; tokens * h],
            proj_out: vec![0.0; tokens * d],
            mlp_up: vec![0.0; tokens * mlp],
            mlp_act: vec![0.0; tokens * mlp],
            mlp_out: vec![0.0; tokens * d],
            bigram_out: vec![0.0; tokens * config.bigram_dim],
            bigram_proj_out: vec![0.0; tokens * d],
            ve_out: vec![0.0; tokens * kv_dim],
            ve_embed_out: vec![0.0; tokens * config.ve_dim],
            logits: vec![0.0; tokens * config.vocab_size],

            skips: Vec::new(),
            v0: None,
            ve_cache: None,
        }
    }
}

impl GptModel {
    pub(crate) fn attn_out_gate_enabled(&self) -> bool {
        self.config.attn_out_gate_enabled
    }

    pub(crate) fn sparse_attn_gate_enabled(&self) -> bool {
        self.config.sparse_attn_gate_enabled
    }

    pub(crate) fn parallel_residual_enabled(&self) -> bool {
        self.config.parallel_residual
    }

    pub(crate) fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.config.is_recurrent_layer(layer)
    }

    pub fn new(config: ModelConfig) -> Self {
        let n = config.num_layers;
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;

        let (rope_cos, rope_sin) = pg_kernels::rope::precompute_rope_tables(
            config.train_seq_len,
            config.rope_dims,
            config.rope_base,
        );

        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let ln_scale_factor = if config.ln_scale {
                1.0 / ((i + 1) as f32).sqrt()
            } else {
                1.0
            };
            let use_xsa = config.xsa_last_n > 0 && i >= n.saturating_sub(config.xsa_last_n);
            blocks.push(BlockParams {
                attn_scale: vec![1.0; d],
                mlp_scale: vec![1.0; d],
                resid_mix: {
                    let mut rm = vec![0.0; 2 * d];
                    // resid_mix[0] = ones, resid_mix[1] = zeros
                    for j in 0..d {
                        rm[j] = 1.0;
                    }
                    rm
                },
                q_gain: vec![config.qk_gain_init; config.num_heads],
                attn_gate_weight: vec![0.0; config.num_heads * config.attn_out_gate_width.max(1)],
                attn_gate_bias: vec![0.0; config.num_heads],
                sparse_attn_gate_weight: vec![
                    0.0;
                    config.num_heads
                        * config.sparse_attn_gate_width.max(1)
                ],
                ln_scale_factor,
                use_xsa,
            });
        }

        let num_ve_layers = config.ve_layers.len();

        Self {
            tok_emb: vec![0.0; config.vocab_size * d],
            bigram_embed: vec![0.0; config.bigram_vocab_size * config.bigram_dim],
            bigram_proj: vec![0.0; d * config.bigram_dim],
            bigram_scale: 0.05,
            smear_gate: vec![0.0; d],
            skip_weights: vec![1.0; config.num_skip_weights() * d],
            qo_bank: vec![0.0; 2 * n * d * d],
            kv_bank: vec![0.0; 2 * n * kv * d],
            mlp_up_bank: vec![0.0; n * mlp * d],
            mlp_down_bank: vec![0.0; n * d * mlp],
            blocks,
            ve_embed: vec![0.0; config.vocab_size * config.ve_dim],
            ve_proj: vec![0.0; kv * config.ve_dim],
            ve_scale: 0.1,
            ve_layer_scales: vec![1.0; num_ve_layers],
            rope_cos,
            rope_sin,
            config,
        }
    }

    /// Print model summary.
    pub fn summary(&self) {
        let c = &self.config;
        eprintln!("=== GPT Model Summary ===");
        eprintln!("Layers: {}", c.num_layers);
        eprintln!("Dim: {}", c.model_dim);
        eprintln!("Heads: {} (KV: {})", c.num_heads, c.num_kv_heads);
        eprintln!("MLP dim: {}", c.mlp_dim);
        eprintln!("Vocab: {}", c.vocab_size);
        eprintln!("Seq len: {}", c.train_seq_len);
        eprintln!("RoPE dims: {}/{}", c.rope_dims, c.head_dim);
        eprintln!("XSA last: {}", c.xsa_last_n);
        eprintln!("BigramHash: {} × {}", c.bigram_vocab_size, c.bigram_dim);
        eprintln!("VE: {:?} (dim {})", c.ve_layers, c.ve_dim);
        eprintln!("Total params: ~{:.1}M", c.param_count() as f64 / 1e6);
        eprintln!("Bank shapes:");
        eprintln!("  qo_bank:       {:?}", c.qo_bank_shape());
        eprintln!("  kv_bank:       {:?}", c.kv_bank_shape());
        eprintln!("  mlp_up_bank:   {:?}", c.mlp_up_bank_shape());
        eprintln!("  mlp_down_bank: {:?}", c.mlp_down_bank_shape());
        eprintln!("========================");
    }

    pub fn fill_deterministic(&mut self) {
        fn fill(buf: &mut [f32], mul: usize, add: usize, modu: usize, scale: f32, bias: f32) {
            for (i, v) in buf.iter_mut().enumerate() {
                *v = scale * (((i * mul + add) % modu) as f32) + bias;
            }
        }

        fill(&mut self.tok_emb, 7, 3, 29, 0.008, -0.11);
        fill(&mut self.bigram_embed, 11, 5, 31, 0.007, -0.09);
        fill(&mut self.bigram_proj, 13, 1, 37, 0.006, -0.10);
        fill(&mut self.smear_gate, 5, 2, 19, 0.08, -0.7);
        fill(&mut self.skip_weights, 3, 4, 17, 0.01, 0.92);
        fill(&mut self.qo_bank, 17, 9, 41, 0.005, -0.10);
        fill(&mut self.kv_bank, 19, 7, 43, 0.005, -0.09);
        fill(&mut self.mlp_up_bank, 23, 6, 47, 0.004, -0.08);
        fill(&mut self.mlp_down_bank, 29, 8, 53, 0.004, -0.07);
        fill(&mut self.ve_embed, 31, 3, 59, 0.006, -0.09);
        fill(&mut self.ve_proj, 37, 2, 61, 0.005, -0.08);
        self.bigram_scale = 0.05;
        self.ve_scale = 0.10;

        for (i, scale) in self.ve_layer_scales.iter_mut().enumerate() {
            *scale = 0.85 + 0.05 * ((i % 3) as f32);
        }

        for (layer, block) in self.blocks.iter_mut().enumerate() {
            for (i, v) in block.attn_scale.iter_mut().enumerate() {
                *v = 0.92 + 0.02 * (((layer + i) % 5) as f32);
            }
            for (i, v) in block.mlp_scale.iter_mut().enumerate() {
                *v = 0.90 + 0.015 * (((2 * layer + i) % 7) as f32);
            }
            let d = self.config.model_dim;
            for i in 0..d {
                block.resid_mix[i] = 0.82 + 0.02 * (((layer + i) % 4) as f32);
                block.resid_mix[d + i] = -0.06 + 0.02 * (((layer + 2 * i) % 5) as f32);
            }
            for (head, gain) in block.q_gain.iter_mut().enumerate() {
                *gain = self.config.qk_gain_init * (0.9 + 0.04 * (((layer + head) % 3) as f32));
            }
            if self.config.attn_out_gate_enabled {
                fill(&mut block.attn_gate_weight, 7 + layer, 3, 23, 0.002, -0.022);
                for (head, bias) in block.attn_gate_bias.iter_mut().enumerate() {
                    *bias = -0.04 + 0.02 * (((layer + head) % 5) as f32);
                }
            }
            if self.config.sparse_attn_gate_enabled {
                block.sparse_attn_gate_weight.fill(0.0);
            }
        }
    }

    pub fn forward_with_plan(
        &self,
        plan: &ExecutionPlan,
        input_ids: &[u32],
        buf: &mut ForwardBuffer,
    ) -> PgResult<()> {
        plan.validate_model_config(&self.config)?;
        self.forward(input_ids, buf);
        Ok(())
    }

    // ---- Bank slicing helpers ----

    /// Get Q weight for layer i: qo_bank[i] — shape [model_dim, model_dim]
    pub(crate) fn q_weight(&self, layer: usize) -> &[f32] {
        let d = self.config.model_dim;
        let offset = layer * d * d;
        &self.qo_bank[offset..offset + d * d]
    }

    /// Get O (output projection) weight for layer i: qo_bank[n + i]
    pub(crate) fn o_weight(&self, layer: usize) -> &[f32] {
        let n = self.config.num_layers;
        let d = self.config.model_dim;
        let offset = (n + layer) * d * d;
        &self.qo_bank[offset..offset + d * d]
    }

    /// Get K weight for layer i: kv_bank[i] — shape [kv_dim, model_dim]
    pub(crate) fn k_weight(&self, layer: usize) -> &[f32] {
        let kv = self.config.kv_dim();
        let d = self.config.model_dim;
        let offset = layer * kv * d;
        &self.kv_bank[offset..offset + kv * d]
    }

    /// Get V weight for layer i: kv_bank[n + i]
    pub(crate) fn v_weight(&self, layer: usize) -> &[f32] {
        let n = self.config.num_layers;
        let kv = self.config.kv_dim();
        let d = self.config.model_dim;
        let offset = (n + layer) * kv * d;
        &self.kv_bank[offset..offset + kv * d]
    }

    /// Get MLP up weight for layer i: shape [mlp_dim, model_dim]
    pub(crate) fn mlp_up_weight(&self, layer: usize) -> &[f32] {
        let mlp = self.config.mlp_dim;
        let d = self.config.model_dim;
        let offset = layer * mlp * d;
        &self.mlp_up_bank[offset..offset + mlp * d]
    }

    /// Get MLP down weight for layer i: shape [model_dim, mlp_dim]
    pub(crate) fn mlp_down_weight(&self, layer: usize) -> &[f32] {
        let mlp = self.config.mlp_dim;
        let d = self.config.model_dim;
        let offset = layer * d * mlp;
        &self.mlp_down_bank[offset..offset + d * mlp]
    }

    /// Get skip weight for decoder position i: shape [model_dim]
    pub(crate) fn skip_weight(&self, i: usize) -> &[f32] {
        let d = self.config.model_dim;
        &self.skip_weights[i * d..(i + 1) * d]
    }

    /// Compute value embedding for a layer, using cache.
    fn compute_ve(&self, layer: usize, tokens: &[u32], buf: &mut ForwardBuffer) {
        if !self.config.ve_enabled {
            return;
        }
        let ve_idx = self.config.ve_layers.iter().position(|&l| l == layer);
        if ve_idx.is_none() {
            return;
        }
        let ve_idx = ve_idx.unwrap();
        let t = buf.tokens;
        let ve_dim = self.config.ve_dim;
        let kv_dim = self.config.kv_dim();

        // Compute shared VE base if not cached
        if buf.ve_cache.is_none() {
            // Embed: [tokens, ve_dim]
            for i in 0..t {
                let tok = tokens[i] as usize;
                let src = &self.ve_embed[tok * ve_dim..(tok + 1) * ve_dim];
                buf.ve_embed_out[i * ve_dim..(i + 1) * ve_dim].copy_from_slice(src);
            }
            // Project: [tokens, kv_dim] = [tokens, ve_dim] @ [kv_dim, ve_dim]^T
            let mut projected = vec![0.0f32; t * kv_dim];
            pg_kernels::linear::linear_forward(
                &buf.ve_embed_out[..t * ve_dim],
                &self.ve_proj,
                &mut projected,
                t,
                kv_dim,
                ve_dim,
            );
            // Scale by ve_scale
            for v in projected.iter_mut() {
                *v *= self.ve_scale;
            }
            buf.ve_cache = Some(projected);
        }

        // Apply per-layer scale
        let base = buf.ve_cache.as_ref().unwrap();
        let scale = self.ve_layer_scales[ve_idx];
        for i in 0..t * kv_dim {
            buf.ve_out[i] = base[i] * scale;
        }
    }

    /// Run one application of a transformer block forward.
    pub(crate) fn block_forward_once(&self, layer: usize, tokens: &[u32], buf: &mut ForwardBuffer) {
        let t = buf.tokens;
        let d = buf.model_dim;
        let h = buf.num_heads;
        let hkv = buf.num_kv_heads;
        let hd = buf.head_dim;
        let kv_dim = buf.kv_dim;
        let mlp_dim = buf.mlp_dim;
        let bp = &self.blocks[layer];

        // 1. Residual mixing: x_in = resid_mix[0] * x + resid_mix[1] * x0
        for i in 0..t * d {
            let dim_idx = i % d;
            buf.x_in[i] = bp.resid_mix[dim_idx] * buf.x[i] + bp.resid_mix[d + dim_idx] * buf.x0[i];
        }

        // 2. Attention norm: RMSNorm(x_in) * ln_scale_factor
        pg_kernels::rms_norm::rms_norm_forward_cpu(
            &buf.x_in[..t * d],
            &mut buf.attn_norm_out[..t * d],
            d,
            bp.ln_scale_factor,
            1e-6,
        );

        // 3. Q, K, V projections from banks
        let q_w = self.q_weight(layer);
        let k_w = self.k_weight(layer);
        let v_w = self.v_weight(layer);

        pg_kernels::linear::linear_forward(
            &buf.attn_norm_out[..t * d],
            q_w,
            &mut buf.q[..t * d],
            t,
            d,
            d,
        );
        pg_kernels::linear::linear_forward(
            &buf.attn_norm_out[..t * d],
            k_w,
            &mut buf.k[..t * kv_dim],
            t,
            kv_dim,
            d,
        );
        pg_kernels::linear::linear_forward(
            &buf.attn_norm_out[..t * d],
            v_w,
            &mut buf.v[..t * kv_dim],
            t,
            kv_dim,
            d,
        );

        // 4. Add value embedding if applicable
        self.compute_ve(layer, tokens, buf);
        if self.config.ve_enabled && self.config.ve_layers.contains(&layer) {
            for i in 0..t * kv_dim {
                buf.v[i] += buf.ve_out[i];
            }
        }

        // 5-6. VRL disabled in SOTA config (value_residual=False)

        // 7-9. Fusion C: Q/K norm + partial RoPE + q_gain.
        pg_kernels::fusion::rmsnorm_qk_rope_qgain_fused(
            &mut buf.q[..t * h * hd],
            &mut buf.k[..t * hkv * hd],
            &self.rope_cos,
            &self.rope_sin,
            &bp.q_gain,
            t,
            h,
            hkv,
            hd,
            self.config.rope_dims,
            1e-6,
        );

        // 10. Causal attention
        pg_kernels::attention::causal_attention_forward(
            &buf.q[..t * h * hd],
            &buf.k[..t * hkv * hd],
            &buf.v[..t * hkv * hd],
            &mut buf.attn_out[..t * h * hd],
            t,
            h,
            hkv,
            hd,
        );

        // 11. XSA (last N layers)
        let attn_result_raw = if bp.use_xsa {
            pg_kernels::xsa::xsa_forward(
                &buf.attn_out[..t * h * hd],
                &buf.v[..t * hkv * hd],
                &mut buf.xsa_out[..t * h * hd],
                t,
                h,
                hkv,
                hd,
            );
            &buf.xsa_out[..t * h * hd]
        } else {
            &buf.attn_out[..t * h * hd]
        };

        let attn_result = if self.attn_out_gate_enabled() {
            let width = self.config.attn_out_gate_width;
            debug_assert!(width <= d);
            for tok in 0..t {
                let gate_input = &buf.attn_norm_out[tok * d..tok * d + width];
                for head in 0..h {
                    let mut score = bp.attn_gate_bias[head];
                    let w = &bp.attn_gate_weight[head * width..(head + 1) * width];
                    for j in 0..width {
                        score += w[j] * gate_input[j];
                    }
                    let sig = 1.0 / (1.0 + (-score).exp());
                    let gate = 2.0 * sig;
                    buf.attn_gate[tok * h + head] = gate;
                    let base = (tok * h + head) * hd;
                    for j in 0..hd {
                        buf.attn_gated[base + j] = attn_result_raw[base + j] * gate;
                    }
                }
            }
            &buf.attn_gated[..t * h * hd]
        } else if self.sparse_attn_gate_enabled() {
            let width = self.config.sparse_attn_gate_width;
            let scale = self.config.sparse_attn_gate_scale;
            debug_assert!(width <= d);
            for tok in 0..t {
                let gate_input = &buf.attn_norm_out[tok * d..tok * d + width];
                for head in 0..h {
                    let w = &bp.sparse_attn_gate_weight[head * width..(head + 1) * width];
                    let mut score = 0.0f32;
                    for j in 0..width {
                        score += w[j] * gate_input[j];
                    }
                    let gate = 1.0 / (1.0 + (-(scale * score)).exp());
                    buf.attn_gate[tok * h + head] = gate;
                    let base = (tok * h + head) * hd;
                    for j in 0..hd {
                        buf.attn_gated[base + j] = attn_result_raw[base + j] * gate;
                    }
                }
            }
            &buf.attn_gated[..t * h * hd]
        } else {
            attn_result_raw
        };

        // 12. Reshape [tokens, num_heads, head_dim] → [tokens, model_dim] and output projection
        // attn_result is already [t * d] since d = h * hd
        let o_w = self.o_weight(layer);
        pg_kernels::linear::linear_forward(attn_result, o_w, &mut buf.proj_out[..t * d], t, d, d);

        // 13. Residual after attention.
        for i in 0..t * d {
            let dim_idx = i % d;
            buf.x[i] = buf.x_in[i] + bp.attn_scale[dim_idx] * buf.proj_out[i];
        }

        let mlp_input = if self.parallel_residual_enabled() {
            &buf.x_in[..t * d]
        } else {
            &buf.x[..t * d]
        };

        // 14. MLP norm
        pg_kernels::rms_norm::rms_norm_forward_cpu(
            mlp_input,
            &mut buf.mlp_norm_out[..t * d],
            d,
            bp.ln_scale_factor,
            1e-6,
        );

        // 15. MLP: up → LeakyReLU(0.5)² → down
        let up_w = self.mlp_up_weight(layer);
        let down_w = self.mlp_down_weight(layer);

        pg_kernels::linear::linear_forward(
            &buf.mlp_norm_out[..t * d],
            up_w,
            &mut buf.mlp_up[..t * mlp_dim],
            t,
            mlp_dim,
            d,
        );
        pg_kernels::activations::leaky_relu_sq_forward(
            &buf.mlp_up[..t * mlp_dim],
            &mut buf.mlp_act[..t * mlp_dim],
        );
        pg_kernels::linear::linear_forward(
            &buf.mlp_act[..t * mlp_dim],
            down_w,
            &mut buf.mlp_out[..t * d],
            t,
            d,
            mlp_dim,
        );

        // 16. Residual: x = x + mlp_scale * mlp_out
        for i in 0..t * d {
            let dim_idx = i % d;
            let base = if self.parallel_residual_enabled() {
                buf.x_in[i] + bp.attn_scale[dim_idx] * buf.proj_out[i]
            } else {
                buf.x[i]
            };
            buf.x[i] = base + bp.mlp_scale[dim_idx] * buf.mlp_out[i];
        }
    }

    /// Run a logical transformer block forward, including recurrence when enabled.
    pub(crate) fn block_forward(&self, layer: usize, tokens: &[u32], buf: &mut ForwardBuffer) {
        self.block_forward_once(layer, tokens, buf);
        if self.is_recurrent_layer(layer) {
            self.block_forward_once(layer, tokens, buf);
        }
    }

    /// Full forward pass: returns logits [tokens, vocab_size].
    /// input_ids: [tokens] (u32 token IDs)
    pub fn forward(&self, input_ids: &[u32], buf: &mut ForwardBuffer) {
        self.forward_inner(input_ids, buf, None);
    }

    /// Forward pass with optional cache capture for backward.
    /// When `cache` is Some, saves per-layer hidden states and intermediate activations.
    pub(crate) fn forward_inner(
        &self,
        input_ids: &[u32],
        buf: &mut ForwardBuffer,
        mut cache: Option<&mut crate::backward::ForwardCache>,
    ) {
        let t = input_ids.len();
        let d = self.config.model_dim;
        assert_eq!(t, buf.tokens);

        // 1. Token embedding lookup
        for i in 0..t {
            let tok = input_ids[i] as usize;
            let src = &self.tok_emb[tok * d..(tok + 1) * d];
            buf.x[i * d..(i + 1) * d].copy_from_slice(src);
        }

        // 2. Fusion B: token embedding + bigram gather + projection + add.
        if self.config.bigram_vocab_size > 0 {
            pg_kernels::fusion::bigram_embed_fused(
                input_ids,
                &self.tok_emb,
                &self.bigram_embed,
                &self.bigram_proj,
                self.bigram_scale,
                &mut buf.x[..t * d],
                d,
                self.config.bigram_dim,
                self.config.bigram_vocab_size,
            );
        }

        // Save x_post_embed (before initial RMSNorm) for bigram backward
        if let Some(ref mut c) = cache {
            c.x_post_embed = buf.x[..t * d].to_vec();
        }

        // 3. Initial RMSNorm
        let mut normed = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_forward_cpu(&buf.x[..t * d], &mut normed, d, 1.0, 1e-6);
        buf.x[..t * d].copy_from_slice(&normed);

        // Save x_post_norm (before SmearGate) for SmearGate backward
        if let Some(ref mut c) = cache {
            c.x_post_norm = buf.x[..t * d].to_vec();
        }

        // 4. SmearGate
        {
            let mut smeared = vec![0.0f32; t * d];
            if let Some(boundary) = self.config.smear_gate_boundary_token_id {
                pg_kernels::smear_gate::smear_gate_forward_boundary(
                    &buf.x[..t * d],
                    input_ids,
                    &self.smear_gate,
                    &mut smeared,
                    t,
                    d,
                    boundary,
                );
            } else {
                let mut x_prev = vec![0.0f32; t * d];
                for i in 1..t {
                    x_prev[i * d..(i + 1) * d].copy_from_slice(&buf.x[(i - 1) * d..i * d]);
                }
                pg_kernels::smear_gate::smear_gate_forward(
                    &buf.x[..t * d],
                    &x_prev,
                    &self.smear_gate,
                    &mut smeared,
                    t,
                    d,
                );
            }
            buf.x[..t * d].copy_from_slice(&smeared);
        }

        // 5. Save x0 (anchor for residual mixing)
        buf.x0[..t * d].copy_from_slice(&buf.x[..t * d]);
        if let Some(ref mut c) = cache {
            c.x0.copy_from_slice(&buf.x0[..t * d]);
            c.input_ids = input_ids.to_vec();
            c.tokens = t;
        }
        buf.v0 = None;
        buf.ve_cache = None;
        buf.skips.clear();

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();

        // 6. Encoder layers
        for i in 0..n_enc {
            if let Some(ref mut c) = cache {
                c.layer_x.push(buf.x[..t * d].to_vec());
            }
            self.block_forward(i, input_ids, buf);
            buf.skips.push(buf.x[..t * d].to_vec());
        }

        // Save encoder skips for cache (before decoder pops them)
        if let Some(ref mut c) = cache {
            c.skips = buf.skips.clone();
        }

        // 7. Decoder layers with skip connections
        for i in 0..n_dec {
            let bi = n_enc + i;
            if let Some(skip) = buf.skips.pop() {
                let sw = self.skip_weight(i);
                for j in 0..t * d {
                    let dim_idx = j % d;
                    buf.x[j] += sw[dim_idx] * skip[j];
                }
            }
            if let Some(ref mut c) = cache {
                c.layer_x.push(buf.x[..t * d].to_vec());
            }
            self.block_forward(bi, input_ids, buf);
        }

        // Save x_final for cache
        if let Some(ref mut c) = cache {
            c.x_final.copy_from_slice(&buf.x[..t * d]);
        }

        // 8. Final RMSNorm
        let mut final_normed = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_forward_cpu(
            &buf.x[..t * d],
            &mut final_normed,
            d,
            1.0,
            1e-6,
        );

        // 9. Output projection (tied embeddings: logits = x @ tok_emb^T)
        let vocab = self.config.vocab_size;
        pg_kernels::linear::linear_forward(
            &final_normed,
            &self.tok_emb,
            &mut buf.logits[..t * vocab],
            t,
            vocab,
            d,
        );
    }

    /// Compute loss from forward pass results.
    /// targets: [tokens] — target token IDs for cross-entropy
    /// Note: buf.logits contains RAW logits; softcap is applied inside cross_entropy.
    pub fn compute_loss(&self, targets: &[u32], buf: &ForwardBuffer) -> f32 {
        let t = buf.tokens;
        let vocab = self.config.vocab_size;
        let mut losses = vec![0.0f32; t];
        pg_kernels::cross_entropy::cross_entropy_forward(
            &buf.logits[..t * vocab],
            targets,
            &mut losses,
            vocab,
            self.config.logit_softcap,
        );
        pg_kernels::cross_entropy::mean_loss(&losses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelConfig;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 32,
            num_layers: 2,
            model_dim: 16,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 8,
            mlp_mult: 2.0,
            mlp_dim: 32,
            rope_base: 10000.0,
            rope_dims: 4,
            xsa_last_n: 1,
            logit_softcap: 30.0,
            qk_gain_init: 1.0,
            recurrence_enabled: false,
            recurrence_start_layer: 0,
            recurrence_repeat_layers: 0,
            parallel_residual: false,
            attn_out_gate_enabled: false,
            attn_out_gate_width: 24,
            sparse_attn_gate_enabled: false,
            sparse_attn_gate_width: 12,
            sparse_attn_gate_scale: 1.0,
            smear_gate_boundary_token_id: Some(1),
            vrl_enabled: false,
            ve_enabled: false,
            ve_dim: 4,
            ve_layers: vec![],
            bigram_vocab_size: 8,
            bigram_dim: 4,
            ln_scale: true,
            tie_embeddings: true,
            tied_embed_init_std: 0.005,
            train_seq_len: 16,
            eval_seq_len: 16,
        }
    }

    #[test]
    fn test_forward_pass_runs() {
        let config = tiny_config();
        let model = GptModel::new(config.clone());
        let tokens = vec![1u32, 5, 3, 7, 2, 4];
        let mut buf = ForwardBuffer::new(&config, tokens.len());

        model.forward(&tokens, &mut buf);

        // Logits should be finite
        let vocab = config.vocab_size;
        for i in 0..tokens.len() * vocab {
            assert!(buf.logits[i].is_finite(), "non-finite logit at {}", i);
        }

        // Logits are raw (pre-softcap), should be finite but not necessarily bounded
        // Softcap is applied inside cross_entropy
        assert!(
            buf.logits[0..tokens.len() * vocab]
                .iter()
                .all(|v| v.is_finite()),
            "all logits should be finite"
        );
    }

    #[test]
    fn test_forward_loss_reasonable() {
        let config = tiny_config();
        let model = GptModel::new(config.clone());
        let tokens = vec![1u32, 5, 3, 7];
        let targets = vec![5u32, 3, 7, 2];
        let mut buf = ForwardBuffer::new(&config, tokens.len());

        model.forward(&tokens, &mut buf);
        let loss = model.compute_loss(&targets, &buf);

        // With random weights, loss should be around ln(vocab_size) = ln(32) ≈ 3.47
        assert!(loss.is_finite(), "loss is not finite: {}", loss);
        assert!(loss > 0.0, "loss should be positive: {}", loss);
        eprintln!("Loss with zero-init weights: {}", loss);
    }

    #[test]
    fn test_attn_out_gate_zero_init_is_identity() {
        let mut gated_config = tiny_config();
        gated_config.attn_out_gate_enabled = true;
        gated_config.attn_out_gate_width = 4;
        let mut plain_config = gated_config.clone();
        plain_config.attn_out_gate_enabled = false;

        let mut gated = GptModel::new(gated_config.clone());
        gated.fill_deterministic();
        for block in &mut gated.blocks {
            block.attn_gate_weight.fill(0.0);
            block.attn_gate_bias.fill(0.0);
        }

        let mut plain = GptModel::new(plain_config.clone());
        plain.tok_emb.clone_from(&gated.tok_emb);
        plain.bigram_embed.clone_from(&gated.bigram_embed);
        plain.bigram_proj.clone_from(&gated.bigram_proj);
        plain.bigram_scale = gated.bigram_scale;
        plain.smear_gate.clone_from(&gated.smear_gate);
        plain.skip_weights.clone_from(&gated.skip_weights);
        plain.qo_bank.clone_from(&gated.qo_bank);
        plain.kv_bank.clone_from(&gated.kv_bank);
        plain.mlp_up_bank.clone_from(&gated.mlp_up_bank);
        plain.mlp_down_bank.clone_from(&gated.mlp_down_bank);
        plain.ve_embed.clone_from(&gated.ve_embed);
        plain.ve_proj.clone_from(&gated.ve_proj);
        plain.ve_scale = gated.ve_scale;
        plain.ve_layer_scales.clone_from(&gated.ve_layer_scales);
        for layer in 0..plain.blocks.len() {
            plain.blocks[layer]
                .attn_scale
                .clone_from(&gated.blocks[layer].attn_scale);
            plain.blocks[layer]
                .mlp_scale
                .clone_from(&gated.blocks[layer].mlp_scale);
            plain.blocks[layer]
                .resid_mix
                .clone_from(&gated.blocks[layer].resid_mix);
            plain.blocks[layer]
                .q_gain
                .clone_from(&gated.blocks[layer].q_gain);
        }

        let tokens = vec![1u32, 5, 3, 7, 2, 4];
        let mut gated_buf = ForwardBuffer::new(&gated_config, tokens.len());
        let mut plain_buf = ForwardBuffer::new(&plain_config, tokens.len());
        gated.forward(&tokens, &mut gated_buf);
        plain.forward(&tokens, &mut plain_buf);

        let max_diff = gated_buf
            .logits
            .iter()
            .zip(plain_buf.logits.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "identity gate changed logits by {max_diff}"
        );
    }

    #[test]
    fn test_sparse_attn_gate_zero_init_halves_attention_path() {
        let mut sparse_config = tiny_config();
        sparse_config.sparse_attn_gate_enabled = true;
        sparse_config.sparse_attn_gate_width = 4;
        sparse_config.sparse_attn_gate_scale = 1.0;
        let mut plain_config = sparse_config.clone();
        plain_config.sparse_attn_gate_enabled = false;

        let mut sparse = GptModel::new(sparse_config.clone());
        sparse.fill_deterministic();
        for block in &mut sparse.blocks {
            block.sparse_attn_gate_weight.fill(0.0);
            for scale in &mut block.attn_scale {
                *scale *= 2.0;
            }
        }

        let mut plain = GptModel::new(plain_config.clone());
        plain.tok_emb.clone_from(&sparse.tok_emb);
        plain.bigram_embed.clone_from(&sparse.bigram_embed);
        plain.bigram_proj.clone_from(&sparse.bigram_proj);
        plain.bigram_scale = sparse.bigram_scale;
        plain.smear_gate.clone_from(&sparse.smear_gate);
        plain.skip_weights.clone_from(&sparse.skip_weights);
        plain.qo_bank.clone_from(&sparse.qo_bank);
        plain.kv_bank.clone_from(&sparse.kv_bank);
        plain.mlp_up_bank.clone_from(&sparse.mlp_up_bank);
        plain.mlp_down_bank.clone_from(&sparse.mlp_down_bank);
        plain.ve_embed.clone_from(&sparse.ve_embed);
        plain.ve_proj.clone_from(&sparse.ve_proj);
        plain.ve_scale = sparse.ve_scale;
        plain.ve_layer_scales.clone_from(&sparse.ve_layer_scales);
        for layer in 0..plain.blocks.len() {
            plain.blocks[layer]
                .attn_scale
                .clone_from(&sparse.blocks[layer].attn_scale);
            for scale in &mut plain.blocks[layer].attn_scale {
                *scale *= 0.5;
            }
            plain.blocks[layer]
                .mlp_scale
                .clone_from(&sparse.blocks[layer].mlp_scale);
            plain.blocks[layer]
                .resid_mix
                .clone_from(&sparse.blocks[layer].resid_mix);
            plain.blocks[layer]
                .q_gain
                .clone_from(&sparse.blocks[layer].q_gain);
        }

        let tokens = vec![1u32, 5, 3, 7, 2, 4];
        let mut sparse_buf = ForwardBuffer::new(&sparse_config, tokens.len());
        let mut plain_buf = ForwardBuffer::new(&plain_config, tokens.len());
        sparse.forward(&tokens, &mut sparse_buf);
        plain.forward(&tokens, &mut plain_buf);

        let max_diff = sparse_buf
            .logits
            .iter()
            .zip(plain_buf.logits.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "zero SparseAttnGate with compensated attn_scale changed logits by {max_diff}"
        );
    }

    #[test]
    fn test_bank_slicing() {
        let config = tiny_config();
        let model = GptModel::new(config.clone());

        // Verify bank slice sizes
        let d = config.model_dim;
        let kv = config.kv_dim();
        assert_eq!(model.q_weight(0).len(), d * d);
        assert_eq!(model.o_weight(0).len(), d * d);
        assert_eq!(model.k_weight(0).len(), kv * d);
        assert_eq!(model.v_weight(0).len(), kv * d);
        assert_eq!(model.mlp_up_weight(0).len(), config.mlp_dim * d);
        assert_eq!(model.mlp_down_weight(0).len(), d * config.mlp_dim);

        // Verify different layers give different slices
        let q0_ptr = model.q_weight(0).as_ptr();
        let q1_ptr = model.q_weight(1).as_ptr();
        assert_ne!(q0_ptr, q1_ptr);
    }

    #[test]
    fn test_xsa_applied_to_last_layer() {
        let config = tiny_config();
        let model = GptModel::new(config);
        assert!(!model.blocks[0].use_xsa); // layer 0: not XSA
        assert!(model.blocks[1].use_xsa); // layer 1: XSA (last 1)
    }
}
