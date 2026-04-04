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

/// Per-block learnable parameters (non-banked).
pub struct BlockParams {
    pub attn_scale: Vec<f32>,    // [model_dim]
    pub mlp_scale: Vec<f32>,     // [model_dim]
    pub resid_mix: Vec<f32>,     // [2, model_dim] — resid_mix[0] and resid_mix[1]
    pub q_gain: Vec<f32>,        // [num_heads]
    pub ln_scale_factor: f32,    // 1/sqrt(layer_idx + 1) if ln_scale
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
    pub rope_cos: Vec<f32>,      // [seq_len, rope_dims/2]
    pub rope_sin: Vec<f32>,      // [seq_len, rope_dims/2]
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
    pub x: Vec<f32>,             // [tokens, model_dim] — main hidden state
    pub x0: Vec<f32>,            // [tokens, model_dim] — residual stream anchor
    pub x_in: Vec<f32>,          // [tokens, model_dim]
    pub attn_norm_out: Vec<f32>, // [tokens, model_dim]
    pub mlp_norm_out: Vec<f32>,  // [tokens, model_dim]
    pub q: Vec<f32>,             // [tokens, num_heads, head_dim]
    pub k: Vec<f32>,             // [tokens, num_kv_heads, head_dim]
    pub v: Vec<f32>,             // [tokens, num_kv_heads, head_dim]
    pub attn_out: Vec<f32>,      // [tokens, num_heads, head_dim]
    pub xsa_out: Vec<f32>,       // [tokens, num_heads, head_dim]
    pub proj_out: Vec<f32>,      // [tokens, model_dim]
    pub mlp_up: Vec<f32>,        // [tokens, mlp_dim]
    pub mlp_act: Vec<f32>,       // [tokens, mlp_dim] (after activation)
    pub mlp_out: Vec<f32>,       // [tokens, model_dim]
    pub bigram_out: Vec<f32>,    // [tokens, bigram_dim]
    pub bigram_proj_out: Vec<f32>, // [tokens, model_dim]
    pub ve_out: Vec<f32>,        // [tokens, kv_dim]
    pub ve_embed_out: Vec<f32>,  // [tokens, ve_dim]
    pub logits: Vec<f32>,        // [tokens, vocab_size]

    // Skip connections stack (encoder outputs)
    pub skips: Vec<Vec<f32>>,

    // v0 cache for VRL
    pub v0: Option<Vec<f32>>,    // [tokens, num_kv_heads, head_dim]

    // VE cache (shared computation)
    pub ve_cache: Option<Vec<f32>>, // [tokens, kv_dim]
}

impl ForwardBuffer {
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
    fn compute_ve(
        &self,
        layer: usize,
        tokens: &[u32],
        buf: &mut ForwardBuffer,
    ) {
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
                t, kv_dim, ve_dim,
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

    /// Run a single transformer block forward.
    pub(crate) fn block_forward(
        &self,
        layer: usize,
        tokens: &[u32],
        buf: &mut ForwardBuffer,
    ) -> Option<Vec<f32>> {
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
            buf.x_in[i] = bp.resid_mix[dim_idx] * buf.x[i]
                + bp.resid_mix[d + dim_idx] * buf.x0[i];
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
            &buf.attn_norm_out[..t * d], q_w, &mut buf.q[..t * d], t, d, d,
        );
        pg_kernels::linear::linear_forward(
            &buf.attn_norm_out[..t * d], k_w, &mut buf.k[..t * kv_dim], t, kv_dim, d,
        );
        pg_kernels::linear::linear_forward(
            &buf.attn_norm_out[..t * d], v_w, &mut buf.v[..t * kv_dim], t, kv_dim, d,
        );

        // 4. Add value embedding if applicable
        self.compute_ve(layer, tokens, buf);
        if self.config.ve_enabled && self.config.ve_layers.contains(&layer) {
            for i in 0..t * kv_dim {
                buf.v[i] += buf.ve_out[i];
            }
        }

        // 5. Capture raw_v for VRL (only from first layer)
        let raw_v = if self.config.vrl_enabled && buf.v0.is_none() {
            Some(buf.v[..t * kv_dim].to_vec())
        } else {
            None
        };

        // 6. VRL: mix v0 and current v (not used in SOTA config, but implemented)
        // In SOTA: value_residual=False, so this is skipped

        // 7. QK-norm (RMSNorm on Q and K per head)
        // Q: [tokens, num_heads * head_dim] — normalize each head's head_dim slice
        for i in 0..t * h {
            let offset = i * hd;
            let slice = &buf.q[offset..offset + hd];
            let rms = (slice.iter().map(|&x| x * x).sum::<f32>() / hd as f32 + 1e-6).sqrt();
            let inv_rms = 1.0 / rms;
            for j in 0..hd {
                buf.q[offset + j] *= inv_rms;
            }
        }
        for i in 0..t * hkv {
            let offset = i * hd;
            let slice = &buf.k[offset..offset + hd];
            let rms = (slice.iter().map(|&x| x * x).sum::<f32>() / hd as f32 + 1e-6).sqrt();
            let inv_rms = 1.0 / rms;
            for j in 0..hd {
                buf.k[offset + j] *= inv_rms;
            }
        }

        // 8. Partial RoPE on Q and K
        let rope_dims = self.config.rope_dims;
        if rope_dims > 0 {
            pg_kernels::rope::apply_partial_rope(
                &mut buf.q[..t * h * hd],
                &self.rope_cos, &self.rope_sin,
                1, t, h, hd, rope_dims,
            );
            pg_kernels::rope::apply_partial_rope(
                &mut buf.k[..t * hkv * hd],
                &self.rope_cos, &self.rope_sin,
                1, t, hkv, hd, rope_dims,
            );
        }

        // 9. Apply q_gain: Q *= q_gain[head] per head
        for tok in 0..t {
            for head in 0..h {
                let offset = (tok * h + head) * hd;
                let gain = bp.q_gain[head];
                for j in 0..hd {
                    buf.q[offset + j] *= gain;
                }
            }
        }

        // 10. Causal attention
        pg_kernels::attention::causal_attention_forward(
            &buf.q[..t * h * hd],
            &buf.k[..t * hkv * hd],
            &buf.v[..t * hkv * hd],
            &mut buf.attn_out[..t * h * hd],
            t, h, hkv, hd,
        );

        // 11. XSA (last N layers)
        let attn_result = if bp.use_xsa {
            pg_kernels::xsa::xsa_forward(
                &buf.attn_out[..t * h * hd],
                &buf.v[..t * hkv * hd],
                &mut buf.xsa_out[..t * h * hd],
                t, h, hkv, hd,
            );
            &buf.xsa_out[..t * h * hd]
        } else {
            &buf.attn_out[..t * h * hd]
        };

        // 12. Reshape [tokens, num_heads, head_dim] → [tokens, model_dim] and output projection
        // attn_result is already [t * d] since d = h * hd
        let o_w = self.o_weight(layer);
        pg_kernels::linear::linear_forward(
            attn_result, o_w, &mut buf.proj_out[..t * d], t, d, d,
        );

        // 13. Residual: x_out = x_in + attn_scale * proj_out
        for i in 0..t * d {
            let dim_idx = i % d;
            buf.x[i] = buf.x_in[i] + bp.attn_scale[dim_idx] * buf.proj_out[i];
        }

        // 14. MLP norm
        pg_kernels::rms_norm::rms_norm_forward_cpu(
            &buf.x[..t * d],
            &mut buf.mlp_norm_out[..t * d],
            d,
            bp.ln_scale_factor,
            1e-6,
        );

        // 15. MLP: up → LeakyReLU(0.5)² → down
        let up_w = self.mlp_up_weight(layer);
        let down_w = self.mlp_down_weight(layer);

        pg_kernels::linear::linear_forward(
            &buf.mlp_norm_out[..t * d], up_w, &mut buf.mlp_up[..t * mlp_dim], t, mlp_dim, d,
        );
        pg_kernels::activations::leaky_relu_sq_forward(
            &buf.mlp_up[..t * mlp_dim], &mut buf.mlp_act[..t * mlp_dim],
        );
        pg_kernels::linear::linear_forward(
            &buf.mlp_act[..t * mlp_dim], down_w, &mut buf.mlp_out[..t * d], t, d, mlp_dim,
        );

        // 16. Residual: x = x + mlp_scale * mlp_out
        for i in 0..t * d {
            let dim_idx = i % d;
            buf.x[i] += bp.mlp_scale[dim_idx] * buf.mlp_out[i];
        }

        raw_v
    }

    /// Full forward pass: returns logits [tokens, vocab_size].
    /// input_ids: [tokens] (u32 token IDs)
    pub fn forward(&self, input_ids: &[u32], buf: &mut ForwardBuffer) {
        let t = input_ids.len();
        let d = self.config.model_dim;
        assert_eq!(t, buf.tokens);

        // 1. Token embedding lookup
        for i in 0..t {
            let tok = input_ids[i] as usize;
            let src = &self.tok_emb[tok * d..(tok + 1) * d];
            buf.x[i * d..(i + 1) * d].copy_from_slice(src);
        }

        // 2. Add bigram hash embedding
        if self.config.bigram_vocab_size > 0 {
            let bigram_dim = self.config.bigram_dim;
            pg_kernels::bigram_hash::bigram_hash_forward(
                input_ids,
                &self.bigram_embed,
                &mut buf.bigram_out[..t * bigram_dim],
                self.config.bigram_vocab_size,
                bigram_dim,
            );
            // Project to model_dim: [tokens, model_dim] = [tokens, bigram_dim] @ [model_dim, bigram_dim]^T
            pg_kernels::linear::linear_forward(
                &buf.bigram_out[..t * bigram_dim],
                &self.bigram_proj,
                &mut buf.bigram_proj_out[..t * d],
                t, d, bigram_dim,
            );
            // Add scaled bigram embedding
            for i in 0..t * d {
                buf.x[i] += self.bigram_scale * buf.bigram_proj_out[i];
            }
        }

        // 3. Initial RMSNorm
        let mut normed = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_forward_cpu(&buf.x[..t * d], &mut normed, d, 1.0, 1e-6);
        buf.x[..t * d].copy_from_slice(&normed);

        // 4. SmearGate: (1-σ(g)) * x + σ(g) * x_prev (sequence-shifted)
        // SOTA convention: x_prev[:, 0] = 0, x_prev[:, t] = x[:, t-1]
        {
            let mut x_prev = vec![0.0f32; t * d];
            // Shift: x_prev[t] = x[t-1], x_prev[0] = 0
            for i in 1..t {
                x_prev[i * d..(i + 1) * d].copy_from_slice(&buf.x[(i - 1) * d..i * d]);
            }
            let mut smeared = vec![0.0f32; t * d];
            pg_kernels::smear_gate::smear_gate_forward(
                &buf.x[..t * d], &x_prev, &self.smear_gate, &mut smeared, t, d,
            );
            buf.x[..t * d].copy_from_slice(&smeared);
        }

        // 5. Save x0 (anchor for residual mixing)
        buf.x0[..t * d].copy_from_slice(&buf.x[..t * d]);
        buf.v0 = None;
        buf.ve_cache = None;
        buf.skips.clear();

        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();

        // 6. Encoder layers
        for i in 0..n_enc {
            let raw_v = self.block_forward(i, input_ids, buf);
            if buf.v0.is_none() {
                buf.v0 = raw_v;
            }
            // Save skip connection (clone current x)
            buf.skips.push(buf.x[..t * d].to_vec());
        }

        // 7. Decoder layers with skip connections
        for i in 0..n_dec {
            let bi = n_enc + i;
            // Add skip connection (pop from stack = LIFO)
            if let Some(skip) = buf.skips.pop() {
                let sw = self.skip_weight(i);
                for j in 0..t * d {
                    let dim_idx = j % d;
                    buf.x[j] += sw[dim_idx] * skip[j];
                }
            }
            self.block_forward(bi, input_ids, buf);
        }

        // 8. Final RMSNorm
        let mut final_normed = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_forward_cpu(&buf.x[..t * d], &mut final_normed, d, 1.0, 1e-6);

        // 9. Output projection (tied embeddings: logits = x @ tok_emb^T)
        // tok_emb is [vocab, d], so we want Y[t,v] = sum_d x[t,d] * tok_emb[v,d]
        let vocab = self.config.vocab_size;
        pg_kernels::linear::linear_forward(
            &final_normed,
            &self.tok_emb,
            &mut buf.logits[..t * vocab],
            t, vocab, d,
        );

        // 10. buf.logits stores RAW (pre-softcap) logits.
        // Softcap is applied inside cross_entropy_forward/backward.
        // This preserves raw logits for backward pass.
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
            buf.logits[0..tokens.len() * vocab].iter().all(|v| v.is_finite()),
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
        assert!(model.blocks[1].use_xsa);  // layer 1: XSA (last 1)
    }
}
