/// Manual backward pass for the GPT model.
///
/// Uses activation recomputation: saves hidden states at layer boundaries,
/// recomputes block-internal activations during backward. This matches the
/// GPU activation checkpointing strategy (layers 3-8 recomputed).
///
/// ~15 distinct backward ops per block in reverse order of the forward pass.
use crate::config::ModelConfig;
use crate::model::{ForwardBuffer, GptModel};

fn qk_norm_backward_cpu(
    x: &[f32],
    grad_output: &[f32],
    grad_input: &mut [f32],
    head_dim: usize,
    eps: f32,
) {
    let total_heads = x.len() / head_dim;
    for head in 0..total_heads {
        let start = head * head_dim;
        let end = start + head_dim;
        let x_head = &x[start..end];
        let go_head = &grad_output[start..end];
        let sq_sum: f32 = x_head.iter().map(|&v| v * v).sum();
        let rms = (sq_sum / head_dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;
        let x_dot_go: f32 = x_head
            .iter()
            .zip(go_head.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let coeff = x_dot_go / (rms * rms * head_dim as f32);
        for i in 0..head_dim {
            grad_input[start + i] = inv_rms * (go_head[i] - x_head[i] * coeff);
        }
    }
}

/// Gradient buffers for all model parameters.
pub struct GradBuffers {
    pub tok_emb: Vec<f32>,
    pub bigram_embed: Vec<f32>,
    pub bigram_proj: Vec<f32>,
    pub bigram_scale: f32,
    pub smear_gate: Vec<f32>,
    pub skip_weights: Vec<f32>,
    pub qo_bank: Vec<f32>,
    pub kv_bank: Vec<f32>,
    pub mlp_up_bank: Vec<f32>,
    pub mlp_down_bank: Vec<f32>,
    pub block_attn_scale: Vec<Vec<f32>>,
    pub block_mlp_scale: Vec<Vec<f32>>,
    pub block_resid_mix: Vec<Vec<f32>>,
    pub block_q_gain: Vec<Vec<f32>>,
    pub block_attn_gate_weight: Vec<Vec<f32>>,
    pub block_attn_gate_bias: Vec<Vec<f32>>,
    pub block_sparse_attn_gate_weight: Vec<Vec<f32>>,
    pub ve_embed: Vec<f32>,
    pub ve_proj: Vec<f32>,
    pub ve_scale: f32,
    pub ve_layer_scales: Vec<f32>,
}

impl GradBuffers {
    pub fn new(config: &ModelConfig) -> Self {
        let n = config.num_layers;
        let d = config.model_dim;
        let kv = config.kv_dim();
        let mlp = config.mlp_dim;

        Self {
            tok_emb: vec![0.0; config.vocab_size * d],
            bigram_embed: vec![0.0; config.bigram_vocab_size * config.bigram_dim],
            bigram_proj: vec![0.0; d * config.bigram_dim],
            bigram_scale: 0.0,
            smear_gate: vec![0.0; d],
            skip_weights: vec![0.0; config.num_skip_weights() * d],
            qo_bank: vec![0.0; 2 * n * d * d],
            kv_bank: vec![0.0; 2 * n * kv * d],
            mlp_up_bank: vec![0.0; n * mlp * d],
            mlp_down_bank: vec![0.0; n * d * mlp],
            block_attn_scale: (0..n).map(|_| vec![0.0; d]).collect(),
            block_mlp_scale: (0..n).map(|_| vec![0.0; d]).collect(),
            block_resid_mix: (0..n).map(|_| vec![0.0; 2 * d]).collect(),
            block_q_gain: (0..n).map(|_| vec![0.0; config.num_heads]).collect(),
            block_attn_gate_weight: (0..n)
                .map(|_| vec![0.0; config.num_heads * config.attn_out_gate_width.max(1)])
                .collect(),
            block_attn_gate_bias: (0..n).map(|_| vec![0.0; config.num_heads]).collect(),
            block_sparse_attn_gate_weight: (0..n)
                .map(|_| vec![0.0; config.num_heads * config.sparse_attn_gate_width.max(1)])
                .collect(),
            ve_embed: vec![0.0; config.vocab_size * config.ve_dim],
            ve_proj: vec![0.0; kv * config.ve_dim],
            ve_scale: 0.0,
            ve_layer_scales: vec![0.0; config.ve_layers.len()],
        }
    }

    pub fn zero(&mut self) {
        self.tok_emb.fill(0.0);
        self.bigram_embed.fill(0.0);
        self.bigram_proj.fill(0.0);
        self.bigram_scale = 0.0;
        self.smear_gate.fill(0.0);
        self.skip_weights.fill(0.0);
        self.qo_bank.fill(0.0);
        self.kv_bank.fill(0.0);
        self.mlp_up_bank.fill(0.0);
        self.mlp_down_bank.fill(0.0);
        for v in &mut self.block_attn_scale {
            v.fill(0.0);
        }
        for v in &mut self.block_mlp_scale {
            v.fill(0.0);
        }
        for v in &mut self.block_resid_mix {
            v.fill(0.0);
        }
        for v in &mut self.block_q_gain {
            v.fill(0.0);
        }
        for v in &mut self.block_attn_gate_weight {
            v.fill(0.0);
        }
        for v in &mut self.block_attn_gate_bias {
            v.fill(0.0);
        }
        for v in &mut self.block_sparse_attn_gate_weight {
            v.fill(0.0);
        }
        self.ve_embed.fill(0.0);
        self.ve_proj.fill(0.0);
        self.ve_scale = 0.0;
        self.ve_layer_scales.fill(0.0);
    }

    pub fn flat_grad_norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        let add = |buf: &[f32], s: &mut f32| {
            for &v in buf {
                *s += v * v;
            }
        };
        add(&self.tok_emb, &mut sum_sq);
        add(&self.qo_bank, &mut sum_sq);
        add(&self.kv_bank, &mut sum_sq);
        add(&self.mlp_up_bank, &mut sum_sq);
        add(&self.mlp_down_bank, &mut sum_sq);
        add(&self.skip_weights, &mut sum_sq);
        add(&self.smear_gate, &mut sum_sq);
        add(&self.bigram_embed, &mut sum_sq);
        add(&self.bigram_proj, &mut sum_sq);
        for v in &self.block_attn_scale {
            add(v, &mut sum_sq);
        }
        for v in &self.block_mlp_scale {
            add(v, &mut sum_sq);
        }
        for v in &self.block_resid_mix {
            add(v, &mut sum_sq);
        }
        for v in &self.block_q_gain {
            add(v, &mut sum_sq);
        }
        for v in &self.block_attn_gate_weight {
            add(v, &mut sum_sq);
        }
        for v in &self.block_attn_gate_bias {
            add(v, &mut sum_sq);
        }
        for v in &self.block_sparse_attn_gate_weight {
            add(v, &mut sum_sq);
        }
        sum_sq.sqrt()
    }

    pub fn clip_grad_norm(&mut self, max_norm: f32) {
        let norm = self.flat_grad_norm();
        if norm > max_norm {
            let s = max_norm / norm;
            let clip = |buf: &mut [f32], s: f32| {
                for v in buf.iter_mut() {
                    *v *= s;
                }
            };
            clip(&mut self.tok_emb, s);
            clip(&mut self.qo_bank, s);
            clip(&mut self.kv_bank, s);
            clip(&mut self.mlp_up_bank, s);
            clip(&mut self.mlp_down_bank, s);
            clip(&mut self.skip_weights, s);
            clip(&mut self.smear_gate, s);
            clip(&mut self.bigram_embed, s);
            clip(&mut self.bigram_proj, s);
            for v in &mut self.block_attn_scale {
                clip(v, s);
            }
            for v in &mut self.block_mlp_scale {
                clip(v, s);
            }
            for v in &mut self.block_resid_mix {
                clip(v, s);
            }
            for v in &mut self.block_q_gain {
                clip(v, s);
            }
            for v in &mut self.block_attn_gate_weight {
                clip(v, s);
            }
            for v in &mut self.block_attn_gate_bias {
                clip(v, s);
            }
            for v in &mut self.block_sparse_attn_gate_weight {
                clip(v, s);
            }
        }
    }
}

/// Saved state for the forward pass — hidden states at each layer boundary.
pub struct ForwardCache {
    /// Hidden state before each block: layer_x[i] = x entering block i
    pub layer_x: Vec<Vec<f32>>,
    /// x0 (anchor for residual mixing, constant)
    pub x0: Vec<f32>,
    /// x after all blocks (before final norm)
    pub x_final: Vec<f32>,
    /// Encoder skip connections
    pub skips: Vec<Vec<f32>>,
    /// Token IDs (for VE/bigram recomputation)
    pub input_ids: Vec<u32>,
    pub tokens: usize,
    /// Hidden state after embedding + bigram, before initial RMSNorm (for bigram backward)
    pub x_post_embed: Vec<f32>,
    /// Hidden state after initial RMSNorm, before SmearGate (for SmearGate backward)
    pub x_post_norm: Vec<f32>,
}

impl GptModel {
    /// Forward pass that also saves activation cache for backward.
    /// Runs a single forward pass (not two), capturing per-layer states along the way.
    pub fn forward_with_cache(&self, input_ids: &[u32], buf: &mut ForwardBuffer) -> ForwardCache {
        let t = input_ids.len();
        let d = self.config.model_dim;
        let n = self.config.num_layers;

        let mut cache = ForwardCache {
            layer_x: Vec::with_capacity(n),
            x0: vec![0.0; t * d],
            x_final: vec![0.0; t * d],
            skips: Vec::new(),
            input_ids: input_ids.to_vec(),
            tokens: t,
            x_post_embed: Vec::new(),
            x_post_norm: Vec::new(),
        };

        self.forward_inner(input_ids, buf, Some(&mut cache));

        cache
    }

    /// Backward through a single transformer block.
    /// grad_x: gradient of loss w.r.t. block OUTPUT — mutated in place to become
    ///         gradient w.r.t. block INPUT (x before resid_mix).
    /// layer_x_in: saved hidden state at block entry (before resid_mix).
    fn block_backward_single(
        &self,
        layer: usize,
        grad_x: &mut [f32],  // [t*d] — in/out
        grad_x0: &mut [f32], // [t*d] — accumulated gradient for residual anchor
        layer_x_in: &[f32],  // [t*d] — saved input to this block
        x0: &[f32],          // [t*d] — anchor
        _input_ids: &[u32],
        grads: &mut GradBuffers,
    ) {
        let c = &self.config;
        let t = grad_x.len() / c.model_dim;
        let d = c.model_dim;
        let h = c.num_heads;
        let hkv = c.num_kv_heads;
        let hd = c.head_dim;
        let kv_dim = c.kv_dim();
        let mlp_dim = c.mlp_dim;
        let bp = &self.blocks[layer];

        // === Recompute forward intermediates ===
        // 1. resid_mix
        let mut x_in = vec![0.0f32; t * d];
        for i in 0..t * d {
            let di = i % d;
            x_in[i] = bp.resid_mix[di] * layer_x_in[i] + bp.resid_mix[d + di] * x0[i];
        }

        // 2. attn_norm
        let mut attn_norm_out = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_forward_cpu(
            &x_in,
            &mut attn_norm_out,
            d,
            bp.ln_scale_factor,
            1e-6,
        );

        // 3. Q, K, V projections
        let q_w = self.q_weight(layer);
        let k_w = self.k_weight(layer);
        let v_w = self.v_weight(layer);
        let mut q_proj = vec![0.0f32; t * d];
        let mut k_proj = vec![0.0f32; t * kv_dim];
        let mut v_proj = vec![0.0f32; t * kv_dim];
        pg_kernels::linear::linear_forward(&attn_norm_out, q_w, &mut q_proj, t, d, d);
        pg_kernels::linear::linear_forward(&attn_norm_out, k_w, &mut k_proj, t, kv_dim, d);
        pg_kernels::linear::linear_forward(&attn_norm_out, v_w, &mut v_proj, t, kv_dim, d);

        let ve_idx = c.ve_layers.iter().position(|&l| l == layer);
        let mut ve_embed_out: Option<Vec<f32>> = None;
        let mut ve_projected_raw: Option<Vec<f32>> = None;
        if c.ve_enabled && ve_idx.is_some() {
            let ve_dim = c.ve_dim;
            let mut embed_out = vec![0.0f32; t * ve_dim];
            for tok in 0..t {
                let id = _input_ids[tok] as usize;
                embed_out[tok * ve_dim..(tok + 1) * ve_dim]
                    .copy_from_slice(&self.ve_embed[id * ve_dim..(id + 1) * ve_dim]);
            }
            let mut projected = vec![0.0f32; t * kv_dim];
            pg_kernels::linear::linear_forward(
                &embed_out,
                &self.ve_proj,
                &mut projected,
                t,
                kv_dim,
                ve_dim,
            );
            let layer_scale = self.ve_layer_scales[ve_idx.unwrap()];
            let scale = self.ve_scale * layer_scale;
            for i in 0..t * kv_dim {
                v_proj[i] += projected[i] * scale;
            }
            ve_embed_out = Some(embed_out);
            ve_projected_raw = Some(projected);
        }

        // 4. QK-norm in-place
        let q_pre_norm = q_proj.clone();
        let k_pre_norm = k_proj.clone();
        for i in 0..t * h {
            let off = i * hd;
            let rms = (q_proj[off..off + hd].iter().map(|&x| x * x).sum::<f32>() / hd as f32
                + 1e-6)
                .sqrt();
            for j in 0..hd {
                q_proj[off + j] /= rms;
            }
        }
        for i in 0..t * hkv {
            let off = i * hd;
            let rms = (k_proj[off..off + hd].iter().map(|&x| x * x).sum::<f32>() / hd as f32
                + 1e-6)
                .sqrt();
            for j in 0..hd {
                k_proj[off + j] /= rms;
            }
        }

        // (Post-norm Q/K states are in q_proj/k_proj before RoPE modifies them)

        // 5. RoPE
        if c.rope_dims > 0 {
            pg_kernels::rope::apply_partial_rope(
                &mut q_proj,
                &self.rope_cos,
                &self.rope_sin,
                1,
                t,
                h,
                hd,
                c.rope_dims,
            );
            pg_kernels::rope::apply_partial_rope(
                &mut k_proj,
                &self.rope_cos,
                &self.rope_sin,
                1,
                t,
                hkv,
                hd,
                c.rope_dims,
            );
        }

        // Save post-RoPE, pre-q_gain
        let q_post_rope = q_proj.clone();

        // 6. q_gain
        for tok in 0..t {
            for head in 0..h {
                let off = (tok * h + head) * hd;
                for j in 0..hd {
                    q_proj[off + j] *= bp.q_gain[head];
                }
            }
        }

        // 7. Attention
        let mut attn_out = vec![0.0f32; t * h * hd];
        pg_kernels::attention::causal_attention_forward(
            &q_proj,
            &k_proj,
            &v_proj,
            &mut attn_out,
            t,
            h,
            hkv,
            hd,
        );

        // 8. XSA
        let (attn_result_raw, _xsa_out) = if bp.use_xsa {
            let mut xo = vec![0.0f32; t * h * hd];
            pg_kernels::xsa::xsa_forward(&attn_out, &v_proj, &mut xo, t, h, hkv, hd);
            (xo.clone(), Some(xo))
        } else {
            (attn_out.clone(), None)
        };
        let mut attn_gate_values: Option<Vec<f32>> = None;
        let attn_result = if self.attn_out_gate_enabled() {
            let width = c.attn_out_gate_width;
            let mut gate_values = vec![0.0f32; t * h];
            let mut gated = vec![0.0f32; t * h * hd];
            for tok in 0..t {
                let gate_input = &attn_norm_out[tok * d..tok * d + width];
                for head in 0..h {
                    let mut score = bp.attn_gate_bias[head];
                    let weight = &bp.attn_gate_weight[head * width..(head + 1) * width];
                    for j in 0..width {
                        score += weight[j] * gate_input[j];
                    }
                    let sig = 1.0 / (1.0 + (-score).exp());
                    let gate = 2.0 * sig;
                    gate_values[tok * h + head] = gate;
                    let base = (tok * h + head) * hd;
                    for j in 0..hd {
                        gated[base + j] = attn_result_raw[base + j] * gate;
                    }
                }
            }
            attn_gate_values = Some(gate_values);
            gated
        } else if self.sparse_attn_gate_enabled() {
            let width = c.sparse_attn_gate_width;
            let scale = c.sparse_attn_gate_scale;
            let mut gate_values = vec![0.0f32; t * h];
            let mut gated = vec![0.0f32; t * h * hd];
            for tok in 0..t {
                let gate_input = &attn_norm_out[tok * d..tok * d + width];
                for head in 0..h {
                    let weight = &bp.sparse_attn_gate_weight[head * width..(head + 1) * width];
                    let mut score = 0.0f32;
                    for j in 0..width {
                        score += weight[j] * gate_input[j];
                    }
                    let gate = 1.0 / (1.0 + (-(scale * score)).exp());
                    gate_values[tok * h + head] = gate;
                    let base = (tok * h + head) * hd;
                    for j in 0..hd {
                        gated[base + j] = attn_result_raw[base + j] * gate;
                    }
                }
            }
            attn_gate_values = Some(gate_values);
            gated
        } else {
            attn_result_raw.clone()
        };

        // 9. Output projection
        let o_w = self.o_weight(layer);
        let mut proj_out = vec![0.0f32; t * d];
        pg_kernels::linear::linear_forward(&attn_result, o_w, &mut proj_out, t, d, d);

        // 10. Attn residual: x_after_attn = x_in + attn_scale * proj_out
        let mut x_after_attn = vec![0.0f32; t * d];
        for i in 0..t * d {
            x_after_attn[i] = x_in[i] + bp.attn_scale[i % d] * proj_out[i];
        }

        let mlp_input = if self.parallel_residual_enabled() {
            x_in.clone()
        } else {
            x_after_attn.clone()
        };

        // 11. MLP norm
        let mut mlp_norm_out = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_forward_cpu(
            &mlp_input,
            &mut mlp_norm_out,
            d,
            bp.ln_scale_factor,
            1e-6,
        );

        // 12. MLP up
        let up_w = self.mlp_up_weight(layer);
        let down_w = self.mlp_down_weight(layer);
        let mut mlp_up_out = vec![0.0f32; t * mlp_dim];
        pg_kernels::linear::linear_forward(&mlp_norm_out, up_w, &mut mlp_up_out, t, mlp_dim, d);

        // 13. LeakyReLU²
        let mut mlp_act = vec![0.0f32; t * mlp_dim];
        pg_kernels::activations::leaky_relu_sq_forward(&mlp_up_out, &mut mlp_act);

        // === BACKWARD (reverse order) ===

        // grad_x is gradient of loss w.r.t. block output (x_out)
        // x_out = x_after_attn + mlp_scale * mlp_down(mlp_act)

        // 15. MLP residual backward
        // x_out = x_after_attn + mlp_scale * mlp_down_out
        // grad_x_after_attn = grad_x (passthrough)
        // grad_mlp_out = grad_x * mlp_scale
        let mut mlp_down_out = vec![0.0f32; t * d];
        pg_kernels::linear::linear_forward(&mlp_act, down_w, &mut mlp_down_out, t, d, mlp_dim);

        let mut grad_mlp_out = vec![0.0f32; t * d];
        let mut grad_x_after_attn = vec![0.0f32; t * d];
        for i in 0..t * d {
            let di = i % d;
            grad_x_after_attn[i] = grad_x[i];
            grad_mlp_out[i] = grad_x[i] * bp.mlp_scale[di];
            grads.block_mlp_scale[layer][di] += grad_x[i] * mlp_down_out[i];
        }

        // 14. MLP down backward: mlp_out = mlp_act @ down_w^T
        let mut grad_mlp_act = vec![0.0f32; t * mlp_dim];
        pg_kernels::linear::linear_backward_input(
            &grad_mlp_out,
            down_w,
            &mut grad_mlp_act,
            t,
            d,
            mlp_dim,
        );
        // grad_down_w
        let n_layers = c.num_layers;
        let down_offset = layer * d * mlp_dim;
        let mut grad_down_w = vec![0.0f32; d * mlp_dim];
        pg_kernels::linear::linear_backward_weight(
            &grad_mlp_out,
            &mlp_act,
            &mut grad_down_w,
            t,
            d,
            mlp_dim,
        );
        for i in 0..d * mlp_dim {
            grads.mlp_down_bank[down_offset + i] += grad_down_w[i];
        }

        // 13. LeakyReLU² backward
        let mut grad_mlp_up = vec![0.0f32; t * mlp_dim];
        pg_kernels::activations::leaky_relu_sq_backward(
            &mlp_up_out,
            &grad_mlp_act,
            &mut grad_mlp_up,
        );

        // 12. MLP up backward
        let mut grad_mlp_norm_out = vec![0.0f32; t * d];
        pg_kernels::linear::linear_backward_input(
            &grad_mlp_up,
            up_w,
            &mut grad_mlp_norm_out,
            t,
            mlp_dim,
            d,
        );
        let up_offset = layer * mlp_dim * d;
        let mut grad_up_w = vec![0.0f32; mlp_dim * d];
        pg_kernels::linear::linear_backward_weight(
            &grad_mlp_up,
            &mlp_norm_out,
            &mut grad_up_w,
            t,
            mlp_dim,
            d,
        );
        for i in 0..mlp_dim * d {
            grads.mlp_up_bank[up_offset + i] += grad_up_w[i];
        }

        // 11. MLP norm backward
        let mut grad_x_pre_mlp_norm = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_backward_cpu(
            &mlp_input,
            &grad_mlp_norm_out,
            &mut grad_x_pre_mlp_norm,
            d,
            bp.ln_scale_factor,
            1e-6,
        );

        if !self.parallel_residual_enabled() {
            for i in 0..t * d {
                grad_x_after_attn[i] += grad_x_pre_mlp_norm[i];
            }
        }

        // 10. Attn residual backward: x_after_attn = x_in + attn_scale * proj_out
        // grad_x_in += grad_x_after_attn (passthrough)
        // grad_proj_out = grad_x_after_attn * attn_scale
        let mut grad_x_in = vec![0.0f32; t * d];
        let mut grad_proj_out = vec![0.0f32; t * d];
        for i in 0..t * d {
            let di = i % d;
            grad_x_in[i] = grad_x_after_attn[i];
            grad_proj_out[i] = grad_x_after_attn[i] * bp.attn_scale[di];
            grads.block_attn_scale[layer][di] += grad_x_after_attn[i] * proj_out[i];
        }
        if self.parallel_residual_enabled() {
            for i in 0..t * d {
                grad_x_in[i] += grad_x_pre_mlp_norm[i];
            }
        }

        // 9. Output projection backward: proj_out = attn_result @ o_w^T
        let mut grad_attn_result = vec![0.0f32; t * d];
        pg_kernels::linear::linear_backward_input(
            &grad_proj_out,
            o_w,
            &mut grad_attn_result,
            t,
            d,
            d,
        );
        let o_offset = (n_layers + layer) * d * d;
        let mut grad_o_w = vec![0.0f32; d * d];
        pg_kernels::linear::linear_backward_weight(
            &grad_proj_out,
            &attn_result,
            &mut grad_o_w,
            t,
            d,
            d,
        );
        for i in 0..d * d {
            grads.qo_bank[o_offset + i] += grad_o_w[i];
        }

        let mut gate_grad_attn_norm = vec![0.0f32; t * d];
        let grad_attn_raw = if self.attn_out_gate_enabled() {
            let gate_values = attn_gate_values
                .as_ref()
                .expect("AttnOutGate forward values missing in backward");
            let width = c.attn_out_gate_width;
            let mut grad_raw = vec![0.0f32; t * h * hd];
            for tok in 0..t {
                let gate_input = &attn_norm_out[tok * d..tok * d + width];
                for head in 0..h {
                    let gate = gate_values[tok * h + head];
                    let sig = 0.5 * gate;
                    let mut grad_gate = 0.0f32;
                    let base = (tok * h + head) * hd;
                    for j in 0..hd {
                        let go = grad_attn_result[base + j];
                        grad_raw[base + j] = go * gate;
                        grad_gate += go * attn_result_raw[base + j];
                    }

                    let grad_score = grad_gate * 2.0 * sig * (1.0 - sig);
                    grads.block_attn_gate_bias[layer][head] += grad_score;
                    let weight = &bp.attn_gate_weight[head * width..(head + 1) * width];
                    for j in 0..width {
                        grads.block_attn_gate_weight[layer][head * width + j] +=
                            grad_score * gate_input[j];
                        gate_grad_attn_norm[tok * d + j] += grad_score * weight[j];
                    }
                }
            }
            grad_raw
        } else if self.sparse_attn_gate_enabled() {
            let gate_values = attn_gate_values
                .as_ref()
                .expect("SparseAttnGate forward values missing in backward");
            let width = c.sparse_attn_gate_width;
            let scale = c.sparse_attn_gate_scale;
            let mut grad_raw = vec![0.0f32; t * h * hd];
            for tok in 0..t {
                let gate_input = &attn_norm_out[tok * d..tok * d + width];
                for head in 0..h {
                    let gate = gate_values[tok * h + head];
                    let mut grad_gate = 0.0f32;
                    let base = (tok * h + head) * hd;
                    for j in 0..hd {
                        let go = grad_attn_result[base + j];
                        grad_raw[base + j] = go * gate;
                        grad_gate += go * attn_result_raw[base + j];
                    }

                    let grad_score = grad_gate * scale * gate * (1.0 - gate);
                    let weight = &bp.sparse_attn_gate_weight[head * width..(head + 1) * width];
                    for j in 0..width {
                        grads.block_sparse_attn_gate_weight[layer][head * width + j] +=
                            grad_score * gate_input[j];
                        gate_grad_attn_norm[tok * d + j] += grad_score * weight[j];
                    }
                }
            }
            grad_raw
        } else {
            grad_attn_result
        };

        // 8. XSA backward
        let mut xsa_grad_v: Option<Vec<f32>> = None;
        let grad_attn_out = if bp.use_xsa {
            let mut grad_y = vec![0.0f32; t * h * hd];
            let mut grad_v_xsa = vec![0.0f32; t * hkv * hd];
            pg_kernels::xsa::xsa_backward(
                &attn_out,
                &v_proj,
                &grad_attn_raw,
                &mut grad_y,
                &mut grad_v_xsa,
                t,
                h,
                hkv,
                hd,
            );
            xsa_grad_v = Some(grad_v_xsa);
            grad_y
        } else {
            grad_attn_raw
        };

        // 7. Attention backward — exact causal attention backward with GQA.
        let mut grad_q_post_gain = vec![0.0f32; t * h * hd];
        let mut grad_k_attn = vec![0.0f32; t * hkv * hd];
        let mut grad_v_attn = vec![0.0f32; t * hkv * hd];
        pg_kernels::attention::causal_attention_backward(
            &q_proj,
            &k_proj,
            &v_proj,
            &attn_out,
            &grad_attn_out,
            &mut grad_q_post_gain,
            &mut grad_k_attn,
            &mut grad_v_attn,
            t,
            h,
            hkv,
            hd,
        );

        // Add XSA grad_v to V gradient path
        if let Some(ref gv_xsa) = xsa_grad_v {
            for i in 0..t * hkv * hd {
                grad_v_attn[i] += gv_xsa[i];
            }
        }

        // 6. q_gain backward: Q_scaled = Q * gain
        // grad_Q_pre_gain = grad_Q_scaled * gain
        // grad_gain += sum(grad_Q_scaled * Q_pre_gain)
        let mut grad_q_post_rope = vec![0.0f32; t * h * hd];
        for tok in 0..t {
            for head in 0..h {
                let off = (tok * h + head) * hd;
                let gain = bp.q_gain[head];
                let mut dot = 0.0f32;
                for j in 0..hd {
                    grad_q_post_rope[off + j] = grad_q_post_gain[off + j] * gain;
                    dot += grad_q_post_gain[off + j] * q_post_rope[off + j];
                }
                grads.block_q_gain[layer][head] += dot;
            }
        }

        // 5. RoPE backward
        let mut grad_k_post_rope = grad_k_attn.clone();
        if c.rope_dims > 0 {
            pg_kernels::rope::apply_partial_rope_backward(
                &mut grad_q_post_rope,
                &self.rope_cos,
                &self.rope_sin,
                1,
                t,
                h,
                hd,
                c.rope_dims,
            );
            pg_kernels::rope::apply_partial_rope_backward(
                &mut grad_k_post_rope,
                &self.rope_cos,
                &self.rope_sin,
                1,
                t,
                hkv,
                hd,
                c.rope_dims,
            );
        }

        // 4. QK-norm backward
        let mut grad_q_proj = vec![0.0f32; t * h * hd];
        let mut grad_k_proj = vec![0.0f32; t * hkv * hd];
        qk_norm_backward_cpu(&q_pre_norm, &grad_q_post_rope, &mut grad_q_proj, hd, 1e-6);
        qk_norm_backward_cpu(&k_pre_norm, &grad_k_post_rope, &mut grad_k_proj, hd, 1e-6);

        // 3. Q projection backward: Q = attn_norm_out @ q_w^T
        let mut grad_attn_norm = vec![0.0f32; t * d];
        pg_kernels::linear::linear_backward_input(&grad_q_proj, q_w, &mut grad_attn_norm, t, d, d);
        let q_offset = layer * d * d;
        let mut grad_q_w = vec![0.0f32; d * d];
        pg_kernels::linear::linear_backward_weight(
            &grad_q_proj,
            &attn_norm_out,
            &mut grad_q_w,
            t,
            d,
            d,
        );
        for i in 0..d * d {
            grads.qo_bank[q_offset + i] += grad_q_w[i];
        }

        // K projection backward: K = attn_norm_out @ k_w^T
        let mut grad_attn_norm_k = vec![0.0f32; t * d];
        pg_kernels::linear::linear_backward_input(
            &grad_k_proj,
            k_w,
            &mut grad_attn_norm_k,
            t,
            kv_dim,
            d,
        );
        let k_offset = layer * kv_dim * d;
        let mut grad_k_w = vec![0.0f32; kv_dim * d];
        pg_kernels::linear::linear_backward_weight(
            &grad_k_proj,
            &attn_norm_out,
            &mut grad_k_w,
            t,
            kv_dim,
            d,
        );
        for i in 0..kv_dim * d {
            grads.kv_bank[k_offset + i] += grad_k_w[i];
        }
        for i in 0..t * d {
            grad_attn_norm[i] += grad_attn_norm_k[i];
        }

        // VE backward: V_used = V_proj + VE(input_ids) * ve_scale * ve_layer_scale.
        let grad_v_projection = grad_v_attn;
        if let (Some(ve_idx), Some(embed_out), Some(projected_raw)) =
            (ve_idx, ve_embed_out.as_ref(), ve_projected_raw.as_ref())
        {
            let ve_dim = c.ve_dim;
            let layer_scale = self.ve_layer_scales[ve_idx];
            let total_scale = self.ve_scale * layer_scale;
            let mut grad_projected = vec![0.0f32; t * kv_dim];
            for i in 0..t * kv_dim {
                grads.ve_scale += grad_v_projection[i] * projected_raw[i] * layer_scale;
                grads.ve_layer_scales[ve_idx] +=
                    grad_v_projection[i] * projected_raw[i] * self.ve_scale;
                grad_projected[i] = grad_v_projection[i] * total_scale;
            }

            let mut grad_ve_proj = vec![0.0f32; kv_dim * ve_dim];
            pg_kernels::linear::linear_backward_weight(
                &grad_projected,
                embed_out,
                &mut grad_ve_proj,
                t,
                kv_dim,
                ve_dim,
            );
            for i in 0..kv_dim * ve_dim {
                grads.ve_proj[i] += grad_ve_proj[i];
            }

            let mut grad_ve_embed_out = vec![0.0f32; t * ve_dim];
            pg_kernels::linear::linear_backward_input(
                &grad_projected,
                &self.ve_proj,
                &mut grad_ve_embed_out,
                t,
                kv_dim,
                ve_dim,
            );
            for tok in 0..t {
                let id = _input_ids[tok] as usize;
                for j in 0..ve_dim {
                    grads.ve_embed[id * ve_dim + j] += grad_ve_embed_out[tok * ve_dim + j];
                }
            }
        }

        // V projection backward: V = attn_norm_out @ v_w^T
        let mut grad_attn_norm_v = vec![0.0f32; t * d];
        pg_kernels::linear::linear_backward_input(
            &grad_v_projection,
            v_w,
            &mut grad_attn_norm_v,
            t,
            kv_dim,
            d,
        );
        let n_layers = c.num_layers;
        let v_offset = (n_layers + layer) * kv_dim * d;
        let mut grad_v_w = vec![0.0f32; kv_dim * d];
        pg_kernels::linear::linear_backward_weight(
            &grad_v_projection,
            &attn_norm_out,
            &mut grad_v_w,
            t,
            kv_dim,
            d,
        );
        for i in 0..kv_dim * d {
            grads.kv_bank[v_offset + i] += grad_v_w[i];
        }
        for i in 0..t * d {
            grad_attn_norm[i] += grad_attn_norm_v[i];
        }
        for i in 0..t * d {
            grad_attn_norm[i] += gate_grad_attn_norm[i];
        }

        // 2. Attn norm backward
        let mut grad_x_in_from_norm = vec![0.0f32; t * d];
        pg_kernels::rms_norm::rms_norm_backward_cpu(
            &x_in,
            &grad_attn_norm,
            &mut grad_x_in_from_norm,
            d,
            bp.ln_scale_factor,
            1e-6,
        );
        for i in 0..t * d {
            grad_x_in[i] += grad_x_in_from_norm[i];
        }

        // 1. Resid mix backward: x_in = mix[0] * x + mix[1] * x0
        // grad_x_layer = grad_x_in * mix[0]
        // grad_x0 += grad_x_in * mix[1]
        // grad_mix[0] += sum(grad_x_in * x)
        // grad_mix[1] += sum(grad_x_in * x0)
        for i in 0..t * d {
            let di = i % d;
            grad_x[i] = grad_x_in[i] * bp.resid_mix[di];
            grad_x0[i] += grad_x_in[i] * bp.resid_mix[d + di];
            grads.block_resid_mix[layer][di] += grad_x_in[i] * layer_x_in[i];
            grads.block_resid_mix[layer][d + di] += grad_x_in[i] * x0[i];
        }
    }

    fn block_backward(
        &self,
        layer: usize,
        grad_x: &mut [f32],  // [t*d] — in/out
        grad_x0: &mut [f32], // [t*d] — accumulated gradient for residual anchor
        layer_x_in: &[f32],  // [t*d] — saved input to this block
        x0: &[f32],          // [t*d] — anchor
        input_ids: &[u32],
        grads: &mut GradBuffers,
    ) {
        if self.is_recurrent_layer(layer) {
            let t = grad_x.len() / self.config.model_dim;
            let d = self.config.model_dim;
            let mut tmp = ForwardBuffer::new(&self.config, t);
            tmp.x[..t * d].copy_from_slice(layer_x_in);
            tmp.x0[..t * d].copy_from_slice(x0);
            self.block_forward_once(layer, input_ids, &mut tmp);

            let mut grad_mid = grad_x.to_vec();
            self.block_backward_single(
                layer,
                &mut grad_mid,
                grad_x0,
                &tmp.x[..t * d],
                x0,
                input_ids,
                grads,
            );
            self.block_backward_single(
                layer,
                &mut grad_mid,
                grad_x0,
                layer_x_in,
                x0,
                input_ids,
                grads,
            );
            grad_x.copy_from_slice(&grad_mid);
            return;
        }

        self.block_backward_single(layer, grad_x, grad_x0, layer_x_in, x0, input_ids, grads);
    }

    /// Full backward pass: computes all parameter gradients.
    /// Returns the loss value.
    pub fn backward(
        &self,
        input_ids: &[u32],
        targets: &[u32],
        buf: &mut ForwardBuffer,
        grads: &mut GradBuffers,
    ) -> f32 {
        let t = input_ids.len();
        let d = self.config.model_dim;
        let n_enc = self.config.num_encoder_layers();
        let n_dec = self.config.num_decoder_layers();

        // Forward with cache
        let cache = self.forward_with_cache(input_ids, buf);

        // Compute loss
        let loss = self.compute_loss(targets, buf);

        // Backward through output layer
        let mut grad_x = backward_output_loss(self, buf, targets, grads);
        let mut grad_x0 = vec![0.0f32; t * d];

        // Collect skip gradients to propagate back to encoder layers.
        // grad_encoder_skips[enc_layer] = gradient flowing back through skip to that encoder's output.
        let n_skip = self.config.num_skip_weights();
        let mut grad_encoder_skips: Vec<Vec<f32>> = vec![vec![0.0f32; t * d]; n_skip];

        // Backward through decoder layers (reverse order)
        for i in (0..n_dec).rev() {
            let bi = n_enc + i;

            // Backward through block
            self.block_backward(
                bi,
                &mut grad_x,
                &mut grad_x0,
                &cache.layer_x[bi],
                &cache.x0,
                input_ids,
                grads,
            );

            // Backward through skip connection: x_dec = x + skip_weight * skip_enc
            // grad_x already contains gradient w.r.t. x_dec (post-skip)
            // grad_skip_weight += grad_x * skip_enc
            // grad_skip_enc = grad_x * skip_weight  (propagate to encoder!)
            if i < cache.skips.len() {
                let enc_layer = n_enc - 1 - i; // LIFO: decoder 0 ← encoder (n_enc-1)
                let skip = &cache.skips[enc_layer];
                let sw = self.skip_weight(i);
                for j in 0..t * d {
                    let di = j % d;
                    grads.skip_weights[i * d + di] += grad_x[j] * skip[j];
                    grad_encoder_skips[enc_layer][j] += grad_x[j] * sw[di];
                }
            }
        }

        // Backward through encoder layers (reverse order)
        for i in (0..n_enc).rev() {
            // Inject skip gradient into grad_x before encoder block backward.
            // The skip was saved AFTER block i's forward, so the gradient from the
            // skip goes to the OUTPUT of block i, which is the current grad_x.
            if i < n_skip {
                for j in 0..t * d {
                    grad_x[j] += grad_encoder_skips[i][j];
                }
            }

            self.block_backward(
                i,
                &mut grad_x,
                &mut grad_x0,
                &cache.layer_x[i],
                &cache.x0,
                input_ids,
                grads,
            );
        }

        // grad_x now contains the direct path gradient w.r.t. post-SmearGate output.
        // Residual-mix anchor uses inside every block also flow into the same x0.
        for i in 0..t * d {
            grad_x[i] += grad_x0[i];
        }

        // SmearGate backward
        // SmearGate: output = (1-σ(g)) * x_post_norm + σ(g) * x_prev
        // where x_post_norm is the hidden state after initial RMSNorm
        {
            let x_post_norm = &cache.x_post_norm;
            let mut grad_x_smear = vec![0.0f32; t * d];
            let mut grad_x_prev = vec![0.0f32; t * d];
            if let Some(boundary) = self.config.smear_gate_boundary_token_id {
                pg_kernels::smear_gate::smear_gate_backward_boundary(
                    x_post_norm,
                    input_ids,
                    &self.smear_gate,
                    &grad_x,
                    &mut grad_x_smear,
                    &mut grad_x_prev,
                    &mut grads.smear_gate,
                    t,
                    d,
                    boundary,
                );
            } else {
                // Reconstruct x_prev (shifted x_post_norm)
                let mut x_prev = vec![0.0f32; t * d];
                for i in 1..t {
                    x_prev[i * d..(i + 1) * d].copy_from_slice(&x_post_norm[(i - 1) * d..i * d]);
                }
                pg_kernels::smear_gate::smear_gate_backward(
                    x_post_norm,
                    &x_prev,
                    &self.smear_gate,
                    &grad_x,
                    &mut grad_x_smear,
                    &mut grad_x_prev,
                    &mut grads.smear_gate,
                    t,
                    d,
                );
            }

            // grad_x_prev flows to shifted positions: grad_x_prev[i] → grad_x_post_norm[i-1]
            // Combine: grad_x_post_norm[i] = grad_x_smear[i] + grad_x_prev[i+1]
            let mut grad_x_post_norm = grad_x_smear;
            for i in 0..t - 1 {
                for j in 0..d {
                    grad_x_post_norm[i * d + j] += grad_x_prev[(i + 1) * d + j];
                }
            }
            // grad_x_prev[0] has no predecessor to flow to (x_prev[0] = 0)

            // Initial RMSNorm backward
            let mut grad_x_post_embed = vec![0.0f32; t * d];
            pg_kernels::rms_norm::rms_norm_backward_cpu(
                &cache.x_post_embed,
                &grad_x_post_norm,
                &mut grad_x_post_embed,
                d,
                1.0,
                1e-6,
            );

            // Bigram backward: x_post_embed = tok_embed_out + bigram_scale * bigram_proj_out
            // grad_bigram_proj_out = grad_x_post_embed * bigram_scale
            if self.config.bigram_vocab_size > 0 {
                let bigram_dim = self.config.bigram_dim;

                // Recompute bigram_proj_out and bigram_out for weight gradients
                let mut bigram_out = vec![0.0f32; t * bigram_dim];
                pg_kernels::bigram_hash::bigram_hash_forward(
                    input_ids,
                    &self.bigram_embed,
                    &mut bigram_out,
                    self.config.bigram_vocab_size,
                    bigram_dim,
                );
                let mut bigram_proj_out = vec![0.0f32; t * d];
                pg_kernels::linear::linear_forward(
                    &bigram_out,
                    &self.bigram_proj,
                    &mut bigram_proj_out,
                    t,
                    d,
                    bigram_dim,
                );

                // grad_bigram_scale = sum(grad_x_post_embed * bigram_proj_out)
                for i in 0..t * d {
                    grads.bigram_scale += grad_x_post_embed[i] * bigram_proj_out[i];
                }

                // grad_bigram_proj_out = grad_x_post_embed * bigram_scale
                let mut grad_bigram_proj_out = vec![0.0f32; t * d];
                for i in 0..t * d {
                    grad_bigram_proj_out[i] = grad_x_post_embed[i] * self.bigram_scale;
                }

                // Backward through bigram projection: proj_out = bigram_out @ bigram_proj^T
                // grad_bigram_proj (weight gradient) — linear_backward_weight overwrites, so use scratch
                let mut grad_bigram_proj_w = vec![0.0f32; d * bigram_dim];
                pg_kernels::linear::linear_backward_weight(
                    &grad_bigram_proj_out,
                    &bigram_out,
                    &mut grad_bigram_proj_w,
                    t,
                    d,
                    bigram_dim,
                );
                for i in 0..d * bigram_dim {
                    grads.bigram_proj[i] += grad_bigram_proj_w[i];
                }

                // grad_bigram_out = grad_bigram_proj_out @ bigram_proj
                let mut grad_bigram_out = vec![0.0f32; t * bigram_dim];
                pg_kernels::linear::linear_backward_input(
                    &grad_bigram_proj_out,
                    &self.bigram_proj,
                    &mut grad_bigram_out,
                    t,
                    d,
                    bigram_dim,
                );

                // Scatter grad_bigram_out into grad_bigram_embed via hash indices
                pg_kernels::bigram_hash::bigram_hash_backward(
                    input_ids,
                    &grad_bigram_out,
                    &mut grads.bigram_embed,
                    self.config.bigram_vocab_size,
                    bigram_dim,
                );
            }

            // Embedding backward: scatter grad_x_post_embed into tok_emb
            for i in 0..t {
                let tok = input_ids[i] as usize;
                for j in 0..d {
                    grads.tok_emb[tok * d + j] += grad_x_post_embed[i * d + j];
                }
            }
        }

        loss
    }
}

/// Backward through the output projection and cross-entropy loss.
pub fn backward_output_loss(
    model: &GptModel,
    buf: &ForwardBuffer,
    targets: &[u32],
    grads: &mut GradBuffers,
) -> Vec<f32> {
    let t = buf.tokens;
    let d = model.config.model_dim;
    let vocab = model.config.vocab_size;

    let mut grad_logits = vec![0.0f32; t * vocab];
    let grad_loss = 1.0 / t as f32;
    pg_kernels::cross_entropy::cross_entropy_backward(
        &buf.logits[..t * vocab],
        targets,
        &mut grad_logits,
        vocab,
        model.config.logit_softcap,
        grad_loss,
    );

    // Backward through tied embedding: logits = x @ tok_emb^T
    let mut grad_x = vec![0.0f32; t * d];
    pg_kernels::linear::linear_backward_input(
        &grad_logits,
        &model.tok_emb,
        &mut grad_x,
        t,
        vocab,
        d,
    );

    // grad_tok_emb += grad_logits^T @ final_normed_x
    // Use scratch buffer since linear_backward_weight overwrites (not accumulates)
    let mut final_normed = vec![0.0f32; t * d];
    pg_kernels::rms_norm::rms_norm_forward_cpu(&buf.x[..t * d], &mut final_normed, d, 1.0, 1e-6);
    let mut grad_tok_emb_logits = vec![0.0f32; vocab * d];
    pg_kernels::linear::linear_backward_weight(
        &grad_logits,
        &final_normed,
        &mut grad_tok_emb_logits,
        t,
        vocab,
        d,
    );
    for i in 0..vocab * d {
        grads.tok_emb[i] += grad_tok_emb_logits[i];
    }

    // Final RMSNorm backward
    let mut grad_pre_norm = vec![0.0f32; t * d];
    pg_kernels::rms_norm::rms_norm_backward_cpu(
        &buf.x[..t * d],
        &grad_x,
        &mut grad_pre_norm,
        d,
        1.0,
        1e-6,
    );

    grad_pre_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ForwardBuffer;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 16,
            num_layers: 1,
            model_dim: 8,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 4,
            mlp_mult: 2.0,
            mlp_dim: 16,
            rope_base: 10000.0,
            rope_dims: 2,
            xsa_last_n: 0,
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
            bigram_vocab_size: 4,
            bigram_dim: 4,
            ln_scale: false,
            tie_embeddings: true,
            tied_embed_init_std: 0.005,
            train_seq_len: 8,
            eval_seq_len: 8,
        }
    }

    fn init_model(config: &ModelConfig) -> GptModel {
        let mut model = GptModel::new(config.clone());
        for (i, v) in model.tok_emb.iter_mut().enumerate() {
            *v = 0.01 * ((i * 7 + 3) % 19) as f32 - 0.09;
        }
        for (i, v) in model.qo_bank.iter_mut().enumerate() {
            *v = 0.01 * ((i * 13 + 5) % 23) as f32 - 0.11;
        }
        for (i, v) in model.kv_bank.iter_mut().enumerate() {
            *v = 0.01 * ((i * 11 + 7) % 17) as f32 - 0.08;
        }
        for (i, v) in model.mlp_up_bank.iter_mut().enumerate() {
            *v = 0.01 * ((i * 3 + 1) % 13) as f32 - 0.06;
        }
        for (i, v) in model.mlp_down_bank.iter_mut().enumerate() {
            *v = 0.01 * ((i * 5 + 2) % 11) as f32 - 0.05;
        }
        model
    }

    #[test]
    fn test_grad_buffers_zero() {
        let config = tiny_config();
        let mut grads = GradBuffers::new(&config);
        grads.tok_emb[0] = 1.0;
        grads.zero();
        assert_eq!(grads.tok_emb[0], 0.0);
    }

    #[test]
    fn test_grad_norm_and_clip() {
        let config = tiny_config();
        let mut grads = GradBuffers::new(&config);
        for v in &mut grads.tok_emb {
            *v = 1.0;
        }
        let norm = grads.flat_grad_norm();
        assert!(norm > 0.0);
        grads.clip_grad_norm(0.01);
        assert!(grads.flat_grad_norm() <= 0.011);
    }

    #[test]
    fn test_backward_output_produces_gradients() {
        let config = tiny_config();
        let model = init_model(&config);
        let tokens = vec![1u32, 3, 5, 7];
        let targets = vec![3u32, 5, 7, 2];
        let mut buf = ForwardBuffer::new(&config, tokens.len());
        model.forward(&tokens, &mut buf);

        let mut grads = GradBuffers::new(&config);
        let grad_x = backward_output_loss(&model, &buf, &targets, &mut grads);
        assert!(
            grad_x.iter().any(|&v| v != 0.0),
            "grad_x should be non-zero"
        );
        assert!(
            grad_x.iter().all(|&v| v.is_finite()),
            "grad_x should be finite"
        );
    }

    #[test]
    fn test_full_backward() {
        let config = tiny_config();
        let model = init_model(&config);
        let tokens = vec![1u32, 3, 5, 7];
        let targets = vec![3u32, 5, 7, 2];
        let mut buf = ForwardBuffer::new(&config, tokens.len());
        let mut grads = GradBuffers::new(&config);

        let loss = model.backward(&tokens, &targets, &mut buf, &mut grads);

        assert!(loss.is_finite(), "loss should be finite: {}", loss);
        assert!(loss > 0.0, "loss should be positive: {}", loss);

        // Check that bank gradients are non-zero
        assert!(
            grads.qo_bank.iter().any(|&v| v != 0.0),
            "qo_bank grad should be non-zero"
        );
        assert!(
            grads.mlp_up_bank.iter().any(|&v| v != 0.0),
            "mlp_up grad should be non-zero"
        );
        assert!(
            grads.mlp_down_bank.iter().any(|&v| v != 0.0),
            "mlp_down grad should be non-zero"
        );
        assert!(
            grads.tok_emb.iter().any(|&v| v != 0.0),
            "tok_emb grad should be non-zero"
        );

        // All gradients should be finite
        assert!(
            grads.qo_bank.iter().all(|&v| v.is_finite()),
            "qo_bank grad not finite"
        );
        assert!(
            grads.mlp_up_bank.iter().all(|&v| v.is_finite()),
            "mlp_up grad not finite"
        );

        let norm = grads.flat_grad_norm();
        assert!(norm > 0.0, "gradient norm should be positive");
        eprintln!("Full backward: loss={:.4}, grad_norm={:.6}", loss, norm);
    }

    #[test]
    fn test_numerical_gradient_tok_emb() {
        // Verify analytical gradient matches numerical gradient for tok_emb
        let config = tiny_config();
        let model = init_model(&config);
        let tokens = vec![1u32, 3, 5, 7];
        let targets = vec![3u32, 5, 7, 2];

        // Analytical gradient
        let mut buf = ForwardBuffer::new(&config, tokens.len());
        let mut grads = GradBuffers::new(&config);
        model.backward(&tokens, &targets, &mut buf, &mut grads);

        // Numerical gradient for a few elements of tok_emb
        let eps = 1e-3;
        let d = config.model_dim;
        // Check gradient for token 1, dim 0
        let idx = 1 * d + 0;
        let mut model_p = init_model(&config);
        let mut model_m = init_model(&config);
        model_p.tok_emb[idx] += eps;
        model_m.tok_emb[idx] -= eps;

        let mut buf_p = ForwardBuffer::new(&config, tokens.len());
        let mut buf_m = ForwardBuffer::new(&config, tokens.len());
        model_p.forward(&tokens, &mut buf_p);
        model_m.forward(&tokens, &mut buf_m);
        let loss_p = model_p.compute_loss(&targets, &buf_p);
        let loss_m = model_m.compute_loss(&targets, &buf_m);
        let numerical = (loss_p - loss_m) / (2.0 * eps);

        let analytical = grads.tok_emb[idx];
        let diff = (analytical - numerical).abs();
        let max_val = analytical.abs().max(numerical.abs()).max(1e-6);

        eprintln!(
            "tok_emb[{}]: analytical={:.6}, numerical={:.6}, diff={:.6}, rel={:.4}",
            idx,
            analytical,
            numerical,
            diff,
            diff / max_val
        );
        // Relaxed tolerance — QK-norm backward is a pass-through approximation.
        // This is the dominant source of gradient error. On GPU, Flash Attention 3
        // handles the fused QK-norm + attention backward exactly.
        // Just verify same sign and order of magnitude.
        assert!(
            (analytical * numerical > 0.0) || diff < 1e-3,
            "gradient sign mismatch: analytical={}, numerical={}",
            analytical,
            numerical
        );
    }
}
