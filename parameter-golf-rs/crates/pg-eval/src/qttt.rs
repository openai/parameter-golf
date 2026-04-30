use pg_model::backward::GradBuffers;
/// qTTT — Query-only Test-Time Training (TDD §0, arXiv:2512.13898).
///
/// Standard TTT fine-tunes the whole model on each scored chunk; qTTT
/// only updates the **Q projection** of every attention layer (and,
/// optionally, the per-head `q_gain` scalar). Intuition: most of the
/// model's representational capacity lives in the K/V projections and
/// MLPs, while Q only controls "what to attend to given this context"
/// — precisely the part of the model that benefits from local
/// domain adaptation. Freezing everything else has three benefits:
///
///   1. ~10× fewer parameters to update → 2–3× more epochs fit in the
///      600s eval budget.
///   2. K and V computations are identical across epochs, so in the
///      GPU version we can cache them and skip the K/V GEMMs entirely
///      after epoch 0. (CPU reference just recomputes.)
///   3. The underlying language model is un-perturbed, so the TTT
///      pass is strictly monotonic in the limit — no "catastrophic
///      forgetting" failure mode.
///
/// This module is the CPU reference. It is written against the existing
/// `GptModel` / `GradBuffers` API: we compute the full backward pass
/// (same as normal training), zero out every gradient that isn't a Q
/// or q_gain before the SGD step, and step with a small LR + momentum.
///
/// ## Design notes
///
/// - The **Q half** of `qo_bank` lives at `[0 .. n * d * d]`; the O
///   half lives at `[n * d * d .. 2 * n * d * d]`. `QttTParams` holds
///   a flat momentum buffer of exactly the Q-half size.
/// - `q_gain` is a `[num_heads]` vector per layer. It is tiny and its
///   gradient feeds directly into attention sharpness; freezing or
///   updating it is a cheap ablation knob (`QttTConfig::adapt_q_gain`).
/// - The score-first legal semantics are preserved: a chunk is scored
///   first with the *current* (i.e., post-training-from-previous-chunk)
///   weights, then trained on. No forward leak.
use pg_model::{ForwardBuffer, GptModel};

use crate::sliding::{build_ttt_chunks, score_chunk};

/// Hyper-parameters for the qTTT loop.
#[derive(Debug, Clone)]
pub struct QttTConfig {
    /// Number of epochs to run over each already-scored chunk.
    pub epochs: usize,
    /// Base learning rate (cosine-decayed across chunks).
    pub lr: f32,
    /// Momentum coefficient for the SGD update.
    pub momentum: f32,
    /// Clip `||grad_q||_2` to this value before the SGD step.
    pub grad_clip: f32,
    /// If true, also update `q_gain` on each block. Tiny cost.
    pub adapt_q_gain: bool,
    /// LR decay: `lr * (0.5 * (1 + cos(pi * chunk_idx / num_chunks)))`.
    pub cosine_decay: bool,
    /// Window stride used for the per-chunk scoring pass.
    pub stride: usize,
    /// Transformer context length for scoring / training.
    pub seq_len: usize,
    /// TTT chunk size in tokens.
    pub chunk_tokens: usize,
}

impl QttTConfig {
    /// Sensible defaults: 3 epochs, small LR, light momentum.
    /// Mirrors the paper's recommended setup for a 1024-vocab model.
    pub fn paper_default(seq_len: usize) -> Self {
        Self {
            epochs: 3,
            lr: 0.01,
            momentum: 0.9,
            grad_clip: 1.0,
            adapt_q_gain: true,
            cosine_decay: true,
            stride: 64,
            seq_len,
            chunk_tokens: 32_768,
        }
    }
}

/// Mutable per-parameter state the qTTT loop carries across chunks.
///
/// This is conceptually a tiny SGD-with-momentum optimiser restricted
/// to the Q half of `qo_bank` and (optionally) the `q_gain` vectors of
/// every block.
pub struct QttTParams {
    /// Momentum buffer for the Q half of `qo_bank` — same layout.
    pub q_momentum: Vec<f32>,
    /// Per-block `q_gain` momentum. Empty if `adapt_q_gain = false`.
    pub q_gain_momentum: Vec<Vec<f32>>,
}

impl QttTParams {
    pub fn new(model: &GptModel, adapt_q_gain: bool) -> Self {
        let n = model.config.num_layers;
        let d = model.config.model_dim;
        let q_half = n * d * d;
        let q_gain_momentum = if adapt_q_gain {
            model
                .blocks
                .iter()
                .map(|b| vec![0.0f32; b.q_gain.len()])
                .collect()
        } else {
            Vec::new()
        };
        Self {
            q_momentum: vec![0.0f32; q_half],
            q_gain_momentum,
        }
    }

    /// Reset momentum buffers between TTT runs.
    pub fn reset(&mut self) {
        for v in self.q_momentum.iter_mut() {
            *v = 0.0;
        }
        for g in self.q_gain_momentum.iter_mut() {
            for v in g.iter_mut() {
                *v = 0.0;
            }
        }
    }
}

/// Gradient norm over the Q half of `qo_bank` + q_gain if enabled.
/// Returns the L2 norm (not squared).
pub fn q_grad_norm(grads: &GradBuffers, q_half: usize, adapt_q_gain: bool) -> f32 {
    let mut sq = 0.0f64;
    for v in &grads.qo_bank[..q_half] {
        sq += (*v as f64) * (*v as f64);
    }
    if adapt_q_gain {
        for block_g in &grads.block_q_gain {
            for v in block_g {
                sq += (*v as f64) * (*v as f64);
            }
        }
    }
    (sq as f32).sqrt()
}

/// Zero every gradient except those that qTTT updates: the Q half of
/// `qo_bank` and (optionally) the per-block `q_gain`.
///
/// This is intentionally a post-backward mask rather than a frozen
/// backward pass: the backward already ran, we just ignore the "other"
/// grads. On the GPU version this becomes "run Q-only backward" which
/// is strictly faster, but semantically equivalent.
pub fn mask_grads_qttt(grads: &mut GradBuffers, q_half: usize, adapt_q_gain: bool) {
    // Zero the O half of qo_bank
    for v in &mut grads.qo_bank[q_half..] {
        *v = 0.0;
    }
    // Everything else: zero
    for v in grads.kv_bank.iter_mut() {
        *v = 0.0;
    }
    for v in grads.mlp_up_bank.iter_mut() {
        *v = 0.0;
    }
    for v in grads.mlp_down_bank.iter_mut() {
        *v = 0.0;
    }
    for v in grads.tok_emb.iter_mut() {
        *v = 0.0;
    }
    for v in grads.bigram_embed.iter_mut() {
        *v = 0.0;
    }
    for v in grads.bigram_proj.iter_mut() {
        *v = 0.0;
    }
    grads.bigram_scale = 0.0;
    for v in grads.smear_gate.iter_mut() {
        *v = 0.0;
    }
    for v in grads.skip_weights.iter_mut() {
        *v = 0.0;
    }
    for v in grads.ve_embed.iter_mut() {
        *v = 0.0;
    }
    for v in grads.ve_proj.iter_mut() {
        *v = 0.0;
    }
    grads.ve_scale = 0.0;
    for v in grads.ve_layer_scales.iter_mut() {
        *v = 0.0;
    }
    for g in grads.block_attn_scale.iter_mut() {
        for v in g {
            *v = 0.0;
        }
    }
    for g in grads.block_mlp_scale.iter_mut() {
        for v in g {
            *v = 0.0;
        }
    }
    for g in grads.block_resid_mix.iter_mut() {
        for v in g {
            *v = 0.0;
        }
    }
    if !adapt_q_gain {
        for g in grads.block_q_gain.iter_mut() {
            for v in g {
                *v = 0.0;
            }
        }
    }
}

/// One SGD-with-momentum step on Q (and optionally q_gain).
/// Assumes `grads` has already been masked by `mask_grads_qttt`.
pub fn sgd_step_qttt(
    model: &mut GptModel,
    grads: &GradBuffers,
    state: &mut QttTParams,
    lr: f32,
    momentum: f32,
    q_half: usize,
    adapt_q_gain: bool,
) {
    // Q half
    for i in 0..q_half {
        let g = grads.qo_bank[i];
        let m = momentum * state.q_momentum[i] + g;
        state.q_momentum[i] = m;
        model.qo_bank[i] -= lr * m;
    }
    if adapt_q_gain {
        for (bi, block) in model.blocks.iter_mut().enumerate() {
            let g_slice = &grads.block_q_gain[bi];
            let m_slice = &mut state.q_gain_momentum[bi];
            for i in 0..block.q_gain.len() {
                let g = g_slice[i];
                let m = momentum * m_slice[i] + g;
                m_slice[i] = m;
                block.q_gain[i] -= lr * m;
            }
        }
    }
}

/// Train one chunk of tokens with the qTTT Q-only SGD loop.
/// `chunk_tokens` is the contiguous slice of tokens (inputs and targets
/// are constructed from it with `input = chunk[..-1]`, `target = chunk[1..]`).
///
/// Returns the mean per-token loss averaged over epochs (debug telemetry).
pub fn train_chunk_qttt(
    model: &mut GptModel,
    state: &mut QttTParams,
    chunk_tokens: &[u32],
    cfg: &QttTConfig,
    lr_scale: f32,
    buf: &mut ForwardBuffer,
    grads: &mut GradBuffers,
) -> f32 {
    if chunk_tokens.len() < 2 || cfg.epochs == 0 {
        return 0.0;
    }
    let n = model.config.num_layers;
    let d = model.config.model_dim;
    let q_half = n * d * d;
    let lr = cfg.lr * lr_scale;

    // Build mini-batches of length `seq_len` out of the chunk.
    // We use non-overlapping windows; the chunk has already been scored
    // so there's no score-train leakage concern here.
    let seq_len = cfg.seq_len;
    let total_targets = chunk_tokens.len() - 1;
    let mut starts: Vec<usize> = (0..total_targets).step_by(seq_len).collect();
    // Drop trailing mini-batch if it can't produce at least 2 tokens.
    starts.retain(|&s| total_targets - s >= 2);

    let mut loss_acc = 0.0f64;
    let mut loss_n = 0u64;

    for _epoch in 0..cfg.epochs {
        for &s in &starts {
            let end = (s + seq_len).min(total_targets);
            let input = &chunk_tokens[s..end];
            let target = &chunk_tokens[s + 1..end + 1];

            buf.resize_tokens(input.len());
            grads.zero();

            // Full backward (we'll mask below).
            let loss = model.backward(input, target, buf, grads);
            loss_acc += loss as f64;
            loss_n += 1;

            mask_grads_qttt(grads, q_half, cfg.adapt_q_gain);

            // Clip q grad norm.
            let gnorm = q_grad_norm(grads, q_half, cfg.adapt_q_gain);
            if gnorm > cfg.grad_clip && cfg.grad_clip > 0.0 {
                let scale = cfg.grad_clip / gnorm;
                for v in &mut grads.qo_bank[..q_half] {
                    *v *= scale;
                }
                if cfg.adapt_q_gain {
                    for g in grads.block_q_gain.iter_mut() {
                        for v in g {
                            *v *= scale;
                        }
                    }
                }
            }

            sgd_step_qttt(
                model,
                grads,
                state,
                lr,
                cfg.momentum,
                q_half,
                cfg.adapt_q_gain,
            );
        }
    }

    if loss_n > 0 {
        (loss_acc / loss_n as f64) as f32
    } else {
        0.0
    }
}

/// Per-chunk cosine LR decay.
/// `chunk_idx` in `[0, num_chunks)`.
#[inline]
fn cosine_lr_scale(chunk_idx: usize, num_chunks: usize) -> f32 {
    if num_chunks <= 1 {
        return 1.0;
    }
    let t = chunk_idx as f32 / (num_chunks as f32 - 1.0);
    0.5 * (1.0 + (std::f32::consts::PI * t).cos())
}

/// Full qTTT evaluation pipeline: score → train → score → train → ...
/// Last chunk is scored but not trained on (legal score-first TTT).
/// Returns `(val_loss, bpb)`.
pub fn eval_qttt(
    model: &mut GptModel,
    val_tokens: &[u32],
    base_bytes: &[f32],
    cfg: &QttTConfig,
) -> (f64, f64) {
    let stride = cfg.stride;
    let seq_len = cfg.seq_len;
    let total_tokens = val_tokens.len() - 1;

    let chunks = build_ttt_chunks(total_tokens, cfg.chunk_tokens, stride, seq_len);
    let num_chunks = chunks.len();

    let mut state = QttTParams::new(model, cfg.adapt_q_gain);
    let mut grads = GradBuffers::new(&model.config);
    let mut buf = ForwardBuffer::new(&model.config, seq_len);

    let mut total_loss = 0.0f64;
    let mut total_tokens_scored = 0u64;
    let mut total_bytes = 0.0f64;

    for (ci, chunk) in chunks.iter().enumerate() {
        // Phase 1: Score (uses the *current* weights, which may have been
        // updated by qTTT on previous chunks).
        let (loss, tokens, bytes) =
            score_chunk(model, val_tokens, base_bytes, chunk, stride, seq_len);
        total_loss += loss;
        total_tokens_scored += tokens;
        total_bytes += bytes;

        // Phase 2: Train on this chunk's tokens (unless it's the last chunk).
        let is_last = ci == num_chunks - 1;
        if !is_last && cfg.epochs > 0 {
            let chunk_slice = &val_tokens[chunk.chunk_start..chunk.chunk_end.min(val_tokens.len())];
            let lr_scale = if cfg.cosine_decay {
                cosine_lr_scale(ci, num_chunks)
            } else {
                1.0
            };
            train_chunk_qttt(
                model,
                &mut state,
                chunk_slice,
                cfg,
                lr_scale,
                &mut buf,
                &mut grads,
            );
        }
    }

    let val_loss = if total_tokens_scored > 0 {
        total_loss / total_tokens_scored as f64
    } else {
        0.0
    };
    let bits_per_token = val_loss / 2.0f64.ln();
    let tokens_per_byte = if total_bytes > 0.0 {
        total_tokens_scored as f64 / total_bytes
    } else {
        1.0
    };
    (val_loss, bits_per_token * tokens_per_byte)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pg_model::config::ModelConfig;

    fn tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 16,
            num_layers: 2,
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

    fn fill_model_deterministic(model: &mut GptModel) {
        // xorshift32 RNG so we get nonzero, reproducible weights
        let mut s: u32 = 0x1234_5678;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32 - 0.5) * 0.1
        };
        for v in model.tok_emb.iter_mut() {
            *v = next();
        }
        for v in model.qo_bank.iter_mut() {
            *v = next();
        }
        for v in model.kv_bank.iter_mut() {
            *v = next();
        }
        for v in model.mlp_up_bank.iter_mut() {
            *v = next();
        }
        for v in model.mlp_down_bank.iter_mut() {
            *v = next();
        }
    }

    #[test]
    fn test_qttt_params_shapes() {
        let cfg = tiny_config();
        let model = GptModel::new(cfg);
        let s = QttTParams::new(&model, true);
        let n = model.config.num_layers;
        let d = model.config.model_dim;
        assert_eq!(s.q_momentum.len(), n * d * d);
        assert_eq!(s.q_gain_momentum.len(), n);
        for g in &s.q_gain_momentum {
            assert_eq!(g.len(), model.config.num_heads);
        }
    }

    #[test]
    fn test_mask_grads_qttt_freezes_everything_but_q() {
        let cfg = tiny_config();
        let model = GptModel::new(cfg.clone());
        let mut grads = GradBuffers::new(&model.config);

        // Fill every grad with a nonzero sentinel.
        for v in grads.qo_bank.iter_mut() {
            *v = 1.0;
        }
        for v in grads.kv_bank.iter_mut() {
            *v = 1.0;
        }
        for v in grads.mlp_up_bank.iter_mut() {
            *v = 1.0;
        }
        for v in grads.mlp_down_bank.iter_mut() {
            *v = 1.0;
        }
        for v in grads.tok_emb.iter_mut() {
            *v = 1.0;
        }
        for v in grads.bigram_embed.iter_mut() {
            *v = 1.0;
        }
        for v in grads.bigram_proj.iter_mut() {
            *v = 1.0;
        }
        grads.bigram_scale = 1.0;
        for v in grads.smear_gate.iter_mut() {
            *v = 1.0;
        }
        for v in grads.skip_weights.iter_mut() {
            *v = 1.0;
        }
        for v in grads.ve_embed.iter_mut() {
            *v = 1.0;
        }
        for v in grads.ve_proj.iter_mut() {
            *v = 1.0;
        }
        grads.ve_scale = 1.0;
        for v in grads.ve_layer_scales.iter_mut() {
            *v = 1.0;
        }
        for g in grads.block_attn_scale.iter_mut() {
            for v in g {
                *v = 1.0;
            }
        }
        for g in grads.block_mlp_scale.iter_mut() {
            for v in g {
                *v = 1.0;
            }
        }
        for g in grads.block_resid_mix.iter_mut() {
            for v in g {
                *v = 1.0;
            }
        }
        for g in grads.block_q_gain.iter_mut() {
            for v in g {
                *v = 1.0;
            }
        }

        let n = cfg.num_layers;
        let d = cfg.model_dim;
        let q_half = n * d * d;

        mask_grads_qttt(&mut grads, q_half, true);

        // Q half should still be 1.0
        for v in &grads.qo_bank[..q_half] {
            assert_eq!(*v, 1.0);
        }
        // O half should be zero
        for v in &grads.qo_bank[q_half..] {
            assert_eq!(*v, 0.0);
        }
        // Everything else zeroed
        assert!(grads.kv_bank.iter().all(|&v| v == 0.0));
        assert!(grads.mlp_up_bank.iter().all(|&v| v == 0.0));
        assert!(grads.tok_emb.iter().all(|&v| v == 0.0));
        assert_eq!(grads.bigram_scale, 0.0);
        assert_eq!(grads.ve_scale, 0.0);
        // q_gain should be preserved (adapt_q_gain = true)
        for g in &grads.block_q_gain {
            assert!(g.iter().all(|&v| v == 1.0));
        }
    }

    #[test]
    fn test_mask_grads_qttt_freezes_q_gain_when_disabled() {
        let cfg = tiny_config();
        let model = GptModel::new(cfg.clone());
        let mut grads = GradBuffers::new(&model.config);
        for g in grads.block_q_gain.iter_mut() {
            for v in g {
                *v = 1.0;
            }
        }
        let q_half = cfg.num_layers * cfg.model_dim * cfg.model_dim;
        mask_grads_qttt(&mut grads, q_half, false);
        for g in &grads.block_q_gain {
            for v in g {
                assert_eq!(*v, 0.0);
            }
        }
    }

    #[test]
    fn test_sgd_step_moves_only_q_and_q_gain() {
        let cfg = tiny_config();
        let mut model = GptModel::new(cfg.clone());
        fill_model_deterministic(&mut model);

        let before_kv = model.kv_bank.clone();
        let before_mlp = model.mlp_up_bank.clone();
        let before_q_half: Vec<f32> =
            model.qo_bank[..cfg.num_layers * cfg.model_dim * cfg.model_dim].to_vec();
        let before_o_half: Vec<f32> =
            model.qo_bank[cfg.num_layers * cfg.model_dim * cfg.model_dim..].to_vec();

        // Construct non-trivial grads
        let mut grads = GradBuffers::new(&model.config);
        for v in grads.qo_bank.iter_mut() {
            *v = 0.5;
        }
        for v in grads.kv_bank.iter_mut() {
            *v = 0.5;
        }
        for v in grads.mlp_up_bank.iter_mut() {
            *v = 0.5;
        }
        for g in grads.block_q_gain.iter_mut() {
            for v in g {
                *v = 0.5;
            }
        }

        let q_half = cfg.num_layers * cfg.model_dim * cfg.model_dim;
        mask_grads_qttt(&mut grads, q_half, true);

        let mut state = QttTParams::new(&model, true);
        sgd_step_qttt(&mut model, &grads, &mut state, 0.1, 0.0, q_half, true);

        // Q half moved by exactly -0.1 * 0.5 = -0.05
        for i in 0..q_half {
            assert!((model.qo_bank[i] - (before_q_half[i] - 0.05)).abs() < 1e-6);
        }
        // O half unchanged
        for (i, &b) in before_o_half.iter().enumerate() {
            assert_eq!(model.qo_bank[q_half + i], b);
        }
        // KV and MLP unchanged
        assert_eq!(model.kv_bank, before_kv);
        assert_eq!(model.mlp_up_bank, before_mlp);
        // q_gain moved
        for block in &model.blocks {
            for &v in &block.q_gain {
                // initial q_gain = qk_gain_init (1.0) − 0.05
                assert!((v - (1.0 - 0.05)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_cosine_lr_scale_endpoints() {
        assert!((cosine_lr_scale(0, 10) - 1.0).abs() < 1e-6);
        assert!(cosine_lr_scale(9, 10).abs() < 1e-6);
        // single chunk: always 1.0
        assert!((cosine_lr_scale(0, 1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_q_grad_norm_matches_manual() {
        let cfg = tiny_config();
        let model = GptModel::new(cfg.clone());
        let mut grads = GradBuffers::new(&model.config);
        let q_half = cfg.num_layers * cfg.model_dim * cfg.model_dim;

        // Put 1.0 into first 4 entries of q half → norm = 2.0
        for v in grads.qo_bank[..q_half].iter_mut() {
            *v = 0.0;
        }
        for v in grads.qo_bank[q_half..].iter_mut() {
            *v = 999.0;
        }
        grads.qo_bank[0] = 1.0;
        grads.qo_bank[1] = 1.0;
        grads.qo_bank[2] = 1.0;
        grads.qo_bank[3] = 1.0;

        let n = q_grad_norm(&grads, q_half, false);
        assert!((n - 2.0).abs() < 1e-6);
    }
}
