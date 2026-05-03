use pg_model::backward::GradBuffers;
/// LaCT — Large Chunk Test-Time Training (TDD §0, arXiv:2505.23884).
///
/// Standard per-window TTT runs at <5% H100 utilisation because every
/// update processes one 2 k-token mini-batch, the grads are allreduced,
/// then the next mini-batch starts. LaCT observes that a *document* is
/// a natural large boundary and restructures the training loop so that:
///
///   * The chunk granularity is document-sized (~32k–128k tokens).
///   * **All** sub-sequences inside the chunk contribute gradients to
///     a single optimiser step per epoch (gradient accumulation, not
///     immediate step). That gives a single big "NCCL reduce" instead
///     of hundreds of small ones on the GPU version.
///   * A small number of epochs (typically 1–2) walks the whole chunk
///     with the accumulated grad. At H100 utilisation this is ~70%.
///
/// The CPU reference here captures the semantics — gradient accumulation
/// over non-overlapping sub-sequences, one optimiser step per epoch —
/// without simulating the GPU batching. The GPU version replaces the
/// inner `for win in sub_seqs { backward }` with one flash-attention
/// call on the whole chunk and one matmul-backward per layer.
///
/// LaCT is **complementary** to qTTT: LaCT sets the chunk shape and the
/// accumulation boundary, qTTT restricts which parameters move. The two
/// composed give "document-sized gradient accumulation on Q only",
/// which is the configuration recommended by the professor-sync memo
/// and the one we'll run in the ablations.
use pg_model::{ForwardBuffer, GptModel};

use crate::qttt::{QttTConfig, QttTParams, mask_grads_qttt, sgd_step_qttt};
use crate::sliding::{build_ttt_chunks, score_chunk};

/// Hyper-parameters for the LaCT loop.
#[derive(Debug, Clone)]
pub struct LaCtConfig {
    /// Number of epochs to run over each already-scored chunk.
    /// 1–2 is typical; compute budget scales linearly.
    pub epochs: usize,
    /// Learning rate for the accumulated-grad SGD step.
    pub lr: f32,
    /// Momentum coefficient.
    pub momentum: f32,
    /// Clip the accumulated grad norm before the step.
    pub grad_clip: f32,
    /// Non-overlapping sub-sequence length inside each chunk.
    pub sub_seq_len: usize,
    /// Stride used for the scoring pass (same semantics as sliding).
    pub stride: usize,
    /// Document-sized chunk length. 32k is the paper's sweet spot.
    pub chunk_tokens: usize,
    /// If true, restrict gradient updates to the Q projection + q_gain
    /// (i.e. compose with qTTT). If false, update every parameter.
    pub q_only: bool,
    /// Cosine-decay the LR across chunks, just like qTTT.
    pub cosine_decay: bool,
}

impl LaCtConfig {
    /// Paper defaults for the LaCT + qTTT composition.
    pub fn paper_default(sub_seq_len: usize) -> Self {
        Self {
            epochs: 2,
            lr: 0.005,
            momentum: 0.9,
            grad_clip: 1.0,
            sub_seq_len,
            stride: 64,
            chunk_tokens: 32_768,
            q_only: true,
            cosine_decay: true,
        }
    }

    /// Convert to a `QttTConfig` for parameter-state sizing. Only the
    /// `adapt_q_gain` and `seq_len` fields are used by `QttTParams::new`.
    pub fn as_qttt_cfg(&self) -> QttTConfig {
        QttTConfig {
            epochs: self.epochs,
            lr: self.lr,
            momentum: self.momentum,
            grad_clip: self.grad_clip,
            adapt_q_gain: self.q_only,
            cosine_decay: self.cosine_decay,
            stride: self.stride,
            seq_len: self.sub_seq_len,
            chunk_tokens: self.chunk_tokens,
        }
    }
}

/// Accumulate `src` into `dst` in-place: `dst += src`.
#[inline]
fn accumulate(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

/// Scale `dst` by `k` in-place.
#[inline]
fn scale(dst: &mut [f32], k: f32) {
    for v in dst.iter_mut() {
        *v *= k;
    }
}

/// Accumulate one `GradBuffers` into another: `dst += src`.
/// Handles every parameter group in the model, including scalars.
pub fn accumulate_grads(dst: &mut GradBuffers, src: &GradBuffers) {
    accumulate(&mut dst.tok_emb, &src.tok_emb);
    accumulate(&mut dst.bigram_embed, &src.bigram_embed);
    accumulate(&mut dst.bigram_proj, &src.bigram_proj);
    dst.bigram_scale += src.bigram_scale;
    accumulate(&mut dst.smear_gate, &src.smear_gate);
    accumulate(&mut dst.skip_weights, &src.skip_weights);
    accumulate(&mut dst.qo_bank, &src.qo_bank);
    accumulate(&mut dst.kv_bank, &src.kv_bank);
    accumulate(&mut dst.mlp_up_bank, &src.mlp_up_bank);
    accumulate(&mut dst.mlp_down_bank, &src.mlp_down_bank);
    accumulate(&mut dst.ve_embed, &src.ve_embed);
    accumulate(&mut dst.ve_proj, &src.ve_proj);
    dst.ve_scale += src.ve_scale;
    accumulate(&mut dst.ve_layer_scales, &src.ve_layer_scales);
    for (a, b) in dst
        .block_attn_scale
        .iter_mut()
        .zip(src.block_attn_scale.iter())
    {
        accumulate(a, b);
    }
    for (a, b) in dst
        .block_mlp_scale
        .iter_mut()
        .zip(src.block_mlp_scale.iter())
    {
        accumulate(a, b);
    }
    for (a, b) in dst
        .block_resid_mix
        .iter_mut()
        .zip(src.block_resid_mix.iter())
    {
        accumulate(a, b);
    }
    for (a, b) in dst.block_q_gain.iter_mut().zip(src.block_q_gain.iter()) {
        accumulate(a, b);
    }
}

/// Scale every element of `grads` by `k` (used to normalise accumulated
/// grads by the number of sub-sequences).
pub fn scale_grads(grads: &mut GradBuffers, k: f32) {
    scale(&mut grads.tok_emb, k);
    scale(&mut grads.bigram_embed, k);
    scale(&mut grads.bigram_proj, k);
    grads.bigram_scale *= k;
    scale(&mut grads.smear_gate, k);
    scale(&mut grads.skip_weights, k);
    scale(&mut grads.qo_bank, k);
    scale(&mut grads.kv_bank, k);
    scale(&mut grads.mlp_up_bank, k);
    scale(&mut grads.mlp_down_bank, k);
    scale(&mut grads.ve_embed, k);
    scale(&mut grads.ve_proj, k);
    grads.ve_scale *= k;
    scale(&mut grads.ve_layer_scales, k);
    for g in grads.block_attn_scale.iter_mut() {
        scale(g, k);
    }
    for g in grads.block_mlp_scale.iter_mut() {
        scale(g, k);
    }
    for g in grads.block_resid_mix.iter_mut() {
        scale(g, k);
    }
    for g in grads.block_q_gain.iter_mut() {
        scale(g, k);
    }
}

/// Full-model momentum state (used when `q_only = false`).
///
/// This is a bare-bones SGD-with-momentum state. When `q_only = true`
/// we reuse `QttTParams` instead and go through `sgd_step_qttt`.
pub struct LaCtMomentum {
    pub qo: Vec<f32>,
    pub kv: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
}

impl LaCtMomentum {
    pub fn new(model: &GptModel) -> Self {
        Self {
            qo: vec![0.0f32; model.qo_bank.len()],
            kv: vec![0.0f32; model.kv_bank.len()],
            mlp_up: vec![0.0f32; model.mlp_up_bank.len()],
            mlp_down: vec![0.0f32; model.mlp_down_bank.len()],
        }
    }

    pub fn reset(&mut self) {
        for v in &mut self.qo {
            *v = 0.0;
        }
        for v in &mut self.kv {
            *v = 0.0;
        }
        for v in &mut self.mlp_up {
            *v = 0.0;
        }
        for v in &mut self.mlp_down {
            *v = 0.0;
        }
    }
}

/// Full-model momentum SGD step on the 4 parameter banks only.
pub fn sgd_step_full(
    model: &mut GptModel,
    grads: &GradBuffers,
    state: &mut LaCtMomentum,
    lr: f32,
    momentum: f32,
) {
    let apply = |w: &mut [f32], g: &[f32], m: &mut [f32]| {
        for i in 0..w.len() {
            let mv = momentum * m[i] + g[i];
            m[i] = mv;
            w[i] -= lr * mv;
        }
    };
    apply(&mut model.qo_bank, &grads.qo_bank, &mut state.qo);
    apply(&mut model.kv_bank, &grads.kv_bank, &mut state.kv);
    apply(
        &mut model.mlp_up_bank,
        &grads.mlp_up_bank,
        &mut state.mlp_up,
    );
    apply(
        &mut model.mlp_down_bank,
        &grads.mlp_down_bank,
        &mut state.mlp_down,
    );
}

/// L2 norm over the banks (used for full-model grad clipping).
pub fn full_grad_norm(grads: &GradBuffers) -> f32 {
    let mut sq = 0.0f64;
    for v in &grads.qo_bank {
        sq += (*v as f64) * (*v as f64);
    }
    for v in &grads.kv_bank {
        sq += (*v as f64) * (*v as f64);
    }
    for v in &grads.mlp_up_bank {
        sq += (*v as f64) * (*v as f64);
    }
    for v in &grads.mlp_down_bank {
        sq += (*v as f64) * (*v as f64);
    }
    (sq as f32).sqrt()
}

fn clip_bank_grads(grads: &mut GradBuffers, max_norm: f32) {
    if max_norm <= 0.0 {
        return;
    }
    let n = full_grad_norm(grads);
    if n > max_norm {
        let s = max_norm / n;
        for v in grads.qo_bank.iter_mut() {
            *v *= s;
        }
        for v in grads.kv_bank.iter_mut() {
            *v *= s;
        }
        for v in grads.mlp_up_bank.iter_mut() {
            *v *= s;
        }
        for v in grads.mlp_down_bank.iter_mut() {
            *v *= s;
        }
    }
}

fn clip_q_grads(grads: &mut GradBuffers, max_norm: f32, q_half: usize, adapt_q_gain: bool) {
    if max_norm <= 0.0 {
        return;
    }
    let n = crate::qttt::q_grad_norm(grads, q_half, adapt_q_gain);
    if n > max_norm {
        let s = max_norm / n;
        for v in &mut grads.qo_bank[..q_half] {
            *v *= s;
        }
        if adapt_q_gain {
            for g in grads.block_q_gain.iter_mut() {
                for v in g {
                    *v *= s;
                }
            }
        }
    }
}

/// Train one chunk with LaCT: accumulate grads across every
/// sub-sequence in the chunk, step once per epoch.
///
/// `q_state` is used when `cfg.q_only = true`, `full_state` otherwise.
/// The unused one is ignored.
pub fn train_chunk_lact(
    model: &mut GptModel,
    chunk_tokens: &[u32],
    cfg: &LaCtConfig,
    lr_scale: f32,
    buf: &mut ForwardBuffer,
    per_win_grads: &mut GradBuffers,
    accum_grads: &mut GradBuffers,
    q_state: &mut QttTParams,
    full_state: &mut LaCtMomentum,
) -> f32 {
    if chunk_tokens.len() < 2 || cfg.epochs == 0 {
        return 0.0;
    }
    let n = model.config.num_layers;
    let d = model.config.model_dim;
    let q_half = n * d * d;
    let lr = cfg.lr * lr_scale;

    let total_targets = chunk_tokens.len() - 1;
    let sub = cfg.sub_seq_len;
    let mut starts: Vec<usize> = (0..total_targets).step_by(sub).collect();
    starts.retain(|&s| total_targets - s >= 2);
    if starts.is_empty() {
        return 0.0;
    }
    let num_sub = starts.len();
    let inv_num = 1.0 / num_sub as f32;

    let mut loss_acc = 0.0f64;
    let mut loss_n = 0u64;

    for _epoch in 0..cfg.epochs {
        accum_grads.zero();

        // --- Gradient accumulation over every sub-sequence in the chunk ---
        for &s in &starts {
            let end = (s + sub).min(total_targets);
            let input = &chunk_tokens[s..end];
            let target = &chunk_tokens[s + 1..end + 1];

            buf.resize_tokens(input.len());
            per_win_grads.zero();

            let loss = model.backward(input, target, buf, per_win_grads);
            loss_acc += loss as f64;
            loss_n += 1;

            accumulate_grads(accum_grads, per_win_grads);
        }

        // Normalise by sub-sequence count so the effective per-token
        // LR is independent of how many windows the chunk got split into.
        scale_grads(accum_grads, inv_num);

        // --- Single optimiser step for this epoch ---
        if cfg.q_only {
            mask_grads_qttt(accum_grads, q_half, true);
            clip_q_grads(accum_grads, cfg.grad_clip, q_half, true);
            sgd_step_qttt(model, accum_grads, q_state, lr, cfg.momentum, q_half, true);
        } else {
            clip_bank_grads(accum_grads, cfg.grad_clip);
            sgd_step_full(model, accum_grads, full_state, lr, cfg.momentum);
        }
    }

    if loss_n > 0 {
        (loss_acc / loss_n as f64) as f32
    } else {
        0.0
    }
}

/// Per-chunk cosine LR decay (same as qTTT).
#[inline]
fn cosine_lr_scale(chunk_idx: usize, num_chunks: usize) -> f32 {
    if num_chunks <= 1 {
        return 1.0;
    }
    let t = chunk_idx as f32 / (num_chunks as f32 - 1.0);
    0.5 * (1.0 + (std::f32::consts::PI * t).cos())
}

/// Full LaCT evaluation pipeline: score → accumulate-grad → step → ...
/// Legal score-first TTT: last chunk scored but never trained on.
pub fn eval_lact(
    model: &mut GptModel,
    val_tokens: &[u32],
    base_bytes: &[f32],
    cfg: &LaCtConfig,
) -> (f64, f64) {
    let stride = cfg.stride;
    let sub_seq_len = cfg.sub_seq_len;
    let total_tokens = val_tokens.len() - 1;

    let chunks = build_ttt_chunks(total_tokens, cfg.chunk_tokens, stride, sub_seq_len);
    let num_chunks = chunks.len();

    let mut q_state = QttTParams::new(model, cfg.q_only);
    let mut full_state = LaCtMomentum::new(model);
    let mut per_win_grads = GradBuffers::new(&model.config);
    let mut accum_grads = GradBuffers::new(&model.config);
    let mut buf = ForwardBuffer::new(&model.config, sub_seq_len);

    let mut total_loss = 0.0f64;
    let mut total_tokens_scored = 0u64;
    let mut total_bytes = 0.0f64;

    for (ci, chunk) in chunks.iter().enumerate() {
        // Score.
        let (loss, toks, bytes) =
            score_chunk(model, val_tokens, base_bytes, chunk, stride, sub_seq_len);
        total_loss += loss;
        total_tokens_scored += toks;
        total_bytes += bytes;

        // Train (not on the last chunk).
        let is_last = ci == num_chunks - 1;
        if !is_last && cfg.epochs > 0 {
            let chunk_slice = &val_tokens[chunk.chunk_start..chunk.chunk_end.min(val_tokens.len())];
            let lr_scale = if cfg.cosine_decay {
                cosine_lr_scale(ci, num_chunks)
            } else {
                1.0
            };
            train_chunk_lact(
                model,
                chunk_slice,
                cfg,
                lr_scale,
                &mut buf,
                &mut per_win_grads,
                &mut accum_grads,
                &mut q_state,
                &mut full_state,
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

    #[test]
    fn test_accumulate_grads_adds_into_dst() {
        let cfg = tiny_config();
        let mut a = GradBuffers::new(&cfg);
        let mut b = GradBuffers::new(&cfg);
        for v in a.qo_bank.iter_mut() {
            *v = 1.0;
        }
        for v in b.qo_bank.iter_mut() {
            *v = 2.0;
        }
        a.ve_scale = 3.0;
        b.ve_scale = 4.0;

        accumulate_grads(&mut a, &b);
        assert!(a.qo_bank.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert!((a.ve_scale - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_grads_halves_everything() {
        let cfg = tiny_config();
        let mut g = GradBuffers::new(&cfg);
        for v in g.qo_bank.iter_mut() {
            *v = 4.0;
        }
        g.ve_scale = 10.0;
        scale_grads(&mut g, 0.5);
        assert!(g.qo_bank.iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!((g.ve_scale - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_full_grad_norm() {
        let cfg = tiny_config();
        let mut g = GradBuffers::new(&cfg);
        g.qo_bank[0] = 3.0;
        g.kv_bank[0] = 4.0;
        // norm = sqrt(9 + 16) = 5
        assert!((full_grad_norm(&g) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_step_full_moves_all_banks() {
        let cfg = tiny_config();
        let mut model = GptModel::new(cfg.clone());
        let before_qo = model.qo_bank.clone();
        let before_kv = model.kv_bank.clone();
        let before_mlp_up = model.mlp_up_bank.clone();
        let before_mlp_dn = model.mlp_down_bank.clone();

        let mut g = GradBuffers::new(&cfg);
        for v in g.qo_bank.iter_mut() {
            *v = 1.0;
        }
        for v in g.kv_bank.iter_mut() {
            *v = 1.0;
        }
        for v in g.mlp_up_bank.iter_mut() {
            *v = 1.0;
        }
        for v in g.mlp_down_bank.iter_mut() {
            *v = 1.0;
        }

        let mut st = LaCtMomentum::new(&model);
        sgd_step_full(&mut model, &g, &mut st, 0.1, 0.0);

        for i in 0..model.qo_bank.len() {
            assert!((model.qo_bank[i] - (before_qo[i] - 0.1)).abs() < 1e-6);
        }
        for i in 0..model.kv_bank.len() {
            assert!((model.kv_bank[i] - (before_kv[i] - 0.1)).abs() < 1e-6);
        }
        for i in 0..model.mlp_up_bank.len() {
            assert!((model.mlp_up_bank[i] - (before_mlp_up[i] - 0.1)).abs() < 1e-6);
        }
        for i in 0..model.mlp_down_bank.len() {
            assert!((model.mlp_down_bank[i] - (before_mlp_dn[i] - 0.1)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cosine_lr_scale() {
        assert!((cosine_lr_scale(0, 4) - 1.0).abs() < 1e-6);
        assert!(cosine_lr_scale(3, 4).abs() < 1e-6);
    }

    #[test]
    fn test_lact_config_defaults_compose_with_qttt() {
        let cfg = LaCtConfig::paper_default(2048);
        assert!(cfg.q_only);
        assert_eq!(cfg.sub_seq_len, 2048);
        let q = cfg.as_qttt_cfg();
        assert!(q.adapt_q_gain);
        assert_eq!(q.seq_len, 2048);
    }
}
