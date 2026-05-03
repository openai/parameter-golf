/// SLOT — Self-Logit Output Transformation (TDD §0, PR #1084).
///
/// Eval-time augmentation that blends the neural next-token distribution
/// with a cheap bigram-LM distribution before computing NLL. Reported at
/// **1.1185 BPB** on the unmodified SOTA stack.
///
/// The intuition: the complementary-training pipeline (Phase 4) already
/// teaches the neural model to ignore tokens the bigram can predict
/// easily. At eval time we then *blend back* the bigram's opinion, and
/// — because the neural model was trained to under-weight those tokens
/// — the blend yields strictly lower NLL than either side alone.
///
/// ## Blend modes
///
/// * **Fixed alpha**: `p_blend = alpha * p_neural + (1 - alpha) * p_bigram`
/// * **Entropy-adaptive**: `alpha` is a function of the bigram row
///   entropy. High-entropy rows (bigram is uncertain) use more neural,
///   low-entropy rows (bigram is confident) use more bigram. This is
///   the PR #1084 + order-adaptive-gating recipe.
///
/// Both modes normalise to a proper probability distribution before
/// taking `log`, so NLL is always well-defined.
///
/// ## Integration point
///
/// This module exposes a `slot_nll(...)` function that takes a slice
/// of (already-softcapped) neural logits, the previous token, and a
/// `BigramStats` reference, and returns the SLOT NLL for the target
/// token. The sliding-window and qTTT scorers call it instead of the
/// raw cross-entropy when a `SlotConfig` is supplied.
use pg_kernels::complementary::BigramStats;

/// Blending strategy for SLOT.
#[derive(Debug, Clone, Copy)]
pub enum BlendMode {
    /// Constant `alpha` for every position.
    /// `alpha = 1.0` is pure neural, `alpha = 0.0` is pure bigram.
    Fixed(f32),
    /// `alpha` scales linearly with the bigram row entropy. Low entropy
    /// (certain bigram) → low alpha (trust bigram). High entropy → high
    /// alpha (trust neural).
    ///
    /// Concretely:
    ///   `alpha = clamp(alpha_min + (alpha_max - alpha_min) * e / ln(V), alpha_min, alpha_max)`
    /// where `e` is the row entropy in nats and `ln(V)` is the maximum
    /// possible entropy (uniform distribution).
    EntropyAdaptive { alpha_min: f32, alpha_max: f32 },
}

/// Evaluation-time SLOT config.
#[derive(Debug, Clone)]
pub struct SlotConfig {
    /// How to combine neural and bigram distributions.
    pub mode: BlendMode,
    /// Softcap to apply to neural logits before the blend (matches the
    /// training loss).
    pub softcap: f32,
    /// Floor for neural probability mass (numerical stability).
    pub eps: f32,
}

impl Default for SlotConfig {
    fn default() -> Self {
        Self {
            mode: BlendMode::EntropyAdaptive {
                alpha_min: 0.60,
                alpha_max: 0.95,
            },
            softcap: 30.0,
            eps: 1e-12,
        }
    }
}

/// Softcap + softmax into a pre-allocated output buffer. Returns the
/// row maximum *after* softcap so callers can re-use it if they want.
pub fn softcap_softmax(logits: &[f32], softcap: f32, out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len());
    if softcap > 0.0 {
        let inv = 1.0 / softcap;
        for (l, o) in logits.iter().zip(out.iter_mut()) {
            *o = softcap * (*l * inv).tanh();
        }
    } else {
        out.copy_from_slice(logits);
    }
    let m = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in out.iter_mut() {
        *v = ((*v - m) as f64).exp() as f32;
        sum += *v as f64;
    }
    let inv_sum = if sum > 0.0 { (1.0 / sum) as f32 } else { 0.0 };
    for v in out.iter_mut() {
        *v *= inv_sum;
    }
}

/// Populate `out[v]` with the bigram probability `P(v | prev)`.
/// Uses the add-one smoothed row from `BigramStats::prob`.
pub fn bigram_row_prob(stats: &BigramStats, prev: u32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), stats.vocab_size);
    for (curr, o) in out.iter_mut().enumerate() {
        *o = stats.prob(prev, curr as u32);
    }
    // Re-normalise (add-one smoothing means exact prob; this just
    // forgives any floating drift).
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in out.iter_mut() {
            *v *= inv;
        }
    }
}

/// Compute the SLOT alpha for a given bigram row under the configured
/// blend mode.
pub fn blend_alpha(stats: &BigramStats, prev: u32, mode: BlendMode) -> f32 {
    match mode {
        BlendMode::Fixed(a) => a.clamp(0.0, 1.0),
        BlendMode::EntropyAdaptive {
            alpha_min,
            alpha_max,
        } => {
            let e = stats.row_entropy(prev);
            let ln_v = (stats.vocab_size as f32).ln().max(1e-6);
            let t = (e / ln_v).clamp(0.0, 1.0);
            alpha_min + (alpha_max - alpha_min) * t
        }
    }
}

/// Compute SLOT-blended NLL for a single token position.
///
/// * `logits`: `[vocab_size]` raw pre-softcap neural logits
/// * `prev`: previous-token id (context for the bigram LM)
/// * `target`: next-token id (what we're scoring)
/// * `stats`: pre-built bigram stats table
/// * `cfg`: blending config
/// * `scratch_neural`, `scratch_bigram`: pre-allocated `[vocab_size]`
///   scratch buffers, reused across tokens by the caller.
pub fn slot_nll(
    logits: &[f32],
    prev: u32,
    target: usize,
    stats: &BigramStats,
    cfg: &SlotConfig,
    scratch_neural: &mut [f32],
    scratch_bigram: &mut [f32],
) -> f32 {
    softcap_softmax(logits, cfg.softcap, scratch_neural);
    bigram_row_prob(stats, prev, scratch_bigram);
    let alpha = blend_alpha(stats, prev, cfg.mode);
    let beta = 1.0 - alpha;
    let p_blend = alpha * scratch_neural[target] + beta * scratch_bigram[target];
    -((p_blend.max(cfg.eps)).ln())
}

/// Compute SLOT-blended NLL for a whole sequence of positions that
/// share the same logit buffer. `logits` is `[num_tokens, vocab_size]`,
/// `prev_tokens[t]` is the context token for position `t`, and
/// `target_tokens[t]` is the token we're scoring.
///
/// Uses one scratch buffer pair (reused across positions).
pub fn slot_nll_sequence(
    logits: &[f32],
    prev_tokens: &[u32],
    target_tokens: &[u32],
    stats: &BigramStats,
    cfg: &SlotConfig,
    vocab_size: usize,
    nlls_out: &mut [f32],
) {
    assert_eq!(prev_tokens.len(), target_tokens.len());
    assert_eq!(nlls_out.len(), target_tokens.len());
    assert_eq!(logits.len(), target_tokens.len() * vocab_size);

    let mut scratch_neural = vec![0.0f32; vocab_size];
    let mut scratch_bigram = vec![0.0f32; vocab_size];

    for t in 0..target_tokens.len() {
        let row = &logits[t * vocab_size..(t + 1) * vocab_size];
        let nll = slot_nll(
            row,
            prev_tokens[t],
            target_tokens[t] as usize,
            stats,
            cfg,
            &mut scratch_neural,
            &mut scratch_bigram,
        );
        nlls_out[t] = nll;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_stats() -> BigramStats {
        // vocab=4, "0 -> 1" very strong, "2 -> 1|3" uniform
        let mut s = BigramStats::new(4);
        for _ in 0..100 {
            s.add_sequence(&[0, 1]);
        }
        s.add_sequence(&[0, 2]);
        for _ in 0..5 {
            s.add_sequence(&[2, 1]);
            s.add_sequence(&[2, 3]);
        }
        s
    }

    fn delta_logits(vocab: usize, target: usize, magnitude: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; vocab];
        v[target] = magnitude;
        v
    }

    #[test]
    fn test_softcap_softmax_sums_to_one() {
        let logits = vec![1.0f32, -0.5, 2.0, 0.3];
        let mut out = vec![0.0f32; 4];
        softcap_softmax(&logits, 30.0, &mut out);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(out.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_bigram_row_prob_normalised() {
        let s = toy_stats();
        let mut r = vec![0.0f32; 4];
        bigram_row_prob(&s, 0, &mut r);
        let sum: f32 = r.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Class 1 should dominate.
        assert!(r[1] > 0.9);
    }

    #[test]
    fn test_blend_alpha_fixed() {
        let s = toy_stats();
        let a = blend_alpha(&s, 0, BlendMode::Fixed(0.7));
        assert!((a - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_blend_alpha_entropy_adaptive_peaked_row_low_alpha() {
        let s = toy_stats();
        // Row 0 is peaked → low entropy → small alpha (trust bigram more)
        let a0 = blend_alpha(
            &s,
            0,
            BlendMode::EntropyAdaptive {
                alpha_min: 0.5,
                alpha_max: 1.0,
            },
        );
        // Row 3 is unseen → uniform → high entropy → alpha ≈ alpha_max
        let a3 = blend_alpha(
            &s,
            3,
            BlendMode::EntropyAdaptive {
                alpha_min: 0.5,
                alpha_max: 1.0,
            },
        );
        assert!(
            a0 < a3,
            "peaked row should have smaller alpha: a0={}, a3={}",
            a0,
            a3
        );
        assert!(a0 >= 0.5);
        assert!(a3 <= 1.0);
    }

    #[test]
    fn test_slot_nll_matches_neural_at_alpha_one() {
        let s = toy_stats();
        let logits = delta_logits(4, 2, 5.0);
        let mut sn = vec![0.0f32; 4];
        let mut sb = vec![0.0f32; 4];
        let cfg = SlotConfig {
            mode: BlendMode::Fixed(1.0),
            softcap: 30.0,
            eps: 1e-12,
        };
        let nll_slot = slot_nll(&logits, 0, 2, &s, &cfg, &mut sn, &mut sb);
        // Compare to a direct softcap-softmax NLL
        let mut probs = vec![0.0f32; 4];
        softcap_softmax(&logits, 30.0, &mut probs);
        let nll_direct = -probs[2].ln();
        assert!((nll_slot - nll_direct).abs() < 1e-5);
    }

    #[test]
    fn test_slot_nll_matches_bigram_at_alpha_zero() {
        let s = toy_stats();
        // Neural logits that WOULD assign near-zero to target 2
        let logits = delta_logits(4, 0, 100.0);
        let mut sn = vec![0.0f32; 4];
        let mut sb = vec![0.0f32; 4];
        let cfg = SlotConfig {
            mode: BlendMode::Fixed(0.0),
            softcap: 30.0,
            eps: 1e-12,
        };
        let nll_slot = slot_nll(&logits, 0, 2, &s, &cfg, &mut sn, &mut sb);
        // Pure bigram P(2|0) ≈ 2/(101+4) ≈ 0.019
        let p = s.prob(0, 2);
        let nll_direct = -p.ln();
        assert!((nll_slot - nll_direct).abs() < 1e-4);
    }

    #[test]
    fn test_slot_nll_improves_when_bigram_is_correct_and_neural_is_wrong() {
        // Bigram row 0 strongly favours target 1. Neural is uniform.
        let s = toy_stats();
        let logits = vec![0.0f32; 4]; // uniform (nll = ln 4)
        let mut sn = vec![0.0f32; 4];
        let mut sb = vec![0.0f32; 4];

        let cfg_neural = SlotConfig {
            mode: BlendMode::Fixed(1.0),
            softcap: 30.0,
            eps: 1e-12,
        };
        let cfg_mixed = SlotConfig {
            mode: BlendMode::Fixed(0.5),
            softcap: 30.0,
            eps: 1e-12,
        };
        let nll_neural = slot_nll(&logits, 0, 1, &s, &cfg_neural, &mut sn, &mut sb);
        let nll_mixed = slot_nll(&logits, 0, 1, &s, &cfg_mixed, &mut sn, &mut sb);
        assert!(
            nll_mixed < nll_neural,
            "SLOT should reduce NLL when bigram is correct and neural is uniform: mixed={}, neural={}",
            nll_mixed,
            nll_neural
        );
    }

    #[test]
    fn test_slot_nll_sequence_matches_per_position() {
        let s = toy_stats();
        let vocab = 4;
        // 3 token positions, arbitrary logits
        let logits = vec![
            0.0f32, 0.5, -0.1, 0.2, // position 0
            1.0, -0.3, 0.0, 0.4, // position 1
            -0.2, 0.3, 0.9, -0.1, // position 2
        ];
        let prev = vec![0u32, 0, 2];
        let target = vec![1u32, 2, 3];
        let mut nlls = vec![0.0f32; 3];

        let cfg = SlotConfig::default();
        slot_nll_sequence(&logits, &prev, &target, &s, &cfg, vocab, &mut nlls);

        let mut sn = vec![0.0f32; vocab];
        let mut sb = vec![0.0f32; vocab];
        for t in 0..3 {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let nll_manual = slot_nll(row, prev[t], target[t] as usize, &s, &cfg, &mut sn, &mut sb);
            assert!((nlls[t] - nll_manual).abs() < 1e-6);
        }
    }
}
