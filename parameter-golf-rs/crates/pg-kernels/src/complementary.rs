/// Complementary Training (TDD §0, post-professor addendum).
///
/// Tokens that are already predictable by simple bigram statistics get a
/// lower training-loss weight, so the neural model specialises on what
/// n-grams cannot predict. At eval time this frees the n-gram / bandit
/// mixer to use a higher alpha on the tokens where it was already winning,
/// without hurting the tokens where the neural model is strong.
///
/// This module is *data-structure only* — it does not touch the model
/// forward or backward. The training loop is expected to:
///
///   1. Build / load a `BigramStats` table from training-set token shards.
///   2. Each step, compute `complementary_weights(prev, curr, stats, alpha)`
///      producing one weight per target token.
///   3. Multiply per-token losses and `grad_logits` rows by those weights
///      before calling backward / computing the mean loss.
///
/// The dense representation is fine for vocab=1024
/// (1024×1024×4B ≈ 4 MB), which is well below the budget. For larger
/// vocab we would switch to a HashMap<(u32,u32), u32>.
use std::io::Read;

/// Add-one smoothed bigram counts. `counts[prev*V + curr]` is the number
/// of times token `curr` followed `prev`. Row sums are maintained
/// incrementally so `prob()` is O(1).
#[derive(Debug, Clone)]
pub struct BigramStats {
    pub vocab_size: usize,
    /// `counts[prev * vocab_size + curr]`
    pub counts: Vec<u32>,
    /// `row_sums[prev]` = sum of `counts[prev * vocab_size ..]`
    pub row_sums: Vec<u64>,
}

impl BigramStats {
    /// Empty table, all rows sum to zero.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            counts: vec![0u32; vocab_size * vocab_size],
            row_sums: vec![0u64; vocab_size],
        }
    }

    /// Add all adjacent pairs from `tokens` to the table.
    pub fn add_sequence(&mut self, tokens: &[u32]) {
        if tokens.len() < 2 {
            return;
        }
        let v = self.vocab_size;
        for w in tokens.windows(2) {
            let prev = w[0] as usize;
            let curr = w[1] as usize;
            debug_assert!(prev < v && curr < v);
            self.counts[prev * v + curr] = self.counts[prev * v + curr].saturating_add(1);
            self.row_sums[prev] = self.row_sums[prev].saturating_add(1);
        }
    }

    /// Merge another `BigramStats` of identical vocab into `self`.
    pub fn merge(&mut self, other: &BigramStats) {
        assert_eq!(self.vocab_size, other.vocab_size);
        for (a, b) in self.counts.iter_mut().zip(other.counts.iter()) {
            *a = a.saturating_add(*b);
        }
        for (a, b) in self.row_sums.iter_mut().zip(other.row_sums.iter()) {
            *a = a.saturating_add(*b);
        }
    }

    /// Add-one-smoothed probability `P(curr | prev)`.
    /// Handles unseen rows (sum=0) by returning a uniform prior.
    #[inline]
    pub fn prob(&self, prev: u32, curr: u32) -> f32 {
        let v = self.vocab_size as u64;
        let prev = prev as usize;
        let curr = curr as usize;
        let row_sum = self.row_sums[prev];
        let c = self.counts[prev * self.vocab_size + curr] as u64;
        (c as f64 + 1.0).min(f64::MAX) as f32 / ((row_sum + v) as f64).max(1.0) as f32
    }

    /// Raw count of `curr` following `prev`, no smoothing.
    #[inline]
    pub fn count(&self, prev: u32, curr: u32) -> u32 {
        self.counts[prev as usize * self.vocab_size + curr as usize]
    }

    /// Entropy of row `prev` in nats (with add-one smoothing). Used by the
    /// order-adaptive gating code in the hybrid track.
    pub fn row_entropy(&self, prev: u32) -> f32 {
        let v = self.vocab_size;
        let prev_idx = prev as usize;
        let row = &self.counts[prev_idx * v..(prev_idx + 1) * v];
        let denom = (self.row_sums[prev_idx] + v as u64) as f64;
        if denom <= 0.0 {
            return (v as f32).ln();
        }
        let mut h = 0.0f64;
        for &c in row {
            let p = (c as f64 + 1.0) / denom;
            h -= p * p.ln();
        }
        h as f32
    }

    /// Serialize to a little-endian byte vector:
    ///   [u32 vocab_size][u64 row_sums[v]][u32 counts[v*v]]
    pub fn to_bytes(&self) -> Vec<u8> {
        let v = self.vocab_size;
        let mut out = Vec::with_capacity(4 + 8 * v + 4 * v * v);
        out.extend_from_slice(&(v as u32).to_le_bytes());
        for &r in &self.row_sums {
            out.extend_from_slice(&r.to_le_bytes());
        }
        for &c in &self.counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    /// Parse the same format written by `to_bytes`.
    pub fn from_bytes(mut bytes: &[u8]) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let mut v_buf = [0u8; 4];
        bytes.read_exact(&mut v_buf)?;
        let v = u32::from_le_bytes(v_buf) as usize;

        let mut row_sums = vec![0u64; v];
        for r in row_sums.iter_mut() {
            let mut b = [0u8; 8];
            bytes.read_exact(&mut b)?;
            *r = u64::from_le_bytes(b);
        }
        let mut counts = vec![0u32; v * v];
        for c in counts.iter_mut() {
            let mut b = [0u8; 4];
            bytes.read_exact(&mut b)?;
            *c = u32::from_le_bytes(b);
        }
        if !bytes.is_empty() {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "trailing bytes in BigramStats blob",
            ));
        }
        Ok(Self {
            vocab_size: v,
            counts,
            row_sums,
        })
    }
}

/// Shape of the complementary weighting curve.
#[derive(Debug, Clone, Copy)]
pub enum WeightShape {
    /// `w = 1 - alpha * p`. Linear falloff in the bigram probability.
    Linear,
    /// `w = 1 - alpha * p^power`. Sharper near p=1, gentler near p=0.
    /// Typical `power=2` which gives full weight to medium-confidence
    /// tokens and only down-weights the very confident ones.
    Power(f32),
    /// `w = clamp(1 - alpha * max(0, p - floor) / (1 - floor), min_w, 1)`.
    /// Floor-then-linear: tokens with p ≤ floor are unchanged.
    Floored { floor: f32, min_w: f32 },
}

impl Default for WeightShape {
    fn default() -> Self {
        // From the PR #1083 / X-WING branch defaults — tokens below p=0.2
        // are left alone so the bigram can't "poison" the interesting tail.
        WeightShape::Floored {
            floor: 0.2,
            min_w: 0.25,
        }
    }
}

#[inline]
fn apply_shape(p: f32, alpha: f32, shape: WeightShape) -> f32 {
    match shape {
        WeightShape::Linear => (1.0 - alpha * p).max(0.0),
        WeightShape::Power(k) => (1.0 - alpha * p.powf(k)).max(0.0),
        WeightShape::Floored { floor, min_w } => {
            if p <= floor {
                1.0
            } else {
                let norm = (p - floor) / (1.0 - floor).max(1e-6);
                (1.0 - alpha * norm).max(min_w)
            }
        }
    }
}

/// Compute per-target complementary weights.
///
/// `prev_tokens`: token IDs at positions `0..T` (the "context" token)
/// `curr_tokens`: target token IDs at positions `1..=T` (what we predict)
/// `stats`: built bigram table
/// `alpha`: down-weighting strength in `[0, 1]`. `alpha=0` disables.
/// `shape`: the weighting curve. Default is `Floored{floor=0.2, min_w=0.25}`.
/// `weights`: output buffer, length = `curr_tokens.len()`
pub fn complementary_weights(
    prev_tokens: &[u32],
    curr_tokens: &[u32],
    stats: &BigramStats,
    alpha: f32,
    shape: WeightShape,
    weights: &mut [f32],
) {
    assert_eq!(prev_tokens.len(), curr_tokens.len());
    assert_eq!(weights.len(), curr_tokens.len());
    if alpha <= 0.0 {
        weights.fill(1.0);
        return;
    }
    for (i, (&p, &c)) in prev_tokens.iter().zip(curr_tokens.iter()).enumerate() {
        let prob = stats.prob(p, c);
        weights[i] = apply_shape(prob, alpha, shape);
    }
}

/// Mean of per-token losses, weighted by `weights`. The mean is normalized
/// by the *sum* of weights (so down-weighted tokens also shrink the denom).
pub fn weighted_mean_loss(losses: &[f32], weights: &[f32]) -> f32 {
    assert_eq!(losses.len(), weights.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (l, w) in losses.iter().zip(weights.iter()) {
        num += (*l as f64) * (*w as f64);
        den += *w as f64;
    }
    if den <= 1e-12 {
        0.0
    } else {
        (num / den) as f32
    }
}

/// Scale each row of `grad_logits` by the corresponding per-token weight.
/// Call this *after* `cross_entropy_backward` to match the weighted-mean
/// semantics used by `weighted_mean_loss` (but with the gradient version
/// of 1/sum_w, which is already baked into the caller's `grad_loss`).
pub fn scale_grad_logits_by_weight(grad_logits: &mut [f32], weights: &[f32], vocab_size: usize) {
    assert_eq!(grad_logits.len(), weights.len() * vocab_size);
    for (t, &w) in weights.iter().enumerate() {
        let off = t * vocab_size;
        let row = &mut grad_logits[off..off + vocab_size];
        for g in row.iter_mut() {
            *g *= w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_stats() -> BigramStats {
        // vocab = 4, sequence emphasises "0 -> 1" so P(1|0) is very high.
        let mut s = BigramStats::new(4);
        // prev=0: [_, 1] × 100, [_, 2] × 1
        for _ in 0..100 {
            s.add_sequence(&[0, 1]);
        }
        s.add_sequence(&[0, 2]);
        // prev=2: uniform over [1,3]
        for _ in 0..5 {
            s.add_sequence(&[2, 1]);
            s.add_sequence(&[2, 3]);
        }
        s
    }

    #[test]
    fn test_bigram_prob_monotone() {
        let s = toy_stats();
        let p_hi = s.prob(0, 1); // should be ~ 100/(101+4) ≈ 0.96
        let p_lo = s.prob(0, 2); // should be ~ 1/(101+4) ≈ 0.019
        assert!(p_hi > p_lo);
        assert!(p_hi > 0.9);
        assert!(p_lo < 0.1);
    }

    #[test]
    fn test_bigram_prob_unseen_row_uniform() {
        let s = toy_stats();
        // prev=3 never appeared: should be uniform 1/V = 0.25 after add-one.
        let p = s.prob(3, 0);
        assert!((p - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_row_entropy_uniform_vs_peaked() {
        let s = toy_stats();
        let h_peaked = s.row_entropy(0); // mostly "1"
        let h_uniform = s.row_entropy(3); // unseen row -> uniform
        assert!(h_uniform > h_peaked);
        // Unseen row should be close to ln(V).
        let ln_v = (4f32).ln();
        assert!((h_uniform - ln_v).abs() < 1e-4);
    }

    #[test]
    fn test_weight_shapes() {
        // Linear: p=1, alpha=0.5 -> 0.5
        let w = apply_shape(1.0, 0.5, WeightShape::Linear);
        assert!((w - 0.5).abs() < 1e-6);

        // Power(2): p=0.5, alpha=1.0 -> 1 - 0.25 = 0.75
        let w = apply_shape(0.5, 1.0, WeightShape::Power(2.0));
        assert!((w - 0.75).abs() < 1e-6);

        // Floored: p=0.1, floor=0.2 -> 1.0 (below floor, unchanged)
        let w = apply_shape(
            0.1,
            0.9,
            WeightShape::Floored {
                floor: 0.2,
                min_w: 0.25,
            },
        );
        assert!((w - 1.0).abs() < 1e-6);

        // Floored: p=0.6, floor=0.2, alpha=1.0
        //   norm = (0.6-0.2)/0.8 = 0.5
        //   w = 1 - 1*0.5 = 0.5
        let w = apply_shape(
            0.6,
            1.0,
            WeightShape::Floored {
                floor: 0.2,
                min_w: 0.25,
            },
        );
        assert!((w - 0.5).abs() < 1e-6);

        // Floored: p=1.0, alpha=10.0 -> clamped at min_w
        let w = apply_shape(
            1.0,
            10.0,
            WeightShape::Floored {
                floor: 0.2,
                min_w: 0.25,
            },
        );
        assert!((w - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_complementary_weights_downweights_predictable() {
        let s = toy_stats();
        let prev = vec![0u32, 0, 3];
        let curr = vec![1u32, 2, 0];
        let mut w = vec![0.0f32; 3];
        complementary_weights(
            &prev,
            &curr,
            &s,
            0.9,
            WeightShape::Floored {
                floor: 0.2,
                min_w: 0.1,
            },
            &mut w,
        );
        // (0,1): very high prob -> heavily down-weighted
        // (0,2): very low prob  -> unchanged (below floor)
        // (3,0): unseen row     -> p=0.25, barely above floor
        assert!(w[0] < w[1], "predictable token should have lower weight");
        assert!(w[0] < w[2]);
        assert!((w[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_complementary_weights_alpha_zero_is_noop() {
        let s = toy_stats();
        let prev = vec![0u32, 0, 3];
        let curr = vec![1u32, 2, 0];
        let mut w = vec![0.0f32; 3];
        complementary_weights(&prev, &curr, &s, 0.0, WeightShape::default(), &mut w);
        for v in w {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weighted_mean_loss() {
        let losses = vec![1.0f32, 2.0, 3.0];
        let weights = vec![1.0f32, 0.0, 1.0];
        // ignores the middle token entirely
        let m = weighted_mean_loss(&losses, &weights);
        assert!((m - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_grad_logits_by_weight() {
        let mut g = vec![1.0f32; 6]; // 2 tokens × 3 vocab
        let w = vec![0.5f32, 2.0];
        scale_grad_logits_by_weight(&mut g, &w, 3);
        assert_eq!(&g[..3], &[0.5, 0.5, 0.5]);
        assert_eq!(&g[3..], &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_bigram_stats_serde_roundtrip() {
        let s = toy_stats();
        let bytes = s.to_bytes();
        let s2 = BigramStats::from_bytes(&bytes).expect("decode");
        assert_eq!(s.vocab_size, s2.vocab_size);
        assert_eq!(s.counts, s2.counts);
        assert_eq!(s.row_sums, s2.row_sums);
    }

    #[test]
    fn test_bigram_stats_merge() {
        let mut a = BigramStats::new(4);
        let mut b = BigramStats::new(4);
        a.add_sequence(&[0, 1, 2]);
        b.add_sequence(&[0, 1, 3]);
        a.merge(&b);
        // Both sequences pass through (0,1), so count should be 2.
        assert_eq!(a.count(0, 1), 2);
        assert_eq!(a.count(1, 2), 1);
        assert_eq!(a.count(1, 3), 1);
        // row_sums updated
        assert_eq!(a.row_sums[0], 2);
        assert_eq!(a.row_sums[1], 2);
    }
}
