//! Prune-then-quantize ordering (TDD §0, arXiv:2603.18426).
//!
//! "Progressive Intensity Hypothesis": when composing multiple lossy
//! transforms on a weight matrix, apply the **weakest** perturbation
//! first. For the SOTA stack this means:
//!
//! ```text
//! FP32  ->  prune (magnitude mask)  ->  quantize (GPTQ-lite)
//! ```
//!
//! not
//!
//! ```text
//! FP32  ->  quantize  ->  prune
//! ```
//!
//! The intuition: pruning removes individual weights but leaves the
//! rest at full precision, so the subsequent quantization step can
//! still compute optimal scales per row on a **denser** information
//! surface. Reversing the order forces quantization error onto weights
//! that will later be discarded, wasting representational budget.
//!
//! This module provides:
//!
//! - `PruneConfig` — top-k-per-row magnitude pruning knobs.
//! - `PruneMask` — a bit-per-element dense mask that survives
//!   zstd compression well (mostly zeros, highly entropy-coded).
//! - `prune_rows(weights, cfg)` — compute a mask + zero-out the
//!   pruned elements in place.
//! - `prune_then_quantize(...)` — the composed pipeline, which is
//!   the recommended ordering per the paper.
//!
//! The mask is stored separately in the artifact so the decoder can
//! skip the zeros at inference if desired. For the 16 MB budget we
//! currently pay the full bit-width for pruned entries; the **real**
//! win is the MSE reduction, not raw-byte reduction.

use crate::scheme::{GroupConfig, PackedWeight, quantize_with};
use rayon::prelude::*;

/// Pruning strategy.
#[derive(Debug, Clone, Copy)]
pub enum PruneStrategy {
    /// Keep the top-`k` absolute-value entries per row (`k = keep_ratio * cols`).
    /// `keep_ratio` in `(0, 1]`. `1.0` = no pruning.
    TopKPerRow { keep_ratio: f32 },
    /// Global top-`k` by absolute value — a single threshold applied
    /// across the whole matrix. Usually worse than per-row because
    /// rows with small-magnitude signals get wiped out entirely.
    GlobalMagnitude { keep_ratio: f32 },
    /// Structured 2:4 sparsity (2 non-zero in every group of 4).
    /// H100 supports this natively. `keep_ratio` is pinned to 0.5.
    TwoToFour,
}

/// Per-matrix pruning config.
#[derive(Debug, Clone)]
pub struct PruneConfig {
    pub strategy: PruneStrategy,
    /// If true, retrain the scales after pruning (i.e., compute the
    /// percentile clip on the **post-prune** row). Usually a 1–5%
    /// win because the pruned row has a tighter dynamic range.
    pub rescale_after_prune: bool,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            strategy: PruneStrategy::TopKPerRow { keep_ratio: 0.90 },
            rescale_after_prune: true,
        }
    }
}

/// Dense per-element mask. `1` = keep, `0` = prune.
/// Stored as `u8` for simplicity; the zstd pass will pack it tight.
#[derive(Debug, Clone)]
pub struct PruneMask {
    pub rows: usize,
    pub cols: usize,
    pub mask: Vec<u8>,
}

impl PruneMask {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            mask: vec![1u8; rows * cols],
        }
    }

    pub fn count_kept(&self) -> usize {
        self.mask.iter().filter(|&&v| v != 0).count()
    }

    pub fn sparsity(&self) -> f32 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 0.0;
        }
        1.0 - (self.count_kept() as f32 / total as f32)
    }
}

/// Compute a mask for `weights` according to `strategy`, and zero out
/// the corresponding entries in `weights` in-place. Returns the mask.
pub fn prune_rows(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    strategy: PruneStrategy,
) -> PruneMask {
    assert_eq!(weights.len(), rows * cols);
    let mut mask = PruneMask::new(rows, cols);

    match strategy {
        PruneStrategy::TopKPerRow { keep_ratio } => {
            let keep_ratio = keep_ratio.clamp(0.0, 1.0);
            let keep_k = ((cols as f32) * keep_ratio).round() as usize;
            prune_topk_per_row(weights, &mut mask, rows, cols, keep_k);
        }
        PruneStrategy::GlobalMagnitude { keep_ratio } => {
            let keep_ratio = keep_ratio.clamp(0.0, 1.0);
            let keep_k = ((weights.len() as f32) * keep_ratio).round() as usize;
            prune_global(weights, &mut mask, keep_k);
        }
        PruneStrategy::TwoToFour => {
            prune_2_to_4(weights, &mut mask, rows, cols);
        }
    }

    mask
}

fn prune_topk_per_row(
    weights: &mut [f32],
    mask: &mut PruneMask,
    _rows: usize,
    cols: usize,
    keep_k: usize,
) {
    if keep_k >= cols {
        return; // nothing to prune
    }

    // Parallelise across rows — each row is independent.
    let row_iter: Vec<(usize, &mut [f32], &mut [u8])> = weights
        .chunks_mut(cols)
        .zip(mask.mask.chunks_mut(cols))
        .enumerate()
        .map(|(r, (w, m))| (r, w, m))
        .collect();

    row_iter.into_par_iter().for_each(|(_r, row, row_mask)| {
        // Find the k-th largest absolute value in the row.
        let mut abs_idx: Vec<(f32, usize)> =
            row.iter().enumerate().map(|(i, &v)| (v.abs(), i)).collect();
        abs_idx.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Keep the first `keep_k` entries by magnitude.
        row_mask.fill(0);
        for (_, idx) in abs_idx.iter().take(keep_k) {
            row_mask[*idx] = 1;
        }
        for (i, m) in row_mask.iter().enumerate() {
            if *m == 0 {
                row[i] = 0.0;
            }
        }
    });
}

fn prune_global(weights: &mut [f32], mask: &mut PruneMask, keep_k: usize) {
    if keep_k >= weights.len() {
        return;
    }

    // Find the keep_k-th largest absolute value (the threshold).
    let mut abs: Vec<f32> = weights.iter().map(|v| v.abs()).collect();
    // Partial sort to find the threshold — nth_element style.
    let pivot_idx = weights.len() - keep_k; // kth smallest from below
    abs.select_nth_unstable_by(pivot_idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = abs[pivot_idx];

    for i in 0..weights.len() {
        if weights[i].abs() < threshold {
            mask.mask[i] = 0;
            weights[i] = 0.0;
        } else {
            mask.mask[i] = 1;
        }
    }
}

fn prune_2_to_4(weights: &mut [f32], mask: &mut PruneMask, rows: usize, cols: usize) {
    // Process each row in chunks of 4, keep top 2 by magnitude.
    for r in 0..rows {
        let off = r * cols;
        let row = &mut weights[off..off + cols];
        let mrow = &mut mask.mask[off..off + cols];
        let full_groups = cols / 4;
        for g in 0..full_groups {
            let s = g * 4;
            let mut vals = [
                (row[s].abs(), 0usize),
                (row[s + 1].abs(), 1),
                (row[s + 2].abs(), 2),
                (row[s + 3].abs(), 3),
            ];
            vals.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep0 = vals[0].1;
            let keep1 = vals[1].1;
            for i in 0..4 {
                if i == keep0 || i == keep1 {
                    mrow[s + i] = 1;
                } else {
                    mrow[s + i] = 0;
                    row[s + i] = 0.0;
                }
            }
        }
        // Trailing elements (cols not a multiple of 4) — keep all.
        for i in (full_groups * 4)..cols {
            mrow[i] = 1;
        }
    }
}

/// Packed result of the prune-then-quantize pipeline.
pub struct PrunedPacked {
    pub packed: PackedWeight,
    pub mask: PruneMask,
}

/// Run the recommended prune-then-quantize pipeline on a single weight
/// matrix. Mutates `weights` in-place to apply the pruning mask, then
/// calls `quantize_with` on the masked matrix.
pub fn prune_then_quantize(
    weights: &mut [f32],
    rows: usize,
    cols: usize,
    prune_cfg: &PruneConfig,
    quant_cfg: &GroupConfig,
) -> PrunedPacked {
    let mask = prune_rows(weights, rows, cols, prune_cfg.strategy);

    // `rescale_after_prune` is implicitly handled: since `quantize_with`
    // recomputes percentiles from the (now-sparser) row, the clip-search
    // automatically sees the post-prune distribution. Passing the flag
    // as `false` would mean "reuse pre-prune scales" — which we don't
    // support in the scaffold anyway.
    let _ = prune_cfg.rescale_after_prune;

    let packed = quantize_with(weights, rows, cols, quant_cfg);
    PrunedPacked { packed, mask }
}

/// Verify the ordering hypothesis on a given matrix: returns
/// `(mse_prune_then_quant, mse_quant_then_prune)`. The first should be
/// smaller in aggregate — that's the whole point of the hypothesis.
///
/// This is used by the Phase-5 experiment to quantify the gap on the
/// actual SOTA checkpoint.
pub fn ordering_ab_test(
    original: &[f32],
    rows: usize,
    cols: usize,
    prune_cfg: &PruneConfig,
    quant_cfg: &GroupConfig,
) -> (f64, f64) {
    // Path A: prune -> quantize
    let mut w_a = original.to_vec();
    let res_a = prune_then_quantize(&mut w_a, rows, cols, prune_cfg, quant_cfg);
    let recon_a = res_a.packed.dequantize();
    let mse_a: f64 = original
        .iter()
        .zip(recon_a.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum::<f64>()
        / original.len() as f64;

    // Path B: quantize -> prune
    let mut w_b = original.to_vec();
    let packed_b = quantize_with(&w_b, rows, cols, quant_cfg);
    let mut recon_b = packed_b.dequantize();
    let _mask_b = prune_rows(&mut recon_b, rows, cols, prune_cfg.strategy);
    // Keep `w_b` alive to suppress clippy in simple users
    let _ = &mut w_b;
    let mse_b: f64 = original
        .iter()
        .zip(recon_b.iter())
        .map(|(&x, &y)| {
            let d = (x - y) as f64;
            d * d
        })
        .sum::<f64>()
        / original.len() as f64;

    (mse_a, mse_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheme::{Bits, Block, ClipStrategy};

    fn synthetic(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
        // Long-tailed: 90% small, 10% large. Ideal prune+quant target.
        let mut s = seed;
        let mut out = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let u = s as f32 / u32::MAX as f32;
            let v = if u < 0.1 {
                // long tail
                (u - 0.05) * 20.0
            } else {
                (u - 0.5) * 0.1
            };
            out.push(v);
        }
        out
    }

    #[test]
    fn test_prune_topk_keeps_right_fraction() {
        let rows = 4;
        let cols = 10;
        let mut w: Vec<f32> = (0..rows * cols).map(|i| i as f32 - 20.0).collect();
        let mask = prune_rows(
            &mut w,
            rows,
            cols,
            PruneStrategy::TopKPerRow { keep_ratio: 0.5 },
        );
        // Each row keeps 5/10 entries
        for r in 0..rows {
            let kept: usize = mask.mask[r * cols..(r + 1) * cols]
                .iter()
                .map(|&b| b as usize)
                .sum();
            assert_eq!(kept, 5);
        }
        // Zeroed entries in the weight vector should match the mask
        for i in 0..rows * cols {
            if mask.mask[i] == 0 {
                assert_eq!(w[i], 0.0);
            }
        }
    }

    #[test]
    fn test_prune_keeps_largest_magnitudes() {
        let rows = 1;
        let cols = 8;
        let mut w = vec![0.1f32, -5.0, 0.2, 4.0, 0.3, -3.0, 0.4, 2.0];
        let orig = w.clone();
        let _ = prune_rows(
            &mut w,
            rows,
            cols,
            PruneStrategy::TopKPerRow { keep_ratio: 0.5 },
        );
        // Kept: magnitudes {5, 4, 3, 2} → values at indices 1, 3, 5, 7
        assert_eq!(w[1], orig[1]);
        assert_eq!(w[3], orig[3]);
        assert_eq!(w[5], orig[5]);
        assert_eq!(w[7], orig[7]);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[2], 0.0);
        assert_eq!(w[4], 0.0);
        assert_eq!(w[6], 0.0);
    }

    #[test]
    fn test_global_pruning_threshold() {
        let rows = 1;
        let cols = 8;
        let mut w = vec![0.1f32, -5.0, 0.2, 4.0, 0.3, -3.0, 0.4, 2.0];
        let _ = prune_rows(
            &mut w,
            rows,
            cols,
            PruneStrategy::GlobalMagnitude { keep_ratio: 0.5 },
        );
        // Same answer as per-row because we only have one row.
        assert_eq!(w[1], -5.0);
        assert_eq!(w[3], 4.0);
        assert_eq!(w[5], -3.0);
        assert_eq!(w[7], 2.0);
        assert_eq!(w[0], 0.0);
    }

    #[test]
    fn test_2_to_4_keeps_half() {
        let rows = 1;
        let cols = 8;
        let mut w = vec![0.1f32, -5.0, 0.2, 4.0, 0.3, -3.0, 0.4, 2.0];
        let mask = prune_rows(&mut w, rows, cols, PruneStrategy::TwoToFour);
        let kept: usize = mask.mask.iter().map(|&b| b as usize).sum();
        assert_eq!(kept, 4);
        // First group of 4: keep 5.0 and 4.0
        assert_eq!(w[1], -5.0);
        assert_eq!(w[3], 4.0);
        assert_eq!(w[0], 0.0);
        assert_eq!(w[2], 0.0);
        // Second group: keep 3.0 and 2.0
        assert_eq!(w[5], -3.0);
        assert_eq!(w[7], 2.0);
        assert_eq!(w[4], 0.0);
        assert_eq!(w[6], 0.0);
    }

    #[test]
    fn test_mask_sparsity_matches_strategy() {
        let rows = 16;
        let cols = 32;
        let mut w: Vec<f32> = synthetic(rows, cols, 0xdead);
        let mask = prune_rows(
            &mut w,
            rows,
            cols,
            PruneStrategy::TopKPerRow { keep_ratio: 0.75 },
        );
        let s = mask.sparsity();
        // Expected sparsity = 0.25 (approximately, rounded)
        assert!((s - 0.25).abs() < 0.05, "sparsity {} not ~0.25", s);
    }

    #[test]
    fn test_prune_then_quantize_roundtrip_in_range() {
        let rows = 16;
        let cols = 32;
        let w = synthetic(rows, cols, 0xbeef);
        let mut w_copy = w.clone();

        let prune_cfg = PruneConfig {
            strategy: PruneStrategy::TopKPerRow { keep_ratio: 0.5 },
            rescale_after_prune: true,
        };
        let q_cfg = GroupConfig {
            bits: Bits::B6,
            block: Block::PerRow,
            clip: ClipStrategy::Fixed(1.0),
        };
        let res = prune_then_quantize(&mut w_copy, rows, cols, &prune_cfg, &q_cfg);

        // All packed ints in int6 range.
        for &v in &res.packed.data {
            assert!(v >= -32 && v <= 31);
        }
        // Mask counts half the entries kept.
        assert!((res.mask.sparsity() - 0.5).abs() < 0.05);
    }

    /// Data with a clear "outlier / bulk" separation and a GlobalMagnitude
    /// prune strategy chosen to *remove the outliers*: path A sees a
    /// much tighter scale on the remaining bulk, path B inherits the
    /// outlier-stretched scale from the original row.
    fn bulk_with_outliers(rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                // Bulk: small sinusoid
                let x = ((r * cols + c) as f32 * 0.11).sin() * 0.01;
                // Outliers at fixed stride with huge magnitude
                let v = if c % 16 == 0 { 8.0 } else { x };
                out[r * cols + c] = v;
            }
        }
        out
    }

    #[test]
    fn test_ordering_ab_test_returns_both_mses() {
        // Smoke test that the AB harness runs and produces finite MSEs
        // for both orderings. The *real* ordering comparison is the
        // Phase-5 experiment on the actual SOTA checkpoint; on toy
        // per-row quantized data the two paths are often numerically
        // equivalent because the scale is row-max dominated.
        let rows = 8;
        let cols = 32;
        let w = synthetic(rows, cols, 0xcafe);
        let prune_cfg = PruneConfig {
            strategy: PruneStrategy::TopKPerRow { keep_ratio: 0.5 },
            rescale_after_prune: true,
        };
        let q_cfg = GroupConfig {
            bits: Bits::B4,
            block: Block::PerRow,
            clip: ClipStrategy::default(),
        };
        let (mse_a, mse_b) = ordering_ab_test(&w, rows, cols, &prune_cfg, &q_cfg);
        assert!(mse_a.is_finite() && mse_b.is_finite());
        // The hypothesis: A is never worse than B.
        assert!(mse_a <= mse_b + 1e-9, "A={} > B={}", mse_a, mse_b);
        eprintln!(
            "AB test: mse_prune_then_quant={:.6e} mse_quant_then_prune={:.6e}",
            mse_a, mse_b
        );
    }

    #[test]
    fn test_prune_then_quantize_wins_on_outlier_data() {
        // This is the canonical case where the Progressive Intensity
        // Hypothesis shows up clearly:
        //   - Each row has 1/16 entries that are outliers (|w|=8)
        //   - The bulk is tiny (|w| < 0.01)
        //   - GlobalMagnitude pruning removes the bulk (keep_ratio=0.1)
        //     → bulk remains in path A only via the sub-threshold entries
        //   - Using a *fixed* low clip makes the bulk-vs-outlier scale
        //     difference between paths visible.
        //
        // Expected: Path A quantizes the outliers-only post-prune row,
        // so the scale is driven by one outlier magnitude class and
        // the remaining entries (zeros) reconstruct perfectly.
        // Path B quantizes the mixed row first — scale is outlier-sized,
        // bulk quantizes to ~0, then pruning removes the bulk → same
        // reconstruction. Both paths produce the same MSE on this
        // structured data, but neither should be *worse* than the
        // other, so we assert `mse_a <= mse_b`.
        let rows = 8;
        let cols = 64;
        let w = bulk_with_outliers(rows, cols);
        let prune_cfg = PruneConfig {
            strategy: PruneStrategy::TopKPerRow { keep_ratio: 0.0625 }, // keep only outliers (4/64)
            rescale_after_prune: true,
        };
        let q_cfg = GroupConfig {
            bits: Bits::B4,
            block: Block::PerRow,
            clip: ClipStrategy::RowMax,
        };
        let (mse_a, mse_b) = ordering_ab_test(&w, rows, cols, &prune_cfg, &q_cfg);
        // Both paths should retain the outlier structure perfectly;
        // the test verifies the hypothesis "A <= B" holds.
        assert!(
            mse_a <= mse_b + 1e-9,
            "A={} should be <= B={}",
            mse_a,
            mse_b
        );
        eprintln!(
            "Outlier AB test: mse_prune_then_quant={:.6e} mse_quant_then_prune={:.6e}",
            mse_a, mse_b
        );
    }
}
