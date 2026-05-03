/// Quantization-Scheme Compiler — runtime scaffold (TDD §1, Innovation 1).
///
/// This is the runtime side of what will eventually become a
/// `#[derive(QuantKernels)]` proc macro. The macro will read a struct of
/// const-generic markers (B4/B5/B6/.., PerRow/Block32/..) and emit one
/// specialized CubeCL kernel per (bits, block) pair found in the struct.
///
/// Until the macro exists, we represent schemes as plain runtime configs and
/// dispatch to a generic per-row quantizer parameterized on bit width and
/// block size. This is enough to:
///   1. Sweep ~50 schemes via `sweep::run_sweep`
///   2. Validate the math against the existing `int6.rs` GPTQ-lite path
///   3. Lock down the data layout that the future packed format will use
///
/// All schemes here are *uniform per-row* in this scaffold. Block-quantized
/// variants are listed in the config space but quantize as `PerRow` for now —
/// they will land when the packing kernels are implemented.
use rayon::prelude::*;

/// Bit width for one weight group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bits {
    B4,
    B5,
    B6,
    B7,
    B8,
}

impl Bits {
    /// Maximum representable signed value, i.e. `2^(bits-1) - 1`.
    pub fn qmax(self) -> i32 {
        match self {
            Bits::B4 => 7,
            Bits::B5 => 15,
            Bits::B6 => 31,
            Bits::B7 => 63,
            Bits::B8 => 127,
        }
    }

    /// Minimum representable signed value, i.e. `-2^(bits-1)`.
    pub fn qmin(self) -> i32 {
        -(self.qmax() + 1)
    }

    /// Bits-per-element. Used by the packed format size estimator.
    pub fn nbits(self) -> usize {
        match self {
            Bits::B4 => 4,
            Bits::B5 => 5,
            Bits::B6 => 6,
            Bits::B7 => 7,
            Bits::B8 => 8,
        }
    }
}

/// Block size for sharing scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Block {
    PerRow,
    B16,
    B32,
    B64,
    B128,
}

impl Block {
    /// Element count covered by one scale. `PerRow` returns the row width.
    pub fn elements_per_scale(self, row_width: usize) -> usize {
        match self {
            Block::PerRow => row_width,
            Block::B16 => 16,
            Block::B32 => 32,
            Block::B64 => 64,
            Block::B128 => 128,
        }
    }
}

/// Clip percentile search strategy. The first entry is the value used when
/// `ClipStrategy::Fixed(_)` is selected.
#[derive(Debug, Clone)]
pub enum ClipStrategy {
    /// Try each percentile, pick the one with min reconstruction MSE.
    PercentileSearch(Vec<f32>),
    /// Use a fixed clip percentile (no search).
    Fixed(f32),
    /// Always use the row maximum (1.0 percentile).
    RowMax,
}

impl Default for ClipStrategy {
    fn default() -> Self {
        ClipStrategy::PercentileSearch(vec![0.999, 0.9995, 0.9999, 0.99999, 1.0])
    }
}

/// Per-group config — one of these per "weight type" in the scheme.
#[derive(Debug, Clone)]
pub struct GroupConfig {
    pub bits: Bits,
    pub block: Block,
    pub clip: ClipStrategy,
}

impl GroupConfig {
    pub fn new(bits: Bits, block: Block) -> Self {
        Self {
            bits,
            block,
            clip: ClipStrategy::default(),
        }
    }
}

/// Full quantization scheme — one entry per weight group in the model.
#[derive(Debug, Clone)]
pub struct Scheme {
    pub attn_q: GroupConfig,
    pub attn_k: GroupConfig,
    pub attn_v: GroupConfig,
    pub attn_o: GroupConfig,
    pub mlp_up: GroupConfig,
    pub mlp_down: GroupConfig,
    pub embed: GroupConfig,
}

impl Scheme {
    /// Current SOTA consensus: int6 attention, int5 MLP, int8 embeddings.
    pub fn sota_baseline() -> Self {
        let int6 = GroupConfig::new(Bits::B6, Block::PerRow);
        let int5 = GroupConfig::new(Bits::B5, Block::PerRow);
        let int8 = GroupConfig::new(Bits::B8, Block::PerRow);
        Self {
            attn_q: int6.clone(),
            attn_k: int6.clone(),
            attn_v: int6.clone(),
            attn_o: int6,
            mlp_up: int5.clone(),
            mlp_down: int5,
            embed: int8,
        }
    }

    /// All groups use the same config — for fast scans.
    pub fn uniform(bits: Bits) -> Self {
        let g = GroupConfig::new(bits, Block::PerRow);
        Self {
            attn_q: g.clone(),
            attn_k: g.clone(),
            attn_v: g.clone(),
            attn_o: g.clone(),
            mlp_up: g.clone(),
            mlp_down: g.clone(),
            embed: g,
        }
    }

    /// Inverted: int5 attention, int6 MLP — tests the alternate split.
    pub fn inverted_split() -> Self {
        let int5 = GroupConfig::new(Bits::B5, Block::PerRow);
        let int6 = GroupConfig::new(Bits::B6, Block::PerRow);
        let int8 = GroupConfig::new(Bits::B8, Block::PerRow);
        Self {
            attn_q: int5.clone(),
            attn_k: int5.clone(),
            attn_v: int5.clone(),
            attn_o: int5,
            mlp_up: int6.clone(),
            mlp_down: int6,
            embed: int8,
        }
    }

    /// Aggressive: 4-bit attention, 5-bit MLP, 6-bit embeddings — frontier.
    pub fn aggressive() -> Self {
        let int4 = GroupConfig::new(Bits::B4, Block::PerRow);
        let int5 = GroupConfig::new(Bits::B5, Block::PerRow);
        let int6 = GroupConfig::new(Bits::B6, Block::PerRow);
        Self {
            attn_q: int4.clone(),
            attn_k: int4.clone(),
            attn_v: int4.clone(),
            attn_o: int4,
            mlp_up: int5.clone(),
            mlp_down: int5,
            embed: int6,
        }
    }

    /// Estimate compressed size in bytes assuming a 0.65 zstd-22 ratio.
    /// Useful for filtering schemes that are obviously over budget.
    pub fn estimate_size(
        &self,
        attn_qkvo_elems: usize,
        mlp_up_elems: usize,
        mlp_down_elems: usize,
        embed_elems: usize,
        zstd_ratio: f32,
    ) -> usize {
        let raw_bits = (attn_qkvo_elems / 4)
            * (self.attn_q.bits.nbits()
                + self.attn_k.bits.nbits()
                + self.attn_v.bits.nbits()
                + self.attn_o.bits.nbits())
            + mlp_up_elems * self.mlp_up.bits.nbits()
            + mlp_down_elems * self.mlp_down.bits.nbits()
            + embed_elems * self.embed.bits.nbits();
        let raw_bytes = (raw_bits + 7) / 8;
        (raw_bytes as f32 * zstd_ratio) as usize
    }
}

/// A single quantized weight buffer + the scales it needs to be dequantized.
pub struct PackedWeight {
    pub bits: Bits,
    pub block: Block,
    pub data: Vec<i8>,    // signed integers in [bits.qmin(), bits.qmax()]
    pub scales: Vec<f32>, // one per scale group
    pub rows: usize,
    pub cols: usize,
    pub mse: f64, // sum-squared reconstruction error
}

impl PackedWeight {
    /// Reconstruct the f32 weight matrix from the packed integers and scales.
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.rows * self.cols];
        let stride = self.block.elements_per_scale(self.cols);
        for r in 0..self.rows {
            let row_off = r * self.cols;
            for c in 0..self.cols {
                // Per-row schemes use one scale per row; block schemes use one per
                // (row, c/block) tile.
                let scale_idx = match self.block {
                    Block::PerRow => r,
                    _ => r * ((self.cols + stride - 1) / stride) + (c / stride),
                };
                out[row_off + c] = self.data[row_off + c] as f32 * self.scales[scale_idx];
            }
        }
        out
    }

    /// Bytes occupied by the packed payload (raw, before zstd).
    pub fn raw_bytes(&self) -> usize {
        // We store the integers as i8 in this scaffold even when bits < 8.
        // The eventual GPU/serialized form will pack tighter; this estimator
        // uses the *theoretical* packed size so the sweep is realistic.
        let weight_bits = self.rows * self.cols * self.bits.nbits();
        let weight_bytes = (weight_bits + 7) / 8;
        let scale_bytes = self.scales.len() * 2; // f16
        weight_bytes + scale_bytes
    }
}

/// Quantize a single row at the requested bit width using percentile clip search.
fn quantize_row_bits(row: &[f32], bits: Bits, clip: &ClipStrategy) -> (Vec<i8>, f32, f64) {
    let n = row.len();
    let qmax = bits.qmax();
    let qmin = bits.qmin();
    let qrange = qmax as f32; // we treat the positive limit as the clamp anchor

    let mut sorted_abs: Vec<f32> = row.iter().map(|x| x.abs()).collect();
    sorted_abs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let percentiles: Vec<f32> = match clip {
        ClipStrategy::PercentileSearch(p) => p.clone(),
        ClipStrategy::Fixed(p) => vec![*p],
        ClipStrategy::RowMax => vec![1.0],
    };

    let mut best_q: Option<Vec<i8>> = None;
    let mut best_scale = 1.0f32;
    let mut best_mse = f64::MAX;

    for pct in percentiles {
        let idx = ((n as f32 * pct) as usize).min(n - 1);
        let clip_val = sorted_abs[idx].max(1e-8);
        let scale = (clip_val / qrange).max(1e-8);

        let mut mse = 0.0f64;
        let q: Vec<i8> = row
            .iter()
            .map(|&x| {
                let v = (x / scale).round().clamp(qmin as f32, qmax as f32) as i8;
                let recon = v as f32 * scale;
                let e = (x - recon) as f64;
                mse += e * e;
                v
            })
            .collect();

        if mse < best_mse {
            best_mse = mse;
            best_q = Some(q);
            best_scale = scale;
        }
    }

    (best_q.unwrap(), best_scale, best_mse)
}

/// Quantize a 2D weight matrix according to a `GroupConfig`. Per-row only in
/// this scaffold (block is recorded in the result but treated as PerRow).
pub fn quantize_with(weights: &[f32], rows: usize, cols: usize, cfg: &GroupConfig) -> PackedWeight {
    assert_eq!(weights.len(), rows * cols);

    let row_results: Vec<(Vec<i8>, f32, f64)> = (0..rows)
        .into_par_iter()
        .map(|r| quantize_row_bits(&weights[r * cols..(r + 1) * cols], cfg.bits, &cfg.clip))
        .collect();

    let mut data = Vec::with_capacity(rows * cols);
    let mut scales = Vec::with_capacity(rows);
    let mut total_mse = 0.0f64;
    for (q, s, m) in row_results {
        data.extend_from_slice(&q);
        scales.push(s);
        total_mse += m;
    }

    PackedWeight {
        bits: cfg.bits,
        block: cfg.block,
        data,
        scales,
        rows,
        cols,
        mse: total_mse,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|i| (i as f32 * 0.123).sin() * 0.5)
            .collect()
    }

    #[test]
    fn test_bits_qmax_qmin() {
        assert_eq!(Bits::B4.qmax(), 7);
        assert_eq!(Bits::B4.qmin(), -8);
        assert_eq!(Bits::B6.qmax(), 31);
        assert_eq!(Bits::B6.qmin(), -32);
        assert_eq!(Bits::B8.qmax(), 127);
        assert_eq!(Bits::B8.qmin(), -128);
    }

    #[test]
    fn test_quantize_with_int6_round_trip() {
        let rows = 32;
        let cols = 64;
        let w = synthetic(rows, cols);
        let cfg = GroupConfig::new(Bits::B6, Block::PerRow);
        let q = quantize_with(&w, rows, cols, &cfg);

        for &v in &q.data {
            assert!(v >= -32 && v <= 31, "int6 out of range: {}", v);
        }

        let recon = q.dequantize();
        let mse: f64 = w
            .iter()
            .zip(recon.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / (rows * cols) as f64;
        assert!(mse < 0.01, "int6 mse too high: {}", mse);
    }

    #[test]
    fn test_int8_better_than_int4() {
        let rows = 32;
        let cols = 64;
        let w = synthetic(rows, cols);
        let q4 = quantize_with(&w, rows, cols, &GroupConfig::new(Bits::B4, Block::PerRow));
        let q8 = quantize_with(&w, rows, cols, &GroupConfig::new(Bits::B8, Block::PerRow));
        // Bit widths monotonic in MSE
        assert!(q8.mse < q4.mse, "int8 should have lower MSE than int4");
    }

    #[test]
    fn test_baseline_size_estimate_under_budget() {
        // SOTA model: ~22M attn + ~17M mlp + 0.5M embed
        let scheme = Scheme::sota_baseline();
        let attn = 2 * 11 * 512 * 512 + 2 * 11 * 256 * 512; // ~8.5M
        let mlp_up = 11 * 1536 * 512; // ~8.6M
        let mlp_down = 11 * 512 * 1536; // ~8.6M
        let embed = 1024 * 512;
        let est = scheme.estimate_size(attn, mlp_up, mlp_down, embed, 0.65);
        // Should be a few MB at 0.65 zstd ratio (the 16MB budget includes scales,
        // headers, etc.; this estimator only counts integers).
        assert!(est < 16_000_000, "size estimate over budget: {} bytes", est);
        eprintln!("Estimated baseline scheme size: {} bytes", est);
    }

    #[test]
    fn test_aggressive_smaller_than_baseline() {
        let attn = 2 * 11 * 512 * 512 + 2 * 11 * 256 * 512;
        let mlp_up = 11 * 1536 * 512;
        let mlp_down = 11 * 512 * 1536;
        let embed = 1024 * 512;
        let baseline = Scheme::sota_baseline().estimate_size(attn, mlp_up, mlp_down, embed, 0.65);
        let aggressive = Scheme::aggressive().estimate_size(attn, mlp_up, mlp_down, embed, 0.65);
        assert!(aggressive < baseline);
    }
}
