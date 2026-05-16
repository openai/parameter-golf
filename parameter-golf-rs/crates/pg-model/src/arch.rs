/// Const-generic architecture specialization (TDD §3, Innovation 3).
///
/// `Arch<...>` is a phantom type whose const generics fully describe an
/// architecture variant. It exists so that the architecture sweep can be
/// expressed as 10 distinct types, each of which generates its own
/// specialized kernels at compile time (CubeCL `#[comptime]` parameters
/// fold the constants into immediate values).
///
/// At the CPU-reference level, `Arch::config()` simply returns a
/// `ModelConfig` whose fields are determined by the const generics. The GPU
/// runner takes the same const generics and uses them as `#[comptime]`
/// parameters into CubeCL kernels.
///
/// ## Example
///
/// ```ignore
/// use pg_model::arch::{Arch, ArchTrait, BaselineArch, WideArch};
///
/// let cfg_baseline = BaselineArch::config();   // d=512, 11 layers
/// let cfg_wide     = WideArch::config();       // d=576, 11 layers
/// ```
///
/// The point is that adding a new architecture is one type alias, not a
/// new code path. The training and eval entry points are generic over
/// `A: ArchTrait`.
use std::marker::PhantomData;

use crate::config::ModelConfig;

/// Architecture descriptor — purely a compile-time type, zero runtime size.
pub struct Arch<
    const D: usize,
    const HEADS: usize,
    const KV_HEADS: usize,
    const LAYERS: usize,
    const MLP_MULT: usize,
    const ROPE_DIMS: usize,
    const XSA_FROM: usize,
    const SEQ_LEN: usize,
> {
    _phantom: PhantomData<()>,
}

/// Trait so we can write training/eval code that's generic over `Arch`.
pub trait ArchTrait {
    const D: usize;
    const HEADS: usize;
    const KV_HEADS: usize;
    const LAYERS: usize;
    const MLP_MULT: usize;
    const ROPE_DIMS: usize;
    const XSA_FROM: usize;
    const SEQ_LEN: usize;

    /// `head_dim = D / HEADS` — must be ≥ ROPE_DIMS.
    const HEAD_DIM: usize = Self::D / Self::HEADS;

    /// `kv_dim = KV_HEADS * HEAD_DIM` (computed from HEAD_DIM).
    const KV_DIM: usize = Self::KV_HEADS * (Self::D / Self::HEADS);

    /// `mlp_dim = MLP_MULT * D`.
    const MLP_DIM: usize = Self::MLP_MULT * Self::D;

    /// Number of XSA layers (last `LAYERS - XSA_FROM` layers).
    /// `XSA_FROM == LAYERS` means XSA is disabled.
    const XSA_LAST_N: usize = if Self::XSA_FROM >= Self::LAYERS {
        0
    } else {
        Self::LAYERS - Self::XSA_FROM
    };

    /// Build a `ModelConfig` whose fields match these const generics.
    /// Other (non-architectural) fields take their values from
    /// `ModelConfig::sota()` so that we only override what changes.
    fn config() -> ModelConfig {
        let base = ModelConfig::sota();
        ModelConfig {
            num_layers: Self::LAYERS,
            model_dim: Self::D,
            num_heads: Self::HEADS,
            num_kv_heads: Self::KV_HEADS,
            head_dim: Self::HEAD_DIM,
            mlp_mult: Self::MLP_MULT as f32,
            mlp_dim: Self::MLP_DIM,
            rope_dims: Self::ROPE_DIMS,
            xsa_last_n: Self::XSA_LAST_N,
            train_seq_len: Self::SEQ_LEN,
            eval_seq_len: Self::SEQ_LEN,
            ..base
        }
    }

    /// Short label for logging / artifact naming.
    fn label() -> String {
        format!(
            "d{}_h{}_kv{}_L{}_m{}_r{}_x{}_s{}",
            Self::D,
            Self::HEADS,
            Self::KV_HEADS,
            Self::LAYERS,
            Self::MLP_MULT,
            Self::ROPE_DIMS,
            Self::XSA_FROM,
            Self::SEQ_LEN,
        )
    }
}

impl<
    const D: usize,
    const HEADS: usize,
    const KV_HEADS: usize,
    const LAYERS: usize,
    const MLP_MULT: usize,
    const ROPE_DIMS: usize,
    const XSA_FROM: usize,
    const SEQ_LEN: usize,
> ArchTrait for Arch<D, HEADS, KV_HEADS, LAYERS, MLP_MULT, ROPE_DIMS, XSA_FROM, SEQ_LEN>
{
    const D: usize = D;
    const HEADS: usize = HEADS;
    const KV_HEADS: usize = KV_HEADS;
    const LAYERS: usize = LAYERS;
    const MLP_MULT: usize = MLP_MULT;
    const ROPE_DIMS: usize = ROPE_DIMS;
    const XSA_FROM: usize = XSA_FROM;
    const SEQ_LEN: usize = SEQ_LEN;
}

// === Sweep variants (TDD §3.5) ===
//
// Adding a new architecture to the sweep = adding one type alias here.
// Every kernel that's generic over `A: ArchTrait` is automatically
// re-specialized.

/// Current SOTA: 512×11×3, partial RoPE 16/64, XSA on last 4 layers.
pub type BaselineArch = Arch<512, 8, 4, 11, 3, 16, 7, 2048>;

/// Wider hidden state — needs more compute per step but fewer steps may suffice.
/// Note: 576/8 = 72 head dim, not 64. Use only if Innovation 1's quant sweep
/// proves int4 viability so the model still fits in 16MB.
pub type WideArch = Arch<576, 8, 4, 11, 3, 16, 7, 2048>;

/// Wider with narrower MLP — same artifact size as baseline.
pub type WideNarrowMlpArch = Arch<576, 8, 4, 11, 2, 16, 7, 2048>;

/// Deeper but narrower — 13 layers at d=512.
pub type DeepArch = Arch<512, 8, 4, 13, 3, 16, 9, 2048>;

/// Very deep, narrow — 16 layers at d=384.
pub type DeepNarrowArch = Arch<384, 6, 3, 16, 3, 12, 12, 2048>;

/// All layers use XSA (XSA_FROM=0).
pub type FullXsaArch = Arch<512, 8, 4, 11, 3, 16, 0, 2048>;

/// Doubled RoPE coverage: 32 of 64 head dims.
pub type MoreRopeArch = Arch<512, 8, 4, 11, 3, 32, 7, 2048>;

/// Higher GQA: 12 query heads, 3 KV heads.
/// Note: 512/12 = 42.67 head dim, so this requires d that's a multiple of HEADS.
/// We use d=480 instead to keep head_dim integral (480/12 = 40).
pub type HighGqaArch = Arch<480, 12, 3, 11, 3, 16, 7, 2048>;

/// Long context: 4096 tokens.
pub type LongCtxArch = Arch<512, 8, 4, 11, 3, 16, 7, 4096>;

/// Aggressive — d=640 only viable at int4 quant.
pub type AggressiveQuantArch = Arch<640, 8, 4, 11, 3, 16, 7, 2048>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_matches_sota() {
        let cfg = BaselineArch::config();
        assert_eq!(cfg.model_dim, 512);
        assert_eq!(cfg.num_heads, 8);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.num_layers, 11);
        assert_eq!(cfg.mlp_dim, 1536);
        assert_eq!(cfg.rope_dims, 16);
        assert_eq!(cfg.xsa_last_n, 4); // 11 - 7 = 4
        assert_eq!(cfg.train_seq_len, 2048);
    }

    #[test]
    fn test_wide_arch() {
        let cfg = WideArch::config();
        assert_eq!(cfg.model_dim, 576);
        assert_eq!(cfg.head_dim, 72); // 576 / 8
        assert_eq!(cfg.mlp_dim, 1728);
        assert_eq!(cfg.kv_dim(), 288); // 4 * 72
    }

    #[test]
    fn test_full_xsa_arch() {
        let cfg = FullXsaArch::config();
        assert_eq!(cfg.xsa_last_n, 11); // all layers
    }

    #[test]
    fn test_deep_narrow_arch() {
        let cfg = DeepNarrowArch::config();
        assert_eq!(cfg.num_layers, 16);
        assert_eq!(cfg.num_heads, 6);
        assert_eq!(cfg.num_kv_heads, 3);
        assert_eq!(cfg.head_dim, 64); // 384 / 6
        assert_eq!(cfg.xsa_last_n, 4); // 16 - 12
    }

    #[test]
    fn test_label() {
        let l = BaselineArch::label();
        assert_eq!(l, "d512_h8_kv4_L11_m3_r16_x7_s2048");
    }

    #[test]
    fn test_xsa_from_disabled() {
        type NoXsa = Arch<512, 8, 4, 11, 3, 16, 11, 2048>;
        let cfg = NoXsa::config();
        assert_eq!(cfg.xsa_last_n, 0);
    }

    #[test]
    fn test_associated_consts_propagate() {
        // Sanity-check that the trait const fns work without ever
        // constructing an instance.
        assert_eq!(<BaselineArch as ArchTrait>::HEAD_DIM, 64);
        assert_eq!(<BaselineArch as ArchTrait>::KV_DIM, 256);
        assert_eq!(<BaselineArch as ArchTrait>::MLP_DIM, 1536);
        assert_eq!(<BaselineArch as ArchTrait>::XSA_LAST_N, 4);
        assert_eq!(<HighGqaArch as ArchTrait>::HEAD_DIM, 40);
        assert_eq!(<HighGqaArch as ArchTrait>::KV_DIM, 120);
    }
}
