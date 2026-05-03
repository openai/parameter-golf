use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{ModelConfig, TrainConfig};
use pg_core::error::{PgError, PgResult};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    #[default]
    Smoke,
    Proxy,
    /// Uses the final record batch arithmetic without doing a full artifact/eval
    /// record attempt. This is the systems benchmark surface for H100 work.
    RecordShapedProxy,
    Record,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainBackend {
    #[default]
    Cpu,
    CudaSingle,
    CudaSingleParity,
    CudaDistributed,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AttentionBackend {
    /// Debug-only parity kernel with duplicated QK work. Kept as a fallback.
    NaiveF32,
    /// F32 online-softmax parity kernel with forward and backward support.
    /// This is still a scalar record-blocking path, not FlashAttention.
    #[default]
    FlashF32,
    /// Intended production backend: fused cuDNN/FlashAttention-style BF16 SDPA.
    /// This is explicit so record runs cannot accidentally report the F32
    /// parity kernel as FlashAttention.
    CudnnSdpaBf16,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelComputePrecision {
    /// Current implemented runtime: f32 storage with optional TF32 cuBLAS math.
    #[default]
    F32Tf32,
    /// Frontier target: BF16/FP16 tensor-core parameter/activation/GEMM graph.
    Bf16TensorCore,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DistributedOptimizerBackend {
    /// Current distributed path: NCCL all-reduce grads, every rank updates a
    /// full replica. Correctness-first, but not Parallel Muon.
    #[default]
    AllReduceReplicatedMuon,
    /// Intended production path: reduce-scatter bank grads, shard-local NS5,
    /// all-gather updated banks.
    ShardedParallelMuon,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum EvalAdaptationBackend {
    #[default]
    None,
    /// Existing CPU q-only score-first TTT reference.
    CpuQOnly,
    /// Intended production frontier evaluator: GPU LoRA/phased score-first TTT.
    GpuLoraPhased,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VariantFamily {
    #[default]
    BaselineSp8192,
    XsaAllSp8192,
    RecurrenceMidSp8192,
    ParallelResidSp8192,
    HybridCompetitiveSp8192,
    #[serde(rename = "frontier_1855_like")]
    Frontier1855Like,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SkipTopology {
    None,
    #[default]
    Unet,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum RopeMode {
    None,
    #[default]
    Partial,
    Yarn,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum QuantScheme {
    None,
    #[default]
    GptqLiteInt6,
    MixedInt5Int6,
    Aggressive,
    TightInt7Int4,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationMode {
    #[default]
    Disabled,
    SelfGenerated,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompressionMode {
    None,
    #[default]
    Zstd22,
    Lzma9,
    Pergroup,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BigramSpec {
    pub enabled: bool,
    pub vocab_size: usize,
    pub dim: usize,
}

impl Default for BigramSpec {
    fn default() -> Self {
        Self {
            enabled: true,
            vocab_size: 3072,
            dim: 112,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RopeSpec {
    pub mode: RopeMode,
    pub dims: usize,
    pub base: f32,
}

impl Default for RopeSpec {
    fn default() -> Self {
        Self {
            mode: RopeMode::Partial,
            dims: 16,
            base: 10_000.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ValueEmbeddingSpec {
    pub enabled: bool,
    pub dim: usize,
    pub layers: Vec<usize>,
}

impl Default for ValueEmbeddingSpec {
    fn default() -> Self {
        Self {
            enabled: true,
            dim: 128,
            layers: vec![9, 10],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RecurrenceSpec {
    pub enabled: bool,
    pub start_layer: usize,
    pub repeat_layers: usize,
}

impl Default for RecurrenceSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            start_layer: 0,
            repeat_layers: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ParallelResidualSpec {
    pub enabled: bool,
    pub split_attention_mlp: bool,
}

impl Default for ParallelResidualSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            split_attention_mlp: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AttnOutGateSpec {
    pub enabled: bool,
    pub width: usize,
}

impl Default for AttnOutGateSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 24,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CaseOpsSpec {
    pub enabled: bool,
    pub byte_sidecar: bool,
}

impl Default for CaseOpsSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            byte_sidecar: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SparseAttnGateSpec {
    pub enabled: bool,
    pub width: usize,
    pub scale: f32,
}

impl Default for SparseAttnGateSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            width: 12,
            scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ModelSpec {
    pub family: VariantFamily,
    pub attention_backend: AttentionBackend,
    pub compute_precision: ModelComputePrecision,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub mlp_mult: f32,
    pub train_seq_len: usize,
    pub eval_seq_len: usize,
    pub bigram: BigramSpec,
    pub xsa_last_n: usize,
    pub value_embedding: ValueEmbeddingSpec,
    pub skip_topology: SkipTopology,
    pub recurrence: RecurrenceSpec,
    pub parallel_residual: ParallelResidualSpec,
    pub attn_out_gate: AttnOutGateSpec,
    pub caseops: CaseOpsSpec,
    pub sparse_attn_gate: SparseAttnGateSpec,
    pub rope: RopeSpec,
    pub smear_gate: bool,
    pub smear_gate_boundary_token_id: Option<u32>,
    pub logit_softcap: f32,
    pub qk_gain_init: f32,
    pub ln_scale: bool,
    pub tie_embeddings: bool,
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self::for_family(VariantFamily::BaselineSp8192)
    }
}

impl ModelSpec {
    pub fn for_family(family: VariantFamily) -> Self {
        let mut spec = Self {
            family,
            attention_backend: AttentionBackend::NaiveF32,
            compute_precision: ModelComputePrecision::F32Tf32,
            vocab_size: 8192,
            num_layers: 11,
            model_dim: 512,
            num_heads: 8,
            num_kv_heads: 4,
            mlp_mult: 3.0,
            train_seq_len: 2048,
            eval_seq_len: 2048,
            bigram: BigramSpec::default(),
            xsa_last_n: 4,
            value_embedding: ValueEmbeddingSpec::default(),
            skip_topology: SkipTopology::Unet,
            recurrence: RecurrenceSpec::default(),
            parallel_residual: ParallelResidualSpec::default(),
            attn_out_gate: AttnOutGateSpec::default(),
            caseops: CaseOpsSpec::default(),
            sparse_attn_gate: SparseAttnGateSpec::default(),
            rope: RopeSpec::default(),
            smear_gate: true,
            smear_gate_boundary_token_id: Some(1),
            logit_softcap: 30.0,
            qk_gain_init: 5.0,
            ln_scale: true,
            tie_embeddings: true,
        };
        match family {
            VariantFamily::BaselineSp8192 => {}
            VariantFamily::XsaAllSp8192 => {
                spec.xsa_last_n = spec.num_layers;
            }
            VariantFamily::RecurrenceMidSp8192 => {
                spec.recurrence.enabled = true;
                spec.recurrence.start_layer = 4;
                spec.recurrence.repeat_layers = 2;
            }
            VariantFamily::ParallelResidSp8192 => {
                spec.parallel_residual.enabled = true;
                spec.parallel_residual.split_attention_mlp = true;
            }
            VariantFamily::HybridCompetitiveSp8192 => {
                spec.xsa_last_n = spec.num_layers;
                spec.qk_gain_init = 5.25;
                spec.value_embedding.enabled = false;
                spec.value_embedding.layers.clear();
                spec.recurrence.enabled = true;
                spec.recurrence.start_layer = 4;
                spec.recurrence.repeat_layers = 2;
                spec.parallel_residual.enabled = true;
                spec.parallel_residual.split_attention_mlp = true;
                spec.caseops.enabled = true;
                spec.caseops.byte_sidecar = true;
                spec.sparse_attn_gate.enabled = true;
                spec.sparse_attn_gate.width = 12;
                spec.sparse_attn_gate.scale = 1.0;
                spec.bigram.vocab_size = 3072;
                spec.bigram.dim = 112;
            }
            VariantFamily::Frontier1855Like => {
                spec.attention_backend = AttentionBackend::CudnnSdpaBf16;
                spec.compute_precision = ModelComputePrecision::Bf16TensorCore;
                spec.mlp_mult = 4.0;
                spec.xsa_last_n = spec.num_layers;
                spec.qk_gain_init = 5.0;
                spec.value_embedding.enabled = false;
                spec.value_embedding.layers.clear();
                spec.recurrence.enabled = true;
                spec.recurrence.start_layer = 4;
                spec.recurrence.repeat_layers = 2;
                spec.parallel_residual.enabled = true;
                spec.parallel_residual.split_attention_mlp = true;
                spec.caseops.enabled = true;
                spec.caseops.byte_sidecar = true;
                spec.sparse_attn_gate.enabled = true;
                spec.sparse_attn_gate.width = 12;
                spec.sparse_attn_gate.scale = 0.5;
                spec.bigram.enabled = false;
                spec.smear_gate_boundary_token_id = Some(1);
            }
        }
        spec
    }

    pub fn to_model_config(&self) -> ModelConfig {
        let head_dim = self.model_dim / self.num_heads;
        let mlp_dim = (self.mlp_mult * self.model_dim as f32) as usize;
        ModelConfig {
            vocab_size: self.vocab_size,
            num_layers: self.num_layers,
            model_dim: self.model_dim,
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim,
            mlp_mult: self.mlp_mult,
            mlp_dim,
            rope_base: self.rope.base,
            rope_dims: self.rope.dims,
            xsa_last_n: self.xsa_last_n,
            logit_softcap: self.logit_softcap,
            qk_gain_init: self.qk_gain_init,
            recurrence_enabled: self.recurrence.enabled,
            recurrence_start_layer: self.recurrence.start_layer,
            recurrence_repeat_layers: self.recurrence.repeat_layers,
            parallel_residual: self.parallel_residual.enabled
                && self.parallel_residual.split_attention_mlp,
            attn_out_gate_enabled: self.attn_out_gate.enabled,
            attn_out_gate_width: self.attn_out_gate.width,
            sparse_attn_gate_enabled: self.sparse_attn_gate.enabled,
            sparse_attn_gate_width: self.sparse_attn_gate.width,
            sparse_attn_gate_scale: self.sparse_attn_gate.scale,
            smear_gate_boundary_token_id: self.smear_gate_boundary_token_id,
            vrl_enabled: false,
            ve_enabled: self.value_embedding.enabled,
            ve_dim: self.value_embedding.dim,
            ve_layers: if self.value_embedding.enabled {
                self.value_embedding.layers.clone()
            } else {
                Vec::new()
            },
            bigram_vocab_size: if self.bigram.enabled {
                self.bigram.vocab_size
            } else {
                0
            },
            bigram_dim: if self.bigram.enabled {
                self.bigram.dim
            } else {
                0
            },
            ln_scale: self.ln_scale,
            tie_embeddings: self.tie_embeddings,
            tied_embed_init_std: 0.005,
            train_seq_len: self.train_seq_len,
            eval_seq_len: self.eval_seq_len,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct TrainSpec {
    pub backend: TrainBackend,
    pub distributed_optimizer_backend: DistributedOptimizerBackend,
    pub batch_tokens: usize,
    pub seq_len: usize,
    pub train_data_pattern: Option<String>,
    pub validation_data_pattern: Option<String>,
    pub rank: usize,
    pub world_size: usize,
    pub artifact_path: String,
    pub fast_bank_updates: bool,
    pub warmup_steps: usize,
    pub total_iterations: usize,
    pub warmdown_iters: usize,
    pub warmdown_frac: f32,
    pub min_lr_scale: f32,
    pub max_wallclock_seconds: f32,
    pub matrix_lr: f32,
    pub scalar_lr: f32,
    pub embed_lr: f32,
    pub tied_embed_lr: f32,
    pub head_lr: f32,
    pub muon_momentum: f32,
    pub muon_momentum_warmup_start: f32,
    pub muon_momentum_warmup_steps: usize,
    pub muon_wd: f32,
    pub adam_wd: f32,
    pub adam_beta2: f32,
    pub ema_decay: f32,
    pub late_qat_threshold: f32,
}

impl Default for TrainSpec {
    fn default() -> Self {
        Self {
            backend: TrainBackend::Cpu,
            distributed_optimizer_backend: DistributedOptimizerBackend::AllReduceReplicatedMuon,
            batch_tokens: 524_288,
            seq_len: 2048,
            train_data_pattern: None,
            validation_data_pattern: None,
            rank: 0,
            world_size: 1,
            artifact_path: "artifact.pgrs".to_string(),
            fast_bank_updates: false,
            warmup_steps: 20,
            total_iterations: 9_000,
            warmdown_iters: 3_500,
            warmdown_frac: 0.0,
            min_lr_scale: 0.0,
            max_wallclock_seconds: 600.0,
            matrix_lr: 0.025,
            scalar_lr: 0.025,
            embed_lr: 0.6,
            tied_embed_lr: 0.035,
            head_lr: 0.008,
            muon_momentum: 0.99,
            muon_momentum_warmup_start: 0.92,
            muon_momentum_warmup_steps: 1_500,
            muon_wd: 0.04,
            adam_wd: 0.04,
            adam_beta2: 0.95,
            ema_decay: 0.997,
            late_qat_threshold: 0.15,
        }
    }
}

impl TrainSpec {
    pub fn to_train_config(&self) -> TrainConfig {
        let warmdown_iters = if self.warmdown_frac > 0.0 {
            ((self.total_iterations as f32) * self.warmdown_frac).round() as usize
        } else {
            self.warmdown_iters
        };
        TrainConfig {
            matrix_lr: self.matrix_lr,
            scalar_lr: self.scalar_lr,
            embed_lr: self.embed_lr,
            tied_embed_lr: self.tied_embed_lr,
            head_lr: self.head_lr,
            muon_momentum: self.muon_momentum,
            muon_momentum_warmup_start: self.muon_momentum_warmup_start,
            muon_momentum_warmup_steps: self.muon_momentum_warmup_steps,
            muon_wd: self.muon_wd,
            newton_schulz_steps: 5,
            adam_beta1: 0.9,
            adam_beta2: self.adam_beta2,
            adam_eps: 1e-8,
            adam_wd: self.adam_wd,
            warmup_steps: self.warmup_steps,
            warmdown_iters,
            total_iterations: self.total_iterations,
            min_lr_scale: self.min_lr_scale,
            max_wallclock_seconds: self.max_wallclock_seconds,
            train_batch_tokens: self.batch_tokens,
            grad_clip_norm: 0.3,
            ema_decay: self.ema_decay,
            swa_enabled: true,
            swa_every: 50,
            late_qat_threshold: self.late_qat_threshold,
            ttt_enabled: false,
            ttt_lr: 0.002,
            ttt_epochs: 3,
            ttt_chunk_tokens: 32_768,
            ttt_freeze_blocks: 0,
            ttt_momentum: 0.9,
            ttt_batch_seqs: 32,
            ttt_grad_clip: 1.0,
            eval_stride: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct QuantSpec {
    pub scheme: QuantScheme,
    pub calibration: CalibrationMode,
    pub prune_keep_ratio: Option<f32>,
    pub lqer: LqerSpec,
    pub compression: CompressionMode,
    pub matrix_bits: u8,
    pub embed_bits: u8,
    pub attn_gate_bits: u8,
    pub mlp_clip_sigmas: f32,
    pub attn_clip_sigmas: f32,
    pub embed_clip_sigmas: f32,
    pub gptq_calibration_batches: usize,
    pub target_artifact_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LqerSpec {
    pub enabled: bool,
    pub rank: usize,
    pub top_k: usize,
    pub a_bits: u8,
    pub b_bits: u8,
    pub group_size: usize,
    pub asymmetric: bool,
}

impl Default for LqerSpec {
    fn default() -> Self {
        Self {
            enabled: false,
            rank: 4,
            top_k: 3,
            a_bits: 2,
            b_bits: 4,
            group_size: 64,
            asymmetric: true,
        }
    }
}

impl Default for QuantSpec {
    fn default() -> Self {
        Self {
            scheme: QuantScheme::GptqLiteInt6,
            calibration: CalibrationMode::Disabled,
            prune_keep_ratio: None,
            lqer: LqerSpec::default(),
            compression: CompressionMode::Zstd22,
            matrix_bits: 6,
            embed_bits: 8,
            attn_gate_bits: 8,
            mlp_clip_sigmas: 12.0,
            attn_clip_sigmas: 13.0,
            embed_clip_sigmas: 14.0,
            gptq_calibration_batches: 0,
            target_artifact_bytes: 16_000_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EvalSpec {
    pub stride: usize,
    pub legal_score_first: bool,
    pub qttt: bool,
    pub adaptation_backend: EvalAdaptationBackend,
    pub lora_rank: usize,
    pub lora_alpha: f32,
    pub phased_ttt_prefix_docs: usize,
    pub phased_ttt_phases: usize,
    pub phased_ttt_weight_decay: f32,
    pub ttt_beta2: f32,
    pub chunk_tokens: usize,
    pub tokenizer_vocab_path: Option<String>,
    pub caseops_byte_sidecar_pattern: Option<String>,
    pub max_tokens: Option<usize>,
}

impl Default for EvalSpec {
    fn default() -> Self {
        Self {
            stride: 64,
            legal_score_first: true,
            qttt: false,
            adaptation_backend: EvalAdaptationBackend::None,
            lora_rank: 128,
            lora_alpha: 144.0,
            phased_ttt_prefix_docs: 2_000,
            phased_ttt_phases: 3,
            phased_ttt_weight_decay: 1.0,
            ttt_beta2: 0.99,
            chunk_tokens: 32_768,
            tokenizer_vocab_path: None,
            caseops_byte_sidecar_pattern: None,
            max_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RunSpec {
    pub name: String,
    pub model: ModelSpec,
    pub train: TrainSpec,
    pub quant: QuantSpec,
    pub eval: EvalSpec,
    pub mode: RunMode,
    pub allow_unsupported_variants: bool,
}

impl Default for RunSpec {
    fn default() -> Self {
        Self::for_family(VariantFamily::BaselineSp8192)
    }
}

impl RunSpec {
    pub fn for_family(family: VariantFamily) -> Self {
        Self {
            name: format!("{family:?}").to_lowercase(),
            model: ModelSpec::for_family(family),
            train: TrainSpec::default(),
            quant: QuantSpec::default(),
            eval: EvalSpec::default(),
            mode: RunMode::Smoke,
            allow_unsupported_variants: false,
        }
    }

    pub fn load(path: &Path) -> PgResult<Self> {
        let body = std::fs::read_to_string(path)?;
        toml::from_str(&body).map_err(|e| PgError::DataFormat(format!("invalid spec TOML: {e}")))
    }

    pub fn save(&self, path: &Path) -> PgResult<()> {
        let body = toml::to_string_pretty(self)
            .map_err(|e| PgError::DataFormat(format!("spec TOML encode failed: {e}")))?;
        std::fs::write(path, body)?;
        Ok(())
    }
}
