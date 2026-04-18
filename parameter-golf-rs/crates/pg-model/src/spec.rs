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
    Record,
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
pub struct ModelSpec {
    pub family: VariantFamily,
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
    pub rope: RopeSpec,
    pub smear_gate: bool,
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
            rope: RopeSpec::default(),
            smear_gate: true,
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
                spec.recurrence.enabled = true;
                spec.recurrence.start_layer = 4;
                spec.recurrence.repeat_layers = 2;
                spec.parallel_residual.enabled = true;
                spec.parallel_residual.split_attention_mlp = true;
                spec.bigram.vocab_size = 3072;
                spec.bigram.dim = 112;
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
            vrl_enabled: false,
            ve_enabled: self.value_embedding.enabled,
            ve_dim: self.value_embedding.dim,
            ve_layers: self.value_embedding.layers.clone(),
            bigram_vocab_size: if self.bigram.enabled { self.bigram.vocab_size } else { 0 },
            bigram_dim: if self.bigram.enabled { self.bigram.dim } else { 0 },
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
    pub batch_tokens: usize,
    pub seq_len: usize,
    pub warmup_steps: usize,
    pub total_iterations: usize,
    pub warmdown_iters: usize,
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
    pub ema_decay: f32,
    pub late_qat_threshold: f32,
}

impl Default for TrainSpec {
    fn default() -> Self {
        Self {
            batch_tokens: 524_288,
            seq_len: 2048,
            warmup_steps: 20,
            total_iterations: 9_000,
            warmdown_iters: 3_500,
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
            ema_decay: 0.997,
            late_qat_threshold: 0.15,
        }
    }
}

impl TrainSpec {
    pub fn to_train_config(&self) -> TrainConfig {
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
            adam_beta2: 0.95,
            adam_eps: 1e-8,
            adam_wd: self.adam_wd,
            warmup_steps: self.warmup_steps,
            warmdown_iters: self.warmdown_iters,
            total_iterations: self.total_iterations,
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
    pub compression: CompressionMode,
    pub target_artifact_bytes: usize,
}

impl Default for QuantSpec {
    fn default() -> Self {
        Self {
            scheme: QuantScheme::GptqLiteInt6,
            calibration: CalibrationMode::Disabled,
            prune_keep_ratio: None,
            compression: CompressionMode::Zstd22,
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
    pub chunk_tokens: usize,
}

impl Default for EvalSpec {
    fn default() -> Self {
        Self {
            stride: 64,
            legal_score_first: true,
            qttt: false,
            chunk_tokens: 32_768,
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
        }
    }

    pub fn load(path: &Path) -> PgResult<Self> {
        let body = std::fs::read_to_string(path)?;
        toml::from_str(&body).map_err(|e| PgError::DataFormat(format!("invalid spec TOML: {e}")))
    }

    pub fn save(&self, path: &Path) -> PgResult<()> {
        let body =
            toml::to_string_pretty(self).map_err(|e| PgError::DataFormat(format!("spec TOML encode failed: {e}")))?;
        std::fs::write(path, body)?;
        Ok(())
    }
}
