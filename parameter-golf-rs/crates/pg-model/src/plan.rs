use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pg_core::error::{PgError, PgResult};

use crate::config::ModelConfig;
use crate::gpu::{bank_shapes, checkpoint_layers, estimate_memory};
use crate::spec::{CompressionMode, ModelSpec, QuantScheme, RopeMode, RunMode, RunSpec, SkipTopology};

#[derive(Debug, Clone)]
pub struct LayerFeatureMask {
    pub xsa: bool,
    pub value_embedding: bool,
    pub checkpointed: bool,
    pub recurrent: bool,
    pub parallel_residual: bool,
}

#[derive(Debug, Clone)]
pub struct BankLayout {
    pub qo_bank: [usize; 3],
    pub kv_bank: [usize; 3],
    pub mlp_up_bank: [usize; 3],
    pub mlp_down_bank: [usize; 3],
    pub qo_bank_elems: usize,
    pub kv_bank_elems: usize,
    pub mlp_up_bank_elems: usize,
    pub mlp_down_bank_elems: usize,
}

#[derive(Debug, Clone)]
pub struct ActivationLayout {
    pub checkpoint_layers: Vec<bool>,
    pub estimated_peak_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct QuantLayout {
    pub scheme: QuantScheme,
    pub compression: CompressionMode,
    pub target_artifact_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct EvalPlan {
    pub stride: usize,
    pub legal_score_first: bool,
    pub qttt: bool,
    pub chunk_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub run_spec: RunSpec,
    pub variant_fingerprint: String,
    pub bank_layout: BankLayout,
    pub layer_schedule: Vec<LayerFeatureMask>,
    pub activation_layout: ActivationLayout,
    pub quant_layout: QuantLayout,
    pub eval_plan: EvalPlan,
}

impl ExecutionPlan {
    pub fn from_run_spec(run_spec: &RunSpec) -> PgResult<Self> {
        validate_model_spec(&run_spec.model)?;
        let config = run_spec.model.to_model_config();
        let bank_shapes = bank_shapes(&config);
        let bank_layout = BankLayout {
            qo_bank: bank_shapes[0],
            kv_bank: bank_shapes[1],
            mlp_up_bank: bank_shapes[2],
            mlp_down_bank: bank_shapes[3],
            qo_bank_elems: bank_shapes[0].iter().product(),
            kv_bank_elems: bank_shapes[1].iter().product(),
            mlp_up_bank_elems: bank_shapes[2].iter().product(),
            mlp_down_bank_elems: bank_shapes[3].iter().product(),
        };

        let checkpointed = checkpoint_layers(&config);
        let xsa_start = config.num_layers.saturating_sub(config.xsa_last_n);
        let mut layer_schedule = Vec::with_capacity(config.num_layers);
        for layer in 0..config.num_layers {
            let value_embedding = run_spec.model.value_embedding.enabled
                && run_spec.model.value_embedding.layers.contains(&layer);
            let recurrent = run_spec.model.recurrence.enabled
                && layer >= run_spec.model.recurrence.start_layer
                && layer < run_spec.model.recurrence.start_layer + run_spec.model.recurrence.repeat_layers;
            layer_schedule.push(LayerFeatureMask {
                xsa: layer >= xsa_start,
                value_embedding,
                checkpointed: checkpointed[layer],
                recurrent,
                parallel_residual: run_spec.model.parallel_residual.enabled,
            });
        }

        let activation_layout = ActivationLayout {
            checkpoint_layers: checkpointed,
            estimated_peak_bytes: estimate_memory(&config, run_spec.train.batch_tokens),
        };
        let quant_layout = QuantLayout {
            scheme: run_spec.quant.scheme,
            compression: run_spec.quant.compression,
            target_artifact_bytes: run_spec.quant.target_artifact_bytes,
        };
        let eval_plan = EvalPlan {
            stride: run_spec.eval.stride,
            legal_score_first: run_spec.eval.legal_score_first,
            qttt: run_spec.eval.qttt,
            chunk_tokens: run_spec.eval.chunk_tokens,
        };

        Ok(Self {
            run_spec: run_spec.clone(),
            variant_fingerprint: fingerprint(run_spec),
            bank_layout,
            layer_schedule,
            activation_layout,
            quant_layout,
            eval_plan,
        })
    }

    pub fn model_spec(&self) -> &ModelSpec {
        &self.run_spec.model
    }

    pub fn mode(&self) -> RunMode {
        self.run_spec.mode
    }

    pub fn artifact_budget_ok(&self, estimated_bytes: usize) -> bool {
        estimated_bytes <= self.quant_layout.target_artifact_bytes
    }

    pub fn has_skip_connections(&self) -> bool {
        self.run_spec.model.skip_topology == SkipTopology::Unet
    }

    pub fn validate_model_config(&self, config: &ModelConfig) -> PgResult<()> {
        let spec = &self.run_spec.model;
        let mismatches = [
            ("vocab_size", spec.vocab_size, config.vocab_size),
            ("num_layers", spec.num_layers, config.num_layers),
            ("model_dim", spec.model_dim, config.model_dim),
            ("num_heads", spec.num_heads, config.num_heads),
            ("num_kv_heads", spec.num_kv_heads, config.num_kv_heads),
            ("train_seq_len", spec.train_seq_len, config.train_seq_len),
            ("eval_seq_len", spec.eval_seq_len, config.eval_seq_len),
            ("bigram_vocab_size", spec.bigram.vocab_size, config.bigram_vocab_size),
            ("bigram_dim", spec.bigram.dim, config.bigram_dim),
            ("ve_dim", spec.value_embedding.dim, config.ve_dim),
            ("xsa_last_n", spec.xsa_last_n, config.xsa_last_n),
            ("rope_dims", spec.rope.dims, config.rope_dims),
        ];

        for (name, expected, got) in mismatches {
            if expected != got {
                return Err(PgError::InvalidOp(format!(
                    "execution plan mismatch for {name}: expected {expected}, got {got}",
                )));
            }
        }

        if (spec.mlp_mult - config.mlp_mult).abs() > f32::EPSILON {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for mlp_mult: expected {}, got {}",
                spec.mlp_mult, config.mlp_mult,
            )));
        }
        if spec.value_embedding.enabled != config.ve_enabled {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for ve_enabled: expected {}, got {}",
                spec.value_embedding.enabled, config.ve_enabled,
            )));
        }
        if spec.ln_scale != config.ln_scale {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for ln_scale: expected {}, got {}",
                spec.ln_scale, config.ln_scale,
            )));
        }
        if spec.tie_embeddings != config.tie_embeddings {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for tie_embeddings: expected {}, got {}",
                spec.tie_embeddings, config.tie_embeddings,
            )));
        }
        if spec.value_embedding.layers != config.ve_layers {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for ve_layers: expected {:?}, got {:?}",
                spec.value_embedding.layers, config.ve_layers,
            )));
        }
        if (spec.rope.base - config.rope_base).abs() > f32::EPSILON {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for rope_base: expected {}, got {}",
                spec.rope.base, config.rope_base,
            )));
        }
        if (spec.logit_softcap - config.logit_softcap).abs() > f32::EPSILON {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for logit_softcap: expected {}, got {}",
                spec.logit_softcap, config.logit_softcap,
            )));
        }
        if (spec.qk_gain_init - config.qk_gain_init).abs() > f32::EPSILON {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for qk_gain_init: expected {}, got {}",
                spec.qk_gain_init, config.qk_gain_init,
            )));
        }

        Ok(())
    }
}

fn validate_model_spec(spec: &ModelSpec) -> PgResult<()> {
    if spec.num_heads == 0 || spec.num_kv_heads == 0 {
        return Err(PgError::InvalidOp("head counts must be non-zero".into()));
    }
    if spec.model_dim % spec.num_heads != 0 {
        return Err(PgError::InvalidOp(format!(
            "model_dim {} must be divisible by num_heads {}",
            spec.model_dim, spec.num_heads
        )));
    }
    if spec.num_heads % spec.num_kv_heads != 0 {
        return Err(PgError::InvalidOp(format!(
            "num_heads {} must be divisible by num_kv_heads {}",
            spec.num_heads, spec.num_kv_heads
        )));
    }
    if spec.bigram.enabled && (spec.bigram.vocab_size == 0 || spec.bigram.dim == 0) {
        return Err(PgError::InvalidOp("bigram enabled but vocab/dim is zero".into()));
    }
    if spec.value_embedding.enabled && spec.value_embedding.dim == 0 {
        return Err(PgError::InvalidOp("value embedding enabled but dim is zero".into()));
    }
    if spec.rope.dims > spec.model_dim / spec.num_heads {
        return Err(PgError::InvalidOp(format!(
            "rope dims {} exceed head dim {}",
            spec.rope.dims,
            spec.model_dim / spec.num_heads
        )));
    }
    if spec.recurrence.enabled {
        return Err(PgError::InvalidOp(
            "recurrence variants are not implemented in the Rust execution runtime yet".into(),
        ));
    }
    if spec.parallel_residual.enabled {
        return Err(PgError::InvalidOp(
            "parallel residual variants are not implemented in the Rust execution runtime yet".into(),
        ));
    }
    if spec.skip_topology != SkipTopology::Unet {
        return Err(PgError::InvalidOp(
            "only the U-Net skip topology is implemented in the Rust execution runtime".into(),
        ));
    }
    if spec.rope.mode == RopeMode::Yarn {
        return Err(PgError::InvalidOp(
            "YaRN RoPE mode is not implemented in the Rust execution runtime yet".into(),
        ));
    }
    if !spec.smear_gate {
        return Err(PgError::InvalidOp(
            "disabling SmearGate is not implemented in the Rust execution runtime".into(),
        ));
    }
    Ok(())
}

fn fingerprint(run_spec: &RunSpec) -> String {
    let mut hasher = DefaultHasher::new();
    run_spec.name.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}
