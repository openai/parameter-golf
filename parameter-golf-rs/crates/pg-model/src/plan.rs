use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use pg_core::error::{PgError, PgResult};

use crate::config::ModelConfig;
use crate::gpu::{bank_shapes, checkpoint_layers, estimate_memory};
use crate::spec::{
    CompressionMode, EvalAdaptationBackend, ModelSpec, QuantScheme, RopeMode, RunMode, RunSpec,
    SkipTopology,
};

#[derive(Debug, Clone)]
pub struct LayerFeatureMask {
    pub xsa: bool,
    pub value_embedding: bool,
    pub checkpointed: bool,
    pub recurrent: bool,
    pub parallel_residual: bool,
    pub attn_out_gate: bool,
    pub sparse_attn_gate: bool,
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
    pub matrix_bits: u8,
    pub embed_bits: u8,
    pub attn_gate_bits: u8,
    pub gptq_calibration_batches: usize,
    pub target_artifact_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct EvalPlan {
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubmissionBudget {
    pub code_bytes: usize,
    pub compressed_model_bytes: usize,
    pub total_bytes: usize,
    pub limit_bytes: usize,
}

impl SubmissionBudget {
    pub fn new(code_bytes: usize, compressed_model_bytes: usize, limit_bytes: usize) -> Self {
        Self {
            code_bytes,
            compressed_model_bytes,
            total_bytes: code_bytes + compressed_model_bytes,
            limit_bytes,
        }
    }

    pub fn ok(&self) -> bool {
        self.total_bytes < self.limit_bytes
    }
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
        validate_run_spec(run_spec)?;
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
                && layer
                    < run_spec.model.recurrence.start_layer
                        + run_spec.model.recurrence.repeat_layers;
            layer_schedule.push(LayerFeatureMask {
                xsa: layer >= xsa_start,
                value_embedding,
                checkpointed: checkpointed[layer],
                recurrent,
                parallel_residual: run_spec.model.parallel_residual.enabled,
                attn_out_gate: run_spec.model.attn_out_gate.enabled,
                sparse_attn_gate: run_spec.model.sparse_attn_gate.enabled,
            });
        }

        let activation_layout = ActivationLayout {
            checkpoint_layers: checkpointed,
            estimated_peak_bytes: estimate_memory(&config, run_spec.train.batch_tokens),
        };
        let quant_layout = QuantLayout {
            scheme: run_spec.quant.scheme,
            compression: run_spec.quant.compression,
            matrix_bits: run_spec.quant.matrix_bits,
            embed_bits: run_spec.quant.embed_bits,
            attn_gate_bits: run_spec.quant.attn_gate_bits,
            gptq_calibration_batches: run_spec.quant.gptq_calibration_batches,
            target_artifact_bytes: run_spec.quant.target_artifact_bytes,
        };
        let eval_plan = EvalPlan {
            stride: run_spec.eval.stride,
            legal_score_first: run_spec.eval.legal_score_first,
            qttt: run_spec.eval.qttt,
            adaptation_backend: run_spec.eval.adaptation_backend,
            lora_rank: run_spec.eval.lora_rank,
            lora_alpha: run_spec.eval.lora_alpha,
            phased_ttt_prefix_docs: run_spec.eval.phased_ttt_prefix_docs,
            phased_ttt_phases: run_spec.eval.phased_ttt_phases,
            phased_ttt_weight_decay: run_spec.eval.phased_ttt_weight_decay,
            ttt_beta2: run_spec.eval.ttt_beta2,
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

    pub fn submission_budget(
        &self,
        code_bytes: usize,
        compressed_model_bytes: usize,
    ) -> SubmissionBudget {
        SubmissionBudget::new(
            code_bytes,
            compressed_model_bytes,
            self.quant_layout.target_artifact_bytes,
        )
    }

    pub fn submission_budget_ok(&self, code_bytes: usize, compressed_model_bytes: usize) -> bool {
        self.submission_budget(code_bytes, compressed_model_bytes)
            .ok()
    }

    pub fn total_submission_budget_ok(&self, total_bytes: usize) -> bool {
        total_bytes < self.quant_layout.target_artifact_bytes
    }

    pub fn has_skip_connections(&self) -> bool {
        self.run_spec.model.skip_topology == SkipTopology::Unet
    }

    pub fn validate_model_config(&self, config: &ModelConfig) -> PgResult<()> {
        let spec = &self.run_spec.model;
        let expected_bigram_vocab = if spec.bigram.enabled {
            spec.bigram.vocab_size
        } else {
            0
        };
        let expected_bigram_dim = if spec.bigram.enabled {
            spec.bigram.dim
        } else {
            0
        };
        let mismatches = [
            ("vocab_size", spec.vocab_size, config.vocab_size),
            ("num_layers", spec.num_layers, config.num_layers),
            ("model_dim", spec.model_dim, config.model_dim),
            ("num_heads", spec.num_heads, config.num_heads),
            ("num_kv_heads", spec.num_kv_heads, config.num_kv_heads),
            ("train_seq_len", spec.train_seq_len, config.train_seq_len),
            ("eval_seq_len", spec.eval_seq_len, config.eval_seq_len),
            (
                "bigram_vocab_size",
                expected_bigram_vocab,
                config.bigram_vocab_size,
            ),
            ("bigram_dim", expected_bigram_dim, config.bigram_dim),
            ("ve_dim", spec.value_embedding.dim, config.ve_dim),
            ("xsa_last_n", spec.xsa_last_n, config.xsa_last_n),
            (
                "recurrence_start_layer",
                spec.recurrence.start_layer,
                config.recurrence_start_layer,
            ),
            (
                "recurrence_repeat_layers",
                spec.recurrence.repeat_layers,
                config.recurrence_repeat_layers,
            ),
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
        if spec.recurrence.enabled != config.recurrence_enabled {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for recurrence_enabled: expected {}, got {}",
                spec.recurrence.enabled, config.recurrence_enabled,
            )));
        }
        if (spec.parallel_residual.enabled && spec.parallel_residual.split_attention_mlp)
            != config.parallel_residual
        {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for parallel_residual: expected {}, got {}",
                spec.parallel_residual.enabled && spec.parallel_residual.split_attention_mlp,
                config.parallel_residual,
            )));
        }
        if spec.attn_out_gate.enabled != config.attn_out_gate_enabled {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for attn_out_gate_enabled: expected {}, got {}",
                spec.attn_out_gate.enabled, config.attn_out_gate_enabled,
            )));
        }
        if spec.attn_out_gate.width != config.attn_out_gate_width {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for attn_out_gate_width: expected {}, got {}",
                spec.attn_out_gate.width, config.attn_out_gate_width,
            )));
        }
        if spec.sparse_attn_gate.enabled != config.sparse_attn_gate_enabled {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for sparse_attn_gate_enabled: expected {}, got {}",
                spec.sparse_attn_gate.enabled, config.sparse_attn_gate_enabled,
            )));
        }
        if spec.sparse_attn_gate.width != config.sparse_attn_gate_width {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for sparse_attn_gate_width: expected {}, got {}",
                spec.sparse_attn_gate.width, config.sparse_attn_gate_width,
            )));
        }
        if (spec.sparse_attn_gate.scale - config.sparse_attn_gate_scale).abs() > f32::EPSILON {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for sparse_attn_gate_scale: expected {}, got {}",
                spec.sparse_attn_gate.scale, config.sparse_attn_gate_scale,
            )));
        }
        if spec.smear_gate_boundary_token_id != config.smear_gate_boundary_token_id {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for smear_gate_boundary_token_id: expected {:?}, got {:?}",
                spec.smear_gate_boundary_token_id, config.smear_gate_boundary_token_id,
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
        let expected_ve_layers = if spec.value_embedding.enabled {
            spec.value_embedding.layers.as_slice()
        } else {
            &[]
        };
        if expected_ve_layers != config.ve_layers.as_slice() {
            return Err(PgError::InvalidOp(format!(
                "execution plan mismatch for ve_layers: expected {:?}, got {:?}",
                expected_ve_layers, config.ve_layers,
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

fn validate_run_spec(run_spec: &RunSpec) -> PgResult<()> {
    let spec = &run_spec.model;
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
        return Err(PgError::InvalidOp(
            "bigram enabled but vocab/dim is zero".into(),
        ));
    }
    if spec.value_embedding.enabled && spec.value_embedding.dim == 0 {
        return Err(PgError::InvalidOp(
            "value embedding enabled but dim is zero".into(),
        ));
    }
    if spec.rope.dims > spec.model_dim / spec.num_heads {
        return Err(PgError::InvalidOp(format!(
            "rope dims {} exceed head dim {}",
            spec.rope.dims,
            spec.model_dim / spec.num_heads
        )));
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
    if spec.attn_out_gate.enabled {
        if spec.attn_out_gate.width == 0 {
            return Err(PgError::InvalidOp(
                "AttnOutGate is enabled but width is zero".into(),
            ));
        }
        if spec.attn_out_gate.width > spec.model_dim {
            return Err(PgError::InvalidOp(format!(
                "AttnOutGate width {} exceeds model_dim {}",
                spec.attn_out_gate.width, spec.model_dim
            )));
        }
    }
    if spec.sparse_attn_gate.enabled {
        if spec.attn_out_gate.enabled {
            return Err(PgError::InvalidOp(
                "SparseAttnGate and AttnOutGate are mutually exclusive in the Rust frontier runtime".into(),
            ));
        }
        if spec.sparse_attn_gate.width == 0 {
            return Err(PgError::InvalidOp(
                "SparseAttnGate is enabled but width is zero".into(),
            ));
        }
        if spec.sparse_attn_gate.width > spec.model_dim {
            return Err(PgError::InvalidOp(format!(
                "SparseAttnGate width {} exceeds model_dim {}",
                spec.sparse_attn_gate.width, spec.model_dim
            )));
        }
        if !spec.sparse_attn_gate.scale.is_finite() || spec.sparse_attn_gate.scale <= 0.0 {
            return Err(PgError::InvalidOp(format!(
                "SparseAttnGate scale must be positive and finite, got {}",
                spec.sparse_attn_gate.scale
            )));
        }
    }
    for (field, bits) in [
        ("quant.matrix_bits", run_spec.quant.matrix_bits),
        ("quant.embed_bits", run_spec.quant.embed_bits),
        ("quant.attn_gate_bits", run_spec.quant.attn_gate_bits),
        ("quant.lqer.a_bits", run_spec.quant.lqer.a_bits),
        ("quant.lqer.b_bits", run_spec.quant.lqer.b_bits),
    ] {
        if !(4..=8).contains(&bits) && !field.starts_with("quant.lqer.") {
            return Err(PgError::InvalidOp(format!(
                "{field}={bits} is unsupported; expected 4..=8"
            )));
        }
        if field.starts_with("quant.lqer.") && !(2..=8).contains(&bits) {
            return Err(PgError::InvalidOp(format!(
                "{field}={bits} is unsupported; expected 2..=8"
            )));
        }
    }
    if run_spec.quant.lqer.enabled && run_spec.quant.lqer.top_k == 0 {
        return Err(PgError::InvalidOp(
            "quant.lqer.top_k must be non-zero when LQER is enabled".into(),
        ));
    }
    if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
        && !run_spec.eval.legal_score_first
    {
        return Err(PgError::InvalidOp(
            "GPU phased LoRA TTT requires legal_score_first=true".into(),
        ));
    }
    Ok(())
}

fn fingerprint(run_spec: &RunSpec) -> String {
    let mut hasher = DefaultHasher::new();
    format!("{:?}", run_spec.model).hash(&mut hasher);
    format!("{:?}", run_spec.quant).hash(&mut hasher);
    format!("{:?}", run_spec.eval).hash(&mut hasher);
    format!("{:?}", run_spec.train.backend).hash(&mut hasher);
    format!("{:?}", run_spec.train.distributed_optimizer_backend).hash(&mut hasher);
    run_spec.train.batch_tokens.hash(&mut hasher);
    run_spec.train.seq_len.hash(&mut hasher);
    run_spec.train.fast_bank_updates.hash(&mut hasher);
    run_spec.train.warmup_steps.hash(&mut hasher);
    run_spec.train.total_iterations.hash(&mut hasher);
    run_spec.train.warmdown_iters.hash(&mut hasher);
    run_spec.train.warmdown_frac.to_bits().hash(&mut hasher);
    run_spec
        .train
        .max_wallclock_seconds
        .to_bits()
        .hash(&mut hasher);
    run_spec.train.matrix_lr.to_bits().hash(&mut hasher);
    run_spec.train.scalar_lr.to_bits().hash(&mut hasher);
    run_spec.train.embed_lr.to_bits().hash(&mut hasher);
    run_spec.train.tied_embed_lr.to_bits().hash(&mut hasher);
    run_spec.train.head_lr.to_bits().hash(&mut hasher);
    run_spec.train.muon_momentum.to_bits().hash(&mut hasher);
    run_spec
        .train
        .muon_momentum_warmup_start
        .to_bits()
        .hash(&mut hasher);
    run_spec.train.muon_momentum_warmup_steps.hash(&mut hasher);
    run_spec.train.muon_wd.to_bits().hash(&mut hasher);
    run_spec.train.adam_wd.to_bits().hash(&mut hasher);
    run_spec.train.adam_beta2.to_bits().hash(&mut hasher);
    run_spec.train.ema_decay.to_bits().hash(&mut hasher);
    run_spec
        .train
        .late_qat_threshold
        .to_bits()
        .hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompressionMode, EvalAdaptationBackend, RunSpec, VariantFamily};

    #[test]
    fn hybrid_competitive_plan_accepts_sparse_attn_gate() {
        let spec = RunSpec::for_family(VariantFamily::HybridCompetitiveSp8192);
        let plan = ExecutionPlan::from_run_spec(&spec).unwrap();
        let config = spec.model.to_model_config();
        assert!(!config.attn_out_gate_enabled);
        assert!(config.sparse_attn_gate_enabled);
        assert_eq!(config.sparse_attn_gate_width, 12);
        plan.validate_model_config(&config).unwrap();
    }

    #[test]
    fn attn_out_gate_rejects_invalid_width() {
        let mut spec = RunSpec::for_family(VariantFamily::HybridCompetitiveSp8192);
        spec.model.sparse_attn_gate.enabled = false;
        spec.model.attn_out_gate.enabled = true;
        spec.model.attn_out_gate.width = spec.model.model_dim + 1;
        let err = ExecutionPlan::from_run_spec(&spec).unwrap_err();
        assert!(
            err.to_string().contains("AttnOutGate width"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sparse_attn_gate_rejects_attn_out_gate_combo() {
        let mut spec = RunSpec::for_family(VariantFamily::HybridCompetitiveSp8192);
        spec.model.attn_out_gate.enabled = true;
        let err = ExecutionPlan::from_run_spec(&spec).unwrap_err();
        assert!(
            err.to_string().contains("mutually exclusive"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn disabled_value_embedding_clears_runtime_layers() {
        let mut spec = RunSpec::for_family(VariantFamily::HybridCompetitiveSp8192);
        spec.model.value_embedding.enabled = false;
        spec.model.value_embedding.layers = vec![9, 10];
        let plan = ExecutionPlan::from_run_spec(&spec).unwrap();
        let config = spec.model.to_model_config();
        assert!(config.ve_layers.is_empty());
        plan.validate_model_config(&config).unwrap();
        assert!(
            plan.layer_schedule
                .iter()
                .all(|layer| !layer.value_embedding)
        );
    }

    #[test]
    fn submission_budget_counts_code_and_model_bytes_strictly() {
        let spec = RunSpec::for_family(VariantFamily::HybridCompetitiveSp8192);
        let plan = ExecutionPlan::from_run_spec(&spec).unwrap();
        assert!(plan.submission_budget_ok(100, 15_999_899));
        assert!(!plan.submission_budget_ok(100, 15_999_900));
        let budget = plan.submission_budget(4_000_000, 11_999_999);
        assert_eq!(budget.total_bytes, 15_999_999);
        assert!(budget.ok());
    }

    #[test]
    fn frontier_1855_spec_loads_critical_hyperparameters() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../specs/frontier_1855_merged_target.toml");
        let spec = RunSpec::load(&path).unwrap();
        assert_eq!(spec.name, "frontier_1855_merged_target");
        assert_eq!(spec.model.family, VariantFamily::Frontier1855Like);
        assert_eq!(spec.model.mlp_mult, 4.0);
        assert_eq!(spec.model.qk_gain_init, 5.0);
        assert!(spec.model.caseops.byte_sidecar);
        assert_eq!(spec.model.sparse_attn_gate.scale, 0.5);
        assert_eq!(spec.train.batch_tokens, 786_432);
        assert_eq!(spec.train.world_size, 8);
        assert_eq!(spec.train.warmdown_frac, 0.85);
        assert_eq!(spec.train.to_train_config().warmdown_iters, 7_650);
        assert_eq!(spec.train.adam_beta2, 0.99);
        assert_eq!(spec.quant.compression, CompressionMode::Pergroup);
        assert_eq!(spec.quant.matrix_bits, 6);
        assert_eq!(spec.quant.embed_bits, 7);
        assert_eq!(spec.quant.attn_gate_bits, 8);
        assert_eq!(spec.quant.gptq_calibration_batches, 16);
        assert_eq!(spec.quant.lqer.top_k, 3);
        assert_eq!(
            spec.eval.adaptation_backend,
            EvalAdaptationBackend::GpuLoraPhased
        );
        assert_eq!(spec.eval.lora_rank, 80);
        assert_eq!(spec.eval.ttt_beta2, 0.99);
    }

    #[test]
    fn fingerprint_changes_for_frontier_critical_train_knobs() {
        let spec = RunSpec::for_family(VariantFamily::Frontier1855Like);
        let base = fingerprint(&spec);

        let mut changed = spec.clone();
        changed.train.warmdown_frac += 0.01;
        assert_ne!(base, fingerprint(&changed));

        let mut changed = spec.clone();
        changed.train.adam_beta2 = 0.98;
        assert_ne!(base, fingerprint(&changed));
    }
}
