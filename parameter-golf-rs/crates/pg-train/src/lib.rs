use std::time::Instant;

use pg_model::{ExecutionPlan, ForwardBuffer, GptModel, RunMode, RunSpec};
use pg_model::backward::GradBuffers;
use pg_optim::adamw::{AdamW, AdamWState};
use pg_optim::ema::{Ema, Swa};
use pg_optim::muon::Muon;
use pg_optim::scheduler;

use pg_core::PgResult;

#[derive(Debug, Clone)]
pub struct VariantResult {
    pub run_name: String,
    pub mode: RunMode,
    pub variant_fingerprint: String,
    pub steps_completed: usize,
    pub train_loss: f32,
    pub proxy_bpb: Option<f64>,
    pub final_bpb: Option<f64>,
    pub artifact_bytes: Option<usize>,
    pub ms_per_step: f64,
    pub wallclock_seconds: f64,
}

pub struct VariantRunner {
    pub run_spec: RunSpec,
    pub plan: ExecutionPlan,
}

impl VariantRunner {
    pub fn new(run_spec: RunSpec) -> PgResult<Self> {
        let plan = ExecutionPlan::from_run_spec(&run_spec)?;
        Ok(Self { run_spec, plan })
    }

    pub fn run(&self, mode: RunMode) -> PgResult<VariantResult> {
        let model_config = self.run_spec.model.to_model_config();
        let train_config = self.run_spec.train.to_train_config();
        let mut model = GptModel::new(model_config.clone());
        let active_tokens = match mode {
            RunMode::Smoke => self.run_spec.train.seq_len.min(16).min(model_config.train_seq_len),
            RunMode::Proxy => self.run_spec.train.seq_len.min(64).min(model_config.train_seq_len),
            RunMode::Record => self.run_spec.train.seq_len.min(model_config.train_seq_len),
        };
        let mut buf = ForwardBuffer::new(&model_config, active_tokens);
        let mut grads = GradBuffers::new(&model_config);

        let n = model_config.num_layers;
        let d = model_config.model_dim;
        let kv = model_config.kv_dim();
        let mlp = model_config.mlp_dim;
        let bank_shapes: Vec<[usize; 3]> = vec![
            [2 * n, d, d],
            [2 * n, kv, d],
            [n, mlp, d],
            [n, d, mlp],
        ];

        let mut muon = Muon::new(
            train_config.matrix_lr,
            train_config.muon_momentum,
            train_config.newton_schulz_steps,
            true,
            train_config.muon_wd,
            &bank_shapes,
        );
        let mut adamw_embed = AdamW::new(
            train_config.embed_lr,
            train_config.adam_beta1,
            train_config.adam_beta2,
            train_config.adam_eps,
            train_config.adam_wd,
        );
        let mut adamw_scalar = AdamW::new(
            train_config.scalar_lr,
            train_config.adam_beta1,
            train_config.adam_beta2,
            train_config.adam_eps,
            train_config.adam_wd,
        );

        let mut state_tok_emb = AdamWState::new(model.tok_emb.len());
        let mut state_bigram_embed = AdamWState::new(model.bigram_embed.len());
        let mut state_bigram_proj = AdamWState::new(model.bigram_proj.len());
        let mut state_smear_gate = AdamWState::new(model.smear_gate.len());
        let mut state_skip_weights = AdamWState::new(model.skip_weights.len());
        let mut state_ve_embed = AdamWState::new(model.ve_embed.len());
        let mut state_ve_proj = AdamWState::new(model.ve_proj.len());
        let mut state_ve_scale = AdamWState::new(1);
        let mut state_ve_layer_scales = AdamWState::new(model.ve_layer_scales.len());
        let mut state_bigram_scale = AdamWState::new(1);
        let mut state_attn_scale: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(d)).collect();
        let mut state_mlp_scale: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(d)).collect();
        let mut state_resid_mix: Vec<AdamWState> = (0..n).map(|_| AdamWState::new(2 * d)).collect();
        let mut state_q_gain: Vec<AdamWState> =
            (0..n).map(|_| AdamWState::new(model_config.num_heads)).collect();

        let total_params = count_params(&model);
        let mut ema = Ema::new(train_config.ema_decay, total_params);
        let mut swa = Swa::new(total_params);
        let mut flat_buf = vec![0.0f32; total_params];

        let requested_steps = match mode {
            RunMode::Smoke => 4usize,
            RunMode::Proxy => 32usize,
            RunMode::Record => train_config.total_iterations.min(256),
        };
        let max_steps = requested_steps.min(train_config.total_iterations);

        let start = Instant::now();
        let mut final_loss = 0.0f32;
        let mut steps_completed = 0usize;
        for step in 0..max_steps {
            let elapsed = start.elapsed().as_secs_f32();
            if elapsed > train_config.max_wallclock_seconds {
                break;
            }
            let lr_scale = scheduler::lr_scale(
                step,
                train_config.warmup_steps,
                train_config.total_iterations,
                train_config.warmdown_iters,
            );
            muon.lr = train_config.matrix_lr * lr_scale;
            muon.momentum = train_config.muon_momentum_at(step);
            adamw_embed.lr = train_config.embed_lr * lr_scale;
            adamw_scalar.lr = train_config.scalar_lr * lr_scale;

            let input_ids: Vec<u32> =
                (0..buf.tokens).map(|i| (i % model_config.vocab_size) as u32).collect();
            let targets: Vec<u32> =
                (1..=buf.tokens).map(|i| (i % model_config.vocab_size) as u32).collect();

            grads.zero();
            final_loss = model.backward(&input_ids, &targets, &mut buf, &mut grads);
            grads.clip_grad_norm(train_config.grad_clip_norm);

            muon.step_bank(0, &mut model.qo_bank, &grads.qo_bank, &bank_shapes[0]);
            muon.step_bank(1, &mut model.kv_bank, &grads.kv_bank, &bank_shapes[1]);
            muon.step_bank(2, &mut model.mlp_up_bank, &grads.mlp_up_bank, &bank_shapes[2]);
            muon.step_bank(3, &mut model.mlp_down_bank, &grads.mlp_down_bank, &bank_shapes[3]);

            adamw_embed.step(&mut model.tok_emb, &grads.tok_emb, &mut state_tok_emb);
            adamw_embed.step(&mut model.bigram_embed, &grads.bigram_embed, &mut state_bigram_embed);
            adamw_embed.step(&mut model.bigram_proj, &grads.bigram_proj, &mut state_bigram_proj);
            adamw_embed.step(&mut model.ve_embed, &grads.ve_embed, &mut state_ve_embed);

            adamw_scalar.step(&mut model.smear_gate, &grads.smear_gate, &mut state_smear_gate);
            adamw_scalar.step(&mut model.skip_weights, &grads.skip_weights, &mut state_skip_weights);
            adamw_scalar.step(&mut model.ve_proj, &grads.ve_proj, &mut state_ve_proj);
            {
                let mut ve_scale_slice = [model.ve_scale];
                let grad_ve_scale_slice = [grads.ve_scale];
                adamw_scalar.step(&mut ve_scale_slice, &grad_ve_scale_slice, &mut state_ve_scale);
                model.ve_scale = ve_scale_slice[0];
            }
            adamw_scalar.step(&mut model.ve_layer_scales, &grads.ve_layer_scales, &mut state_ve_layer_scales);
            {
                let mut bigram_scale_slice = [model.bigram_scale];
                let grad_bigram_scale_slice = [grads.bigram_scale];
                adamw_scalar.step(&mut bigram_scale_slice, &grad_bigram_scale_slice, &mut state_bigram_scale);
                model.bigram_scale = bigram_scale_slice[0];
            }
            for i in 0..n {
                adamw_scalar.step(&mut model.blocks[i].attn_scale, &grads.block_attn_scale[i], &mut state_attn_scale[i]);
                adamw_scalar.step(&mut model.blocks[i].mlp_scale, &grads.block_mlp_scale[i], &mut state_mlp_scale[i]);
                adamw_scalar.step(&mut model.blocks[i].resid_mix, &grads.block_resid_mix[i], &mut state_resid_mix[i]);
                adamw_scalar.step(&mut model.blocks[i].q_gain, &grads.block_q_gain[i], &mut state_q_gain[i]);
            }

            flatten_params_into(&model, &mut flat_buf);
            ema.update(&flat_buf);
            if train_config.should_swa(step) {
                swa.accumulate(&flat_buf);
            }
            steps_completed = step + 1;
        }

        let wallclock_seconds = start.elapsed().as_secs_f64();
        let ms_per_step = if steps_completed > 0 {
            (wallclock_seconds * 1000.0) / steps_completed as f64
        } else {
            0.0
        };

        let artifact_bytes = match mode {
            RunMode::Smoke => None,
            _ => {
                let artifact_path = std::path::Path::new("artifact.pgrs");
                pg_quant::export::export_model(&model, artifact_path).ok()
            }
        };

        Ok(VariantResult {
            run_name: self.run_spec.name.clone(),
            mode,
            variant_fingerprint: self.plan.variant_fingerprint.clone(),
            steps_completed,
            train_loss: final_loss,
            proxy_bpb: if matches!(mode, RunMode::Proxy) {
                Some(final_loss as f64 / std::f64::consts::LN_2)
            } else {
                None
            },
            final_bpb: None,
            artifact_bytes,
            ms_per_step,
            wallclock_seconds,
        })
    }
}

fn count_params(model: &GptModel) -> usize {
    let mut total = 0;
    total += model.tok_emb.len();
    total += model.bigram_embed.len();
    total += model.bigram_proj.len();
    total += 1;
    total += model.smear_gate.len();
    total += model.skip_weights.len();
    total += model.qo_bank.len();
    total += model.kv_bank.len();
    total += model.mlp_up_bank.len();
    total += model.mlp_down_bank.len();
    for bp in &model.blocks {
        total += bp.attn_scale.len();
        total += bp.mlp_scale.len();
        total += bp.resid_mix.len();
        total += bp.q_gain.len();
    }
    total += model.ve_embed.len();
    total += model.ve_proj.len();
    total += 1;
    total += model.ve_layer_scales.len();
    total
}

fn flatten_params_into(model: &GptModel, flat: &mut [f32]) {
    let mut pos = 0;
    fn copy(src: &[f32], dst: &mut [f32], pos: &mut usize) {
        dst[*pos..*pos + src.len()].copy_from_slice(src);
        *pos += src.len();
    }

    copy(&model.tok_emb, flat, &mut pos);
    copy(&model.bigram_embed, flat, &mut pos);
    copy(&model.bigram_proj, flat, &mut pos);
    flat[pos] = model.bigram_scale;
    pos += 1;
    copy(&model.smear_gate, flat, &mut pos);
    copy(&model.skip_weights, flat, &mut pos);
    copy(&model.qo_bank, flat, &mut pos);
    copy(&model.kv_bank, flat, &mut pos);
    copy(&model.mlp_up_bank, flat, &mut pos);
    copy(&model.mlp_down_bank, flat, &mut pos);
    for bp in &model.blocks {
        copy(&bp.attn_scale, flat, &mut pos);
        copy(&bp.mlp_scale, flat, &mut pos);
        copy(&bp.resid_mix, flat, &mut pos);
        copy(&bp.q_gain, flat, &mut pos);
    }
    copy(&model.ve_embed, flat, &mut pos);
    copy(&model.ve_proj, flat, &mut pos);
    flat[pos] = model.ve_scale;
    pos += 1;
    copy(&model.ve_layer_scales, flat, &mut pos);
}
