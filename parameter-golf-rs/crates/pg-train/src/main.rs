/// CPU-reference training loop for Parameter Golf.
///
/// Wires together: forward → backward → grad clip → Muon (banks) + AdamW (scalars) →
/// EMA every step → SWA during warmdown → WSD LR schedule → late QAT.
///
/// The GPU version replaces inner loops with cuBLASLt/CubeCL kernels
/// and adds NCCL for multi-GPU, but the control flow is identical.

use std::time::Instant;

use pg_model::config::{ModelConfig, TrainConfig};
use pg_model::model::{GptModel, ForwardBuffer};
use pg_model::backward::GradBuffers;
use pg_optim::adamw::{AdamW, AdamWState};
use pg_optim::ema::{Ema, Swa};
use pg_optim::muon::Muon;
use pg_optim::scheduler;

fn main() {
    env_logger::init();

    let model_config = ModelConfig::sota();
    let train_config = TrainConfig::sota();

    let mut model = GptModel::new(model_config.clone());
    model.summary();

    // Bank shapes for Muon: [batch, rows, cols]
    let n = model_config.num_layers;
    let d = model_config.model_dim;
    let kv = model_config.kv_dim();
    let mlp = model_config.mlp_dim;
    let bank_shapes: Vec<[usize; 3]> = vec![
        [2 * n, d, d],    // qo_bank
        [2 * n, kv, d],   // kv_bank
        [n, mlp, d],      // mlp_up_bank
        [n, d, mlp],      // mlp_down_bank
    ];

    // Muon for the 4 parameter banks
    let mut muon = Muon::new(
        train_config.matrix_lr,
        train_config.muon_momentum,
        train_config.newton_schulz_steps,
        true, // nesterov
        train_config.muon_wd,
        &bank_shapes,
    );

    // AdamW base configs (LR is overridden per-step with WSD schedule)

    // AdamW states for each parameter group
    let mut state_tok_emb = AdamWState::new(model.tok_emb.len());
    let mut state_bigram_embed = AdamWState::new(model.bigram_embed.len());
    let mut state_bigram_proj = AdamWState::new(model.bigram_proj.len());
    let mut state_smear_gate = AdamWState::new(model.smear_gate.len());
    let mut state_skip_weights = AdamWState::new(model.skip_weights.len());

    // Per-block scalar states
    let mut state_attn_scale: Vec<AdamWState> = (0..n)
        .map(|_| AdamWState::new(d))
        .collect();
    let mut state_mlp_scale: Vec<AdamWState> = (0..n)
        .map(|_| AdamWState::new(d))
        .collect();
    let mut state_resid_mix: Vec<AdamWState> = (0..n)
        .map(|_| AdamWState::new(2 * d))
        .collect();
    let mut state_q_gain: Vec<AdamWState> = (0..n)
        .map(|_| AdamWState::new(model_config.num_heads))
        .collect();

    // EMA
    let total_params = count_params(&model);
    let mut ema = Ema::new(train_config.ema_decay, total_params);
    let mut swa = Swa::new(total_params);

    // Forward buffer (reused across steps)
    let seq_len = model_config.train_seq_len;
    let mut buf = ForwardBuffer::new(&model_config, seq_len);
    let mut grads = GradBuffers::new(&model_config);

    eprintln!("Training config:");
    eprintln!("  Total iterations: {}", train_config.total_iterations);
    eprintln!("  Batch tokens: {}", train_config.train_batch_tokens);
    eprintln!("  Seq len: {}", seq_len);
    eprintln!("  LR (matrix): {}", train_config.matrix_lr);
    eprintln!("  Warmdown: {} steps", train_config.warmdown_iters);
    eprintln!("  Total params: {}", total_params);

    // TODO: Load data shards
    // let mut data = DistributedTokenLoader::new("data/datasets/fineweb10B_sp1024/fineweb_train_*.bin", 0, 1).unwrap();

    let start = Instant::now();

    for step in 0..train_config.total_iterations {
        let elapsed = start.elapsed().as_secs_f32();
        if elapsed > train_config.max_wallclock_seconds {
            eprintln!("Wall-clock limit reached at step {}", step);
            break;
        }

        // LR schedule
        let lr_scale = scheduler::lr_scale(
            step,
            train_config.warmup_steps,
            train_config.total_iterations,
            train_config.warmdown_iters,
        );
        let momentum = train_config.muon_momentum_at(step);
        muon.lr = train_config.matrix_lr * lr_scale;
        muon.momentum = momentum;

        // Get batch (placeholder — in production, from DistributedTokenLoader)
        // For now, generate synthetic tokens for compilation testing
        let input_ids: Vec<u32> = (0..seq_len).map(|i| (i % model_config.vocab_size) as u32).collect();
        let targets: Vec<u32> = (1..=seq_len).map(|i| (i % model_config.vocab_size) as u32).collect();

        // Zero gradients
        grads.zero();

        // Forward + backward
        let loss = model.backward(&input_ids, &targets, &mut buf, &mut grads);

        // Grad clipping
        grads.clip_grad_norm(train_config.grad_clip_norm);

        // Optimizer step: Muon for banks
        muon.step_bank(0, &mut model.qo_bank, &grads.qo_bank, &bank_shapes[0]);
        muon.step_bank(1, &mut model.kv_bank, &grads.kv_bank, &bank_shapes[1]);
        muon.step_bank(2, &mut model.mlp_up_bank, &grads.mlp_up_bank, &bank_shapes[2]);
        muon.step_bank(3, &mut model.mlp_down_bank, &grads.mlp_down_bank, &bank_shapes[3]);

        // Optimizer step: AdamW for scalar/embedding params
        let embed_lr_scaled = train_config.embed_lr * lr_scale;
        let scalar_lr_scaled = train_config.scalar_lr * lr_scale;
        {
            let adamw_e = AdamW::new(embed_lr_scaled, train_config.adam_beta1, train_config.adam_beta2, train_config.adam_eps, train_config.adam_wd);
            adamw_e.step(&mut model.tok_emb, &grads.tok_emb, &mut state_tok_emb);
            adamw_e.step(&mut model.bigram_embed, &grads.bigram_embed, &mut state_bigram_embed);
            adamw_e.step(&mut model.bigram_proj, &grads.bigram_proj, &mut state_bigram_proj);
        }
        {
            let adamw_s = AdamW::new(scalar_lr_scaled, train_config.adam_beta1, train_config.adam_beta2, train_config.adam_eps, train_config.adam_wd);
            adamw_s.step(&mut model.smear_gate, &grads.smear_gate, &mut state_smear_gate);
            adamw_s.step(&mut model.skip_weights, &grads.skip_weights, &mut state_skip_weights);

            for i in 0..n {
                adamw_s.step(&mut model.blocks[i].attn_scale, &grads.block_attn_scale[i], &mut state_attn_scale[i]);
                adamw_s.step(&mut model.blocks[i].mlp_scale, &grads.block_mlp_scale[i], &mut state_mlp_scale[i]);
                adamw_s.step(&mut model.blocks[i].resid_mix, &grads.block_resid_mix[i], &mut state_resid_mix[i]);
                adamw_s.step(&mut model.blocks[i].q_gain, &grads.block_q_gain[i], &mut state_q_gain[i]);
            }
        }

        // EMA
        let flat = flatten_params(&model);
        ema.update(&flat);

        // SWA during warmdown
        if train_config.should_swa(step) {
            swa.accumulate(&flat);
        }

        // Late QAT (placeholder — int6 STE quantization during warmdown)
        if train_config.qat_active(step) {
            // TODO: Apply int6 quantize-dequantize to bank params
        }

        // Logging
        if step % 100 == 0 || step < 5 {
            let bpb = (loss as f64 / std::f64::consts::LN_2) * 1.0; // approximate (tokens≈bytes for sp1024)
            eprintln!(
                "step {:5} | loss {:.4} | bpb ~{:.4} | lr_scale {:.4} | elapsed {:.1}s",
                step, loss, bpb, lr_scale, elapsed
            );
        }
    }

    // Final weights: prefer SWA if collected, else EMA
    let final_params = if swa.count > 0 {
        eprintln!("Using SWA average ({} checkpoints)", swa.count);
        swa.average()
    } else {
        eprintln!("Using EMA weights");
        ema.shadow.clone()
    };

    // Load final weights back into model
    unflatten_params(&mut model, &final_params);

    let total_time = start.elapsed().as_secs_f32();
    eprintln!("Training complete in {:.1}s", total_time);

    // TODO: Quantize model to int6 (16MB artifact limit)
    // TODO: Run eval with TTT
}

/// Count total trainable parameters.
fn count_params(model: &GptModel) -> usize {
    let mut total = 0;
    total += model.tok_emb.len();
    total += model.bigram_embed.len();
    total += model.bigram_proj.len();
    total += 1; // bigram_scale
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
    total += 1; // ve_scale
    total += model.ve_layer_scales.len();
    total
}

/// Flatten all model parameters into a single contiguous buffer.
fn flatten_params(model: &GptModel) -> Vec<f32> {
    let mut flat = Vec::with_capacity(count_params(model));
    flat.extend_from_slice(&model.tok_emb);
    flat.extend_from_slice(&model.bigram_embed);
    flat.extend_from_slice(&model.bigram_proj);
    flat.push(model.bigram_scale);
    flat.extend_from_slice(&model.smear_gate);
    flat.extend_from_slice(&model.skip_weights);
    flat.extend_from_slice(&model.qo_bank);
    flat.extend_from_slice(&model.kv_bank);
    flat.extend_from_slice(&model.mlp_up_bank);
    flat.extend_from_slice(&model.mlp_down_bank);
    for bp in &model.blocks {
        flat.extend_from_slice(&bp.attn_scale);
        flat.extend_from_slice(&bp.mlp_scale);
        flat.extend_from_slice(&bp.resid_mix);
        flat.extend_from_slice(&bp.q_gain);
    }
    flat.extend_from_slice(&model.ve_embed);
    flat.extend_from_slice(&model.ve_proj);
    flat.push(model.ve_scale);
    flat.extend_from_slice(&model.ve_layer_scales);
    flat
}

/// Unflatten a contiguous parameter buffer back into the model.
fn unflatten_params(model: &mut GptModel, flat: &[f32]) {
    let mut pos = 0;

    fn take(dest: &mut [f32], src: &[f32], pos: &mut usize) {
        dest.copy_from_slice(&src[*pos..*pos + dest.len()]);
        *pos += dest.len();
    }

    take(&mut model.tok_emb, flat, &mut pos);
    take(&mut model.bigram_embed, flat, &mut pos);
    take(&mut model.bigram_proj, flat, &mut pos);
    model.bigram_scale = flat[pos]; pos += 1;
    take(&mut model.smear_gate, flat, &mut pos);
    take(&mut model.skip_weights, flat, &mut pos);
    take(&mut model.qo_bank, flat, &mut pos);
    take(&mut model.kv_bank, flat, &mut pos);
    take(&mut model.mlp_up_bank, flat, &mut pos);
    take(&mut model.mlp_down_bank, flat, &mut pos);
    for bp in &mut model.blocks {
        take(&mut bp.attn_scale, flat, &mut pos);
        take(&mut bp.mlp_scale, flat, &mut pos);
        take(&mut bp.resid_mix, flat, &mut pos);
        take(&mut bp.q_gain, flat, &mut pos);
    }
    take(&mut model.ve_embed, flat, &mut pos);
    take(&mut model.ve_proj, flat, &mut pos);
    model.ve_scale = flat[pos]; pos += 1;
    take(&mut model.ve_layer_scales, flat, &mut pos);
    assert_eq!(pos, flat.len(), "unflatten size mismatch");
}
