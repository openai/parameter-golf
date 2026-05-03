use std::path::PathBuf;

use pg_model::{
    AttentionBackend, DistributedOptimizerBackend, EvalAdaptationBackend, QuantScheme, RunMode,
    RunSpec, TrainBackend, VariantFamily,
};
use pg_train::VariantRunner;

fn timing_per_step(value_ms: f64, timing_steps: usize) -> f64 {
    if timing_steps > 0 {
        value_ms / timing_steps as f64
    } else {
        0.0
    }
}

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        std::process::exit(2);
    };

    match command.as_str() {
        "run" => {
            let mut spec_path: Option<PathBuf> = None;
            let mut builtin: Option<VariantFamily> = None;
            let mut mode: Option<RunMode> = None;
            let mut backend: Option<TrainBackend> = None;
            let mut train_data_pattern: Option<String> = None;
            let mut validation_data_pattern: Option<String> = None;
            let mut artifact_path: Option<String> = None;
            let mut rank: Option<usize> = None;
            let mut world_size: Option<usize> = None;
            let mut batch_tokens: Option<usize> = None;
            let mut seq_len: Option<usize> = None;
            let mut total_iterations: Option<usize> = None;
            let mut max_wallclock_seconds: Option<f32> = None;
            let mut min_lr_scale: Option<f32> = None;
            let mut tokenizer_vocab_path: Option<String> = None;
            let mut caseops_byte_sidecar_pattern: Option<String> = None;
            let mut eval_max_tokens: Option<usize> = None;
            let mut eval_stride: Option<usize> = None;
            let mut attention_backend: Option<AttentionBackend> = None;
            let mut distributed_optimizer_backend: Option<DistributedOptimizerBackend> = None;
            let mut eval_adaptation_backend: Option<EvalAdaptationBackend> = None;
            let mut attn_out_gate: Option<bool> = None;
            let mut attn_out_gate_width: Option<usize> = None;
            let mut quant_scheme: Option<QuantScheme> = None;
            let mut prune_keep_ratio: Option<f32> = None;
            let mut fast_bank_updates = false;
            let mut allow_unsupported_variants = false;

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--spec" => spec_path = args.next().map(PathBuf::from),
                    "--builtin" => {
                        builtin = args.next().as_deref().and_then(parse_family);
                    }
                    "--mode" => {
                        mode = args.next().as_deref().and_then(parse_mode);
                    }
                    "--backend" => {
                        backend = args.next().as_deref().and_then(parse_backend);
                    }
                    "--train-data" => train_data_pattern = args.next(),
                    "--val-data" => validation_data_pattern = args.next(),
                    "--artifact" => artifact_path = args.next(),
                    "--rank" => rank = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--world-size" => {
                        world_size = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--batch-tokens" => {
                        batch_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--seq-len" => seq_len = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--total-iterations" | "--max-steps" => {
                        total_iterations = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--max-wallclock-seconds" => {
                        max_wallclock_seconds = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--min-lr-scale" => {
                        min_lr_scale = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--tokenizer-vocab" => tokenizer_vocab_path = args.next(),
                    "--caseops-byte-sidecar" => caseops_byte_sidecar_pattern = args.next(),
                    "--eval-max-tokens" => {
                        eval_max_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--eval-stride" => {
                        eval_stride = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--attention-backend" => {
                        attention_backend = args.next().as_deref().and_then(parse_attention_backend)
                    }
                    "--distributed-optimizer" => {
                        distributed_optimizer_backend = args
                            .next()
                            .as_deref()
                            .and_then(parse_distributed_optimizer_backend)
                    }
                    "--eval-adaptation" => {
                        eval_adaptation_backend = args
                            .next()
                            .as_deref()
                            .and_then(parse_eval_adaptation_backend)
                    }
                    "--attn-out-gate" => attn_out_gate = Some(true),
                    "--attn-out-gate-width" => {
                        attn_out_gate_width = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--quant-scheme" => {
                        quant_scheme = args.next().as_deref().and_then(parse_quant_scheme)
                    }
                    "--prune-keep-ratio" => {
                        prune_keep_ratio = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--fast-bank-updates" => fast_bank_updates = true,
                    "--allow-unsupported-variants" => allow_unsupported_variants = true,
                    _ => {}
                }
            }

            let mut run_spec = if let Some(path) = spec_path {
                RunSpec::load(&path).expect("failed to load spec")
            } else {
                RunSpec::for_family(builtin.unwrap_or(VariantFamily::BaselineSp8192))
            };
            if let Some(mode) = mode {
                run_spec.mode = mode;
            }
            if let Some(backend) = backend {
                run_spec.train.backend = backend;
            }
            if let Some(pattern) = train_data_pattern {
                run_spec.train.train_data_pattern = Some(pattern);
            }
            if let Some(pattern) = validation_data_pattern {
                run_spec.train.validation_data_pattern = Some(pattern);
            }
            if let Some(path) = artifact_path {
                run_spec.train.artifact_path = path;
            }
            if let Some(value) = rank {
                run_spec.train.rank = value;
            }
            if let Some(value) = world_size {
                run_spec.train.world_size = value;
            }
            if let Some(value) = batch_tokens {
                run_spec.train.batch_tokens = value;
            }
            if let Some(value) = seq_len {
                run_spec.train.seq_len = value;
                run_spec.model.train_seq_len = value.max(run_spec.model.train_seq_len);
            }
            if let Some(value) = total_iterations {
                run_spec.train.total_iterations = value;
            }
            if let Some(value) = max_wallclock_seconds {
                run_spec.train.max_wallclock_seconds = value;
            }
            if let Some(value) = min_lr_scale {
                run_spec.train.min_lr_scale = value;
            }
            if let Some(path) = tokenizer_vocab_path {
                run_spec.eval.tokenizer_vocab_path = Some(path);
            }
            if let Some(pattern) = caseops_byte_sidecar_pattern {
                run_spec.eval.caseops_byte_sidecar_pattern = Some(pattern);
            }
            if let Some(value) = eval_max_tokens {
                run_spec.eval.max_tokens = Some(value);
            }
            if let Some(value) = eval_stride {
                run_spec.eval.stride = value;
            }
            if let Some(value) = attention_backend {
                run_spec.model.attention_backend = value;
            }
            if let Some(value) = distributed_optimizer_backend {
                run_spec.train.distributed_optimizer_backend = value;
            }
            if let Some(value) = eval_adaptation_backend {
                run_spec.eval.adaptation_backend = value;
                run_spec.eval.qttt = value != EvalAdaptationBackend::None;
            }
            if let Some(value) = attn_out_gate {
                run_spec.model.attn_out_gate.enabled = value;
            }
            if let Some(value) = attn_out_gate_width {
                run_spec.model.attn_out_gate.width = value;
            }
            if let Some(value) = quant_scheme {
                run_spec.quant.scheme = value;
            }
            if let Some(value) = prune_keep_ratio {
                run_spec.quant.prune_keep_ratio = Some(value);
            }
            if fast_bank_updates {
                run_spec.train.fast_bank_updates = true;
            }
            if allow_unsupported_variants {
                run_spec.allow_unsupported_variants = true;
            }
            let result = match VariantRunner::new(run_spec.clone())
                .and_then(|runner| runner.run(run_spec.mode))
            {
                Ok(result) => result,
                Err(err) => {
                    eprintln!("variant run failed: {err}");
                    std::process::exit(1);
                }
            };
            println!("run_name={}", result.run_name);
            println!("mode={:?}", result.mode);
            println!("train_backend={:?}", result.train_backend);
            println!("variant_fingerprint={}", result.variant_fingerprint);
            println!("steps_completed={}", result.steps_completed);
            println!("train_loss={:.6}", result.train_loss);
            println!("train_loss_source={}", result.train_loss_source);
            println!("ms_per_step={:.3}", result.ms_per_step);
            println!("wallclock_seconds={:.3}", result.wallclock_seconds);
            println!("timing_steps={}", result.timing_steps);
            println!(
                "timing_measured_ms_per_step={:.3}",
                result.timing_measured_ms_per_step
            );
            println!("rank={}", result.rank);
            println!("world_size={}", result.world_size);
            println!("seq_len={}", result.seq_len);
            println!("global_batch_tokens={}", result.global_batch_tokens);
            println!(
                "local_microbatches_per_step={}",
                result.local_microbatches_per_step
            );
            println!("tokens_seen_global={}", result.tokens_seen_global);
            println!("distributed_sync={}", result.distributed_sync);
            println!("attention_backend={}", result.attention_backend);
            println!(
                "distributed_optimizer_backend={}",
                result.distributed_optimizer_backend
            );
            println!("eval_adaptation_backend={}", result.eval_adaptation_backend);
            println!("frontier_record_ready={}", result.frontier_record_ready);
            println!(
                "leaderboard_algorithm_ready={}",
                result.leaderboard_algorithm_ready
            );
            println!("record_shape={}", result.record_shape);
            println!("record_attention_grade={}", result.record_attention_grade);
            println!("microbatch_serial_loop={}", result.microbatch_serial_loop);
            println!("bank_update_backend={}", result.bank_update_backend);
            println!("train_data_source={}", result.train_data_source);
            println!("bpb_byte_source={}", result.bpb_byte_source);
            println!("timing_backend={}", result.timing_backend);
            println!("timing_cuda_stage_fields_unit=accumulated_over_timing_steps");
            println!(
                "timing_data_sampling_ms={:.3}",
                result.timing_data_sampling_ms
            );
            println!("timing_train_step_ms={:.3}", result.timing_train_step_ms);
            println!(
                "timing_cuda_zero_grads_ms={:.3}",
                result.timing_cuda_zero_grads_ms
            );
            println!("timing_cuda_h2d_ms={:.3}", result.timing_cuda_h2d_ms);
            println!(
                "timing_cuda_backward_ms={:.3}",
                result.timing_cuda_backward_ms
            );
            println!(
                "timing_cuda_backward_forward_ms={:.3}",
                result.timing_cuda_backward_forward_ms
            );
            println!(
                "timing_cuda_backward_forward_embed_ms={:.3}",
                result.timing_cuda_backward_forward_embed_ms
            );
            println!(
                "timing_cuda_backward_forward_encoder_ms={:.3}",
                result.timing_cuda_backward_forward_encoder_ms
            );
            println!(
                "timing_cuda_backward_forward_encoder_layer_max_ms={:.3}",
                result.timing_cuda_backward_forward_encoder_layer_max_ms
            );
            println!(
                "timing_cuda_backward_forward_decoder_ms={:.3}",
                result.timing_cuda_backward_forward_decoder_ms
            );
            println!(
                "timing_cuda_backward_forward_decoder_layer_max_ms={:.3}",
                result.timing_cuda_backward_forward_decoder_layer_max_ms
            );
            println!(
                "timing_cuda_backward_forward_logits_ms={:.3}",
                result.timing_cuda_backward_forward_logits_ms
            );
            println!(
                "timing_cuda_backward_forward_block_pre_attn_ms={:.3}",
                result.timing_cuda_backward_forward_block_pre_attn_ms
            );
            println!(
                "timing_cuda_backward_forward_block_attention_ms={:.3}",
                result.timing_cuda_backward_forward_block_attention_ms
            );
            println!(
                "timing_cuda_backward_forward_block_post_attn_ms={:.3}",
                result.timing_cuda_backward_forward_block_post_attn_ms
            );
            println!(
                "timing_cuda_backward_forward_block_mlp_ms={:.3}",
                result.timing_cuda_backward_forward_block_mlp_ms
            );
            println!(
                "timing_cuda_backward_block_recompute_ms={:.3}",
                result.timing_cuda_backward_block_recompute_ms
            );
            println!(
                "timing_cuda_backward_block_mlp_ms={:.3}",
                result.timing_cuda_backward_block_mlp_ms
            );
            println!(
                "timing_cuda_backward_block_mlp_residual_ms={:.3}",
                result.timing_cuda_backward_block_mlp_residual_ms
            );
            println!(
                "timing_cuda_backward_block_mlp_down_ms={:.3}",
                result.timing_cuda_backward_block_mlp_down_ms
            );
            println!(
                "timing_cuda_backward_block_mlp_act_ms={:.3}",
                result.timing_cuda_backward_block_mlp_act_ms
            );
            println!(
                "timing_cuda_backward_block_mlp_up_ms={:.3}",
                result.timing_cuda_backward_block_mlp_up_ms
            );
            println!(
                "timing_cuda_backward_block_mlp_norm_ms={:.3}",
                result.timing_cuda_backward_block_mlp_norm_ms
            );
            println!(
                "timing_cuda_backward_block_attn_out_ms={:.3}",
                result.timing_cuda_backward_block_attn_out_ms
            );
            println!(
                "timing_cuda_backward_block_attn_out_residual_ms={:.3}",
                result.timing_cuda_backward_block_attn_out_residual_ms
            );
            println!(
                "timing_cuda_backward_block_attn_out_proj_ms={:.3}",
                result.timing_cuda_backward_block_attn_out_proj_ms
            );
            println!(
                "timing_cuda_backward_block_attn_out_gate_xsa_ms={:.3}",
                result.timing_cuda_backward_block_attn_out_gate_xsa_ms
            );
            println!(
                "timing_cuda_backward_block_attention_ms={:.3}",
                result.timing_cuda_backward_block_attention_ms
            );
            println!(
                "timing_cuda_backward_block_attention_sdpa_ms={:.3}",
                result.timing_cuda_backward_block_attention_sdpa_ms
            );
            println!(
                "timing_cuda_backward_block_attention_xsa_accum_ms={:.3}",
                result.timing_cuda_backward_block_attention_xsa_accum_ms
            );
            println!(
                "timing_cuda_backward_block_qkv_ms={:.3}",
                result.timing_cuda_backward_block_qkv_ms
            );
            println!(
                "timing_cuda_backward_block_qkv_rope_ms={:.3}",
                result.timing_cuda_backward_block_qkv_rope_ms
            );
            println!(
                "timing_cuda_backward_block_qkv_proj_ms={:.3}",
                result.timing_cuda_backward_block_qkv_proj_ms
            );
            println!(
                "timing_cuda_backward_block_qkv_ve_ms={:.3}",
                result.timing_cuda_backward_block_qkv_ve_ms
            );
            println!(
                "timing_cuda_backward_block_qkv_norm_resid_ms={:.3}",
                result.timing_cuda_backward_block_qkv_norm_resid_ms
            );
            println!(
                "timing_cuda_backward_output_ms={:.3}",
                result.timing_cuda_backward_output_ms
            );
            println!(
                "timing_cuda_backward_decoder_ms={:.3}",
                result.timing_cuda_backward_decoder_ms
            );
            println!(
                "timing_cuda_backward_encoder_ms={:.3}",
                result.timing_cuda_backward_encoder_ms
            );
            println!(
                "timing_cuda_backward_tail_ms={:.3}",
                result.timing_cuda_backward_tail_ms
            );
            println!(
                "timing_cuda_non_bank_sync_ms={:.3}",
                result.timing_cuda_non_bank_sync_ms
            );
            println!(
                "timing_cuda_bank_update_ms={:.3}",
                result.timing_cuda_bank_update_ms
            );
            println!(
                "timing_cuda_non_bank_update_ms={:.3}",
                result.timing_cuda_non_bank_update_ms
            );
            println!(
                "timing_post_train_sync_ms={:.3}",
                result.timing_post_train_sync_ms
            );
            println!(
                "timing_artifact_export_ms={:.3}",
                result.timing_artifact_export_ms
            );
            println!("timing_eval_ms={:.3}", result.timing_eval_ms);
            println!(
                "timing_data_sampling_ms_per_step={:.3}",
                timing_per_step(result.timing_data_sampling_ms, result.timing_steps)
            );
            println!(
                "timing_train_step_ms_per_step={:.3}",
                timing_per_step(result.timing_train_step_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_zero_grads_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_zero_grads_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_h2d_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_h2d_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_backward_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_backward_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_backward_forward_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_backward_forward_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_backward_forward_logits_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_forward_logits_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_mlp_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_mlp_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_mlp_residual_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_mlp_residual_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_mlp_down_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_mlp_down_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_mlp_act_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_mlp_act_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_mlp_up_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_mlp_up_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_mlp_norm_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_mlp_norm_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attn_out_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attn_out_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attn_out_residual_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attn_out_residual_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attn_out_proj_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attn_out_proj_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attn_out_gate_xsa_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attn_out_gate_xsa_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attention_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attention_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attention_sdpa_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attention_sdpa_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_attention_xsa_accum_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_attention_xsa_accum_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_qkv_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_qkv_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_qkv_rope_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_qkv_rope_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_qkv_proj_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_qkv_proj_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_qkv_ve_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_qkv_ve_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_block_qkv_norm_resid_ms_per_step={:.3}",
                timing_per_step(
                    result.timing_cuda_backward_block_qkv_norm_resid_ms,
                    result.timing_steps
                )
            );
            println!(
                "timing_cuda_backward_output_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_backward_output_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_non_bank_sync_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_non_bank_sync_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_bank_update_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_bank_update_ms, result.timing_steps)
            );
            println!(
                "timing_cuda_non_bank_update_ms_per_step={:.3}",
                timing_per_step(result.timing_cuda_non_bank_update_ms, result.timing_steps)
            );
            if let Some(bytes) = result.artifact_bytes {
                println!("artifact_bytes={bytes}");
            }
            if let Some(bytes) = result.submission_code_bytes {
                println!("submission_code_bytes={bytes}");
            }
            if let Some(bytes) = result.submission_total_bytes {
                println!("submission_total_bytes={bytes}");
            }
            if let Some(ok) = result.artifact_budget_ok {
                println!("artifact_budget_ok={ok}");
            }
            if let Some(bpb) = result.proxy_bpb {
                println!("proxy_bpb={bpb:.6}");
            }
            if let Some(source) = result.proxy_metric_source {
                println!("proxy_metric_source={source}");
            }
            if let Some(tokens) = result.eval_tokens {
                println!("eval_tokens={tokens}");
            }
            if let Some(loss) = result.eval_loss {
                println!("eval_loss={loss:.6}");
            }
            if let Some(bpb) = result.final_bpb {
                println!("final_bpb={bpb:.6}");
            }
        }
        "sweep" => {
            let mut mode = RunMode::Smoke;
            let mut backend: Option<TrainBackend> = None;
            let mut train_data_pattern: Option<String> = None;
            let mut validation_data_pattern: Option<String> = None;
            let mut rank: Option<usize> = None;
            let mut world_size: Option<usize> = None;
            let mut batch_tokens: Option<usize> = None;
            let mut seq_len: Option<usize> = None;
            let mut total_iterations: Option<usize> = None;
            let mut max_wallclock_seconds: Option<f32> = None;
            let mut min_lr_scale: Option<f32> = None;
            let mut tokenizer_vocab_path: Option<String> = None;
            let mut caseops_byte_sidecar_pattern: Option<String> = None;
            let mut eval_max_tokens: Option<usize> = None;
            let mut eval_stride: Option<usize> = None;
            let mut attention_backend: Option<AttentionBackend> = None;
            let mut distributed_optimizer_backend: Option<DistributedOptimizerBackend> = None;
            let mut eval_adaptation_backend: Option<EvalAdaptationBackend> = None;
            let mut attn_out_gate: Option<bool> = None;
            let mut attn_out_gate_width: Option<usize> = None;
            let mut quant_scheme: Option<QuantScheme> = None;
            let mut prune_keep_ratio: Option<f32> = None;
            let mut fast_bank_updates = false;
            let mut allow_unsupported_variants = false;
            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--mode" => {
                        if let Some(raw) = args.next() {
                            mode = parse_mode(&raw).unwrap_or(mode);
                        }
                    }
                    "--backend" => backend = args.next().as_deref().and_then(parse_backend),
                    "--train-data" => train_data_pattern = args.next(),
                    "--val-data" => validation_data_pattern = args.next(),
                    "--rank" => rank = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--world-size" => {
                        world_size = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--batch-tokens" => {
                        batch_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--seq-len" => seq_len = args.next().and_then(|v| v.parse::<usize>().ok()),
                    "--total-iterations" | "--max-steps" => {
                        total_iterations = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--max-wallclock-seconds" => {
                        max_wallclock_seconds = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--min-lr-scale" => {
                        min_lr_scale = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--tokenizer-vocab" => tokenizer_vocab_path = args.next(),
                    "--caseops-byte-sidecar" => caseops_byte_sidecar_pattern = args.next(),
                    "--eval-max-tokens" => {
                        eval_max_tokens = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--eval-stride" => {
                        eval_stride = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--attention-backend" => {
                        attention_backend = args.next().as_deref().and_then(parse_attention_backend)
                    }
                    "--distributed-optimizer" => {
                        distributed_optimizer_backend = args
                            .next()
                            .as_deref()
                            .and_then(parse_distributed_optimizer_backend)
                    }
                    "--eval-adaptation" => {
                        eval_adaptation_backend = args
                            .next()
                            .as_deref()
                            .and_then(parse_eval_adaptation_backend)
                    }
                    "--attn-out-gate" => attn_out_gate = Some(true),
                    "--attn-out-gate-width" => {
                        attn_out_gate_width = args.next().and_then(|v| v.parse::<usize>().ok())
                    }
                    "--quant-scheme" => {
                        quant_scheme = args.next().as_deref().and_then(parse_quant_scheme)
                    }
                    "--prune-keep-ratio" => {
                        prune_keep_ratio = args.next().and_then(|v| v.parse::<f32>().ok())
                    }
                    "--fast-bank-updates" => fast_bank_updates = true,
                    "--allow-unsupported-variants" => allow_unsupported_variants = true,
                    _ => {}
                }
            }
            let default_families = [VariantFamily::BaselineSp8192, VariantFamily::XsaAllSp8192];
            let all_families = [
                VariantFamily::BaselineSp8192,
                VariantFamily::XsaAllSp8192,
                VariantFamily::RecurrenceMidSp8192,
                VariantFamily::ParallelResidSp8192,
                VariantFamily::HybridCompetitiveSp8192,
                VariantFamily::Frontier1855Like,
            ];
            let families: &[VariantFamily] = if allow_unsupported_variants {
                &all_families
            } else {
                &default_families
            };
            for &family in families {
                let mut run_spec = RunSpec::for_family(family);
                run_spec.mode = mode;
                if let Some(backend) = backend {
                    run_spec.train.backend = backend;
                }
                if allow_unsupported_variants {
                    run_spec.allow_unsupported_variants = true;
                }
                run_spec.train.artifact_path =
                    format!("/tmp/pg_{family:?}_{mode:?}.pgrs").to_lowercase();
                if let Some(pattern) = train_data_pattern.clone() {
                    run_spec.train.train_data_pattern = Some(pattern);
                }
                if let Some(pattern) = validation_data_pattern.clone() {
                    run_spec.train.validation_data_pattern = Some(pattern);
                }
                if let Some(value) = rank {
                    run_spec.train.rank = value;
                }
                if let Some(value) = world_size {
                    run_spec.train.world_size = value;
                }
                if let Some(value) = batch_tokens {
                    run_spec.train.batch_tokens = value;
                }
                if let Some(value) = seq_len {
                    run_spec.train.seq_len = value;
                    run_spec.model.train_seq_len = value.max(run_spec.model.train_seq_len);
                }
                if let Some(value) = total_iterations {
                    run_spec.train.total_iterations = value;
                }
                if let Some(value) = max_wallclock_seconds {
                    run_spec.train.max_wallclock_seconds = value;
                }
                if let Some(value) = min_lr_scale {
                    run_spec.train.min_lr_scale = value;
                }
                if let Some(path) = tokenizer_vocab_path.clone() {
                    run_spec.eval.tokenizer_vocab_path = Some(path);
                }
                if let Some(pattern) = caseops_byte_sidecar_pattern.clone() {
                    run_spec.eval.caseops_byte_sidecar_pattern = Some(pattern);
                }
                if let Some(value) = eval_max_tokens {
                    run_spec.eval.max_tokens = Some(value);
                }
                if let Some(value) = eval_stride {
                    run_spec.eval.stride = value;
                }
                if let Some(value) = attention_backend {
                    run_spec.model.attention_backend = value;
                }
                if let Some(value) = distributed_optimizer_backend {
                    run_spec.train.distributed_optimizer_backend = value;
                }
                if let Some(value) = eval_adaptation_backend {
                    run_spec.eval.adaptation_backend = value;
                    run_spec.eval.qttt = value != EvalAdaptationBackend::None;
                }
                if let Some(value) = attn_out_gate {
                    run_spec.model.attn_out_gate.enabled = value;
                }
                if let Some(value) = attn_out_gate_width {
                    run_spec.model.attn_out_gate.width = value;
                }
                if let Some(value) = quant_scheme {
                    run_spec.quant.scheme = value;
                }
                if let Some(value) = prune_keep_ratio {
                    run_spec.quant.prune_keep_ratio = Some(value);
                }
                if fast_bank_updates {
                    run_spec.train.fast_bank_updates = true;
                }
                match VariantRunner::new(run_spec.clone()).and_then(|runner| runner.run(mode)) {
                    Ok(result) => {
                        println!(
                            "variant={:?} status=ok backend={:?} fingerprint={} steps={} loss={:.6} train_loss_source={} ms_per_step={:.3} timing_steps={} timing_measured_ms_per_step={:.3} rank={} world_size={} seq_len={} global_batch_tokens={} local_microbatches_per_step={} tokens_seen_global={} distributed_sync={} attention_backend={} distributed_optimizer_backend={} eval_adaptation_backend={} frontier_record_ready={} leaderboard_algorithm_ready={} record_shape={} record_attention_grade={} microbatch_serial_loop={} bank_update_backend={} train_data_source={} bpb_byte_source={} timing_backend={} timing_data_sampling_ms={:.3} timing_train_step_ms={:.3} timing_cuda_zero_grads_ms={:.3} timing_cuda_h2d_ms={:.3} timing_cuda_backward_ms={:.3} timing_cuda_backward_forward_ms={:.3} timing_cuda_backward_forward_embed_ms={:.3} timing_cuda_backward_forward_encoder_ms={:.3} timing_cuda_backward_forward_encoder_layer_max_ms={:.3} timing_cuda_backward_forward_decoder_ms={:.3} timing_cuda_backward_forward_decoder_layer_max_ms={:.3} timing_cuda_backward_forward_logits_ms={:.3} timing_cuda_backward_forward_block_pre_attn_ms={:.3} timing_cuda_backward_forward_block_attention_ms={:.3} timing_cuda_backward_forward_block_post_attn_ms={:.3} timing_cuda_backward_forward_block_mlp_ms={:.3} timing_cuda_backward_block_recompute_ms={:.3} timing_cuda_backward_block_mlp_ms={:.3} timing_cuda_backward_block_attn_out_ms={:.3} timing_cuda_backward_block_attention_ms={:.3} timing_cuda_backward_block_qkv_ms={:.3} timing_cuda_backward_output_ms={:.3} timing_cuda_backward_decoder_ms={:.3} timing_cuda_backward_encoder_ms={:.3} timing_cuda_backward_tail_ms={:.3} timing_cuda_non_bank_sync_ms={:.3} timing_cuda_bank_update_ms={:.3} timing_cuda_non_bank_update_ms={:.3} timing_post_train_sync_ms={:.3} timing_artifact_export_ms={:.3} timing_eval_ms={:.3} proxy_bpb={} proxy_metric_source={} final_bpb={} artifact_bytes={} submission_total_bytes={} artifact_budget_ok={}",
                            family,
                            result.train_backend,
                            result.variant_fingerprint,
                            result.steps_completed,
                            result.train_loss,
                            result.train_loss_source,
                            result.ms_per_step,
                            result.timing_steps,
                            result.timing_measured_ms_per_step,
                            result.rank,
                            result.world_size,
                            result.seq_len,
                            result.global_batch_tokens,
                            result.local_microbatches_per_step,
                            result.tokens_seen_global,
                            result.distributed_sync,
                            result.attention_backend,
                            result.distributed_optimizer_backend,
                            result.eval_adaptation_backend,
                            result.frontier_record_ready,
                            result.leaderboard_algorithm_ready,
                            result.record_shape,
                            result.record_attention_grade,
                            result.microbatch_serial_loop,
                            result.bank_update_backend,
                            result.train_data_source,
                            result.bpb_byte_source,
                            result.timing_backend,
                            result.timing_data_sampling_ms,
                            result.timing_train_step_ms,
                            result.timing_cuda_zero_grads_ms,
                            result.timing_cuda_h2d_ms,
                            result.timing_cuda_backward_ms,
                            result.timing_cuda_backward_forward_ms,
                            result.timing_cuda_backward_forward_embed_ms,
                            result.timing_cuda_backward_forward_encoder_ms,
                            result.timing_cuda_backward_forward_encoder_layer_max_ms,
                            result.timing_cuda_backward_forward_decoder_ms,
                            result.timing_cuda_backward_forward_decoder_layer_max_ms,
                            result.timing_cuda_backward_forward_logits_ms,
                            result.timing_cuda_backward_forward_block_pre_attn_ms,
                            result.timing_cuda_backward_forward_block_attention_ms,
                            result.timing_cuda_backward_forward_block_post_attn_ms,
                            result.timing_cuda_backward_forward_block_mlp_ms,
                            result.timing_cuda_backward_block_recompute_ms,
                            result.timing_cuda_backward_block_mlp_ms,
                            result.timing_cuda_backward_block_attn_out_ms,
                            result.timing_cuda_backward_block_attention_ms,
                            result.timing_cuda_backward_block_qkv_ms,
                            result.timing_cuda_backward_output_ms,
                            result.timing_cuda_backward_decoder_ms,
                            result.timing_cuda_backward_encoder_ms,
                            result.timing_cuda_backward_tail_ms,
                            result.timing_cuda_non_bank_sync_ms,
                            result.timing_cuda_bank_update_ms,
                            result.timing_cuda_non_bank_update_ms,
                            result.timing_post_train_sync_ms,
                            result.timing_artifact_export_ms,
                            result.timing_eval_ms,
                            result
                                .proxy_bpb
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .proxy_metric_source
                                .clone()
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .final_bpb
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .artifact_bytes
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .submission_total_bytes
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "none".to_string()),
                            result
                                .artifact_budget_ok
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                        );
                    }
                    Err(err) => {
                        println!("variant={:?} status=skipped reason={}", family, err);
                    }
                }
            }
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!(
        "  pg-train run [--spec spec.toml] [--builtin baseline_sp8192] [--mode smoke|proxy|record-shaped-proxy|record]"
    );
    eprintln!("               [--backend cpu|cuda-single|cuda-single-parity|cuda-distributed]");
    eprintln!("               [--train-data glob] [--val-data glob] [--artifact path]");
    eprintln!("               [--rank n] [--world-size n] [--batch-tokens n] [--seq-len n]");
    eprintln!(
        "               [--total-iterations n|--max-steps n] [--max-wallclock-seconds n] [--min-lr-scale f]"
    );
    eprintln!("               [--tokenizer-vocab path] [--eval-max-tokens n] [--eval-stride n]");
    eprintln!("               [--attention-backend naive_f32|flash_f32|cudnn_sdpa_bf16]");
    eprintln!(
        "               [--distributed-optimizer all_reduce_replicated_muon|sharded_parallel_muon]"
    );
    eprintln!("               [--eval-adaptation none|cpu_q_only|gpu_lora_phased]");
    eprintln!("               [--attn-out-gate] [--attn-out-gate-width n]");
    eprintln!(
        "               [--quant-scheme gptq_lite_int6|mixed_int5_int6|aggressive|tight_int7_int4]"
    );
    eprintln!("               [--prune-keep-ratio f]");
    eprintln!("               [--fast-bank-updates] [--allow-unsupported-variants]");
    eprintln!("               record requires --backend cuda-distributed and real --train-data");
    eprintln!("  pg-train sweep [--mode smoke|proxy|record-shaped-proxy]");
    eprintln!("                 [--backend cpu|cuda-single|cuda-single-parity|cuda-distributed]");
    eprintln!("                 [--train-data glob] [--val-data glob]");
    eprintln!("                 [--rank n] [--world-size n] [--batch-tokens n] [--seq-len n]");
    eprintln!(
        "                 [--total-iterations n|--max-steps n] [--max-wallclock-seconds n] [--min-lr-scale f]"
    );
    eprintln!("                 [--tokenizer-vocab path] [--eval-max-tokens n] [--eval-stride n]");
    eprintln!("                 [--attention-backend naive_f32|flash_f32|cudnn_sdpa_bf16]");
    eprintln!(
        "                 [--distributed-optimizer all_reduce_replicated_muon|sharded_parallel_muon]"
    );
    eprintln!("                 [--eval-adaptation none|cpu_q_only|gpu_lora_phased]");
    eprintln!("                 [--attn-out-gate] [--attn-out-gate-width n]");
    eprintln!(
        "                 [--quant-scheme gptq_lite_int6|mixed_int5_int6|aggressive|tight_int7_int4]"
    );
    eprintln!("                 [--prune-keep-ratio f]");
    eprintln!("                 [--fast-bank-updates] [--allow-unsupported-variants]");
    eprintln!("  env: PG_SUBMISSION_CODE_BYTES overrides executable-size budget accounting");
}

fn parse_mode(raw: &str) -> Option<RunMode> {
    match raw {
        "smoke" => Some(RunMode::Smoke),
        "proxy" => Some(RunMode::Proxy),
        "record-shaped-proxy" | "record_shaped_proxy" => Some(RunMode::RecordShapedProxy),
        "record" => Some(RunMode::Record),
        _ => None,
    }
}

fn parse_backend(raw: &str) -> Option<TrainBackend> {
    match raw {
        "cpu" => Some(TrainBackend::Cpu),
        "cuda-single" => Some(TrainBackend::CudaSingle),
        "cuda-single-parity" => Some(TrainBackend::CudaSingleParity),
        "cuda-distributed" => Some(TrainBackend::CudaDistributed),
        _ => None,
    }
}

fn parse_attention_backend(raw: &str) -> Option<AttentionBackend> {
    match raw {
        "naive_f32" => Some(AttentionBackend::NaiveF32),
        "flash_f32" => Some(AttentionBackend::FlashF32),
        "cudnn_sdpa_bf16" => Some(AttentionBackend::CudnnSdpaBf16),
        _ => None,
    }
}

fn parse_distributed_optimizer_backend(raw: &str) -> Option<DistributedOptimizerBackend> {
    match raw {
        "all_reduce_replicated_muon" => Some(DistributedOptimizerBackend::AllReduceReplicatedMuon),
        "sharded_parallel_muon" => Some(DistributedOptimizerBackend::ShardedParallelMuon),
        _ => None,
    }
}

fn parse_eval_adaptation_backend(raw: &str) -> Option<EvalAdaptationBackend> {
    match raw {
        "none" => Some(EvalAdaptationBackend::None),
        "cpu_q_only" => Some(EvalAdaptationBackend::CpuQOnly),
        "gpu_lora_phased" => Some(EvalAdaptationBackend::GpuLoraPhased),
        _ => None,
    }
}

fn parse_quant_scheme(raw: &str) -> Option<QuantScheme> {
    match raw {
        "none" => Some(QuantScheme::None),
        "gptq_lite_int6" => Some(QuantScheme::GptqLiteInt6),
        "mixed_int5_int6" => Some(QuantScheme::MixedInt5Int6),
        "aggressive" => Some(QuantScheme::Aggressive),
        "tight_int7_int4" => Some(QuantScheme::TightInt7Int4),
        _ => None,
    }
}

fn parse_family(raw: &str) -> Option<VariantFamily> {
    match raw {
        "baseline_sp8192" => Some(VariantFamily::BaselineSp8192),
        "xsa_all_sp8192" => Some(VariantFamily::XsaAllSp8192),
        "recurrence_mid_sp8192" => Some(VariantFamily::RecurrenceMidSp8192),
        "parallel_resid_sp8192" => Some(VariantFamily::ParallelResidSp8192),
        "hybrid_competitive_sp8192" => Some(VariantFamily::HybridCompetitiveSp8192),
        "frontier_1855_like" => Some(VariantFamily::Frontier1855Like),
        _ => None,
    }
}
