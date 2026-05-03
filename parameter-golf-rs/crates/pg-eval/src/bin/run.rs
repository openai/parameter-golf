use std::path::PathBuf;
use std::time::Instant;

use pg_model::{EvalAdaptationBackend, ExecutionPlan, GptModel, RunSpec, VariantFamily};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut artifact: Option<PathBuf> = None;
    let mut spec = None;
    let mut builtin = VariantFamily::BaselineSp8192;
    let mut max_tokens: Option<usize> = None;
    let mut validation_data_pattern: Option<String> = None;
    let mut stride: Option<usize> = None;
    let mut tokenizer_vocab_path: Option<String> = None;
    let mut caseops_byte_sidecar_pattern: Option<String> = None;
    let mut leaderboard_mode = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--spec" => spec = args.next(),
            "--val-data" => validation_data_pattern = args.next(),
            "--max-tokens" => max_tokens = args.next().and_then(|v| v.parse::<usize>().ok()),
            "--stride" => stride = args.next().and_then(|v| v.parse::<usize>().ok()),
            "--tokenizer-vocab" => tokenizer_vocab_path = args.next(),
            "--caseops-byte-sidecar" => caseops_byte_sidecar_pattern = args.next(),
            "--leaderboard" => leaderboard_mode = true,
            "--builtin" => {
                if let Some(name) = args.next() {
                    builtin = match name.as_str() {
                        "xsa_all_sp8192" => VariantFamily::XsaAllSp8192,
                        "recurrence_mid_sp8192" => VariantFamily::RecurrenceMidSp8192,
                        "parallel_resid_sp8192" => VariantFamily::ParallelResidSp8192,
                        "hybrid_competitive_sp8192" => VariantFamily::HybridCompetitiveSp8192,
                        "frontier_1855_like" => VariantFamily::Frontier1855Like,
                        _ => VariantFamily::BaselineSp8192,
                    };
                }
            }
            _ => {}
        }
    }

    let mut run_spec = spec
        .map(PathBuf::from)
        .map(|p| RunSpec::load(&p).expect("failed to load spec"))
        .unwrap_or_else(|| RunSpec::for_family(builtin));
    if let Some(pattern) = validation_data_pattern {
        run_spec.train.validation_data_pattern = Some(pattern);
    }
    if let Some(value) = stride {
        run_spec.eval.stride = value;
    }
    if let Some(path) = tokenizer_vocab_path {
        run_spec.eval.tokenizer_vocab_path = Some(path);
    }
    if let Some(pattern) = caseops_byte_sidecar_pattern {
        run_spec.eval.caseops_byte_sidecar_pattern = Some(pattern);
    }
    validate_leaderboard_eval_request(&run_spec, artifact.as_ref(), max_tokens, leaderboard_mode);
    if leaderboard_mode {
        if let Some(path) = artifact.as_ref() {
            if !path.is_file() {
                fail(&format!(
                    "leaderboard eval artifact does not exist or is not a regular file: {}",
                    path.display()
                ));
            }
        }
    }
    let plan = ExecutionPlan::from_run_spec(&run_spec).expect("failed to build execution plan");

    let artifact_bytes = artifact
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len() as usize);

    println!("variant_fingerprint={}", plan.variant_fingerprint);
    println!("eval_stride={}", plan.eval_plan.stride);
    println!("legal_score_first={}", plan.eval_plan.legal_score_first);
    println!("qttt={}", plan.eval_plan.qttt);
    println!(
        "eval_adaptation_backend={:?}",
        plan.eval_plan.adaptation_backend
    );
    println!("lora_rank={}", plan.eval_plan.lora_rank);
    println!("lora_alpha={:.3}", plan.eval_plan.lora_alpha);
    println!(
        "phased_ttt_prefix_docs={}",
        plan.eval_plan.phased_ttt_prefix_docs
    );
    println!("phased_ttt_phases={}", plan.eval_plan.phased_ttt_phases);
    println!(
        "phased_ttt_weight_decay={:.3}",
        plan.eval_plan.phased_ttt_weight_decay
    );
    if let Some(bytes) = artifact_bytes {
        let code_bytes = current_executable_bytes();
        let total_bytes = bytes + code_bytes;
        println!("artifact_bytes={bytes}");
        println!("submission_code_bytes={code_bytes}");
        println!("submission_total_bytes={total_bytes}");
        println!(
            "artifact_budget_ok={}",
            plan.submission_budget_ok(code_bytes, bytes)
        );
        if leaderboard_mode && !plan.submission_budget_ok(code_bytes, bytes) {
            fail(&format!(
                "leaderboard eval artifact budget failed: artifact_bytes={bytes} code_bytes={code_bytes} total_bytes={total_bytes} limit={}",
                plan.run_spec.quant.target_artifact_bytes
            ));
        }
    } else {
        println!("artifact_bytes=unknown");
    }

    if let Some(pattern) = run_spec.train.validation_data_pattern.as_deref() {
        let mut model = GptModel::new(run_spec.model.to_model_config());
        model.fill_deterministic();
        if let Some(path) = artifact.as_ref() {
            pg_quant::export::load_artifact(path, &mut model).expect("failed to load artifact");
        }
        let tokens = pg_data::token_stream::load_validation_tokens_limited(
            pattern,
            max_tokens.map(|limit| limit.max(2)),
        )
        .expect("failed to load validation tokens")
        .into_iter()
        .map(|v| v as u32)
        .collect::<Vec<_>>();
        let bpb_luts = if let Some(path) = run_spec.eval.tokenizer_vocab_path.as_deref() {
            pg_data::bpb::BpbLuts::from_vocab_file(std::path::Path::new(path))
                .expect("failed to load tokenizer vocab")
        } else {
            pg_data::bpb::BpbLuts::placeholder(run_spec.model.vocab_size)
        };
        let target_bytes =
            eval_target_byte_counts(&run_spec, &tokens, &bpb_luts, max_tokens.map(|v| v.max(2)))
                .expect("failed to load eval byte counts");
        let seq_len = run_spec
            .model
            .eval_seq_len
            .min(tokens.len().saturating_sub(1))
            .max(1);
        let eval_t0 = Instant::now();
        let eval_world_size = eval_gpu_world_size(&run_spec, leaderboard_mode);
        let (loss, bpb) =
            if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased {
                eval_gpu_lora_phased(&model, &plan, &tokens, &target_bytes, eval_world_size)
                    .expect("GPU LoRA/phased TTT eval failed")
            } else if run_spec.eval.qttt {
                let mut cfg = pg_eval::qttt::QttTConfig::paper_default(seq_len);
                cfg.stride = run_spec.eval.stride;
                cfg.seq_len = seq_len;
                cfg.chunk_tokens = run_spec.eval.chunk_tokens;
                pg_eval::qttt::eval_qttt(&mut model, &tokens, &target_bytes, &cfg)
            } else {
                pg_eval::sliding::eval_sliding(
                    &model,
                    &tokens,
                    &target_bytes,
                    run_spec.eval.stride,
                    seq_len,
                )
            };
        let eval_wallclock_seconds = eval_t0.elapsed().as_secs_f64();
        println!("eval_wallclock_seconds={eval_wallclock_seconds:.3}");
        let max_eval_wallclock_seconds = leaderboard_eval_max_wallclock_seconds();
        println!("eval_max_wallclock_seconds={max_eval_wallclock_seconds:.3}");
        if leaderboard_mode
            && max_eval_wallclock_seconds > 0.0
            && eval_wallclock_seconds > max_eval_wallclock_seconds
        {
            fail(&format!(
                "leaderboard eval exceeded wallclock budget: eval_wallclock_seconds={eval_wallclock_seconds:.3} max_eval_wallclock_seconds={max_eval_wallclock_seconds:.3}"
            ));
        }
        println!("eval_tokens={}", tokens.len());
        println!("eval_loss={loss:.6}");
        println!(
            "eval_bpb_kind={}",
            if run_spec.eval.caseops_byte_sidecar_pattern.is_some() {
                "caseops_byte_sidecar"
            } else if run_spec.eval.tokenizer_vocab_path.is_some() {
                "tokenizer_vocab"
            } else {
                "placeholder"
            }
        );
        println!("eval_bpb={bpb:.6}");
        println!(
            "eval_audit_json={{\"leaderboard_mode\":{},\"full_validation\":{},\"eval_tokens\":{},\"legal_score_first\":{},\"eval_adaptation_backend\":\"{:?}\",\"bpb_kind\":\"{}\",\"placeholder_bpb\":{},\"artifact_budget_checked\":{},\"eval_wallclock_seconds\":{:.3},\"eval_max_wallclock_seconds\":{:.3}}}",
            leaderboard_mode,
            max_tokens.is_none(),
            tokens.len(),
            plan.eval_plan.legal_score_first,
            plan.eval_plan.adaptation_backend,
            if run_spec.eval.caseops_byte_sidecar_pattern.is_some() {
                "caseops_byte_sidecar"
            } else if run_spec.eval.tokenizer_vocab_path.is_some() {
                "tokenizer_vocab"
            } else {
                "placeholder"
            },
            run_spec.eval.caseops_byte_sidecar_pattern.is_none()
                && run_spec.eval.tokenizer_vocab_path.is_none(),
            artifact_bytes.is_some(),
            eval_wallclock_seconds,
            max_eval_wallclock_seconds,
        );
        println!("eval_gpu_world_size={}", eval_world_size);
    } else {
        println!("eval_status=plan_only_no_validation_data");
    }
}

fn eval_gpu_world_size(run_spec: &RunSpec, leaderboard_mode: bool) -> usize {
    if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased {
        if leaderboard_mode {
            std::env::var("PG_EVAL_GPU_WORLD_SIZE")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(1)
        } else {
            1
        }
    } else {
        0
    }
}

fn leaderboard_eval_max_wallclock_seconds() -> f64 {
    std::env::var("PG_EVAL_MAX_WALLCLOCK_SECONDS")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(600.0)
}

fn validate_leaderboard_eval_request(
    run_spec: &RunSpec,
    artifact: Option<&PathBuf>,
    max_tokens: Option<usize>,
    leaderboard_mode: bool,
) {
    if let Some(message) =
        leaderboard_eval_error(run_spec, artifact.is_some(), max_tokens, leaderboard_mode)
    {
        fail(message);
    }
}

fn leaderboard_eval_error(
    run_spec: &RunSpec,
    artifact_present: bool,
    max_tokens: Option<usize>,
    leaderboard_mode: bool,
) -> Option<&'static str> {
    if !leaderboard_mode {
        return None;
    }
    if !artifact_present {
        return Some("leaderboard eval requires --artifact");
    }
    if run_spec.train.validation_data_pattern.is_none() {
        return Some("leaderboard eval requires --val-data");
    }
    if max_tokens.is_some() {
        return Some(
            "leaderboard eval cannot use --max-tokens; it must score the full validation stream",
        );
    }
    if !run_spec.eval.legal_score_first {
        return Some("leaderboard eval requires legal_score_first=true");
    }
    if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
        && run_spec.eval.phased_ttt_prefix_docs > 0
        && run_spec.model.smear_gate_boundary_token_id.is_none()
    {
        return Some(
            "leaderboard GPU LoRA/phased prefix-doc TTT requires model.smear_gate_boundary_token_id/BOS token",
        );
    }
    if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
        && eval_gpu_world_size(run_spec, leaderboard_mode) < run_spec.train.world_size.max(1)
    {
        return Some(
            "leaderboard GPU LoRA/phased eval requires distributed eval across the configured H100 world_size; single-GPU eval is a development path only",
        );
    }
    if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased
        && !ttt_score_mutation_guard_requested()
    {
        return Some(
            "leaderboard GPU LoRA/phased eval requires PG_TTT_ASSERT_SCORE_NO_MUTATION=1 so score-before-update is runtime-audited",
        );
    }
    if run_spec.model.caseops.enabled && run_spec.model.caseops.byte_sidecar {
        if run_spec.eval.caseops_byte_sidecar_pattern.is_none() {
            return Some("leaderboard eval with CaseOps requires --caseops-byte-sidecar");
        }
    } else if run_spec.eval.tokenizer_vocab_path.is_none() {
        return Some(
            "leaderboard eval requires --tokenizer-vocab when CaseOps byte sidecar is not active",
        );
    }
    None
}

fn ttt_score_mutation_guard_requested() -> bool {
    matches!(
        std::env::var("PG_TTT_ASSERT_SCORE_NO_MUTATION")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn eval_target_byte_counts(
    run_spec: &RunSpec,
    tokens: &[u32],
    bpb_luts: &pg_data::bpb::BpbLuts,
    max_tokens: Option<usize>,
) -> pg_core::PgResult<Vec<f32>> {
    if let Some(pattern) = run_spec.eval.caseops_byte_sidecar_pattern.as_deref() {
        let sidecar =
            pg_data::token_stream::load_validation_byte_sidecar_limited(pattern, max_tokens)?;
        if sidecar.len() != tokens.len() {
            return Err(pg_core::PgError::DataFormat(format!(
                "CaseOps byte sidecar length {} does not match validation token length {}",
                sidecar.len(),
                tokens.len()
            )));
        }
        return Ok(sidecar.get(1..).unwrap_or(&[]).to_vec());
    }
    Ok(bpb_luts.pair_byte_counts_u32(tokens))
}

#[cfg(feature = "cuda")]
fn eval_gpu_lora_phased(
    model: &GptModel,
    plan: &ExecutionPlan,
    tokens: &[u32],
    target_bytes: &[f32],
    world_size: usize,
) -> pg_core::PgResult<(f64, f64)> {
    let cfg = pg_eval::gpu_lora_ttt::GpuLoraPhasedTttConfig::from_plan(plan, tokens.len());
    pg_eval::gpu_lora_ttt::eval_gpu_lora_phased_ttt_distributed(
        model,
        plan,
        tokens,
        target_bytes,
        &cfg,
        world_size,
    )
}

#[cfg(not(feature = "cuda"))]
fn eval_gpu_lora_phased(
    _model: &GptModel,
    _plan: &ExecutionPlan,
    _tokens: &[u32],
    _target_bytes: &[f32],
    _world_size: usize,
) -> pg_core::PgResult<(f64, f64)> {
    Err(pg_core::PgError::InvalidOp(
        "eval_adaptation_backend=gpu_lora_phased requires pg-eval --features cuda".into(),
    ))
}

fn current_executable_bytes() -> usize {
    if let Ok(value) = std::env::var("PG_SUBMISSION_CODE_BYTES") {
        if let Ok(bytes) = value.parse::<usize>() {
            return bytes;
        }
    }
    if let Ok(path) = std::env::var("PG_SUBMISSION_CODE_DIR") {
        if let Ok(bytes) = directory_regular_file_bytes(std::path::Path::new(&path)) {
            return bytes;
        }
    }
    std::env::current_exe()
        .ok()
        .and_then(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len() as usize)
        .unwrap_or(0)
}

fn directory_regular_file_bytes(path: &std::path::Path) -> std::io::Result<usize> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len() as usize);
    }
    if !metadata.is_dir() {
        return Ok(0);
    }

    let mut total = 0usize;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        total += directory_regular_file_bytes(&entry.path())?;
    }
    Ok(total)
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn leaderboard_eval_requires_full_official_inputs() {
        let mut spec = RunSpec::for_family(VariantFamily::BaselineSp8192);
        assert_eq!(
            leaderboard_eval_error(&spec, false, None, true),
            Some("leaderboard eval requires --artifact")
        );
        assert_eq!(
            leaderboard_eval_error(&spec, true, None, true),
            Some("leaderboard eval requires --val-data")
        );
        spec.train.validation_data_pattern = Some("/val/*.bin".to_string());
        assert_eq!(
            leaderboard_eval_error(&spec, true, Some(4096), true),
            Some(
                "leaderboard eval cannot use --max-tokens; it must score the full validation stream"
            )
        );
        assert_eq!(
            leaderboard_eval_error(&spec, true, None, true),
            Some(
                "leaderboard eval requires --tokenizer-vocab when CaseOps byte sidecar is not active"
            )
        );
        spec.eval.tokenizer_vocab_path = Some("/tok.vocab".to_string());
        assert_eq!(leaderboard_eval_error(&spec, true, None, true), None);
    }

    #[test]
    fn leaderboard_eval_requires_caseops_sidecar_when_caseops_is_active() {
        let mut spec = RunSpec::for_family(VariantFamily::BaselineSp8192);
        spec.train.validation_data_pattern = Some("/val/*.bin".to_string());
        spec.model.caseops.enabled = true;
        spec.model.caseops.byte_sidecar = true;
        spec.eval.tokenizer_vocab_path = Some("/tok.vocab".to_string());
        assert_eq!(
            leaderboard_eval_error(&spec, true, None, true),
            Some("leaderboard eval with CaseOps requires --caseops-byte-sidecar")
        );
        spec.eval.caseops_byte_sidecar_pattern = Some("/val_bytes/*.bin".to_string());
        assert_eq!(leaderboard_eval_error(&spec, true, None, true), None);
    }

    #[test]
    fn leaderboard_eval_accepts_gpu_lora_prefix_doc_ttt_semantics() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_guard = std::env::var("PG_TTT_ASSERT_SCORE_NO_MUTATION").ok();
        unsafe {
            std::env::set_var("PG_TTT_ASSERT_SCORE_NO_MUTATION", "1");
        }
        let mut spec = RunSpec::for_family(VariantFamily::BaselineSp8192);
        spec.train.validation_data_pattern = Some("/val/*.bin".to_string());
        spec.eval.tokenizer_vocab_path = Some("/tok.vocab".to_string());
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        spec.eval.phased_ttt_prefix_docs = 2000;
        assert_eq!(leaderboard_eval_error(&spec, true, None, true), None);
        spec.model.smear_gate_boundary_token_id = None;
        assert_eq!(
            leaderboard_eval_error(&spec, true, None, true),
            Some(
                "leaderboard GPU LoRA/phased prefix-doc TTT requires model.smear_gate_boundary_token_id/BOS token"
            )
        );
        spec.model.smear_gate_boundary_token_id = Some(1);
        spec.eval.phased_ttt_prefix_docs = 0;
        assert_eq!(leaderboard_eval_error(&spec, true, None, true), None);

        match old_guard {
            Some(value) => unsafe { std::env::set_var("PG_TTT_ASSERT_SCORE_NO_MUTATION", value) },
            None => unsafe { std::env::remove_var("PG_TTT_ASSERT_SCORE_NO_MUTATION") },
        }
    }

    #[test]
    fn leaderboard_eval_rejects_single_gpu_lora_when_train_world_size_is_multi_gpu() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_world_size = std::env::var("PG_EVAL_GPU_WORLD_SIZE").ok();
        let old_guard = std::env::var("PG_TTT_ASSERT_SCORE_NO_MUTATION").ok();
        unsafe {
            std::env::remove_var("PG_EVAL_GPU_WORLD_SIZE");
            std::env::set_var("PG_TTT_ASSERT_SCORE_NO_MUTATION", "1");
        }

        let mut spec = RunSpec::for_family(VariantFamily::BaselineSp8192);
        spec.train.world_size = 8;
        spec.train.validation_data_pattern = Some("/val/*.bin".to_string());
        spec.eval.tokenizer_vocab_path = Some("/tok.vocab".to_string());
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        spec.eval.phased_ttt_prefix_docs = 0;

        assert_eq!(
            leaderboard_eval_error(&spec, true, None, true),
            Some(
                "leaderboard GPU LoRA/phased eval requires distributed eval across the configured H100 world_size; single-GPU eval is a development path only"
            )
        );

        unsafe {
            std::env::set_var("PG_EVAL_GPU_WORLD_SIZE", "8");
        }
        assert_eq!(leaderboard_eval_error(&spec, true, None, true), None);

        match old_world_size {
            Some(value) => unsafe { std::env::set_var("PG_EVAL_GPU_WORLD_SIZE", value) },
            None => unsafe { std::env::remove_var("PG_EVAL_GPU_WORLD_SIZE") },
        }
        match old_guard {
            Some(value) => unsafe { std::env::set_var("PG_TTT_ASSERT_SCORE_NO_MUTATION", value) },
            None => unsafe { std::env::remove_var("PG_TTT_ASSERT_SCORE_NO_MUTATION") },
        }
    }

    #[test]
    fn leaderboard_eval_requires_ttt_score_mutation_guard_for_gpu_lora() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let old_world_size = std::env::var("PG_EVAL_GPU_WORLD_SIZE").ok();
        let old_guard = std::env::var("PG_TTT_ASSERT_SCORE_NO_MUTATION").ok();
        unsafe {
            std::env::set_var("PG_EVAL_GPU_WORLD_SIZE", "8");
            std::env::remove_var("PG_TTT_ASSERT_SCORE_NO_MUTATION");
        }

        let mut spec = RunSpec::for_family(VariantFamily::BaselineSp8192);
        spec.train.world_size = 8;
        spec.train.validation_data_pattern = Some("/val/*.bin".to_string());
        spec.eval.tokenizer_vocab_path = Some("/tok.vocab".to_string());
        spec.eval.adaptation_backend = EvalAdaptationBackend::GpuLoraPhased;
        spec.eval.phased_ttt_prefix_docs = 0;

        assert_eq!(
            leaderboard_eval_error(&spec, true, None, true),
            Some(
                "leaderboard GPU LoRA/phased eval requires PG_TTT_ASSERT_SCORE_NO_MUTATION=1 so score-before-update is runtime-audited"
            )
        );
        unsafe {
            std::env::set_var("PG_TTT_ASSERT_SCORE_NO_MUTATION", "1");
        }
        assert_eq!(leaderboard_eval_error(&spec, true, None, true), None);

        match old_world_size {
            Some(value) => unsafe { std::env::set_var("PG_EVAL_GPU_WORLD_SIZE", value) },
            None => unsafe { std::env::remove_var("PG_EVAL_GPU_WORLD_SIZE") },
        }
        match old_guard {
            Some(value) => unsafe { std::env::set_var("PG_TTT_ASSERT_SCORE_NO_MUTATION", value) },
            None => unsafe { std::env::remove_var("PG_TTT_ASSERT_SCORE_NO_MUTATION") },
        }
    }
}
