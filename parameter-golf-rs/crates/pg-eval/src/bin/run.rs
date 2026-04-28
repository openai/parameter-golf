use std::path::PathBuf;

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
            plan.artifact_budget_ok(total_bytes)
        );
        if leaderboard_mode && !plan.artifact_budget_ok(total_bytes) {
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
        let (loss, bpb) =
            if run_spec.eval.adaptation_backend == EvalAdaptationBackend::GpuLoraPhased {
                eval_gpu_lora_phased(&model, &plan, &tokens, &target_bytes)
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
            "eval_audit_json={{\"leaderboard_mode\":{},\"full_validation\":{},\"eval_tokens\":{},\"legal_score_first\":{},\"eval_adaptation_backend\":\"{:?}\",\"bpb_kind\":\"{}\",\"placeholder_bpb\":{},\"artifact_budget_checked\":{}}}",
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
        );
    } else {
        println!("eval_status=plan_only_no_validation_data");
    }
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
) -> pg_core::PgResult<(f64, f64)> {
    let cfg = pg_eval::gpu_lora_ttt::GpuLoraPhasedTttConfig::from_plan(plan, tokens.len());
    pg_eval::gpu_lora_ttt::eval_gpu_lora_phased_ttt(model, plan, tokens, target_bytes, &cfg)
}

#[cfg(not(feature = "cuda"))]
fn eval_gpu_lora_phased(
    _model: &GptModel,
    _plan: &ExecutionPlan,
    _tokens: &[u32],
    _target_bytes: &[f32],
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
}
