use std::path::PathBuf;

use pg_model::{ExecutionPlan, RunSpec, VariantFamily};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut artifact: Option<PathBuf> = None;
    let mut spec = None;
    let mut builtin = VariantFamily::BaselineSp8192;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--artifact" => artifact = args.next().map(PathBuf::from),
            "--spec" => spec = args.next(),
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

    let run_spec = spec
        .map(PathBuf::from)
        .map(|p| RunSpec::load(&p).expect("failed to load spec"))
        .unwrap_or_else(|| RunSpec::for_family(builtin));
    let plan = ExecutionPlan::from_run_spec(&run_spec).expect("failed to build execution plan");

    let artifact_bytes = artifact
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len() as usize);

    println!("variant_fingerprint={}", plan.variant_fingerprint);
    println!("eval_stride={}", plan.eval_plan.stride);
    println!("legal_score_first={}", plan.eval_plan.legal_score_first);
    println!("qttt={}", plan.eval_plan.qttt);
    if let Some(bytes) = artifact_bytes {
        println!("artifact_bytes={bytes}");
        println!("artifact_budget_ok={}", plan.artifact_budget_ok(bytes));
    } else {
        println!("artifact_bytes=unknown");
    }
}
