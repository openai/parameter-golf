use std::path::PathBuf;

use pg_model::{RunMode, RunSpec, VariantFamily};
use pg_train::VariantRunner;

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

            while let Some(arg) = args.next() {
                match arg.as_str() {
                    "--spec" => spec_path = args.next().map(PathBuf::from),
                    "--builtin" => {
                        builtin = args.next().as_deref().and_then(parse_family);
                    }
                    "--mode" => {
                        mode = args.next().as_deref().and_then(parse_mode);
                    }
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
            let runner = VariantRunner::new(run_spec.clone()).expect("failed to create variant runner");
            let result = runner.run(run_spec.mode).expect("variant run failed");
            println!("run_name={}", result.run_name);
            println!("mode={:?}", result.mode);
            println!("variant_fingerprint={}", result.variant_fingerprint);
            println!("steps_completed={}", result.steps_completed);
            println!("train_loss={:.6}", result.train_loss);
            println!("ms_per_step={:.3}", result.ms_per_step);
            println!("wallclock_seconds={:.3}", result.wallclock_seconds);
            if let Some(bytes) = result.artifact_bytes {
                println!("artifact_bytes={bytes}");
            }
            if let Some(bpb) = result.proxy_bpb {
                println!("proxy_bpb={bpb:.6}");
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
    eprintln!("  pg-train run [--spec spec.toml] [--builtin baseline_sp8192] [--mode smoke|proxy|record]");
}

fn parse_mode(raw: &str) -> Option<RunMode> {
    match raw {
        "smoke" => Some(RunMode::Smoke),
        "proxy" => Some(RunMode::Proxy),
        "record" => Some(RunMode::Record),
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
        _ => None,
    }
}
