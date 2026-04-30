//! Smoke test agent for IGLA race.
//!
//! Runs a synthetic 1-step experiment to verify the pipeline is alive.
//! This is the first line of defense against:
//! - "zero steps" (trainer exits without output)
//! - dead workers (no DB connection)
//! - rotated credentials (RAILWAY_TOKEN invalid)
//!
//! Usage:
//!   cargo run -p trios-igla-race --features smoke --bin smoke_agent -- \
//!     --steps 1 --seed 42

use anyhow::Result;
use clap::Parser;
use tracing::info;

#[cfg(feature = "smoke")]
use trios_railway_smoke::{run_local, SmokeConfig};

#[derive(Parser)]
#[command(
    name = "smoke-agent",
    about = "ADR-002: Fast smoke test agent (synthetic, CPU-only, <60s)"
)]
struct Cli {
    /// Number of steps to run (default: 1 for smoke)
    #[arg(long, default_value = "1")]
    steps: u32,
    /// Random seed (default: 42)
    #[arg(long, default_value = "42")]
    seed: u64,
    /// Exit with error code 1 if smoke fails
    #[arg(long)]
    fail_on_error: bool,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter("smoke_agent=info")
        .init();

    let cli = Cli::parse();

    info!("smoke-agent starting | steps={} | seed={}", cli.steps, cli.seed);

    #[cfg(feature = "smoke")]
    {
        let config = SmokeConfig {
            steps: cli.steps,
            seed: cli.seed,
            ..Default::default()
        };

        let result = run_local(&config);

        info!(
            "smoke result: jsonl_lines={}, samples={}",
            result.jsonl_lines,
            result.samples.len()
        );

        if result.jsonl_lines < config.steps as usize {
            eprintln!("❌ SMOKE TEST FAILED: expected {} lines, got {}",
                config.steps, result.jsonl_lines);
            if cli.fail_on_error {
                std::process::exit(1);
            }
        } else {
            println!("✅ SMOKE TEST PASSED");
        }

        Ok(())
    }

    #[cfg(not(feature = "smoke"))]
    {
        anyhow::bail!("smoke-agent requires the 'smoke' feature. Build with: --features smoke");
    }
}
