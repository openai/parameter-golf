//! Example: Run an in-memory smoke test cycle.
//!
//! This demonstrates the full smoke test pipeline without requiring:
//! - Neon database
//! - Railway connection
//! - GPU
//! - Real trainer binary
//!
//! Usage:
//!   cargo run --example smoke_in_memory

use trios_railway_smoke::{run_local, SmokeConfig};

fn main() -> anyhow::Result<()> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                🚀 SMOKE TEST CYCLE 🚀");
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let cfg = SmokeConfig {
        steps: 3,
        ..Default::default()
    };

    println!("[Config] steps={}, batch={}, seed={}", cfg.steps, cfg.batch, cfg.seed);
    println!();

    println!("[Trainer] Running synthetic training...");
    let result = run_local(&cfg);
    println!();

    println!("[Result] jsonl_lines={}, samples={}", result.jsonl_lines, result.samples.len());
    println!();

    if result.jsonl_lines >= cfg.steps as usize {
        println!("═══════════════════════════════════════════════════════════════");
        println!("                    ✅ SMOKE PASSED");
        println!("═══════════════════════════════════════════════════════════════");
        Ok(())
    } else {
        println!("═══════════════════════════════════════════════════════════════");
        println!("                    ❌ SMOKE FAILED");
        println!("═══════════════════════════════════════════════════════════════");
        anyhow::bail!("expected {} JSONL lines, got {}", cfg.steps, result.jsonl_lines);
    }
}
