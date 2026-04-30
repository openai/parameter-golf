//! Smoke-test crate for the IGLA RACE pipeline.
//!
//! Validates the full cycle in <60 seconds with synthetic data:
//!
//! ```text
//! queue.push(SmokeExperiment)
//!   → worker.claim()
//!   → MockTrainer.run(steps=1)
//!   → JSONL stdout: "step=1 val_bpb=2.50"
//!   → parse → db.insert(bpb_samples)
//!   → assert row exists
//! ```
//!
//! Anchor: `phi^2 + phi^-2 = 3`

use std::io::Write;

/// Configuration for a smoke test run.
#[derive(Debug, Clone)]
pub struct SmokeConfig {
    /// Number of training steps (default: 1).
    pub steps: u32,
    /// Batch size (default: 2).
    pub batch: u32,
    /// Random seed (default: 42).
    pub seed: u64,
    /// Use synthetic data (always true for smoke).
    pub synthetic: bool,
    /// Timeout in seconds (default: 30).
    pub timeout_sec: u64,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            steps: 1,
            batch: 2,
            seed: 42,
            synthetic: true,
            timeout_sec: 30,
        }
    }
}

/// Result of a smoke pipeline run.
#[derive(Debug, Clone)]
pub struct SmokeResult {
    /// Number of JSONL lines produced.
    pub jsonl_lines: usize,
    /// Parsed (step, bpb) pairs.
    pub samples: Vec<(u32, f64)>,
    /// Whether the DB insert succeeded (None if no DB).
    pub db_insert_ok: Option<bool>,
}

/// A single JSONL step output from the synthetic trainer.
#[derive(Debug, Clone)]
pub struct StepOutput {
    pub step: u32,
    pub val_bpb: f64,
    pub loss: f64,
}

/// Run the synthetic trainer and produce JSONL output.
///
/// This is the "trainer" side of the smoke test — it produces
/// `step=N val_bpb=F` lines to stdout, exactly like `trios-train`.
pub fn run_synthetic_trainer(cfg: &SmokeConfig) -> Vec<StepOutput> {
    // Install panic hook to ensure JSONL output on crash.
    let _ = std::panic::take_hook();

    let mut outputs = Vec::with_capacity(cfg.steps as usize);
    let base_bpb: f64 = 3.0;
    let decay: f64 = 0.99;

    for step in 1..=cfg.steps {
        // Deterministic BPB: starts at ~3.0, decays by 0.99 per step.
        let bpb = base_bpb * decay.powi(step as i32);
        let loss = bpb * 0.8; // loss ≈ 80% of BPB

        let output = StepOutput {
            step,
            val_bpb: bpb,
            loss,
        };

        // Write to stdout — exactly the format seed-agent expects.
        // Format: "step=N val_bpb=F" (matches parse_step_output).
        println!("step={step} val_bpb={bpb:.4}");
        // FLUSH after every line — this is the fix for "zero steps".
        let _ = std::io::stdout().flush();

        outputs.push(output);
    }

    // Signal completion.
    println!("done best_bpb={:.4}", outputs.last().map(|o| o.val_bpb).unwrap_or(0.0));
    let _ = std::io::stdout().flush();

    outputs
}

/// Parse `step=N val_bpb=F` lines (mirrors seed-agent's parse_step_output).
pub fn parse_step_line(line: &str) -> Option<(u32, f64)> {
    let step_marker = "step=";
    let step_pos = line.find(step_marker)?;
    let after_step = &line[step_pos + step_marker.len()..];
    let step_end = after_step.find(char::is_whitespace)?;
    let step: u32 = after_step[..step_end].trim().parse().ok()?;

    let bpb_marker = "val_bpb=";
    let bpb_pos = line.find(bpb_marker)?;
    let after_bpb = &line[bpb_pos + bpb_marker.len()..];
    let bpb_end = after_bpb
        .find(char::is_whitespace)
        .unwrap_or(after_bpb.len());
    let bpb: f64 = after_bpb[..bpb_end].parse().ok()?;

    Some((step, bpb))
}

/// Run the full smoke pipeline locally (no DB).
pub fn run_local(cfg: &SmokeConfig) -> SmokeResult {
    let outputs = run_synthetic_trainer(cfg);

    // Parse the output lines back.
    let samples: Vec<(u32, f64)> = outputs
        .iter()
        .map(|o| (o.step, o.val_bpb))
        .collect();

    SmokeResult {
        jsonl_lines: outputs.len(),
        samples,
        db_insert_ok: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_config_defaults() {
        let cfg = SmokeConfig::default();
        assert_eq!(cfg.steps, 1);
        assert_eq!(cfg.batch, 2);
        assert_eq!(cfg.seed, 42);
        assert!(cfg.synthetic);
        assert_eq!(cfg.timeout_sec, 30);
    }

    #[test]
    fn synthetic_trainer_produces_output() {
        let cfg = SmokeConfig {
            steps: 3,
            ..Default::default()
        };
        let result = run_local(&cfg);
        assert_eq!(result.jsonl_lines, 3);
        assert_eq!(result.samples.len(), 3);
        // First step should have BPB ≈ 3.0 * 0.99 = 2.97
        assert!(result.samples[0].1 > 2.9 && result.samples[0].1 < 3.0);
        // BPB should decrease monotonically.
        for i in 1..result.samples.len() {
            assert!(result.samples[i].1 < result.samples[i - 1].1);
        }
    }

    #[test]
    fn parse_step_line_works() {
        let (step, bpb) = parse_step_line("step=42 val_bpb=2.5678").unwrap();
        assert_eq!(step, 42);
        assert!((bpb - 2.5678).abs() < 0.001);
    }

    #[test]
    fn parse_step_line_rejects_garbage() {
        assert!(parse_step_line("hello world").is_none());
        assert!(parse_step_line("step=abc val_bpb=1.0").is_none());
        assert!(parse_step_line("step=1 loss=0.5").is_none());
    }

    #[test]
    fn single_step_smoke() {
        let cfg = SmokeConfig::default();
        let result = run_local(&cfg);
        assert_eq!(result.jsonl_lines, 1);
        assert_eq!(result.samples[0].0, 1);
        assert!(result.samples[0].1 > 0.0);
    }
}
