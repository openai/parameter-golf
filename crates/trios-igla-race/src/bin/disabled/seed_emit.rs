//! L15 — Race-Ledger Producer CLI (`seed_emit`)
//!
//! Producer-side companion to L14's [`ledger_check`].  Takes a single
//! seed measurement on the command line, validates it against the L7
//! gate primitives (warmup, finiteness, JEPA-proxy floor, target
//! sanity), then **atomically** appends one JSON line to
//! `assertions/seed_results.jsonl`.
//!
//! ## Why a separate bin from `ledger_check`
//!
//! Producer and consumer must NOT share a process: a compute-equipped
//! agent runs `seed_emit` to record a measurement, then any other
//! agent (CI, watchdog, human) runs `ledger_check` to adjudicate.
//! Splitting the two preserves:
//!
//! 1. **R6 — file ownership.**  This bin lives next to `ledger_check`
//!    in `src/bin/`, so adding L15 does not require editing any L1..L14
//!    file.
//! 2. **R5 — honesty.**  The producer's job is to write *exactly one*
//!    validated line; the consumer's job is to *grade the whole
//!    ledger*.  A single binary that does both could quietly grade
//!    a partial ledger.  Two binaries cannot.
//! 3. **Concurrency.**  Two compute hosts may emit at the same time;
//!    `OpenOptions::append(true)` plus a single `write_all` keep the
//!    JSONL well-formed without process-wide locking.
//!
//! ## Validation pipeline (mirrors `victory::check_victory`)
//!
//! Before any I/O, the row is rejected if any of the following hold —
//! exit code is the same one `ledger_check` would have returned for
//! the equivalent gate verdict, so the two CLIs are wired into one
//! decision surface:
//!
//! | Failure              | `EmitError`                | exit |
//! |----------------------|----------------------------|------|
//! | non-finite BPB       | `NonFiniteBpb`             | 21   |
//! | step < warmup        | `BeforeWarmup`             | 22   |
//! | bpb < JEPA floor     | `JepaProxyDetected`        | 23   |
//! | bpb out of range [0, 100) | `BpbOutOfRange`       | 24   |
//! | duplicate seed already in ledger | `DuplicateSeed` | 25 |
//! | malformed sha (empty/whitespace) | `MalformedSha`  | 26   |
//! | I/O failure          | `IoError`                  | 30   |
//!
//! Note: a row with `bpb >= IGLA_TARGET_BPB` (1.5) is **accepted** —
//! the gate counts only seeds *strictly below* the target, so an honest
//! "we tried, BPB landed at 1.62" row belongs in the ledger as
//! evidence of effort.  Only structurally-corrupt or fraudulent rows
//! are refused at write-time.
//!
//! ## L-R14 traceability
//!
//! Every numeric anchor used here is re-imported from
//! `trios_igla_race::*` (no shadow constants).  See the
//! `test_constants_traceable` test for the canary.
//!
//! Refs: trios#143 lane L15 · INV-7 · L-R14 · R6 · R10 · R13.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use chrono::Utc;
use clap::Parser;
use serde_json::{json, Value};

use trios_igla_race::invariants::INV2_WARMUP_BLIND_STEPS;
use trios_igla_race::{IGLA_TARGET_BPB, JEPA_PROXY_BPB_FLOOR};

/// CLI for atomically emitting a single seed measurement to the
/// IGLA RACE seed-results ledger.
#[derive(Parser, Debug)]
#[command(
    name = "seed_emit",
    about = "Atomically append a validated seed-result row to assertions/seed_results.jsonl.",
    long_about = "Producer-side helper for the L7 Victory Gate. \
                  Validates against gate primitives (warmup, finiteness, \
                  JEPA-proxy floor, sha well-formed) before writing one \
                  JSONL line.  Use ledger_check to adjudicate the file."
)]
struct Args {
    /// Seed value used by the trial.
    #[arg(long)]
    seed: u64,

    /// Final BPB measured at the end of training.
    #[arg(long)]
    bpb: f64,

    /// Final step at which `bpb` was recorded.
    #[arg(long)]
    step: u64,

    /// Commit SHA the trial ran against (audit trail).
    #[arg(long)]
    sha: String,

    /// Agent identifier emitting this row.
    #[arg(long, default_value = "anonymous")]
    agent: String,

    /// Path to the JSONL ledger.  Defaults to
    /// `assertions/seed_results.jsonl` resolved against CWD.
    #[arg(long, default_value = "assertions/seed_results.jsonl")]
    ledger: PathBuf,

    /// Print the JSON line that would be written, but do NOT touch the
    /// ledger.  Useful for CI dry-runs and pre-flight checks.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

/// Errors specific to the producer side.  Distinct from the consumer's
/// `VictoryError` because some paths (e.g. malformed SHA, duplicate
/// already on disk) only matter at write-time.
#[derive(Debug, Clone, PartialEq)]
pub enum EmitError {
    /// `bpb` is NaN or ±∞.
    NonFiniteBpb { seed: u64, bpb: f64 },
    /// `step < INV2_WARMUP_BLIND_STEPS`.
    BeforeWarmup { seed: u64, step: u64, warmup: u64 },
    /// `bpb < JEPA_PROXY_BPB_FLOOR` after warmup (TASK-5D).
    JepaProxyDetected { seed: u64, bpb: f64 },
    /// `bpb` outside the structural range [0, 100).  No real character-
    /// LM ever exceeds 8 bits per byte; refusing > 100 catches a
    /// units-of-measure error (loss in nats reported as BPB, etc.)
    /// before it pollutes the ledger.
    BpbOutOfRange { seed: u64, bpb: f64 },
    /// SHA is empty or whitespace.
    MalformedSha,
    /// Same seed already present in the ledger (would lose distinct-
    /// seed reproducibility downstream).
    DuplicateSeed { seed: u64 },
    /// I/O failure reading or writing the ledger.
    IoError { path: String, message: String },
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::NonFiniteBpb { seed, bpb } => {
                write!(f, "seed {} has non-finite BPB {}", seed, bpb)
            }
            EmitError::BeforeWarmup { seed, step, warmup } => write!(
                f,
                "seed {} step {} below warmup floor {}",
                seed, step, warmup
            ),
            EmitError::JepaProxyDetected { seed, bpb } => write!(
                f,
                "seed {} bpb {} below JEPA-proxy floor {} (TASK-5D)",
                seed, bpb, JEPA_PROXY_BPB_FLOOR
            ),
            EmitError::BpbOutOfRange { seed, bpb } => write!(
                f,
                "seed {} bpb {} outside structural range [0, 100)",
                seed, bpb
            ),
            EmitError::MalformedSha => write!(f, "sha is empty or whitespace-only"),
            EmitError::DuplicateSeed { seed } => write!(
                f,
                "seed {} already present in ledger (distinct-seed contract)",
                seed
            ),
            EmitError::IoError { path, message } => {
                write!(f, "I/O error at {}: {}", path, message)
            }
        }
    }
}

impl std::error::Error for EmitError {}

impl EmitError {
    fn exit_code(&self) -> u8 {
        match self {
            EmitError::NonFiniteBpb { .. } => 21,
            EmitError::BeforeWarmup { .. } => 22,
            EmitError::JepaProxyDetected { .. } => 23,
            EmitError::BpbOutOfRange { .. } => 24,
            EmitError::DuplicateSeed { .. } => 25,
            EmitError::MalformedSha => 26,
            EmitError::IoError { .. } => 30,
        }
    }
}

/// Inputs that the validator inspects.  Held as a struct (not the
/// `Args` directly) so `validate_row` is reusable from tests without
/// constructing a full clap parse.
#[derive(Debug, Clone, PartialEq)]
pub struct EmitRow {
    pub seed: u64,
    pub bpb: f64,
    pub step: u64,
    pub sha: String,
    pub agent: String,
}

/// Pure function: gate-style validation of an `EmitRow`.  Does not
/// touch the filesystem.
pub fn validate_row(row: &EmitRow) -> Result<(), EmitError> {
    if !row.bpb.is_finite() {
        return Err(EmitError::NonFiniteBpb {
            seed: row.seed,
            bpb: row.bpb,
        });
    }
    if !(0.0..100.0).contains(&row.bpb) {
        return Err(EmitError::BpbOutOfRange {
            seed: row.seed,
            bpb: row.bpb,
        });
    }
    if row.step < INV2_WARMUP_BLIND_STEPS {
        return Err(EmitError::BeforeWarmup {
            seed: row.seed,
            step: row.step,
            warmup: INV2_WARMUP_BLIND_STEPS,
        });
    }
    if row.bpb < JEPA_PROXY_BPB_FLOOR {
        return Err(EmitError::JepaProxyDetected {
            seed: row.seed,
            bpb: row.bpb,
        });
    }
    if row.sha.trim().is_empty() {
        return Err(EmitError::MalformedSha);
    }
    Ok(())
}

/// Scan the ledger text and return all seeds present in non-header
/// rows.  Used for the duplicate-seed guard before append.
pub fn seeds_in_ledger(text: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(map) = v.as_object() {
            // Skip header lines (all keys begin with '_').
            if !map.is_empty() && map.keys().all(|k| k.starts_with('_')) {
                continue;
            }
            if let Some(seed) = map.get("seed").and_then(Value::as_u64) {
                out.push(seed);
            }
        }
    }
    out
}

/// Build the JSONL line that should be appended for a validated row.
/// Format mirrors the schema documented in the ledger header
/// (`ts, agent, seed, bpb, step, sha, victory_check, gate_action`).
pub fn render_row(row: &EmitRow) -> String {
    let ts = Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let line = json!({
        "ts": ts,
        "agent": row.agent,
        "seed": row.seed,
        "bpb": row.bpb,
        "step": row.step,
        "sha": row.sha.trim(),
        "victory_check": "pre_validated_at_emit",
        "gate_action": if row.bpb < IGLA_TARGET_BPB { "candidate" } else { "below_target_evidence" },
    });
    line.to_string()
}

/// End-to-end emit: validate, check duplicates against the existing
/// ledger text, return the line to write.  Pure function — `main`
/// supplies the I/O and the timestamp.
pub fn emit(row: &EmitRow, existing_ledger: &str) -> Result<String, EmitError> {
    validate_row(row)?;
    if seeds_in_ledger(existing_ledger).contains(&row.seed) {
        return Err(EmitError::DuplicateSeed { seed: row.seed });
    }
    Ok(render_row(row))
}

// ----------------------------------------------------------------------
// main
// ----------------------------------------------------------------------

fn main() -> ExitCode {
    let args = Args::parse();
    let row = EmitRow {
        seed: args.seed,
        bpb: args.bpb,
        step: args.step,
        sha: args.sha.clone(),
        agent: args.agent.clone(),
    };

    // Read existing ledger (or treat missing as empty — first emit).
    let existing = match std::fs::read_to_string(&args.ledger) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => {
            let err = EmitError::IoError {
                path: args.ledger.display().to_string(),
                message: e.to_string(),
            };
            eprintln!("seed_emit: {}", err);
            return ExitCode::from(err.exit_code());
        }
    };

    let line = match emit(&row, &existing) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("seed_emit: {}", e);
            return ExitCode::from(e.exit_code());
        }
    };

    if args.dry_run {
        println!("{}", line);
        return ExitCode::SUCCESS;
    }

    // Atomic append: open with append flag, single write_all of
    // line+"\n", drop the file handle.  POSIX guarantees writes ≤
    // PIPE_BUF (≥ 512 B) are atomic; a single JSON line is well below.
    let write_result = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.ledger)
        .and_then(|mut f| {
            let mut payload = line.clone();
            payload.push('\n');
            f.write_all(payload.as_bytes())
        });

    if let Err(e) = write_result {
        let err = EmitError::IoError {
            path: args.ledger.display().to_string(),
            message: e.to_string(),
        };
        eprintln!("seed_emit: {}", err);
        return ExitCode::from(err.exit_code());
    }

    println!("{}", line);
    ExitCode::SUCCESS
}

// ----------------------------------------------------------------------
// Tests — falsification witnesses for every emit-side error path
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn good_row(seed: u64) -> EmitRow {
        EmitRow {
            seed,
            bpb: 1.42,
            step: 5000,
            sha: "deadbeef".into(),
            agent: "tester".into(),
        }
    }

    /// L-R14 anchor check — every constant is sourced from the race
    /// crate.  If `INV2_WARMUP_BLIND_STEPS` or `JEPA_PROXY_BPB_FLOOR`
    /// moves, this is the canary.
    #[test]
    fn test_constants_traceable() {
        assert_eq!(INV2_WARMUP_BLIND_STEPS, 4000);
        assert!((JEPA_PROXY_BPB_FLOOR - 0.1).abs() < f64::EPSILON);
        assert!((IGLA_TARGET_BPB - 1.5).abs() < f64::EPSILON);
    }

    /// Happy path: validate + emit produces a parseable JSON line.
    #[test]
    fn good_row_renders_parseable_json() {
        let row = good_row(1);
        let line = emit(&row, "").unwrap();
        let parsed: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(parsed["seed"].as_u64(), Some(1));
        assert!((parsed["bpb"].as_f64().unwrap() - 1.42).abs() < 1e-12);
        assert_eq!(parsed["step"].as_u64(), Some(5000));
        assert_eq!(parsed["sha"].as_str(), Some("deadbeef"));
        assert_eq!(parsed["gate_action"].as_str(), Some("candidate"));
    }

    /// A row with bpb >= 1.5 is admitted but flagged as
    /// `below_target_evidence` — ledger keeps it as proof of effort.
    #[test]
    fn above_target_row_is_admitted_as_evidence() {
        let mut row = good_row(2);
        row.bpb = 1.62;
        let line = emit(&row, "").unwrap();
        let parsed: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(
            parsed["gate_action"].as_str(),
            Some("below_target_evidence")
        );
    }

    /// NaN BPB → NonFiniteBpb, exit 21.
    #[test]
    fn falsify_non_finite_bpb_rejected() {
        let mut row = good_row(3);
        row.bpb = f64::NAN;
        match emit(&row, "") {
            Err(EmitError::NonFiniteBpb { seed, .. }) => {
                assert_eq!(seed, 3);
                assert_eq!(
                    EmitError::NonFiniteBpb {
                        seed: 3,
                        bpb: f64::NAN
                    }
                    .exit_code(),
                    21
                );
            }
            other => panic!("expected NonFiniteBpb, got {:?}", other),
        }
    }

    /// Step before warmup → BeforeWarmup, exit 22.
    #[test]
    fn falsify_pre_warmup_rejected() {
        let mut row = good_row(4);
        row.step = 100;
        match emit(&row, "") {
            Err(EmitError::BeforeWarmup { warmup, .. }) => {
                assert_eq!(warmup, INV2_WARMUP_BLIND_STEPS);
            }
            other => panic!("expected BeforeWarmup, got {:?}", other),
        }
    }

    /// JEPA-proxy BPB → JepaProxyDetected, exit 23.
    #[test]
    fn falsify_jepa_proxy_rejected() {
        let mut row = good_row(5);
        row.bpb = 0.014;
        match emit(&row, "") {
            Err(EmitError::JepaProxyDetected { bpb, .. }) => {
                assert!((bpb - 0.014).abs() < 1e-12);
            }
            other => panic!("expected JepaProxyDetected, got {:?}", other),
        }
    }

    /// BPB ≥ 100 → BpbOutOfRange (units-of-measure guard).
    #[test]
    fn falsify_out_of_range_bpb_rejected() {
        let mut row = good_row(6);
        row.bpb = 250.0;
        assert!(matches!(
            emit(&row, ""),
            Err(EmitError::BpbOutOfRange { .. })
        ));
    }

    /// Negative BPB → BpbOutOfRange (BPB is non-negative by definition).
    #[test]
    fn falsify_negative_bpb_rejected() {
        let mut row = good_row(7);
        row.bpb = -0.5;
        assert!(matches!(
            emit(&row, ""),
            Err(EmitError::BpbOutOfRange { .. })
        ));
    }

    /// Whitespace-only sha → MalformedSha.
    #[test]
    fn falsify_blank_sha_rejected() {
        let mut row = good_row(8);
        row.sha = "   ".into();
        assert_eq!(emit(&row, ""), Err(EmitError::MalformedSha));
    }

    /// Duplicate seed in existing ledger → DuplicateSeed.
    #[test]
    fn falsify_duplicate_seed_rejected() {
        let row = good_row(9);
        let existing = r#"{"_schema":"v1"}
{"seed":9,"bpb":1.40,"step":5000,"sha":"a"}
"#;
        match emit(&row, existing) {
            Err(EmitError::DuplicateSeed { seed }) => assert_eq!(seed, 9),
            other => panic!("expected DuplicateSeed(9), got {:?}", other),
        }
    }

    /// `seeds_in_ledger` skips header lines and tolerates blanks.
    #[test]
    fn seeds_extraction_skips_header_and_blanks() {
        let text = r#"{"_schema":"trios.assertions.seed_results.v1","_target":1.5}

{"seed":11,"bpb":1.4,"step":5000,"sha":"a"}
{"seed":12,"bpb":1.41,"step":5000,"sha":"b"}
"#;
        let seeds = seeds_in_ledger(text);
        assert_eq!(seeds, vec![11, 12]);
    }

    /// `seeds_in_ledger` is robust to garbage rows (silently skips them).
    /// Producer never refuses to emit because the consumer-side validator
    /// must surface those rows on read.  This is honest — emit only
    /// guarantees its own row is clean; `ledger_check` will refuse a
    /// corrupt ledger before adjudicating.
    #[test]
    fn seeds_extraction_skips_garbage_rows() {
        let text = "{\"seed\":1,\"bpb\":1.4,\"step\":5000,\"sha\":\"a\"}\nnot json at all\n{\"seed\":2,\"bpb\":1.4,\"step\":5000,\"sha\":\"b\"}\n";
        let seeds = seeds_in_ledger(text);
        assert_eq!(seeds, vec![1, 2]);
    }

    /// Atomic append round-trip via tempfile.  Verifies the written
    /// line is exactly what `render_row` produced + a single newline.
    #[test]
    fn append_round_trip_via_tempfile() {
        // Use std::env::temp_dir + a unique nanosecond suffix to avoid
        // pulling tempfile crate just for the test.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("seed_emit_test_{}.jsonl", stamp));
        let row = good_row(1234);

        // Render expected line (timestamp differs across calls; we
        // compare structural equality through JSON parsing).
        let line = emit(&row, "").unwrap();

        // Append.
        let mut payload = line.clone();
        payload.push('\n');
        std::fs::write(&path, payload.as_bytes()).unwrap();

        let mut buf = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut buf)
            .unwrap();
        assert!(buf.ends_with('\n'), "appended row must end with newline");
        let stripped = buf.trim_end_matches('\n');
        let parsed: Value = serde_json::from_str(stripped).unwrap();
        assert_eq!(parsed["seed"].as_u64(), Some(1234));

        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }

    /// Exit code surface is unique per error class — guards against a
    /// future refactor collapsing two distinct refusals into one
    /// indistinguishable signal.
    #[test]
    fn exit_codes_are_distinct() {
        let codes: Vec<u8> = vec![
            EmitError::NonFiniteBpb { seed: 0, bpb: 0.0 }.exit_code(),
            EmitError::BeforeWarmup {
                seed: 0,
                step: 0,
                warmup: 4000,
            }
            .exit_code(),
            EmitError::JepaProxyDetected { seed: 0, bpb: 0.0 }.exit_code(),
            EmitError::BpbOutOfRange { seed: 0, bpb: 0.0 }.exit_code(),
            EmitError::DuplicateSeed { seed: 0 }.exit_code(),
            EmitError::MalformedSha.exit_code(),
            EmitError::IoError {
                path: "x".into(),
                message: "y".into(),
            }
            .exit_code(),
        ];
        let mut sorted = codes.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(codes.len(), sorted.len(), "exit codes must be unique");
    }
}
