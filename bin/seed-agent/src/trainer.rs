//! Trainer abstraction.
//!
//! For ADR-0081 PR-2 (issue #90 promotion-path Step C) we ship two backends:
//!
//! 1. `MockTrainer` — deterministic in-process simulator. Emits a
//!    monotonically-decreasing BPB curve seeded by the experiment's
//!    rng. Used by CI and local smoke tests so the pull loop is
//!    fully exercised without GPUs.
//!
//! 2. `ExternalTrainer` — shells out to the IGLA trainer binary
//!    (`trios-train`, ADR-0001 — src lives in `trios-trainer-igla`,
//!    we only invoke the compiled binary, never edit it).  Reads
//!    text-format `step=N val_bpb=F` lines (and optional `DONE: bpb=F`)
//!    from the subprocess stdout, matching the actual `trios-train`
//!    output protocol (see `crates/trios-igla-race/src/bin/seed_agent.rs`).
//!
//! The contract is intentionally narrow: `step()` advances one
//! training step, `eval_bpb()` returns the current BPB. The pull
//! loop owns Neon I/O, the trainer owns the math.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};

pub trait Trainer: Send {
    fn step(&mut self) -> Result<()>;
    fn eval_bpb(&self) -> f64;
    fn current_step(&self) -> i32;
    fn finished(&self) -> bool;
}

/// Deterministic simulator. BPB starts at `initial_bpb` and decays
/// asymptotically toward `target_bpb` by `step_bpb_delta` per step.
/// Seeded by the experiment's `seed` so two workers given the same
/// row produce identical curves — useful for property tests.
///
/// **Step F (promotion-path):** gated behind `#[cfg(test)]` so release
/// builds only carry `ExternalTrainer`. Mock remains available for CI
/// unit tests and local smoke runs (`cargo test`).
#[cfg(test)]
pub struct MockTrainer {
    #[allow(dead_code)] // referenced in audit logs and future trainer backends
    pub canon_name: String,
    pub seed: i32,
    pub current_step: i32,
    pub max_steps: i32,
    #[allow(dead_code)]
    pub initial_bpb: f64,
    pub target_bpb: f64,
    pub bpb: f64,
    pub decay: f64,
}

#[cfg(test)]
impl MockTrainer {
    pub fn from_config(
        canon_name: &str,
        seed: i32,
        max_steps: i32,
        config: &Value,
    ) -> Result<Self> {
        let initial_bpb = config
            .get("mock_initial_bpb")
            .and_then(Value::as_f64)
            .unwrap_or(3.5);
        let target_bpb = config
            .get("mock_target_bpb")
            .and_then(Value::as_f64)
            .unwrap_or(1.85);
        let decay = config
            .get("mock_decay")
            .and_then(Value::as_f64)
            .unwrap_or(0.0015);
        if !(0.0..1.0).contains(&decay) {
            return Err(anyhow!("mock_decay must be in [0, 1), got {decay}"));
        }
        Ok(Self {
            canon_name: canon_name.to_string(),
            seed,
            current_step: 0,
            max_steps,
            initial_bpb,
            target_bpb,
            bpb: initial_bpb,
            decay,
        })
    }
}

#[cfg(test)]
impl Trainer for MockTrainer {
    fn step(&mut self) -> Result<()> {
        if self.current_step >= self.max_steps {
            return Err(anyhow!("trainer step beyond max_steps"));
        }
        self.current_step += 1;
        // Deterministic exponential decay toward target_bpb plus a
        // tiny seed-dependent jitter so two seeds aren't identical.
        let jitter = f64::from(self.seed).sin().abs() * 0.005;
        self.bpb = self.target_bpb
            + (self.bpb - self.target_bpb) * (1.0 - self.decay)
            + jitter * self.decay;
        Ok(())
    }

    fn eval_bpb(&self) -> f64 {
        self.bpb
    }

    fn current_step(&self) -> i32 {
        self.current_step
    }

    fn finished(&self) -> bool {
        self.current_step >= self.max_steps
    }
}

/// External trainer that shells out to the IGLA `trios-train` binary.
///
/// The subprocess is launched on the first `step()` call. Subsequent
/// `step()` calls block on the next JSONL row from the trainer's
/// stdout: `{ "step": <i32>, "bpb": <f64>, "done": <bool> }`. Any
/// non-JSON line is logged and skipped (R5: surface honestly, do not
/// silently swallow trainer panics). When the trainer exits, the
/// next `step()` marks the trainer `finished` and returns Ok(()) so
/// the pull loop can finalize cleanly.
///
/// Binary path resolution order:
///   1. `TRAINER_BIN` env var,
///   2. fallback `/usr/local/bin/trios-train` (Dockerfile default).
pub struct ExternalTrainer {
    canon_name: String,
    seed: i32,
    max_steps: i32,
    current_step: i32,
    bpb: f64,
    finished: bool,
    trainer_path: PathBuf,
    config: Value,
    child: Option<Child>,
    reader: Option<BufReader<ChildStdout>>,
}

#[derive(Debug)]
struct TrainerStepOutput {
    step: i32,
    bpb: f64,
    done: bool,
}

impl ExternalTrainer {
    pub fn new(canon_name: &str, seed: i32, max_steps: i32, config: &Value) -> Result<Self> {
        let trainer_path: PathBuf = std::env::var("TRAINER_BIN")
            .unwrap_or_else(|_| "/usr/local/bin/trios-train".to_string())
            .into();
        Self::with_trainer_path(canon_name, seed, max_steps, config, trainer_path)
    }

    /// Construct with an explicit binary path — used by tests so they do
    /// not race on the `TRAINER_BIN` env var when run in parallel.
    pub fn with_trainer_path(
        canon_name: &str,
        seed: i32,
        max_steps: i32,
        config: &Value,
        trainer_path: PathBuf,
    ) -> Result<Self> {
        if !trainer_path.exists() {
            return Err(anyhow!(
                "trainer binary not found at {} (set TRAINER_BIN to override)",
                trainer_path.display()
            ));
        }
        Ok(Self {
            canon_name: canon_name.to_string(),
            seed,
            max_steps,
            current_step: 0,
            bpb: f64::NAN, // honest: no BPB before first step
            finished: false,
            trainer_path,
            config: config.clone(),
            child: None,
            reader: None,
        })
    }

    /// Spawn the trainer subprocess.
    ///
    /// Invokes `trios-train` with the same CLI protocol as the old
    /// `seed_agent.rs`: `--seed N --steps N --hidden N --lr F`.
    /// The binary writes `step=N val_bpb=F` lines to stdout.
    fn spawn(&mut self) -> Result<()> {
        // Extract hidden/lr from config JSON (ExperimentConfig fields).
        // Use sensible defaults if fields are absent (e.g. in tests).
        // hidden is always a small integer (512, 1024, etc.) so truncation is impossible.
        #[allow(clippy::cast_possible_truncation)]
        let hidden = self.config["hidden"].as_u64().unwrap_or(512) as usize;
        let lr = self.config["lr"].as_f64().unwrap_or(0.0004);
        let workdir: PathBuf = std::env::var("TRAINER_WORKDIR")
            .unwrap_or_else(|_| "/work".to_string())
            .into();

        let mut cmd = Command::new(&self.trainer_path);
        cmd.arg("--seed")
            .arg(self.seed.to_string())
            .arg("--steps")
            .arg(self.max_steps.to_string())
            .arg("--hidden")
            .arg(hidden.to_string())
            .arg("--lr")
            .arg(format!("{lr:.6}"));
        // Pass --ctx if present in config (default 12).
        // Bisect showed trios-train REQUIRES --ctx to produce output.
        // Experiment #800 (realistic BPB=1.82) had ctx=12 and worked.
        // Without --ctx the trainer subprocess hangs silently.
        if let Some(ctx) = self.config["ctx"].as_u64() {
            cmd.arg("--ctx").arg(ctx.to_string());
        }
        // NOTE(bisect): --format and --attn-layers still REMOVED.
        // These caused BPB≈0 collapse for 8K steps in MEGAASHA experiments.
        // Only --ctx restored as it's required for trainer to function.
        cmd.stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // R5: stream stderr to seed-agent logs
        // Set working directory only when it exists (present in Docker, absent on macOS dev).
        if workdir.is_dir() {
            cmd.current_dir(&workdir);
        }

        tracing::info!(
            canon = %self.canon_name,
            seed = self.seed,
            steps = self.max_steps,
            hidden,
            lr = format!("{lr:.6}"),
            workdir = %workdir.display(),
            trainer = %self.trainer_path.display(),
            "spawning trainer subprocess"
        );

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("failed to spawn {}: {e}", self.trainer_path.display()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("trainer subprocess stdout pipe missing"))?;
        self.reader = Some(BufReader::new(stdout));
        self.child = Some(child);
        Ok(())
    }

    fn read_next(&mut self) -> Result<Option<TrainerStepOutput>> {
        let Some(reader) = self.reader.as_mut() else {
            return Ok(None);
        };
        loop {
            let mut line = String::new();
            let n = reader
                .read_line(&mut line)
                .map_err(|e| anyhow!("read trainer stdout: {e}"))?;
            if n == 0 {
                return Ok(None); // EOF
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Try parsing "step=N val_bpb=F" (trios-train step output).
            if let Some(out) = parse_step_output(trimmed) {
                return Ok(Some(out));
            }
            // Try parsing "DONE: bpb=F" (trios-train final line).
            if let Some(bpb) = parse_done_output(trimmed) {
                return Ok(Some(TrainerStepOutput {
                    step: self.current_step, // keep last known step
                    bpb,
                    done: true,
                }));
            }
            // Unrecognized line — log and skip.
            tracing::warn!(
                canon = %self.canon_name,
                seed = self.seed,
                line = %trimmed,
                "trainer output line not recognized as step or done"
            );
        }
    }
}

/// Parse `step=N val_bpb=F` line from `trios-train` stdout.
/// Mirrors `parse_step_line` in `crates/trios-igla-race/src/bin/seed_agent.rs`.
fn parse_step_output(line: &str) -> Option<TrainerStepOutput> {
    let step_marker = "step=";
    let step_pos = line.find(step_marker)?;
    let after_step = &line[step_pos + step_marker.len()..];
    let step_end = after_step.find(char::is_whitespace)?;
    let step: i32 = after_step[..step_end].trim().parse().ok()?;

    let bpb_marker = "val_bpb=";
    let bpb_pos = line.find(bpb_marker)?;
    let after_bpb = &line[bpb_pos + bpb_marker.len()..];
    let bpb_end = after_bpb
        .find(char::is_whitespace)
        .unwrap_or(after_bpb.len());
    let bpb: f64 = after_bpb[..bpb_end].parse().ok()?;

    Some(TrainerStepOutput {
        step,
        bpb,
        done: false,
    })
}

/// Parse `DONE: bpb=F` line from `trios-train` stdout.
/// Mirrors `parse_done_line` in `crates/trios-igla-race/src/bin/seed_agent.rs`.
fn parse_done_output(line: &str) -> Option<f64> {
    let lower = line.to_ascii_lowercase();
    if !lower.starts_with("done:") {
        return None;
    }
    let bpb_prefix = "bpb=";
    let start = line.find(bpb_prefix)?;
    let rest = &line[start + bpb_prefix.len()..];
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

impl Trainer for ExternalTrainer {
    fn step(&mut self) -> Result<()> {
        if self.child.is_none() {
            self.spawn()?;
        }
        if let Some(out) = self.read_next()? {
            self.current_step = out.step;
            self.bpb = out.bpb;
            if out.done || self.current_step >= self.max_steps {
                self.finished = true;
            }
        } else {
            // Subprocess closed stdout. Reap to honor R9 (no zombies).
            if let Some(mut ch) = self.child.take() {
                if let Ok(s) = ch.wait() {
                    if !s.success() {
                        // Honest: trainer crashed. Mark finished, let
                        // pull loop record failure via outcome.
                        tracing::error!(
                            canon = %self.canon_name,
                            seed = self.seed,
                            exit = ?s.code(),
                            "trainer subprocess exited non-zero"
                        );
                    } else if self.current_step == 0 {
                        // Trainer exited cleanly but produced zero JSONL
                        // output — likely missing corpus or stub binary.
                        tracing::warn!(
                            canon = %self.canon_name,
                            seed = self.seed,
                            "trainer exited cleanly but produced no step output \
                             (step=0, bpb=NaN); experiment will be marked failed"
                        );
                    }
                }
            }
            self.finished = true;
        }
        Ok(())
    }

    fn eval_bpb(&self) -> f64 {
        self.bpb
    }

    fn current_step(&self) -> i32 {
        self.current_step
    }

    fn finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mock_trainer_decays_monotonically() {
        let cfg = json!({"mock_initial_bpb": 3.5, "mock_target_bpb": 1.85, "mock_decay": 0.01});
        let mut t = MockTrainer::from_config("IGLA-X", 42, 1000, &cfg).unwrap();
        let mut prev = t.eval_bpb();
        for _ in 0..500 {
            t.step().unwrap();
            let now = t.eval_bpb();
            assert!(
                now <= prev + 0.01,
                "non-monotone at step {}",
                t.current_step()
            );
            prev = now;
        }
        assert!(t.eval_bpb() < 3.5);
    }

    #[test]
    fn mock_trainer_is_deterministic_for_same_seed() {
        let cfg = json!({});
        let mut a = MockTrainer::from_config("IGLA-X", 42, 100, &cfg).unwrap();
        let mut b = MockTrainer::from_config("IGLA-X", 42, 100, &cfg).unwrap();
        for _ in 0..50 {
            a.step().unwrap();
            b.step().unwrap();
        }
        assert!((a.eval_bpb() - b.eval_bpb()).abs() < 1e-9);
    }

    #[test]
    fn mock_trainer_different_seeds_diverge() {
        let cfg = json!({});
        let mut a = MockTrainer::from_config("IGLA-X", 42, 100, &cfg).unwrap();
        let mut b = MockTrainer::from_config("IGLA-X", 43, 100, &cfg).unwrap();
        for _ in 0..50 {
            a.step().unwrap();
            b.step().unwrap();
        }
        assert!((a.eval_bpb() - b.eval_bpb()).abs() > 0.0);
    }

    #[test]
    fn mock_trainer_rejects_bad_decay() {
        let cfg = json!({"mock_decay": 1.5});
        assert!(MockTrainer::from_config("X", 42, 100, &cfg).is_err());
    }

    #[test]
    fn mock_trainer_finishes_at_max_steps() {
        let cfg = json!({});
        let mut t = MockTrainer::from_config("X", 42, 5, &cfg).unwrap();
        for _ in 0..5 {
            t.step().unwrap();
        }
        assert!(t.finished());
        assert!(t.step().is_err());
    }

    #[test]
    fn external_trainer_rejects_missing_binary() {
        let cfg = json!({});
        let result = ExternalTrainer::with_trainer_path(
            "IGLA-X",
            42,
            100,
            &cfg,
            "/definitely/not/a/real/binary/trios-train".into(),
        );
        let err = match result {
            Ok(_) => panic!("expected error for missing binary, got Ok"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("trainer binary not found"), "got: {err}");
    }

    #[cfg(unix)]
    #[test]
    fn external_trainer_constructs_when_binary_exists() {
        let cfg = json!({"hidden_dim": 384});
        let tr = ExternalTrainer::with_trainer_path("IGLA-X", 42, 100, &cfg, "/usr/bin/true".into())
            .expect("construct");
        assert_eq!(tr.current_step(), 0);
        assert!(!tr.finished());
        assert!(
            tr.eval_bpb().is_nan(),
            "BPB should be NaN before first step"
        );
    }

    #[cfg(unix)]
    #[test]
    fn external_trainer_finalizes_when_subprocess_exits_silently() {
        // /usr/bin/true exits 0 with no stdout; first step() should mark finished.
        let cfg = json!({});
        let mut tr =
            ExternalTrainer::with_trainer_path("IGLA-X", 42, 100, &cfg, "/usr/bin/true".into())
                .expect("construct");
        tr.step().expect("step ok");
        assert!(tr.finished(), "trainer should finalize on EOF");
    }

    // ----- subprocess-based tests using a shell shim. ------------------
    // Skipped on non-Unix because we rely on `/bin/sh` and chmod 0o755.
    #[cfg(unix)]
    fn write_shim(script: &str) -> std::path::PathBuf {
        use std::io::Write as _;
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let path = dir.join(format!("trios-train-shim-{pid}-{nonce}.sh"));
        {
            let mut f = std::fs::File::create(&path).expect("create shim");
            f.write_all(script.as_bytes()).expect("write shim");
        }
        let mut perms = std::fs::metadata(&path).expect("meta").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).expect("chmod shim");
        path
    }

    #[cfg(unix)]
    #[test]
    fn external_trainer_reads_text_stream_from_subprocess() {
        let shim = write_shim(
            r#"#!/bin/sh
echo 'step=1 val_bpb=3.40'
echo 'step=2 val_bpb=3.30'
echo 'step=3 val_bpb=3.20'
echo 'step=4 val_bpb=3.10'
echo 'DONE: bpb=3.10'
"#,
        );
        let cfg = json!({"hidden": 384, "lr": 0.001});
        let mut tr = ExternalTrainer::with_trainer_path("IGLA-T-INT", 42, 100, &cfg, shim.clone())
            .expect("construct");

        tr.step().expect("step 1");
        assert_eq!(tr.current_step(), 1);
        assert!((tr.eval_bpb() - 3.40).abs() < 1e-9);
        assert!(!tr.finished());

        tr.step().expect("step 2");
        tr.step().expect("step 3");
        tr.step().expect("step 4");
        assert_eq!(tr.current_step(), 4);
        assert!((tr.eval_bpb() - 3.10).abs() < 1e-9);
        assert!(!tr.finished(), "step line should not finalize");

        tr.step().expect("DONE line");
        assert!(
            (tr.eval_bpb() - 3.10).abs() < 1e-9,
            "DONE bpb should be 3.10"
        );
        assert!(tr.finished(), "DONE line should finalize");
        let _ = std::fs::remove_file(&shim);
    }

    #[cfg(unix)]
    #[test]
    fn external_trainer_handles_subprocess_crash() {
        let shim = write_shim(
            r#"#!/bin/sh
echo 'step=1 val_bpb=2.99'
exit 7
"#,
        );
        let cfg = json!({});
        let mut tr =
            ExternalTrainer::with_trainer_path("IGLA-T-CRASH", 43, 100, &cfg, shim.clone())
                .expect("construct");
        tr.step().expect("first row consumed");
        assert_eq!(tr.current_step(), 1);
        // Next step should hit EOF and finalize.
        tr.step().expect("EOF after crash");
        assert!(tr.finished());
        let _ = std::fs::remove_file(&shim);
    }

    #[cfg(unix)]
    #[test]
    fn external_trainer_skips_garbage_lines() {
        let shim = write_shim(
            r#"#!/bin/sh
echo 'not-a-step-line-at-all'
echo ''
echo 'step=1 val_bpb=2.50'
echo 'DONE: bpb=2.50'
"#,
        );
        let cfg = json!({});
        let mut tr =
            ExternalTrainer::with_trainer_path("IGLA-T-NOISE", 44, 100, &cfg, shim.clone())
                .expect("construct");
        tr.step().expect("step ok");
        assert_eq!(tr.current_step(), 1);
        assert!(!tr.finished(), "step line should not finalize");
        tr.step().expect("DONE line");
        assert!(tr.finished(), "DONE line should finalize");
        let _ = std::fs::remove_file(&shim);
    }
}
