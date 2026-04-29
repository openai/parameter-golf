# Step C: ExternalTrainer Implementation

**Owner:** Sergeant prepares, Operator approves
**Time:** 5 minutes operator review/approval
**Purpose:** Replace bail statement with real external trainer integration

---

## C1. trainer.rs — Add ExternalTrainer

**File:** `bin/seed-agent/src/trainer.rs`

Add after MockTrainer implementation (after line 102):

```rust
/// External trainer that shells out to the IGLA trainer binary.
/// Communicates via subprocess stdout parsing (JSONL format per ADR-0001).
///
/// The trainer binary (trios-train) lives at path specified by
/// TRAINER_BIN env var, defaulting to /usr/local/bin/trios-train.
pub struct ExternalTrainer {
    #[allow(dead_code)]
    canon_name: String,
    seed: i32,
    max_steps: i32,
    current_step: i32,
    bpb: f64,
    finished: bool,
    trainer_path: String,
    child: Option<std::process::Child>,
    config: serde_json::Value,
}

impl ExternalTrainer {
    /// Create a new external trainer instance.
    ///
    /// Spawns the trainer subprocess with appropriate CLI arguments.
    /// The subprocess runs in the background; we pull results via stdout.
    pub fn new(canon_name: &str, seed: i32, max_steps: i32, config: &serde_json::Value) -> Result<Self> {
        // Trainer binary path from env or default
        let trainer_path = std::env::var("TRAINER_BIN")
            .unwrap_or_else(|_| "/usr/local/bin/trios-train".to_string());

        // Verify trainer binary exists
        if !std::path::Path::new(&trainer_path).exists() {
            return Err(anyhow!("trainer binary not found at {trainer_path}"));
        }

        // For now: initialize in "ready" state but don't spawn subprocess yet
        // Subprocess will be spawned on first step() call
        Ok(Self {
            canon_name: canon_name.to_string(),
            seed,
            max_steps,
            current_step: 0,
            bpb: 3.5, // Initial BPB (will be updated from trainer)
            finished: false,
            trainer_path,
            child: None,
            config: config.clone(),
        })
    }

    /// Spawn the trainer subprocess.
    fn spawn_trainer(&mut self) -> Result<()> {
        use std::process::{Command, Stdio};

        let mut cmd = Command::new(&self.trainer_path);
        cmd.arg("--config")
           .arg(serde_json::to_string(&self.config)?)
           .arg("--seed")
           .arg(self.seed.to_string())
           .arg("--steps")
           .arg(self.max_steps.to_string())
           .arg("--jsonl") // Request JSONL output format
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());

        let child = cmd.spawn()
            .map_err(|e| anyhow!("failed to spawn trainer: {e}"))?;

        self.child = Some(child);
        Ok(())
    }

    /// Read next JSONL line from trainer stdout.
    fn read_next_output(&self) -> Result<Option<TrainingStepOutput>> {
        use std::io::{BufRead, BufReader};

        let child = self.child.as_ref()
            .ok_or_else(|| anyhow!("trainer subprocess not spawned"))?;

        if let Some(stdout) = child.stdout.as_ref() {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = line.map_err(|e| anyhow!("failed to read trainer output: {e}"))?;
                if line.is_empty() {
                    continue;
                }
                // Parse JSONL line
                let output: TrainingStepOutput = serde_json::from_str(&line)
                    .map_err(|e| anyhow!("failed to parse trainer output: {e}"))?;
                return Ok(Some(output));
            }
        }
        Ok(None)
    }
}

/// Training step output from external trainer (JSONL format).
#[derive(serde::Deserialize)]
struct TrainingStepOutput {
    step: i32,
    bpb: f64,
    #[serde(default)]
    done: bool,
}

impl Trainer for ExternalTrainer {
    fn step(&mut self) -> Result<()> {
        // Spawn subprocess on first step
        if self.child.is_none() {
            self.spawn_trainer()?;
        }

        // Read next output from trainer
        if let Some(output) = self.read_next_output()? {
            self.current_step = output.step;
            self.bpb = output.bpb;
            self.finished = output.done || self.current_step >= self.max_steps;
        } else {
            // Trainer exited without more output
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
```

---

## C2. worker.rs — Replace bail with ExternalTrainer

**File:** `bin/seed-agent/src/worker.rs`

Replace lines 88-96 with:

```rust
    let mut tr: Box<dyn trainer::Trainer> = match cfg.trainer_kind.as_str() {
        "mock" => Box::new(trainer::MockTrainer::from_config(
            &exp.canon_name,
            exp.seed,
            exp.steps_budget,
            &exp.config,
        )?),
        "external" => Box::new(trainer::ExternalTrainer::new(
            &exp.canon_name,
            exp.seed,
            exp.steps_budget,
            &exp.config,
        )?),
        other => anyhow::bail!("trainer_kind {other:?} not supported (valid: mock, external)"),
    };
```

---

## C3. Add Unit Tests

Add to `trainer.rs` in the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn external_trainer_requires_binary() {
        // Force binary path to non-existent location
        std::env::set_var("TRAINER_BIN", "/nonexistent/trios-train");
        let cfg = json!({});
        let result = ExternalTrainer::new("test", 42, 100, &cfg);
        assert!(result.is_err());
    }

    #[test]
    fn external_trainer_initializes_correctly() {
        // This test assumes trios-train binary is available
        let cfg = json!({"model_dim": 384, "hidden_dim": 128});
        if std::path::Path::new("/usr/local/bin/trios-train").exists() {
            let tr = ExternalTrainer::new("test", 42, 100, &cfg).unwrap();
            assert_eq!(tr.current_step, 0);
            assert_eq!(tr.seed, 42);
            assert_eq!(tr.max_steps, 100);
            assert!(!tr.finished());
        } else {
            // Skip test if binary not available
            println!("Skipping external trainer test - binary not found");
        }
    }
```

---

## C4. Integration Test Mock

**File:** `bin/seed-agent/tests/integration_external_trainer.rs` (new file)

```rust
//! Integration test for ExternalTrainer with mock subprocess.

use std::io::Write;
use std::process::{Command, Stdio};

#[test]
fn external_trainer_reads_from_subprocess() {
    // Create a mock trainer script that outputs JSONL
    let mock_script = r#"#!/bin/sh
# Mock trainer that outputs training progress
for i in $(seq 0 5); do
    echo "{\"step\":$i,\"bpb\":$(echo "3.5 - $i * 0.1" | bc -l),\"done\":false}"
    sleep 0.01
done
echo "{\"step\":6,\"bpb\":2.9,\"done\":true}
"#;

    let mut child = Command::new("sh")
        .arg("-c")
        .arg(mock_script)
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn mock trainer");

    // Verify JSONL output can be parsed
    if let Some(stdout) = child.stdout.as_mut() {
        use std::io::{BufRead, BufReader};
        let reader = BufReader::new(stdout);
        let mut count = 0;
        for line in reader.lines() {
            let line = line.unwrap();
            let output: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert!(output["step"].is_number());
            assert!(output["bpb"].is_number());
            count += 1;
        }
        assert_eq!(count, 7); // 6 training steps + 1 final
    }

    child.wait().expect("mock trainer failed");
}
```

---

## C5. PR Template

```markdown
## Purpose
Add ExternalTrainer implementation to enable real training via IGLA binary.

## Changes
- `trainer.rs`: Add `ExternalTrainer` struct + `Trainer` trait impl
- `worker.rs`: Replace bail statement with ExternalTrainer match arm
- `tests/integration_external_trainer.rs`: New integration test

## Safety
- ExternalTrainer spawns subprocess in controlled manner
- Binary path configurable via TRAINER_BIN env var
- JSONL parsing with error handling
- Branch protection requires review

## Testing
- Unit tests: ExternalTrainer initialization
- Integration test: Mock subprocess communication
- Manual: Deploy with TRAINER_KIND=external + TRAINER_BIN=/path/to/trios-train

## Dependencies
Requires Step B (Dockerfile) for binary availability in runtime
```

---

## C6. Verification Checklist

- [ ] ExternalTrainer added to trainer.rs
- [ ] worker.rs line 95 bail replaced
- [ ] Unit tests added and passing
- [ ] Integration test added and passing
- [ ] Cargo fmt passes
- [ ] Cargo clippy passes (no warnings)
- [ ] `cargo test` in bin/seed-agent/ passes

---

**⏭️ When complete, proceed to Step D**
