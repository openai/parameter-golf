//! Trainer abstraction.
//!
//! For ADR-0081 PR-1 we ship two backends:
//!
//! 1. `MockTrainer` — deterministic in-process simulator. Emits a
//!    monotonically-decreasing BPB curve seeded by the experiment's
//!    rng. Used by CI and local smoke tests so the pull loop is
//!    fully exercised without GPUs.
//!
//! 2. `external` — placeholder. Will shell out to the IGLA trainer
//!    in a follow-up PR (per ADR-0001 the trainer src lives in
//!    `trios-trainer-igla`; we only invoke its compiled binary, never
//!    edit it).
//!
//! The contract is intentionally narrow: `step()` advances one
//! training step, `eval_bpb()` returns the current BPB. The pull
//! loop owns Neon I/O, the trainer owns the math.
//!
//! Anchor: `phi^2 + phi^-2 = 3`.

use anyhow::{anyhow, Result};
use serde_json::Value;

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
}
