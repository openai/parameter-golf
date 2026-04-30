//! Smoke test for the full pull-to-train loop.
//!
//! Tests the entire experiment pipeline without requiring real GPU or
//! trainer binary. Uses MockTrainer which emits deterministic
//! BPB curves without any external dependencies.
//!
//! This test validates:
//!   1. Worker can register
//!   2. Worker can claim an experiment
//!   3. MockTrainer runs and produces step/BPB data
//!   4. Data is written to bpb_samples table
//!   5. Experiment is marked as done
//!
//! Run with: cargo test -p seed-agent --test smoke_full_cycle
//!
//! Anchor: phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP

use std::time::Duration;
use tokio_postgres::NoTls;
use uuid::Uuid;

use crate::{claim, telemetry, trainer, worker};

/// Run a complete experiment cycle with MockTrainer.
///
/// This is the minimal smoke test that validates the entire
/// pull-queue → trainer → bpb_samples → done pipeline.
#[tokio::test]
#[ignore] // Requires NEON_DATABASE_URL - run with: cargo test -- --ignored
async fn smoke_full_cycle() {
    let neon_url = match std::env::var("NEON_DATABASE_URL") {
        Ok(url) => url,
        Err(_) => {
            eprintln!("Skipping smoke test: NEON_DATABASE_URL not set");
            return;
        }
    };

    // Connect to Neon
    let (client, conn) = tokio_postgres::connect(&neon_url, NoTls)
        .await
        .expect("failed to connect to Neon");
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            eprintln!("Neon connection error: {e}");
        }
    });

    // Create a test worker config
    let worker_id = Uuid::new_v4();
    let cfg = worker::WorkerConfig {
        worker_id,
        railway_acc: "acc0".to_string(),
        railway_svc_id: "smoke-test-worker".to_string(),
        railway_svc_name: "smoke-test-worker".to_string(),
        poll_idle: Duration::from_secs(1),
        early_stop_step: 100, // Smaller for smoke test
        early_stop_bpb_ceiling: 3.0,
        trainer_kind: "mock".to_string(),
    };

    // Register worker
    worker::register_worker(&client, &cfg)
        .await
        .expect("register_worker failed");

    println!("[SMOKE] Worker registered: {}", worker_id);

    // Insert a smoke experiment
    let canon_name = "SMOKE-TEST-h1024-LR002";
    let config_json = r#"{"hidden":1024,"lr":0.002,"ctx":12,"steps":100,"mock_initial_bpb":3.5,"mock_target_bpb":1.85,"mock_decay":0.01}"#;
    let priority = 50_i32;
    let seed = 42_i32;
    let steps_budget = 100_i32;

    let row = client
        .query_one(
            "INSERT INTO experiment_queue \
             (canon_name, config_json, priority, seed, steps_budget, account, status, created_by) \
             VALUES ($1, $2::jsonb, $3, $4, $5, 'acc0', 'pending', 'smoke-test') \
             ON CONFLICT DO NOTHING \
             RETURNING id",
            &[&canon_name, &config_json, &priority, &seed, &steps_budget],
        )
        .await
        .expect("insert experiment failed");
    let exp_id: i64 = row.get(0);

    println!("[SMOKE] Experiment inserted: id={} name={}", exp_id, canon_name);

    // Run one iteration (claim → train → record)
    let result = worker::run_one_iteration(&client, &cfg).await;
    println!("[SMOKE] Iteration result: {:?}", result);

    // Verify outcome
    match result {
        Ok(worker::IterOutcome::Trained(name)) => {
            assert_eq!(name, canon_name, "experiment name mismatch");
            println!("[SMOKE] ✅ Experiment completed successfully");
        }
        Ok(worker::IterOutcome::Pruned(name, reason)) => {
            assert_eq!(name, canon_name, "experiment name mismatch");
            println!("[SMOKE] ⚠️  Experiment pruned: {}", reason);
        }
        Ok(worker::IterOutcome::Idle) => {
            panic!("Expected work, got Idle");
        }
        Err(e) => {
            panic!("Smoke test failed: {e}");
        }
    }

    // Verify bpb_samples were recorded
    let rows = client
        .query(
            "SELECT COUNT(*) as cnt FROM bpb_samples WHERE canon_name = $1",
            &[&canon_name],
        )
        .await
        .expect("query bpb_samples failed");
    let count: i64 = rows[0].get(0);
    assert!(count > 0, "Expected at least one bpb_sample, got {}", count);
    println!("[SMOKE] ✅ BPB samples recorded: {}", count);

    // Verify experiment is marked done
    let row = client
        .query_one(
            "SELECT status, final_step, final_bpb FROM experiment_queue WHERE id = $1",
            &[&exp_id],
        )
        .await
        .expect("query experiment failed");
    let status: String = row.get(0);
    let final_step: i32 = row.get(1);
    let final_bpb: f64 = row.get(2);

    assert_eq!(status, "done", "Expected status=done, got {}", status);
    assert_eq!(final_step, 100, "Expected final_step=100, got {}", final_step);
    assert!(
        final_bpb < 3.5,
        "Expected final_bpb < 3.5, got {}",
        final_bpb
    );
    assert!(final_bpb >= 1.85, "Expected final_bpb >= 1.85, got {}", final_bpb);

    println!(
        "[SMOKE] ✅ All checks passed: status={} step={} bpb={:.2}",
        status, final_step, final_bpb
    );
}

#[test]
fn smoke_mock_trainer_produces_monotonic_decrease() {
    let config = serde_json::json!({
        "mock_initial_bpb": 3.5,
        "mock_target_bpb": 1.85,
        "mock_decay": 0.01
    });

    let mut trainer = trainer::MockTrainer::from_config("SMOKE-TEST", 42, 100, &config)
        .expect("failed to create MockTrainer");

    let mut prev_bpb = f64::MAX;

    for step in 1..=100 {
        trainer.step().expect("step failed");
        let bpb = trainer.eval_bpb();
        let step = trainer.current_step();

        // Verify monotonic decrease (with tiny jitter allowed)
        assert!(
            bpb <= prev_bpb + 0.1,
            "BPB increased at step {}: {:.4} -> {:.4}",
            step - 1,
            prev_bpb,
            bpb
        );

        // Verify we progress toward target
        assert!(
            bpb > 1.5,
            "BPB collapsed below floor at step {}: {:.4}",
            step,
            bpb
        );

        prev_bpb = bpb;
    }

    assert_eq!(trainer.current_step(), 100, "Expected final step 100");
    let final_bpb = trainer.eval_bpb();
    assert!(
        final_bpb < 3.5,
        "Expected final BPB < 3.5, got {:.4}",
        final_bpb
    );
    assert!(
        final_bpb > 1.85,
        "Expected final BPB > 1.85, got {:.4}",
        final_bpb
    );

    println!(
        "[SMOKE] ✅ MockTrainer monotonic: {:.4} → {:.4}",
        3.5,
        final_bpb
    );
}
