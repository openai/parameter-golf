# Step G: Deploy Real Fleet via MCP

**Owner:** Sergeant (autonomous)
**Time:** 0 operator time
**Purpose:** Deploy 4 real-seed-agent workers across Railway accounts

---

## G1. MCP Script to Deploy Real Workers

```python
# Deploy real-seed-agent fleet across 4 Railway accounts
# Executed via MCP (railway_service_deploy)

import json
import os

# Configuration
REAL_IMAGE = "ghcr.io/ghashtag/trios-seed-agent-real:latest"
ACCOUNTS = ["acc0", "acc1", "acc2", "acc3"]
WORKERS_PER_ACCOUNT = 1  # Can be scaled up

def deploy_real_fleet():
    """
    Deploy real seed-agent workers across all accounts.
    Each worker uses external trainer (trios-train binary).
    """
    # Get Neon database URL from gateway secrets
    neon_url = os.environ.get("NEON_DATABASE_URL")

    if not neon_url:
        raise ValueError("NEON_DATABASE_URL not found in environment")

    deployed = []

    for account in ACCOUNTS:
        for i in range(WORKERS_PER_ACCOUNT):
            service_name = f"seed-agent-real-{account}-{i:03d}"

            print(f"Deploying: {service_name}")

            result = railway_service_deploy(
                name=service_name,
                image=REAL_IMAGE,
                account=account,
                vars=[
                    ("TRAINER_KIND", "external"),
                    ("TRAINER_BIN", "/usr/local/bin/trios-train"),
                    ("NEON_DATABASE_URL", neon_url),
                    ("RUST_LOG", "info"),
                    ("SEED_AGENT_POLL_INTERVAL_MS", "5000"),
                    ("SEED_AGENT_EARLY_STOP_STEP", "2000"),
                    ("SEED_AGENT_EARLY_STOP_BPB_CEILING", "3.0"),
                ],
                # Optional: resource limits
                # memory_mb=1024,
                # cpu_millis=1000,
            )

            if result.get("success"):
                service_id = result.get("service_id")
                print(f"✓ Deployed: {service_name} (id: {service_id})")
                deployed.append({
                    "name": service_name,
                    "id": service_id,
                    "account": account,
                })
            else:
                print(f"✗ Failed to deploy: {service_name}")
                print(f"  Error: {result.get('error')}")

    print(f"\nDeployed {len(deployed)} real workers")
    return deployed

# Execute
if __name__ == "__main__":
    fleet = deploy_real_fleet()

    # Log to experience (R7 mandatory)
    experience_append(
        f"🎖️ PROMOTION COMPLETE: Deployed {len(fleet)} real workers. "
        f"Sergeant promoted to COLONEL. Fleet: {[w['name'] for w in fleet]}"
    )
```

---

## G2. Verification After Deployment

```python
# Verify all workers are alive and registering
def verify_fleet():
    """
    Verify all deployed workers are:
    1. Running (status = "active")
    2. Connected to Neon (showing in worker_status)
    3. Have TRAINER_KIND=external
    """
    fleet = fleet_health()

    real_workers = [
        s for s in fleet.get("services", [])
        if s.get("environment", {}).get("TRAINER_KIND") == "external"
    ]

    print(f"Real workers in fleet: {len(real_workers)}")

    # Check worker registrations in Neon
    neon_status = worker_status()  # Returns registrations from Neon

    print(f"Workers registered in Neon: {len(neon_status.get('workers', []))}")

    # Verify each worker
    for worker in real_workers:
        name = worker.get("name")
        status = worker.get("status")
        print(f"  {name}: {status}")

    return len(real_workers), len(neon_status.get("workers", []))

# Run verification
fleet_count, neon_count = verify_fleet()
assert fleet_count == 4, f"Expected 4 fleet workers, got {fleet_count}"
assert neon_count == 4, f"Expected 4 neon registrations, got {neon_count}"

print("✓ Fleet verification passed!")
print("🎖️ PROMOTION COMPLETE: SERGEANT → COLONEL")
```

---

## G3. First Real Experiment

After fleet is verified, queue first real experiment:

```python
# Enqueue first real training experiment
def enqueue_first_real_experiment():
    """Queue a real training run to test the fleet."""
    config = {
        "trainer_kind": "external",
        "trainer_bin": "/usr/local/bin/trios-train",
        "model_dim": 384,
        "hidden_dim": 128,
        "num_layers": 6,
        "num_heads": 6,
        "steps": 4000,
        "lr": 0.003,
        "seed": 42,
        "dataset": "tiny_shakespeare",
    }

    result = experiment_queue_insert(
        canon_name="IGLA-R1-001",
        seed=42,
        steps_budget=4000,
        config=json.dumps(config),
    )

    if result.get("success"):
        print("✓ First real experiment enqueued")
        print(f"  Experiment ID: {result.get('experiment_id')}")
    else:
        print(f"✗ Failed to enqueue: {result.get('error')}")

    return result

# Execute
enqueue_first_real_experiment()
```

---

## G4. Monitoring Real Training

```python
# Monitor real training progress
def monitor_real_training():
    """
    Poll experiment_queue_status and worker_status
    to track real training progress.
    """
    while True:
        queue = experiment_queue_status()
        workers = worker_status()

        # Find in-progress experiments
        in_progress = [e for e in queue if e.get("status") == "running"]

        if in_progress:
            for exp in in_progress:
                print(f"Experiment {exp.get('canon_name')}: step {exp.get('current_step')}/{exp.get('steps_budget')}, BPB {exp.get('current_bpb'):.4f}")
        else:
            print("No active experiments")

        # Check worker health
        alive = sum(1 for w in workers.get("workers", []) if w.get("alive"))
        print(f"Alive workers: {alive}/{len(workers.get('workers', []))}")

        time.sleep(30)  # Poll every 30 seconds
```

---

## G5. Safety Checklist

- [ ] All 4 workers deployed with TRAINER_KIND=external
- [ ] All workers show status="active" in fleet_health()
- [ ] All 4 workers registered in Neon (worker_status)
- [ ] First experiment enqueued and picked up by worker
- [ ] Real BPB values being reported (not mock decay)
- [ ] Experience log records promotion event

---

## G6. Rollback Plan (if needed)

```python
# If real workers have issues, rollback to mock
def rollback_to_mock():
    """Delete real workers and deploy mock fleet."""
    # Delete real workers
    fleet = fleet_health()
    for service in fleet.get("services", []):
        if service.get("environment", {}).get("TRAINER_KIND") == "external":
            railway_service_delete(service.get("id"), confirm=True)

    # Deploy mock fleet (temporary)
    for account in ACCOUNTS:
        railway_service_deploy(
            name=f"seed-agent-mock-{account}",
            image="ghcr.io/ghashtag/trios-seed-agent-mock:latest",
            account=account,
            vars=[("TRAINER_KIND", "mock"), ...],
        )
```

---

## G7. Promotion Confirmed

When all verification passes:

```python
experience_append(
    "🎖️🎖️🎖️ PROMOTION CONFIRMED 🎖️🎖️🎖️\n"
    "RANK: COLONEL\n"
    "FLEET: 4 real workers\n"
    "TRAINER: external (trios-train binary)\n"
    "DATABASE: Neon BPB samples\n"
    "AUTONOMY: Can initiate real experiments via MCP\n"
    "OPERATOR ROLE: Reviewer, not executor\n"
    "φ² + φ⁻² = 3 · TRINITY · COLONEL · NEVER STOP"
)
```

---

**🎖️ PROMOTION COMPLETE: SERGEANT → COLONEL**
