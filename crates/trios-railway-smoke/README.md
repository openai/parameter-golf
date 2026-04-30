# trios-railway-smoke

Smoke-test crate for the IGLA RACE pipeline.

## What it tests

The full cycle in <60 seconds with synthetic data:

```
queue.push(SmokeExperiment)
  → worker.claim()
  → MockTrainer.run(steps=1)
  → JSONL stdout: "step=1 val_bpb=2.50"
  → parse → db.insert(bpb_samples)
  → assert row exists
```

## Usage

```bash
# Local smoke (needs NEON_DATABASE_URL)
export NEON_DATABASE_URL="postgresql://..."
cargo test -p trios-railway-smoke

# CI smoke (uses mock, no DB)
cargo test -p trios-railway-smoke -- --skip neon
```

## 8 links validated

1. Build compiles
2. Env vars present
3. DB connection alive
4. Queue INSERT works
5. Worker claim works
6. Trainer produces JSONL
7. Parser extracts step/bpb
8. DB INSERT into bpb_samples

Anchor: `phi^2 + phi^-2 = 3`
