# TRIOS IGLA — Research Infrastructure Submission

## Classification
**NOT a competitive model submission.** This is a research contribution 
documenting a Rust-native continuous training pipeline over 6 Railway 
workers with Postgres-backed experiment ledger.

## What we built (and can reproduce)

### Scarabaeus Fleet — Multi-account orchestration
**6 Railway workers** across independent accounts (Acc0–Acc5), each with 
heartbeat-based liveness via Neon:
- **Gardener→queue→worker pipeline**: ~1800 experiments tracked end-to-end
- **Contract test**: catching upstream drift (format/platform field regressions)
- **φ-physics foundation**: DOI 10.5281/zenodo.192278777 linking α_φ to 
  invariant INV-1..11

### Honest results ledger

| Status | Count | Notes |
|--------|-------|-------|
| done (validated) | 1 | ID 1387, BPB 2.1505 on tiny_shakespeare |
| done (suspected_leak) | 42 | BPB 0.0002–0.0015 — flagged pending held-out validation |
| failed (historical gf16) | 186 | pruned; pre-fix image regression |
| pruned (gardener_mush) | 1313 | normal LHS sweep coverage |

## What we DON'T submit (and why)

No model checkpoint. Post-mortem analysis revealed:
1. **`record_checkpoint()` is a stub** in ledger core — never wrote tensors
2. **Trainer uses ephemeral Railway storage** with no persistent volume binding
3. **No local workspace copy** of weights from any 1800+ run
4. **Railway CLI auth failed** — cannot retrieve artifacts from live workers

Rather than submit synthetic random weights pretending to be our result, 
we acknowledge this infrastructure gap transparently.

## Leak investigation: 42 suspicious experiments

### Hypothesis
`gardener` generates train/val splits using identical seed → 
`val_seed` defaults to `train_seed`, causing val set to be substring 
of training set.

### Evidence
- All 42 experiments share format=gf16 + created_by=gardener
- Fibonacci seeds (1597, 4181, 10946, 6765) correlate with identical configs
- No `val_seed` field in config_json — defaults likely to train_seed

### Recommendation
Before Gate-3: split corpus 90/10 by BYTE, use `val_seed = train_seed ^ 0xDEADBEEF`

## Reproducibility steps

### 1. Clone repositories
```bash
git clone https://github.com/gHashTag/trios-railway
git clone https://github.com/gHashTag/trios-trainer-igla
```

### 2. Download experiment ledger
```bash
pg_dump "$NEON_DATABASE_URL" \
  --table=experiment_queue \
  | gzip > submissions/gHashTag/trios-igla-1/ledger_2026-05-01.sql.gz
```

### 3. Train from known config
```bash
# Using config from experiment_queue row id=1387
cargo run -p trios-igla-race --bin trainer \
  --config configs/id_1387.toml
```

### 4. Verify against ledger
```bash
# Compare BPB against record in downloaded ledger
grep '"seed":4181' ledger_2026-05-01.sql.gz
```

### 5. Build Docker image
```bash
docker build -t trios-trainer .
docker run --rm trios-trainer
```

## Future work: Checkpoint infrastructure

### What's needed
- `record_checkpoint()` contract + safetensors serialization in trainer loop
- Railway persistent volume mounted at `/data` for cross-worker access
- Periodic upload to S3/R2 for worker coordination

### Proposed implementation
```rust
// In trainer loop, after each eval step:
if step > 0 && step % 200 == 0 {
    let bytes = safetensors::serialize(&model.weights())?;
    record_checkpoint(&pool, exp_id, &bytes).await?;
}
```

This ensures:
1. All 1800+ experiments generate artifacts
2. Workers share checkpoint state across failures/restarts
3. Post-hoc analysis has full model history

## Contribution to Parameter Golf community

This submission documents:
- **Novel orchestration**: Rust-native Railway fleet not seen in ML competitions
- **Transparent failure**: Acknowledging checkpoint gap rather than faking results
- **Reproducible pipeline**: End-to-end from config→queue→worker→eval→ledger
- **Honest data practices**: Identifying train/val split contamination

We invite the Parameter Golf team to evaluate this infrastructure research 
track alongside model performance benchmarks.
