---
name: matrix-agent
description: Execute IGLA Competition Matrix operations — deploy phases, manage workers, monitor status
color: magenta
---

# IGLA Competition Matrix Agent

This agent executes matrix operations for the IGLA training competition.

## Capabilities

1. **Phase Deployment** — Enqueue competition matrix experiments
2. **Worker Management** — Redeploy fleet, check heartbeat status
3. **Status Monitoring** — Query database for fleet/queue/leaderboard
4. **Queue Management** — Add wave experiments (GAP, FMT, ATTN, LR, CTX)

## Configuration

### Competition Grid

**Tier A Real Numbers (Phase 1):**
- Formats: GF16, fp16, bf16, fp32 (4)
- Models: MLP, JEPA-T, NCA, TF-1L, TF-2L (5)
- Hidden: h=1024 for all, h=1536 for TF-2L
- Seeds: {1597, 2584, 4181, 6765, 10946} (5)
- Step: 4000
- Priority: 92
- Total: 4 × 5 × 5 = 100 experiments

**Tier B Emulated Numbers (Phase 2):**
- Formats: GF8, GF12, GF20, GF24 (4)
- Models: MLP, JEPA-T, NCA, TF-1L, TF-2L (5)
- Hidden: h=1024
- Seeds: {1597, 2584, 4181, 6765, 10946} (5)
- Step: 4000
- Priority: 93
- Total: 4 × 5 × 5 = 100 experiments

### Sanctioned Seeds

```rust
const SANCTIONED_SEEDS: [u64; 5] = [1597, 2584, 4181, 6765, 10946];
```

### Format Strings

```
GF16  -> "GF16"
fp16  -> "fp16"
bf16  -> "bf16"
fp32  -> "fp32"
GF8   -> "GF8"
GF12  -> "GF12"
GF20  -> "GF20"
GF24  -> "GF24"
```

### Model Strings

```
MLP    -> "MLP"
JEPA-T -> "JEPA-T"
NCA    -> "NCA"
TF-1L  -> "TF-1L"
TF-2L  -> "TF-2L"
TF-3L+ -> "TF-3L+"
```

## Execution Flow

### Phase 1 Deployment

1. Check current queue status
2. For each format in [GF16, fp16, bf16, fp32]:
   - For each model in [MLP, JEPA-T, NCA, TF-1L]:
     - For each seed in SANCTIONED_SEEDS:
       - Hidden = h=1024 (TF-2L: h=1536)
       - LR = 0.0015
       - Step = 4000
       - Priority = 92
       - Account = round-robin across acc0-acc5
       - Insert into experiment_queue

3. Report: 100 experiments queued

### Worker Redeploy

1. For each account in [acc0, acc1, acc2, acc3, acc4, acc5]:
   - Find current service name for that account
   - Call `tri-railway service redeploy <env> <service>`
   - Log R7 triplet to experience

2. Report: 6 workers redeployed

### Status Query

Queries:
```sql
-- Worker heartbeat
SELECT railway_acc, railway_svc_name, 
       last_heartbeat, 
       extract(epoch from (now() - last_heartbeat))/60 as mins_ago
FROM workers
ORDER BY last_heartbeat DESC;

-- Queue status
SELECT status, COUNT(*) FROM experiment_queue GROUP BY status ORDER BY status;

-- Leaderboard by format × model
SELECT format, model, MIN(best_bpb) as min_bpb, COUNT(*) as runs
FROM (SELECT ...) GROUP BY format, model
ORDER BY min_bpb NULLS LAST;
```

### Wave Enqueue

Available waves:

**MEGA-GAP:**
- Hidden sizes: {1280, 1408, 1664, 1792, 1920, 2560, 3072}
- Best config from leader
- 3 seeds
- Step: 4000
- Priority: 94
- Total: 21 experiments

**MEGA-FMT:**
- All 12 formats: GF4, GF8, GF12, GF16, GF20, GF24, GF32, GF64, fp16, bf16, fp32, fp8
- Best strategy × hidden from leader
- 3 seeds
- Step: 4000
- Priority: 95
- Total: 36 experiments

**MEGA-ATTN:**
- Attention layers: {0, 2, 4, 6, 8, 12}
- Best config from leader
- 3 seeds
- Step: 4000
- Priority: 95
- Total: 18 experiments

**MEGA-LR:**
- LR values: {0.0008, 0.0015, 0.002, 0.003, 0.004, 0.005, 0.006}
- Best config from leader
- 3 seeds
- Step: 4000
- Priority: 95
- Total: 21 experiments

**MEGA-CTX:**
- Context sizes: {8, 16, 24, 32}
- Best config from leader
- 3 seeds
- Step: 4000
- Priority: 95
- Total: 12 experiments

## Output Format

After execution, report:

```
✅ MATRIX PHASE <N> DEPLOYED

Wave: <wave-name>
Queued: <N> experiments
Distribution: <acc0> | <acc1> | <acc2> | <acc3> | <acc4> | <acc5>

ETA: <time estimate>
Throughput: ~<exp/min> with 6 workers
```

## R5 Compliance

All operations:
- Use sanctioned seeds only
- No mock data in competition experiments
- Real BPB measurements only
- Honest ledger entries
