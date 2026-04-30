---
name: matrix
description: Execute IGLA Competition Matrix commands — Phase deployment, monitoring, worker management
---

# IGLA Competition Matrix Commands

Run these commands via: `/matrix <subcommand>`

## Available Commands

### `/matrix phase1`
Deploy Phase 1: Tier A Real Numbers — 4 formats × 5 models × 5 seeds = 100 experiments

**Formats:** GF16, fp16, bf16, fp32
**Models:** MLP, JEPA-T, NCA, TF-1L, TF-2L
**Hidden:** h=1024 for all, h=1536 for TF-2L
**Seeds:** {1597, 2584, 4181, 6765, 10946}
**Step:** 4000
**Priority:** 92

### `/matrix phase2`
Deploy Phase 2: Tier B Emulated Numbers — GF8/12/20/24 × models

**Requires:** parametric quant implementation in trios-trainer-igla

### `/matrix promote <strategy> <format> <hidden>`
Promote winning cell to long run (step=27000-50000)

**Args:**
- strategy: MLP, JEPA-T, NCA, TF-1L, TF-2L, TF-3L+
- format: GF16, fp16, bf16, fp32
- hidden: 512, 768, 828, 1024, 1280, 1536, 2048

### `/matrix status`
Show current fleet and queue status

**Displays:**
- Worker heartbeat (alive < 5m / < 60s / stale)
- Queue counts by status and account
- Leaderboard by format × model
- Gate-2 candidates

### `/matrix redeploy`
Redeploy all 6 workers (acc0-acc5) to recover dead fleet

**Action:** Calls `tri-railway service redeploy` for each account

### `/matrix queue <wave>`
Enqueue a specific wave of experiments

**Waves:**
- `MEGA-GAP` — hidden sizes never tried {1280,1408,1664,1792,1920,2560,3072}
- `MEGA-FMT` — all 11 formats on best configs
- `MEGA-ATTN` — attention layers {0,2,4,6,8,12}
- `MEGA-LR` — LR sweep {0.0008,0.0015,0.002,0.003,0.004,0.005,0.006}
- `MEGA-CTX` — context sweep {8,16,24,32}

Example: `/matrix queue MEGA-GAP`

## Implementation

Commands use `tri-railway` CLI for:
- Service redeploy: `tri-railway service redeploy <account> <service>`
- Queue management: SQL inserts to experiment_queue
- Status queries: SQL SELECT from workers, experiment_queue, bpb_samples

## R5 Honest Contract

All Competition Matrix experiments:
- Use only sanctioned seeds {1597,2584,4181,6765,10946}
- Non-mock data only
- Record real BPB values
- No synthetic results in competition rows
- Tier C placeholders marked as "not_yet_implemented" in ledger
