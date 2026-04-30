---
name: igla
description: This skill should be used when user asks to "update IGLA strategy", "check IGLA race status", "monitor workers", "deploy IGLA workers", "IGLA competition report", or references "IGLA RACE" numbers × models grid. Manages IGLA RACE competition (12 numeric formats × 6 models) monitoring and worker coordination.
version: 0.1.0
---

# IGLA RACE — Competition Management Skill

IGLA RACE is a two-axis competition format: **12 numeric formats × 6 model types** = 72 minimum cells.

## Competition Structure

### Numbers (12 participants)

| Lane# | Format | Bits | EXP:MANT | PHI_BIAS | Status |
|--------|---------|-------|-----------|-----------|--------|
| N1 | GF4 | 4 | 1:2 | 0 (F₀) | extract-only |
| N2 | GF8 | 8 | 3:4 | 1 (L₁) | extract-only |
| N3 | GF12 | 12 | 4:7 | 2 (L₀) | extract-only |
| N4 | GF16 | 16 | 6:9 | 60 (norm) | ✅ production-ready |
| N5 | GF20 | 20 | 7:12 | 289 (17²) | extract-only |
| N6 | GF24 | 24 | 9:14 | 1364 (L₁₅) | extract-only |
| N7 | GF32 | 32 | 12:19 | 0 (F₀) | spec-only |
| N8 | GF64 | 64 | 24:39 | 8388608 | no codegen |
| B1 | fp16 | 16 | 5:10 | — | ✅ IEEE baseline |
| B2 | bf16 | 16 | 8:7 | — | ✅ IEEE baseline |
| B3 | fp32 | 32 | 8:23 | — | ✅ IEEE baseline |
| B4 | fp8 | 8 | 4:3 | — | NVIDIA H100 only |

### Models (6 participants)

| Lane# | Model | Arch | Hidden bands | Notes |
|--------|--------|------|-------------|--------|
| M1 | MLP-baseline | feedforward | {384, 512, 768, 828, 1024, 1280} | reference |
| M2 | JEPA-T | Joint-Embedding Predictive | {512, 768, 1024} | 2-4 ATTN_LAYERS |
| M3 | NCA | Neural Cellular Automata | {384, 512} | 0 ATTN (cellular) |
| M4 | TF-1L | 1-layer Transformer | {512, 768, 1024} | 1 ATTN |
| M5 | TF-2L | 2-layer Transformer | {828, 1024} | 2 ATTN |
| M6 | TF-3L+ | 3+ layer | {1024, 1280} | 3-4 ATTN |

### Competition Grid

12 numbers × 6 models = **72 cells**. With 5 sanctioned seeds = **360 lane-runs**.

## Substitution Strategy

When full grid is not available, use tier-based substitution:

| Tier | Numbers | Method | Timeline |
|------|----------|---------|-----------|
| A (real) | GF16, fp16, bf16, fp32 | Native trainer support | Now |
| B (emulated) | GF8, GF12, GF20, GF24 | Quantize-on-the-fly (fp32 + weight quant) | T+3h |
| C (placeholder) | GF4, GF32, GF64 | Record "not_yet_implemented" | Post-Gate2 |

**Tier B emulation:** Use existing `gf16.rs` with parametric (exp_bits, mant_bits, phi_bias) to synthesize quantization without full backend. ~2-3h work.

## Core Workflows

### Check Competition Status

To check current competition status:

1. Read latest experience file: `.trinity/experience/{date}.md`
2. Check fleet status (workers alive, heartbeat times)
3. Check queue status (done, failed, pending, running)
4. Check gate-2 eligible count
5. Identify top performers by final_bpb

### Generate Competition Report

To generate live standings:

1. Query database for completed experiments with metrics
2. Group by format (number) and model
3. Calculate average BPB per cell
4. Identify champions per format
5. Identify champions per model
6. Check gate-2 eligibility (final_bpb < 1.85, step >= 4000)

Report format:
```
🏁 IGLA RACE — Live Standings @ T+Nh

NUMBERS LEADERBOARD                 MODELS LEADERBOARD
1. GF16   bpb=1.42 (h828)           1. TF-2L  bpb=1.38 (GF16)
...

CHAMPIONS BY CELL:
            M1     M2     M3     M4     M5     M6
GF16     1.71   1.45   1.89   1.52   1.38   1.41
...
```

### Update Worker Strategy

To update worker strategy:

1. Identify phase (Phase 1: 4-number race, Phase 2: 8-number race, Phase 3: full grid)
2. Determine active formats for current phase
3. Determine active models for current phase
4. Calculate lane-run count: formats × models × seeds × steps
5. Update experiment queue with new configurations

### Deploy Workers

To deploy IGLA workers:

1. Verify Railway tokens (acc0-acc5) in `.env`
2. Verify Neon database connectivity
3. Check image pin (current: trixie-slim@sha)
4. Deploy worker image across target accounts
5. Enqueue configurations via queue loader
6. Monitor initial heartbeat responses

## Monitoring Indicators

### Critical WAKE_TRIGGERS

| Trigger | Condition | Severity | Action |
|---------|------------|------------|----------|
| W-1 | fleet_alive < 4/6 | CRITICAL | Log, investigate |
| W-2 | zero bpb_samples (15m) | WARNING | Check TRAINER_KIND |
| W-6 | Φ-1 collapse (identical bpb) | CRITICAL | Stop, investigate |

### Gate-2 Requirements

- Target: final_bpb < 1.85
- Minimum steps: >= 4000
- Minimum samples: >= 5 distinct steps
- Eligible cells count toward Gate-2 progression

## Additional Resources

### Reference Files

For detailed competition rules and patterns, consult:
- **`references/competition-rules.md`** - Gate definitions, validation criteria
- **`references/worker-config.md`** - Worker deployment patterns
- **`references/quantization.md`** - GF format emulation details

### Database Queries

Useful queries for monitoring are in **`references/queries.sql`**:
- Fleet status aggregation
- Gate-2 candidate extraction
- Top performers by format/model
