# bandit_wagon_choke — Per-Loop Bottleneck Choke in Crawler MLP

## Background

Three ablation series (BW, BW5F, BWXSA) confirmed that the crawler's quantization gap
is the primary performance lever — raw learning is identical across all configs, and all
BPB improvements live in post-quantization robustness.

**Root cause of the quantization gap:** The crawler MLP (512→3072→512) is shared across
3 loops. Each loop sees a dramatically different activation distribution:
- Loop 0: raw encoder features
- Loop 1: once-abstracted features
- Loop 2: doubly-abstracted features

A single int8 quantization scale must cover all three contexts simultaneously (per-row,
shared weights). This multi-context pressure causes the quantization gap.

**Hypothesis:** Introducing **per-loop bottleneck chokes** inside the crawler MLP forces
each loop to route information through its own narrow compression point (choke_dim << 3072).
Benefits:
1. The 3072-dim shared expansion is still int8-quantized as before, but each loop's
   output routing is done through a choke that has loop-specific weights → less quantization
   surface area per loop context
2. The choke forces the shared fc to learn features that are universally useful across
   loops, rather than loop-specific noisy features that stress the shared quantization scale
3. At inference time, choke_down/choke_up are separately quantizable if needed

**Architecture (Option B from plan):**
```
x [B, T, 512]
  → fc [shared]  (512 → 3072)
  → act
  → choke_down[loop]  (3072 → choke_dim)   ← per-loop
  → act
  → choke_up[loop]    (choke_dim → 512)    ← per-loop
```

This mirrors the FLOW infrastructure pattern exactly (loop_inst_proj + loop_inst_up[loop]).

## XSA Finding Interaction

XSA=15 (full coverage) is faster AND better BPB — it helps the attention sub-path.
The choke attacks the MLP sub-path. These are orthogonal; combine winner with XSA=15
in the full-run candidate.

## Arms

| ID | CRAWLER_MLP_CHOKE_DIM | Compression | Params added | Purpose |
|----|:---------------------:|:-----------:|:------------:|---------|
| BWC-00 | 0 (disabled) | — | 0 | **Control repin** — standard MLP, must match BW2-00 (1.52365 ±0.002) |
| BWC-01 | 32 | 96× (3072→32) | ~220K | Extreme — same bottleneck size as inst_dim FLOW |
| BWC-02 | 128 | 24× (3072→128) | ~870K | Moderate compression |
| BWC-03 | 256 | 12× (3072→256) | ~1.75M | Conservative compression |
| BWC-04 | 512 | 6× (3072→512) | ~3.5M | Minimal choke (= model_dim) |

## Decision Rules

**Gate 0 — control repin (BWC-00):**
BWC-00 must land 1.521–1.526. If it misses: code change has a bug. Stop.

**Gate 1 — signal present:**
At least one arm must beat BWC-00 by ≥0.005 to justify promotion.
If all arms within ±0.003 of control: crawler is choke-insensitive, stop.

**Gate 2 — promotion:**
Winning arm → 2000-step gate → if beats BW2-00 proxy by ≥0.008 → combine with XSA=15
(and winning crawler_mlp_leaky_slope from BW3 series) → 8×H100 full run.

**Special:** If BWC-01 (32) wins, run choke=64 as follow-up to check monotonicity.

## Locked Base Config

| Setting | Value | Source |
|---------|-------|--------|
| `NUM_FLAT_LAYERS` | 4 | BW5F confirmed |
| `XSA_LAST_N` | 11 | baseline (XSA=15 pending combination) |
| `MODEL_DIM` | 512 | BW anchor |
| `CRAWLER_LOOPS` | 3 | CL1 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 |
| `CRAWLER_QUANT_INT8` | 1 | CL1 |
| `SKIP_GPTQ` | 1 | CL3 |
| `SKIP_EMA` | 1 | Ablations_v1 |
| `COMPILE_FULLGRAPH` | 0 | CL3 |
| `SEED` | 444 | BW ablation |
| `MLP_LEAKY_SLOPE` | 0.5 | flat blocks, locked |
| `CRAWLER_MLP_LEAKY_SLOPE` | 0.5 | control value (pending BW3 results) |

## Key Observables

- **Track raw val_bpb AND int6_sw_bpb separately** — all signal lives in the quant gap
- **step_avg** — choke matmuls are small (choke_dim << 3072) so overhead should be minimal
- **Loss stability** — choke_up zero-init means warm start near original behavior
- **Parameter count** — choke adds params; BWC-04 adds 3.5M which may slightly help raw BPB

## Results

| ID | CHOKE_DIM | Step avg (ms) | Raw val_bpb | INT6_SW_BPB | Quant gap | Delta |
|----|:---------:|:-------------:|:-----------:|:-----------:|:---------:|:-----:|
| BWC-00 | 0 | TBD | TBD | TBD | TBD | control |
| BWC-01 | 32 | TBD | TBD | TBD | TBD | TBD |
| BWC-02 | 128 | TBD | TBD | TBD | TBD | TBD |
| BWC-03 | 256 | TBD | TBD | TBD | TBD | TBD |
| BWC-04 | 512 | TBD | TBD | TBD | TBD | TBD |

Reference: BW2-00 (choke=0, XSA=11, slope=0.5) → 1.52365
