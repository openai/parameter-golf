# Experiment Registry

## Current Target Baseline (to reproduce)

- **Source**: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
- **Author**: @abaybektursun (PR #756)
- **BPB**: 1.1147 (3-seed mean, std 0.0004)
- **Artifact**: ~15.91 MB
- **Hardware**: 8xH100 SXM, 600s
- **Stack**: 11L + XSA-all + BigramHash3072x112 + Full Hessian GPTQ int6 (AR self-gen) + LZMA9 + selective pruning
- **Run command**:
  ```bash
  BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
  TARGET_MB=15.9 SEED=314 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```

## Our Starting Point

- **Smoke test only**: val_bpb=2.3283 (200 steps, MLX, local Mac, baseline script)
- **Gap to SOTA**: ~1.14 BPB (need to reproduce their full stack on 8xH100)

---

## Phase 0: Reproduce SOTA

| ID | Branch | Goal | Status | BPB Result | Delta vs Target |
|----|--------|------|--------|------------|-----------------|
| R-01 | exp/reproduce-sota | Run leader's exact code, seed=314, 8xH100 | PLANNED | - | target: 1.1151 |
| R-02 | exp/reproduce-sota | seed=42 | BLOCKED on R-01 | - | target: 1.1144 |
| R-03 | exp/reproduce-sota | seed=999 | BLOCKED on R-01 | - | target: 1.1148 |

## Phase 1: Innovate (Engram + TurboQuant)

| ID | Branch | Hypothesis | Status | Priority | BPB Result | Delta |
|----|--------|-----------|--------|----------|------------|-------|
| EXP-001 | exp/engram-replace-bigram | Engram trigrams + gating > flat BigramHash | PLANNED | P0 | - | - |
| EXP-002 | exp/hadamard-gptq-int6 | Hadamard rotation reduces GPTQ quant error | PLANNED | P0 | - | - |
| EXP-003 | exp/hadamard-scalar-int5 | Rotation enables int5, freeing 2.5MB for more params | PLANNED | P1 | - | - |
| EXP-004 | exp/engram-hadamard-combo | Best of EXP-001 + EXP-002 combined | BLOCKED on 001,002 | P1 | - | - |

## Completed Experiments (newest first)

| ID | Branch | Result | Delta vs SOTA | Keep/Drop | Notes |
|----|--------|--------|---------------|-----------|-------|

---

## Negative Results Log

| Date | Experiment | Expected | Actual | Why It Failed | Lesson |
|------|-----------|----------|--------|---------------|--------|

---

## Compute Budget

| Date | Experiment | Run Type | Hardware | Duration | Cost | Running Total |
|------|-----------|----------|----------|----------|------|---------------|
