# Experiment Log

This document summarizes the experiments conducted during the development of this submission. Over 60 training runs were performed across RTX 3090, A6000, and 8xH100 SXM hardware.

## Novel Technique Validation (NIGHT_MODE Campaign)

All novel techniques were validated independently on cheap GPUs before stacking on the final architecture.

| Technique | Seeds | Result | Verdict | Description |
|-----------|-------|--------|---------|-------------|
| **Gated Attention** | n=5 | train_loss 1.3711 (champion) | Confirmed win | Per-head sigmoid gate on attention output |
| **NorMuon** | n=2 | train_loss 1.40995 | Confirmed win | Post-NS row normalization (vs pre-NS in standard MuonEq-R) |
| **Norm-PCT-Dropout** | n=2 | train_loss 1.41365 | Confirmed win | Zero top 1% L2-norm FFN rows during training |
| **Parallel Muon** | n=2 | +3% throughput, quality neutral | Confirmed speedup | Batched Newton-Schulz across same-shape params |
| Gated + Legal TTT + N-gram Backoff (stacked) | n=2 | 1.45705 (+0.086 regression) | Stacking hostile | Too many novel techniques degrade each other |
| N-gram Bias Stack | n=3 | Various | Ruled out | Issue #1017 Condition 2 grey area; excluded from submission |
| CMP_QUANT_VALUE_DEDUP | n=2 | Quality neutral, -10-15% artifact size | Validated but not used | Alphabet-snap post-quant compression |

**Key finding**: Novel techniques that work in isolation can interfere when stacked. Our final stack uses only the 4 techniques that survived multi-seed validation AND compose cleanly.

## Phase 2: Speed Optimization (31 Experiments on RTX 3090)

| Exp | Config | ms/step | Speedup vs Baseline | Pre-quant BPB | Notes |
|-----|--------|---------|---------------------|---------------|-------|
| E1 | Baseline (no compile) | 2933 | 1.0x | 3.035 | Shot 0e quant gap 0.022 |
| E2 | torch.compile (default) | 1581 | **1.85x** | 2.920 | torch.compile is the biggest single win |
| E4b | max-autotune-no-cudagraphs | 1526 | **1.92x** | 2.923 | +3.7% over E2 |
| E5 | + cudnn.benchmark | 1514 | **1.94x** | 2.925 | +0.8% incremental |
| E6 | + Parallel Muon | 1369 | **2.14x** | 2.932 | Batched NS across params |
| E8 | + NUM_LOOPS=1 | 1410 | **2.08x** | 2.928 | Speed win but quality trade-off |
| E13 | NUM_LAYERS=8 | 1062 | **2.76x** | 3.052 | Layer reduction — faster but less capacity |
| E17 | NUM_LAYERS=8 + MLP=3 | 983 | **2.98x** | 3.065 | Near-3x baseline |
| E21 | NUM_LAYERS=6 | 856 | **3.43x** | 2.954 | Smaller model, more steps |
| E24 | NUM_LAYERS=6 + MLP=2 | 725 | **4.05x** | 2.971 | Best speed/quality balance |
| E26 | + TRAIN_SEQ_LEN=1024 | 643 | **4.56x** | 2.923 | Pareto optimal on 3090 |
| E29 | MODEL_DIM=256 | 343 | **8.55x** | 2.082 | Speed record but quant 3.64 (unusable) |

**Key insight**: 3090 is compute-bound. Bigger batches are a wash. Only cutting compute (fewer layers, smaller MLP, shorter sequences) or fusing kernels gives real speedups.

## Phase 2: Champion Full-Wallclock Runs (600s Budget)

| Config | Hardware | Steps | Pre-quant BPB | Quant BPB | Quant Gap | Notes |
|--------|----------|-------|---------------|-----------|-----------|-------|
| CHAMP_A (11L + MLP=2 + int6) | 3090 | 515 | 1.600 | 4.603 | **3.00** | Int6 catastrophic failure |
| CHAMP_B (6L + MLP=2 + int6) | 3090 | 813 | 1.399 | 4.966 | **3.57** | Int6 catastrophic failure |
| CHAMP_C (default + int6) | 3090 | 431 | 1.704 | 4.801 | **3.10** | Int6 catastrophic failure |
| **CHAMP_D (6L + MLP=2 + int8)** | 3090 | 813 | **1.398** | **1.399** | **0.001** | **Int8 breakthrough** |

**Critical discovery**: GPTQ int6 has insufficient precision for converged weight distributions on small models. The quant gap goes from ~0.02 (undertrained) to 3+ BPP (converged). Switching to int8 eliminates this entirely for small models.

For the full 11L+4x architecture used in the final submission, int8 doesn't fit the 16MB cap. We use int6 (matching PR #1493) and achieve a quant gap of **10.3 mBPP** — better than PR #1493's **11.7 mBPP**.

## Final Submission Run (8xH100 SXM)

| Retry | Issue | Resolution | Cost |
|-------|-------|------------|------|
| 1 | get_data.sh missing mkdir for cached SP model | Added mkdir -p before cp | ~$1.40 |
| 2 | Bootstrap STEP 3 ran with default config (not our stack) | Skipped bootstrap STEP 3, went straight to submission | ~$3 |
| 3 | Single-GPU (run.sh used python3 not torchrun) | Auto-detect GPU count, use torchrun when >1 | ~$8 |
| 4 | Flash Attention 3 not installed | pip install flash_attn_3 from wheel | ~$5 |
| **5 (final)** | Int8 quant doesn't fit 16MB + catastrophic gap with dedup | Switched to int6 matrices + int8 embeddings (matching PR #1493) | ~$25 |

Total compute cost: ~$60 across 5 retries. Effective (non-wasted) cost: ~$25.
