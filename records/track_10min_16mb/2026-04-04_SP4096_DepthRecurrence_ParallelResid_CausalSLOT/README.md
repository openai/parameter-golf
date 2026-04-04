# Record: SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + Causal SLOT — val_bpb 1.0766 (3-seed mean)

**val_bpb = 1.0766** (3-seed mean, std 0.0004) | **~16.00 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Sliding BPB | **Causal SLOT BPB** | SLOT gain | Artifact |
|------|-------------|---------------------|-----------|----------|
| 42   | 1.0893      | **1.0762**          | -0.0131   | 15,999,461 |
| 314  | 1.0897      | **1.0766**          | -0.0131   | 15,997,932 |
| 999  | 1.0897      | **1.0770**          | -0.0127   | 15,994,941 |
| **Mean** | | **1.0766** | **-0.0130** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0381 BPB**.

## Key Techniques

### Training (8 techniques)

1. **4096-Vocab + MLP 4x + WD 0.090** — PR #1218 @clarkkev, PR #1285 @dexhunter
2. **Depth Recurrence (layers 4,5)** — PR #1204 @msisovic, PR #1260 @dexhunter
3. **Parallel Residuals (from layer 7)** — PR #1204 @msisovic, PR #1289 @MatoTeziTanka
4. **MuonEq-R** — arXiv:2603.28254, PR #1260 @dexhunter
5. **QK-Gain 5.0** — PR #1217 @bigbag
6. **Full GPTQ int6 + Brotli + LZMA Compressed Wrapper**

### Evaluation: Causal SLOT (context-only delta optimization)

Per-batch additive delta vector (dim=512) optimized with AdamW (lr=0.008, 16 steps) on **context-only positions** during sliding-window eval. Only already-scored tokens contribute to the optimization loss. Delta is re-initialized to zeros for each batch. Model weights completely frozen.

This is provably causal: the delta at position t depends only on tokens x_1,...,x_{t-stride} which have all been previously scored. New positions (last stride=64 tokens per window) are scored with the context-adapted delta but do not influence its optimization.

Source: arXiv:2505.12392v2, PR #1306 @resouer (causal variant), PR #1176 @bigbag (SLOT concept).

## Compliance

- **Condition 1** (causal): delta optimized on context-only positions (already scored). New tokens excluded from optimization loss.
- **Condition 2** (full distribution): standard softmax over full 4096-token vocabulary
- **Condition 3** (score-before-update): new tokens scored AFTER delta optimization on context. Delta does not use new token information.
- **Condition 4** (single pass): single left-to-right sliding window, no rescoring
- Model weights frozen during eval — only delta vector optimized per-batch
- GPTQ calibration within training budget
- Total eval: ~520s (sliding ~76s + SLOT ~444s), within 600s budget

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
SLOT_ENABLED=1 SLOT_LR=0.008 SLOT_STEPS=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1218 @clarkkev, PR #1285 @dexhunter, PR #1204 @msisovic, PR #1289 @MatoTeziTanka, PR #1260 @dexhunter, PR #1019 @abaybektursun, PR #1287 @dentity007, PR #1217 @bigbag, PR #493 @parinzee, PR #1306 @resouer (causal SLOT), PR #1176 @bigbag (SLOT concept)
