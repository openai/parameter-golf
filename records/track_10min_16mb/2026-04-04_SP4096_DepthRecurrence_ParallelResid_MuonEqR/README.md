# Record: SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R — val_bpb 1.0897 (3-seed mean)

**val_bpb = 1.0897** (3-seed mean, std 0.0003) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | **Sliding BPB** | Artifact |
|------|-----------------|----------|
| 42   | **1.0894**      | 15,999,533 |
| 314  | **1.0898**      | 15,992,752 |
| 999  | **1.0899**      | 15,988,473 |
| **Mean** | **1.0897** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0250 BPB**.

## Key Techniques

1. **4096-Vocab + MLP 4x + WD 0.090** — PR #1218 @clarkkev, PR #1285 @dexhunter
2. **Depth Recurrence (layers 4,5)** — virtual 13-layer network from 11 physical layers, zero extra params. PR #1204 @msisovic, PR #1260 @dexhunter
3. **Parallel Residuals (from layer 7)** — separate attention/MLP residual lanes with learned merge. PR #1204 @msisovic, PR #1289 @MatoTeziTanka
4. **MuonEq-R** — row-normalized Muon, arXiv:2603.28254. PR #1260 @dexhunter
5. **QK-Gain 5.0** — PR #1217 @bigbag
6. **Full GPTQ int6 + Brotli + LZMA Compressed Wrapper** (~24KB code)

## Compliance (Track A — Fixed Predictor)

- No TTT — no weight updates during evaluation
- No SLOT — no eval-time delta optimization
- No n-gram cache — no eval-time statistics accumulation
- No eval-time adaptation of any kind — model weights completely frozen after training
- GPTQ calibration uses training data within the training time budget
- Standard autoregressive sliding-window eval (stride=64)
- **Condition 1** (causal): standard causal attention, no future token access
- **Condition 2** (full distribution): standard softmax over full 4096-token vocab
- **Condition 3** (score-before-update): no updates during eval
- **Condition 4** (single pass): single sliding-window pass, no rescoring

All improvements are purely train-time architectural/optimizer changes. The eval procedure is standard sliding-window.

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1218 @clarkkev, PR #1285 @dexhunter, PR #1204 @msisovic, PR #1289 @MatoTeziTanka, PR #1260 @dexhunter, PR #1019 @abaybektursun, PR #1287 @dentity007, PR #1217 @bigbag, PR #493 @parinzee
