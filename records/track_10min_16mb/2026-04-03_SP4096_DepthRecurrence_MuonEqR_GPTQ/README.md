# Record: SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R + Full GPTQ — val_bpb 1.0904 (3-seed mean)

**val_bpb = 1.0904** (3-seed mean, std 0.0016) | **~15.98 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | **Sliding BPB** | Artifact |
|------|-------|-----------------|----------|
| 42   | 5,279 | **1.0923**      | 15,965,928 |
| 314  | 5,279 | **1.0894**      | 15,997,318 |
| 999  | 5,279 | **1.0896**      | 15,990,607 |
| **Mean** | | **1.0904** | |

Merged SOTA (PR #1019): **1.1147 BPB**. Delta: **-0.0243 BPB**.

## Changes from Merged SOTA

Five orthogonal improvements:

### 1. 4096-Vocab + MLP 4x + WD 0.090
sp4096 tokenizer, wider MLP (4x vs 3x), higher weight decay for better quantization compression. Source: PR #1218 @clarkkev, PR #1285 @dexhunter.

### 2. Depth Recurrence (layers 4,5)
Virtual 13-layer network from 11 physical layers, zero extra params. Activates step 3000. Source: PR #1204 @msisovic, PR #1260 @dexhunter.

### 3. Parallel Residuals (from layer 7)
From layer 7 onward, attention and MLP operate on separate residual lanes. Attention reads from lane 0, MLP reads from lane 1. A learned `lane_merge` scalar blends the lanes after the final layer. Source: PR #1204 @msisovic, PR #1289 @MatoTeziTanka.

### 4. MuonEq-R
Row-normalized Muon optimizer (arXiv:2603.28254). Source: PR #1260 @dexhunter.

### 5. Full GPTQ int6 + Brotli + Compressed Wrapper
All 66 layers at int6, brotli-11 byte-shuffle, LZMA self-extracting code wrapper (~25KB). Source: PR #1019 @abaybektursun, PR #1218 @clarkkev.

## Architecture

11L/512d/8H/4KV, MLP 4x LeakyReLU(0.5)^2, XSA all, QK-Gain 4.0, Partial RoPE 16d, LN Scale, VE128 (9-10), sigmoid-gated U-Net skips, EMA(0.997), MuonEq-R (lr=0.02, WD=0.090), depth recurrence layers 4,5, parallel residuals from layer 7, full GPTQ int6 + brotli-11.

## Compliance

- No TTT, no SLOT, no n-gram cache, no eval-time adaptation
- GPTQ calibration within training budget
- All four conditions from Issue #1017 satisfied

## Reproduction

```bash
pip install brotli
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 PARALLEL_START_LAYER=7 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1218 @clarkkev (4096-vocab + MLP 4x + brotli)
- PR #1285 @dexhunter (WD 0.090 + all-int6)
- PR #1204 @msisovic (parallel residuals + depth recurrence)
- PR #1289 @MatoTeziTanka (parallel residuals integration)
- PR #1260 @dexhunter (MuonEq-R + depth recurrence impl)
- PR #1019 @abaybektursun (GPTQ + XSA-all)
- PR #1287 @dentity007 (base code)
- PR #493 @parinzee (LeakyReLU^2)
