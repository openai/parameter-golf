# Record: SP4096 + Depth Recurrence + MuonEq-R + Full GPTQ — val_bpb 1.0940 (3-seed mean)

**val_bpb = 1.0940** (3-seed mean, std 0.0005) | **~15.96 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | steps | Pre-quant BPB | **Sliding BPB** | Artifact |
|------|-------|---------------|-----------------|----------|
| 42   | 5,415 | 1.0997        | **1.0942**      | 15,960,147 |
| 314  | 5,415 | 1.0995        | **1.0934**      | 15,963,424 |
| 999  | 5,420 | 1.0996        | **1.0942**      | 15,958,655 |
| **Mean** | | | **1.0940** | |

Merged SOTA (PR #1019): **1.1147 BPB**. This run: **1.0940 BPB**. Delta: **-0.0208 BPB** (Welch t=-61.9). Clears the 0.005-nat threshold by ~3x.

## Changes from Merged SOTA (PR #1019)

This submission combines the PR #1218 4096-vocab architecture with depth recurrence, MuonEq-R, and higher weight decay for better quantization.

### 1. 4096-Vocab + MLP 4x + WD 0.090

Switched from sp1024 to sp4096 tokenizer (4096 BPE tokens vs 1024). Wider MLP (4x expansion vs 3x). Higher weight decay (0.090 vs 0.04) produces smaller weights that compress ~5% better with brotli, allowing all 66 quantized layers at int6 precision.

Source: PR #1218 by @clarkkev (4096-vocab + MLP 4x + WD 0.085), PR #1285 by @dexhunter (WD 0.090 + all-int6).

### 2. Depth Recurrence (layers 4,5 repeated)

Layers 4 and 5 (U-Net hinge point) execute twice during the forward pass using the same physical parameter banks. Virtual 13-layer network from 11-layer parameter budget, zero extra parameters. Activates at step 3000.

Source: PR #1204 by @msisovic (concept), PR #1260 by @dexhunter (implementation).

### 3. MuonEq-R (Row-Normalized Muon)

Row-normalizes gradient matrices before Newton-Schulz orthogonalization for better-conditioned optimization. Zero cost.

Source: arXiv:2603.28254, PR #1260 by @dexhunter.

### 4. Full GPTQ int6 + Brotli + Selective Pruning

Full Hessian GPTQ with training-data calibration. Brotli-11 compression with byte-shuffle. Selective +-1 pruning by reconstruction error to fit under 16MB.

Source: PR #1019 by @abaybektursun (GPTQ), PR #1218 by @clarkkev (brotli + byte-shuffle).

## Architecture

| Component | Setting |
|-----------|---------|
| Vocab | 4096 (sp4096 BPE) |
| Layers | 11 physical (13 virtual with recurrence) |
| Dimensions | 512d, 8H / 4KV (GQA) |
| MLP | 4x (2048), LeakyReLU(0.5)^2 |
| XSA | All 11 layers |
| QK Gain | 4.0 |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Skip gates | Sigmoid-gated U-Net |
| Weight avg | EMA(0.997) |
| Optimizer | MuonEq-R (lr=0.02, WD=0.090) |
| Quantization | Full GPTQ int6 + brotli-11 + byte-shuffle |
| Warmdown | 66.7% of steps |

## Training

- MuonEq-R: lr=0.02, momentum 0.92->0.99/1500 steps, WD=0.090
- Adam for embeddings (lr=0.03) and scalars (lr=0.02)
- Batch 786,432 tokens, seq_len 2048
- Depth recurrence activates at step 3000
- ~5415 steps in 590s (~109ms/step with recurrence)

## Compliance

- No TTT, no SLOT, no n-gram cache, no eval-time adaptation
- GPTQ calibration uses training data within the training time budget
- All seeds within 600s training, <16MB artifact
- Fully legal under all four conditions (Issue #1017)

## Reproduction

```bash
# Download sp4096 data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --skip-manifest

# Train
SEED=42 RECUR_LAYERS=4,5 RECUR_START_STEP=3000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **4096-Vocab + MLP 4x + WD 0.085 + Brotli**: PR #1218 by @clarkkev
- **WD 0.090 + All-Int6**: PR #1285 by @dexhunter
- **Depth Recurrence concept**: PR #1204 by @msisovic
- **MuonEq-R + Depth Recurrence implementation**: PR #1260 by @dexhunter
- **Full GPTQ + XSA-all**: PR #1019 by @abaybektursun
- **Base architecture**: PR #1287 by @dentity007
- **LeakyReLU^2**: PR #493 by @parinzee
- **LN Scale + Partial RoPE**: PR #315 by @jfprincz
