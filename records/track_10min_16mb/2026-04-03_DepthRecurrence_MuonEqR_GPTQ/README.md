# Record: Depth Recurrence + MuonEq-R + AR Self-Gen GPTQ — val_bpb 1.1104 (3-seed mean)

**val_bpb = 1.1104** (3-seed mean, std 0.0009) | **~15.97 MB** | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | EMA bpb | **Sliding bpb** | val_loss (nats) | Artifact |
|------|----------|-------|---------|-----------------|-----------------|----------|
| 42   | 96.7ms   | 6,204 | 1.1300  | **1.1105**      | 1.8751          | 15,974,737 |
| 314  | 96.7ms   | 6,205 | 1.1288  | **1.1094**      | 1.8731          | 15,972,993 |
| 999  | 96.7ms   | 6,205 | 1.1292  | **1.1112**      | 1.8762          | 15,969,221 |
| **Mean** | **96.7ms** | **6,205** | **1.1293** | **1.1104** | **1.8748** | |

SOTA (PR #1019, 3-seed mean): **1.88218 nats**. This run: **1.87481 nats**. Delta: **-0.00737 nats**. Clears the 0.005-nat threshold (Welch t=-7.73, df=2.59).

## Key Innovations

### 1. Depth Recurrence (layers 4,5 repeated)

Layers 4 and 5 (the U-Net hinge point) are executed twice during the forward pass using the same physical parameter banks and block modules. This creates a virtual 13-layer network from an 11-layer parameter budget — zero extra parameters, ~0.003 BPB improvement.

Recurrence activates at step 3000 (after the model has learned basic representations), with the MLP weights primed from the base layers. Step time increases from ~87ms to ~97ms (13 virtual layers vs 11), but the improved per-step learning outweighs the step count reduction.

Lineage: PR #1204 by @msisovic (concept), PR #1260 by @dexhunter (implementation on PR #1218 stack).

### 2. MuonEq-R (Row-Normalized Muon)

Before Newton-Schulz orthogonalization, gradient matrices are row-normalized:

```python
row_norms = update.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
update = update / row_norms.to(update.dtype)
```

This equalizes row norms so Newton-Schulz operates on better-conditioned matrices. Zero additional bytes, ~0.001 BPB improvement. Source: arXiv:2603.28254, PR #1260 by @dexhunter.

### 3. AR Self-Generated Full Hessian GPTQ (from PR #1019)

After training, the model autoregressively generates 64 sequences of 2048 tokens (temperature=0.8) for GPTQ calibration. Full Hessian GPTQ with Cholesky error compensation. No val data, no train data accessed during quantization. Source: PR #1019 by @abaybektursun.

## Architecture (PR #1019 stack + modifications)

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 physical (13 virtual with recurrence) | **This work** |
| Dimensions | 512d, 8 GQA / 4 KV heads | Baseline |
| MLP | 3x (1536), LeakyReLU(0.5)^2 | PR #493 @parinzee |
| Attention | XSA on all 11 layers | PR #1019 @abaybektursun |
| BigramHash | 3072 x 112 | PR #1019 @abaybektursun |
| RoPE | Partial (16/64 dims) | PR #315 @jfprincz |
| LN Scale | 1/sqrt(layer+1) | PR #315 @jfprincz |
| VE128 | Layers 9-10 | PR #374 @unnir |
| SmearGate | Position-mixing gate | PR #65 @aquariouseworkman |
| U-Net skips | Encoder-decoder connections | PR #289 |
| Weight avg | EMA(0.997) + SWA | PR #401 @newjordan |
| Optimizer | **MuonEq-R** (Parallel Muon + row-norm) | **This work** (Muon: PR #399 @abaybektursun) |
| Depth Recurrence | **Layers 4,5 repeated** | **This work** (concept: PR #1204 @msisovic) |
| Quantization | Full Hessian GPTQ int6 (AR self-gen) | PR #1019 @abaybektursun |
| Compression | LZMA preset=9 | PR #160 @ChaseWNorton |
| Warmdown | 4000 iterations | PR #364 @shikhar1729 |
| Late QAT | STE at LR scale < 0.15 | PR #286 @chris-buckley |
| Selective pruning | +/-1 by reconstruction error | PR #609 @saml212 |
| Flash Attention 3 | Hopper kernels | PR #122 @mtybadger |

## Training

- MuonEq-R: lr=0.025, momentum 0.92->0.99/1500 steps, WD=0.04, Newton-Schulz 5 steps
- Adam for embeddings (lr=0.035) and scalars (lr=0.025)
- Batch 786,432 tokens, seq_len 2048, warmdown 4000 iters
- Depth recurrence activates at step 3000
- Late QAT via STE (final 15% wallclock)
- Gradient clipping 0.3

## Quantization Pipeline

| Stage | BPB (seed 314) |
|-------|----------------|
| Pre-quant (post-EMA) | 1.1288 |
| Post-GPTQ int6 roundtrip | 1.1331 |
| Post-GPTQ sliding window | **1.1094** |

## Compliance

- No TTT, no SLOT, no n-gram cache, no eval-time adaptation
- AR self-generated GPTQ calibration (no external data during quantization)
- All seeds within 600s training, <16MB artifact
- Fully legal under all four conditions (Issue #1017)

## Run Command

```bash
SEED=42 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
RECUR_LAYERS=4,5 RECUR_START_STEP=3000 TARGET_MB=15.9 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Base model + GPTQ + XSA-all + BigramHash**: PR #1019 by @abaybektursun
- **Depth Recurrence concept**: PR #1204 by @msisovic
- **MuonEq-R + Depth Recurrence implementation**: PR #1260 by @dexhunter
- **Parallel Muon**: PR #399 by @abaybektursun
- **LeakyReLU^2**: PR #493 by @parinzee, PR #518 by @sofiabod
- **LN Scale + Partial RoPE**: PR #315 by @jfprincz
