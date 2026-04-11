## Record: MuonEq-R + Depth Recurrence + Mixed Int5/Int6 GPTQ (val_bpb: 1.0929)

**val_bpb = 1.0929** (3-seed mean, std 0.0009) | **2.5145 nats** | **~15.96 MB** | 8xH100 SXM, 600s train + ~83s eval | No TTT

Built on [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by @clarkkev (4096-Vocab + 4.0-MLP-mult + 0.085-WD).

Previous: [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (1.1147) -> [PR #1218](https://github.com/openai/parameter-golf/pull/1218) (1.0979) -> this (1.0929)

### Changes from PR #1218

| | PR #1218 | This |
|---|---|---|
| val_bpb | 1.09785 | **1.09290** |
| Optimizer | Muon | **MuonEq-R** (row-norm before NS5) |
| Depth recurrence | None | **Layers 4,5 repeated** (RECUR_LAYERS=4,5) |
| Recurrence MLP sharing | N/A | **Fully shared** (REPEAT_UNTIE_MLP=none) |
| Mixed quantization | No | **Yes** (60 int6 + 6 int5 via Hessian sensitivity) |
| Recurrence activation | N/A | Step 3000 with 20-step warmup |
| Everything else | Same | Same |

### What's New

1. **MuonEq-R** — Row-normalizes gradient matrices before Newton-Schulz orthogonalization in the Muon optimizer. Improves conditioning of the NS5 iteration for non-square weight matrices. Zero-byte cost, ~0.001 BPB improvement.

2. **Depth Recurrence** — Layers 4 and 5 are repeated once after the initial forward pass (virtual layers 12-13 on top of 11 physical layers). MLP weights are fully shared during recurrence (REPEAT_UNTIE_MLP=none), so this adds zero extra parameters. Activated at step 3000 with a 20-step linear warmup. ~0.003 BPB improvement.

3. **Mixed Int5/Int6 GPTQ** — Hessian-based sensitivity ranking determines which layers get int6 (clip_range=31) vs int5 (clip_range=15). The 60 most sensitive layers keep int6 precision; the 6 least sensitive get int5 to save artifact bytes. Combined with full GPTQ and brotli-11 compression.

### Carried from PR #1218

- 4096 SentencePiece BPE vocabulary
- 4.0x MLP multiplier with sigmoid-gated activation
- Weight decay 0.085 (high WD for better compression)
- Full Hessian GPTQ quantization
- XSA-all-11 attention pattern
- BigramHash embedding (2816x160)
- Sigmoid-gated skip connections
- Soft-round QAT
- Split-LR training
- Brotli-11 compression with byte shuffle
- EMA (decay 0.997)

### Configuration

```bash
NCCL_NET=Socket \
DATA_DIR=./data \
SEED=1337 \
MIXED_QUANT=1 \
N_INT6_LAYERS=60 \
RECUR_LAYERS=4,5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, no TTT)

### Core Results

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-------------|-----------------|----------|
| 1337 | 5,541 | 106.5 | 1.1000 | 1.0939 | 2.51667 | 15,933,457 |
| 42 | 5,530 | 106.7 | 1.0987 | 1.0922 | 2.51279 | 15,981,324 |
| 0 | 5,543 | 106.5 | 1.0988 | 1.0927 | 2.51394 | 15,960,050 |
| **Mean** | **5,538** | **106.6** | **1.0992** | **1.0929** | **2.51447** | **15,958,277** |

### Supplemental Diagnostics

| Seed | Post-EMA BPB | Roundtrip BPB | Sliding BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|--------------|---------------|-------------|-----------------|-----------|------------------|------------|-----------|
| 1337 | 1.1000 | 1.1122 | 1.0939 | 2.51667 | 21,084 | 15,933,457 | 590s | 83s |
| 42 | 1.0987 | 1.1106 | 1.0922 | 2.51279 | 21,084 | 15,981,324 | 590s | 83s |
| 0 | 1.0988 | 1.1113 | 1.0927 | 2.51394 | 21,084 | 15,960,050 | 590s | 83s |
| **Mean** | **1.0992** | **1.1114** | **1.0929** | **2.51447** | **21,084** | **15,958,277** | **590s** | **83s** |

### Rule Compliance

- No TTT (no test-time training or adaptation)
- No SLOT (no scored-position lookup table)
- No validation data during training
- No training data during evaluation
- Artifact < 16,000,000 bytes for ALL seeds (max: 15,981,324)
- Train < 600s on 8xH100 SXM (590s)
- Eval < 600s on 8xH100 SXM (~83s)

### Architecture

- 11 layers + 2 virtual (depth recurrence on layers 4,5)
- d_model = 512, MLP 4x (2048), 4 heads
- 4096 SentencePiece BPE vocabulary
- BigramHash(2816x160) token embedding
- Sigmoid-gated skip connections with soft-round QAT
- MuonEq-R optimizer with row normalization
- Full Hessian GPTQ (int6) with mixed int5/int6 via sensitivity ranking

### Requirements

- PyTorch 2.9.1+cu128
- flash-attn 2.8.3
- sentencepiece
- brotli
- 8x H100 SXM 80GB

### Run Command (3-seed loop)

```bash
for SEED in 1337 42 0; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  SEED=$SEED \
  MIXED_QUANT=1 \
  N_INT6_LAYERS=60 \
  RECUR_LAYERS=4,5 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee train_seed${SEED}.log
done
```

### Lineage

PR #1019 (ValCalib + GPTQ + XSA + BigramHash, 1.1147) -> PR #1218 (4096-Vocab + MLP 4x + WD 0.085, 1.0979) -> this (MuonEq-R + Depth Recurrence + Mixed Quant, 1.0929)

### Credits

- @clarkkev for PR #1218 (4096-Vocab + high-WD architecture — the foundation)
- @abaybektursun for PR #1019 (GPTQ + XSA + BigramHash baseline)
- @msisovic for PR #1204 (depth recurrence concept)
- MuonEq-R inspired by equalized gradient normalization literature

### Included Files

- `train_gpt.py` — full training + quantization + evaluation script (21,084 bytes, self-extracting)
- `train_seed1337.log`, `train_seed42.log`, `train_seed0.log` — all seed logs
- `submission.json` — leaderboard metadata
