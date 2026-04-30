## Record: MuonEq-R + 3-Layer Recurrence + WD=0.095 + MLR=0.022 + All-Int6 (val_bpb: 1.0900)

**val_bpb = 1.0900** (3-seed mean, std 0.0005) | **2.5078 nats** | **~15.96 MB** | 8xH100 SXM, 590s train + ~81s eval | No TTT

Built on [PR #1218](https://github.com/openai/parameter-golf/pull/1218) by @clarkkev.

Previous: [PR #1218](https://github.com/openai/parameter-golf/pull/1218) (1.0979) -> [PR #1285](https://github.com/openai/parameter-golf/pull/1285) (1.0912) -> this (1.0900)

### Changes from PR #1285

| | PR #1285 | This |
|---|---|---|
| val_bpb | 1.09124 | **1.08995** |
| Recurrence | Layers 4,5 (2-layer) | **Layers 3,4,5 (3-layer)** |
| Weight decay | 0.090 | **0.095** |
| Matrix LR | 0.020 | **0.022** |
| Everything else | Same | Same |

### Key Innovations

1. **3-Layer Depth Recurrence** — Layers 3, 4, and 5 are repeated (RECUR_LAYERS=3,4,5), creating 14 virtual layers from 11 physical. MLP weights fully shared. ~0.0005 BPP improvement over 2-layer recurrence.

2. **WD=0.095 + MLR=0.022 Synergy** — Higher weight decay (0.095 vs 0.090) compresses weights better, while slightly higher matrix LR (0.022 vs 0.020) recovers quality. The net effect is better BPP at the same artifact budget. The 3-layer recurrence needs WD≥0.093 to fit all-int6 under 16MB.

3. **MuonEq-R + All-Int6 GPTQ** — Row-normalized Muon optimizer with all 66 layers at int6 precision (carried from PR #1285).

### Configuration

```bash
NCCL_NET=Socket DATA_DIR=./data SEED=42 \
MIXED_QUANT=1 N_INT6_LAYERS=66 \
RECUR_LAYERS=3,4,5 MUON_WD=0.095 EMBED_WD=0.095 \
MATRIX_LR=0.022 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, no TTT)

### Core Results

| Seed | Steps | ms/step | Post-EMA BPB | Sliding BPB | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-------------|-----------------|----------|
| 42 | 5,540 | 106.5 | 1.0991 | 1.0898 | 2.50733 | 15,961,029 |
| 0 | 5,536 | 106.6 | 1.0993 | 1.0895 | 2.50672 | 15,955,962 |
| 7 | 5,538 | 106.6 | 1.0999 | 1.0905 | 2.50901 | 15,964,018 |
| **Mean** | **5,538** | **106.6** | **1.0994** | **1.0900** | **2.50769** | **15,960,336** |

### Rule Compliance

- No TTT, no SLOT, no eval-time adaptation
- Artifact < 16,000,000 bytes for ALL seeds (max: 15,964,018)
- Train < 600s, eval < 600s on 8xH100 SXM

### Run Command (3-seed loop)

```bash
for SEED in 42 0 7; do
  NCCL_NET=Socket DATA_DIR=./data SEED=$SEED \
  MIXED_QUANT=1 N_INT6_LAYERS=66 \
  RECUR_LAYERS=3,4,5 MUON_WD=0.095 EMBED_WD=0.095 \
  MATRIX_LR=0.022 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  2>&1 | tee train_seed${SEED}.log
done
```

### Credits

- @clarkkev for PR #1218 (4096-Vocab + high-WD architecture)
- @abaybektursun for PR #1019 (GPTQ + XSA + BigramHash baseline)
- @msisovic for PR #1204 (depth recurrence concept)
- @dexhunter for PR #1285 (WD-quantization synergy discovery)

### Included Files

- `train_gpt.py` — full training + quantization + evaluation (20,302 bytes, self-extracting)
- `train_seed42.log`, `train_seed0.log`, `train_seed7.log`
- `submission.json`
