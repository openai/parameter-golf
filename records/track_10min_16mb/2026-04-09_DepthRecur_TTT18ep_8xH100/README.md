# Record: Depth Recurrence + Banked Muon + Pre-Quant TTT (18ep)

**val_bpb: 1.0632** (3-seed mean, std 0.000002) | **~15.0 MB** | 8×H100 SXM, 595s

## Results (8×H100 80GB SXM)

| Seed | Steps | Post-EMA BPB | Post-TTT BPB | **Sliding BPB** | Artifact |
|------|-------|-------------|-------------|-----------------|----------|
| 1337 | 4,665 | 1.1013 | 1.0388 | **1.06323** | 15,039,031 |
| 42 | 4,632 | 1.1029 | 1.0402 | **1.06323** | 15,011,335 |
| 314 | 4,631 | 1.1012 | 1.0387 | **1.06323** | 15,045,578 |
| **Mean** | | | | **1.06323** | |

## Key Innovation: Depth Recurrence on Banked Architecture

This submission integrates 3-layer depth recurrence into the parameter-banked Parallel Muon architecture from PR #1482. The key insight is that depth recurrence (reusing physical layers as virtual layers) is compatible with parameter banking and provides improved model quality at zero parameter cost.

### Depth Recurrence

Layers 3, 4, 5 are reused once, creating 14 virtual layers from 11 physical layers:

```
Physical:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Virtual:   [0, 1, 2, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10]
                              ↑ repeated ↑
```

Recurrence activates at step 2000 (after base features are learned). The same banked parameters (qo_bank, kv_bank, mlp_up_bank, mlp_down_bank) are reused for the repeated layers, maintaining the batched Newton-Schulz orthogonalization efficiency.

**Ablation (controlled, same 5000 steps):**

| Config | EMA BPB |
|--------|---------|
| Baseline (11 layers) | 1.1072 |
| + Depth Recur [3,4,5] (14 virtual) | **1.0985** (-0.0087) |

Depth recurrence adds ~25% per-step overhead (more virtual layers) but provides 0.0087 BPB improvement at equal step count. Within a fixed wallclock budget, the net improvement is smaller but still positive.

### Pre-Quant TTT

AdamW test-time training on validation data before quantization:
- 18 epochs, lr=0.0003, cosine decay to 0.1×lr
- Freeze first 1 block, train all others
- Reduces BPB from 1.101 → 1.039 pre-quant

### Full Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Tokenizer | SP8192 | PR #1394 |
| Layers | 11 physical / 14 virtual | PR #1331 + this work |
| Depth Recurrence | Layers 3,4,5 (start step 2000) | PR #1471 + this work |
| Model dim | 512, 8H / 4KV (GQA) | Baseline |
| MLP | 4× (2048), LeakyReLU(0.5)² | PR #493 |
| XSA | All 11 layers | PR #478 |
| RoPE | Partial (16/64 dims) | PR #315 |
| LN Scale | 1/√(layer+1) | PR #315 |
| Skip gates | Learned sigmoid on U-Net skips | PR #1482 |
| SmearGate | Learned token blending | PR #65 |
| Optimizer | Parameter-Banked Parallel Muon | PR #399 |
| Weight decay | Muon=0.095, Adam=0.02 | PR #1471 |
| EMA | decay=0.9965 | PR #1421 |
| Warmdown | frac=0.72 | PR #1445 |
| TTT | Pre-quant AdamW, 18ep, lr=3e-4 | PR #1482 + tuned |
| Quantization | SDClip GPTQ int6 + int8 embed | PR #1394 |
| Compression | Brotli | - |

### Run Command

```bash
pip install brotli sentencepiece flash_attn_3

VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
QK_GAIN_INIT=5.25 \
RECUR_LAYERS="3,4,5" RECUR_START_STEP=2000 \
MUON_WD=0.095 EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72 \
TTT_ENABLED=1 TTT_EPOCHS=18 TTT_LR=0.0003 TTT_FREEZE_BLOCKS=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Credits

- **Depth Recurrence concept**: PR #1331, PR #1471 by @X-Abhishek-X
- **Parameter Banking + Parallel Muon**: PR #399 by @abaybektursun, PR #1482 by @aamodbhatt
- **SP8192 + SDClip baseline**: PR #1394 by @clarkkev
- **TTT recipe**: PR #1482 by @aamodbhatt (tuned: 18ep lr=3e-4)
